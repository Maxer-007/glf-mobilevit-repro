import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from glf_mobilevit.data import build_dataloaders
from glf_mobilevit.losses import (
    GramConsistencyLoss,
    LabelSmoothingCrossEntropy,
    MixupCutmix,
    SoftTargetCrossEntropy,
)
from glf_mobilevit.models import create_model, list_models
from glf_mobilevit.utils import (
    AverageMeter,
    CSVLogger,
    accuracy,
    count_parameters,
    estimate_flops,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLF-MobileViT on CIFAR.")
    parser.add_argument("--model", default="glf_base", choices=list_models())
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--drop-path", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention", default="cbs", choices=["cbs", "mha", "none"])
    parser.add_argument("--no-large-kernel", action="store_true")
    parser.add_argument("--no-grn", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--gram-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def make_scheduler(optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float):
    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    args,
    scaler,
    hard_criterion,
    soft_criterion,
    mixup_cutmix,
    gram_criterion,
):
    model.train()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    gram_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for step, (images, targets) in enumerate(loader):
        if isinstance(images, (list, tuple)):
            images, images_view2 = images
            images_view2 = images_view2.to(device, non_blocking=True)
            if args.channels_last:
                images_view2 = images_view2.contiguous(memory_format=torch.channels_last)
        else:
            images_view2 = None
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        gram_target = None
        if args.gram_weight > 0 and images_view2 is not None:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    _, aux_target = model(images_view2, return_aux=True)
                gram_target = aux_target["stage3"]

        if mixup_cutmix.enabled:
            images, soft_targets = mixup_cutmix(images, targets)
            criterion_targets = soft_targets
            criterion = soft_criterion
        else:
            criterion_targets = targets
            criterion = hard_criterion

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            if args.gram_weight > 0:
                logits, aux = model(images, return_aux=True)
            else:
                logits = model(images)
                aux = None
            ce_loss = criterion(logits, criterion_targets)
            loss = ce_loss
            gram_loss = logits.new_tensor(0.0)
            if args.gram_weight > 0 and gram_target is not None and aux is not None:
                gram_loss = gram_criterion(aux["stage3"], gram_target)
                loss = loss + args.gram_weight * gram_loss

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.detach(), targets, topk=(1, min(5, logits.shape[1])))
        losses.update(loss.item(), images.size(0))
        ce_losses.update(ce_loss.item(), images.size(0))
        gram_losses.update(gram_loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:03d} step {step:04d}/{len(loader)} "
                f"loss {losses.avg:.4f} ce {ce_losses.avg:.4f} gram {gram_losses.avg:.4f} "
                f"acc1 {top1.avg:.2f} lr {lr:.3e} time {batch_time.avg:.3f}s"
            )

    return {
        "train_loss": losses.avg,
        "train_ce": ce_losses.avg,
        "train_gram": gram_losses.avg,
        "train_acc1": top1.avg,
        "train_acc5": top5.avg,
    }


@torch.no_grad()
def validate(model, loader, device, criterion, args):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, logits.shape[1])))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
    return {"val_loss": losses.avg, "val_acc1": top1.avg, "val_acc5": top5.avg}


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        print("AMP requested but CUDA is unavailable; running without AMP.")

    run_name = args.run_name or f"{args.model}_{args.dataset}_{args.img_size}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    train_loader, val_loader, num_classes = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        download=args.download,
        random_erasing=args.random_erasing,
        two_views=args.gram_weight > 0,
    )
    model = create_model(
        args.model,
        num_classes=num_classes,
        attention=args.attention,
        use_large_kernel=not args.no_large_kernel,
        use_grn=not args.no_grn,
        use_gate=not args.no_gate,
        drop_path=args.drop_path,
        dropout=args.dropout,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    params = count_parameters(model)
    flops = None
    if not args.skip_flops:
        flops = estimate_flops(model, (1, 3, args.img_size, args.img_size), device=device)
    write_json(
        output_dir / "model_summary.json",
        {
            "model": args.model,
            "num_classes": num_classes,
            "parameters": params,
            "flops_estimate": flops,
            "device": str(device),
        },
    )
    print(f"model={args.model} params={params/1e6:.2f}M flops={(flops or 0)/1e9:.2f}G device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    scaler = make_scaler(args.amp and device.type == "cuda")
    hard_criterion = LabelSmoothingCrossEntropy(args.label_smoothing).to(device)
    soft_criterion = SoftTargetCrossEntropy().to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    mixup_cutmix = MixupCutmix(
        num_classes=num_classes,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        label_smoothing=args.label_smoothing,
    )
    gram_criterion = GramConsistencyLoss().to(device)

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_acc1 = float(checkpoint.get("best_acc1", 0.0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    logger = CSVLogger(
        output_dir / "log.csv",
        [
            "epoch",
            "lr",
            "train_loss",
            "train_ce",
            "train_gram",
            "train_acc1",
            "train_acc5",
            "val_loss",
            "val_acc1",
            "val_acc5",
            "best_acc1",
        ],
    )

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args,
            scaler,
            hard_criterion,
            soft_criterion,
            mixup_cutmix,
            gram_criterion,
        )
        scheduler.step()

        val_stats = {}
        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            val_stats = validate(model, val_loader, device, val_criterion, args)
            best_acc1 = max(best_acc1, val_stats["val_acc1"])
            print(
                f"epoch {epoch:03d} val_loss {val_stats['val_loss']:.4f} "
                f"val_acc1 {val_stats['val_acc1']:.2f} val_acc5 {val_stats['val_acc5']:.2f} "
                f"best {best_acc1:.2f}"
            )
            if val_stats["val_acc1"] >= best_acc1:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc1, args, scaler)

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler, epoch, best_acc1, args, scaler)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "best_acc1": best_acc1,
            **train_stats,
            **val_stats,
        }
        logger.write(row)

    print(f"finished. best_acc1={best_acc1:.2f}. outputs={output_dir}")


if __name__ == "__main__":
    main()
