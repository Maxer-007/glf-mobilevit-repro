import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from glf_mobilevit.data import build_eval_loader
from glf_mobilevit.models import create_model, list_models
from glf_mobilevit.utils import (
    AverageMeter,
    accuracy,
    benchmark_throughput,
    build_confusion_matrix,
    count_parameters,
    estimate_flops,
    load_checkpoint,
    save_confusion_matrix,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GLF-MobileViT.")
    parser.add_argument("--model", default="glf_base", choices=list_models())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attention", default="cbs", choices=["cbs", "mha", "none"])
    parser.add_argument("--no-large-kernel", action="store_true")
    parser.add_argument("--no-grn", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
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
        confusion += build_confusion_matrix(logits.cpu(), targets.cpu(), num_classes)
    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
        "confusion": confusion,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, num_classes = build_eval_loader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        download=args.download,
    )
    model = create_model(
        args.model,
        num_classes=num_classes,
        attention=args.attention,
        use_large_kernel=not args.no_large_kernel,
        use_grn=not args.no_grn,
        use_gate=not args.no_gate,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    load_checkpoint(args.checkpoint, model, map_location=device)

    flops = None
    if not args.skip_flops:
        flops = estimate_flops(model, (1, 3, args.img_size, args.img_size), device=device)
    result = evaluate(model, loader, device, nn.CrossEntropyLoss().to(device), num_classes, args)
    throughput = None
    if args.benchmark:
        throughput = benchmark_throughput(model, device, args.img_size, args.batch_size, amp=args.amp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(out_dir / "confusion_matrix.csv", result["confusion"])
    metrics = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "dataset": args.dataset,
        "img_size": args.img_size,
        "loss": result["loss"],
        "top1": result["top1"],
        "top5": result["top5"],
        "parameters": count_parameters(model),
        "flops_estimate": flops,
        "throughput_img_s": throughput,
        "device": str(device),
    }
    write_json(out_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
