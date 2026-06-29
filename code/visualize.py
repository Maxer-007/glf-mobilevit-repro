import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from glf_mobilevit.data import build_eval_loader
from glf_mobilevit.models import GatedMobileViTBlock, MultiHeadSelfAttention, SharedKVAttention, create_model, list_models
from glf_mobilevit.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Save gate and attention heatmaps.")
    parser.add_argument("--model", default="glf_base", choices=list_models())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attention", default="cbs", choices=["cbs", "mha", "none"])
    parser.add_argument("--no-large-kernel", action="store_true")
    parser.add_argument("--no-grn", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="visual_outputs")
    return parser.parse_args()


def first_gate(model):
    for module in model.modules():
        if isinstance(module, GatedMobileViTBlock) and module.last_gate is not None:
            return module.last_gate
    return None


def first_attention(model):
    for module in model.modules():
        if isinstance(module, (SharedKVAttention, MultiHeadSelfAttention)) and module.last_attn is not None:
            return module.last_attn
    return None


def save_gate_heatmap(gate: torch.Tensor, out_dir: Path, img_size: int) -> None:
    heat = gate.mean(dim=1, keepdim=True)
    heat = F.interpolate(heat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    save_image(heat.clamp(0, 1), out_dir / "gate_heatmap.png", nrow=min(4, heat.shape[0]))


def save_attention_heatmap(attn: torch.Tensor, out_dir: Path, img_size: int) -> None:
    # attn: B, heads, query_tokens, key_tokens. Mean over heads and queries.
    key_importance = attn.mean(dim=(1, 2))
    num_keys = key_importance.shape[-1]
    grid = int(num_keys ** 0.5)
    if grid * grid != num_keys:
        return
    heat = key_importance.reshape(key_importance.shape[0], 1, grid, grid)
    heat = heat / (heat.amax(dim=(2, 3), keepdim=True) + 1e-6)
    heat = F.interpolate(heat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    save_image(heat.clamp(0, 1), out_dir / "attention_key_heatmap.png", nrow=min(4, heat.shape[0]))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, num_classes = build_eval_loader(
        args.dataset,
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.workers,
        args.download,
    )
    model = create_model(
        args.model,
        num_classes=num_classes,
        attention=args.attention,
        use_large_kernel=not args.no_large_kernel,
        use_grn=not args.no_grn,
        use_gate=not args.no_gate,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        model(images)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = first_gate(model)
    attn = first_attention(model)
    if gate is not None:
        save_gate_heatmap(gate.cpu(), out_dir, args.img_size)
    if attn is not None:
        save_attention_heatmap(attn.cpu(), out_dir, args.img_size)
    print(f"saved visual outputs to {out_dir}")


if __name__ == "__main__":
    main()
