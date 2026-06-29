import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from glf_mobilevit.models import create_model, list_models
from glf_mobilevit.utils import count_parameters, estimate_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Random-input smoke test.")
    parser.add_argument("--model", default="glf_tiny", choices=list_models())
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--attention", default="cbs", choices=["cbs", "mha", "none"])
    parser.add_argument("--no-large-kernel", action="store_true")
    parser.add_argument("--no-grn", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        args.model,
        num_classes=args.classes,
        attention=args.attention,
        use_large_kernel=not args.no_large_kernel,
        use_grn=not args.no_grn,
        use_gate=not args.no_gate,
    ).to(device)
    model.eval()
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        logits, aux = model(x, return_aux=True)
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"params={count_parameters(model):,}")
    print(f"logits={tuple(logits.shape)}")
    print({key: tuple(value.shape) for key, value in aux.items()})
    if not args.skip_flops:
        flops = estimate_flops(model, (1, 3, args.img_size, args.img_size), device=device)
        print(f"flops_estimate={flops:,}")


if __name__ == "__main__":
    main()
