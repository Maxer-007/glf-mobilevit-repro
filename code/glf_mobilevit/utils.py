import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / target.numel()))
    return result


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int, int], device: torch.device) -> int:
    hooks = []
    flops = {"total": 0}

    def conv_hook(module: nn.Conv2d, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        out_h, out_w = output.shape[-2:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops["total"] += int(batch * out_h * out_w * module.out_channels * kernel_ops)

    def linear_hook(module: nn.Linear, inputs, output):
        x = inputs[0]
        flops["total"] += int(np.prod(x.shape[:-1]) * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_size, device=device)
        model(dummy)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return flops["total"]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc1: float,
    args,
    scaler=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_acc1": best_acc1,
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


class CSVLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_confusion_matrix(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    indices = target * num_classes + pred
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def save_confusion_matrix(path: Path, matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix.cpu().numpy().astype(np.int64), fmt="%d", delimiter=",")


def benchmark_throughput(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    batch_size: int,
    amp: bool = True,
    warmup: int = 20,
    steps: int = 50,
) -> float:
    model.eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.time() - start
    return batch_size * steps / max(elapsed, 1e-9)
