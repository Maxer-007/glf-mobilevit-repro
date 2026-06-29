import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (confidence * nll + self.smoothing * smooth).mean()


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


class MixupCutmix:
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        label_smoothing: float = 0.1,
        switch_prob: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.switch_prob = switch_prob

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0 or self.cutmix_alpha > 0

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        y = torch.full(
            (target.shape[0], self.num_classes),
            off,
            device=target.device,
            dtype=torch.float32,
        )
        y.scatter_(1, target.unsqueeze(1), on)
        return y

    @staticmethod
    def _rand_bbox(
        size: Tuple[int, int, int, int],
        lam: float,
        device: torch.device,
    ) -> Tuple[int, int, int, int]:
        _, _, height, width = size
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = torch.randint(width, (1,), device=device).item()
        cy = torch.randint(height, (1,), device=device).item()
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return images, self._one_hot(targets)

        batch = images.shape[0]
        perm = torch.randperm(batch, device=images.device)
        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or torch.rand(1, device=images.device).item() < self.switch_prob
        )

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
            x1, y1, x2, y2 = self._rand_bbox(images.shape, lam, images.device)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.shape[-1] * images.shape[-2]))
        else:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            mixed = images * lam + images[perm] * (1.0 - lam)

        y1 = self._one_hot(targets)
        y2 = self._one_hot(targets[perm])
        return mixed, y1 * lam + y2 * (1.0 - lam)


class GramConsistencyLoss(nn.Module):
    """DINO-style view consistency on channel Gram matrices."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def gram(x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        feat = x.flatten(2)
        feat = F.normalize(feat, dim=-1)
        return feat @ feat.transpose(1, 2) / max(height * width, 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_gram = self.gram(source)
        target_gram = self.gram(target).detach()
        return F.mse_loss(source_gram, target_gram)
