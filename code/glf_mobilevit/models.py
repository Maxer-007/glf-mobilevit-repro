import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(value: int, divisor: int = 8) -> int:
    return int(math.ceil(value / divisor) * divisor)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GRN(nn.Module):
    """ConvNeXt V2 global response normalization for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class UIBConvBlock(nn.Module):
    """Universal-inverted-bottleneck inspired mobile convolution block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expansion: float = 4.0,
        kernel_size: int = 3,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(int(in_ch * expansion))
        self.use_residual = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden, 1),
            ConvBNAct(hidden, hidden, kernel_size, stride=stride, groups=hidden),
            ConvBNAct(hidden, out_ch, 1, act=False),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = x + self.drop_path(out)
        return out


class RepLargeKernelBlock(nn.Module):
    """Training-time multi-branch depthwise large-kernel block.

    The branches are intentionally left explicit for the course project so
    ablations can disable the large-kernel path without extra tooling.
    """

    def __init__(
        self,
        channels: int,
        drop_path: float = 0.0,
        use_large_kernel: bool = True,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        self.use_large_kernel = use_large_kernel
        self.dw13 = (
            ConvBNAct(channels, channels, 13, groups=channels, act=False)
            if use_large_kernel
            else None
        )
        self.dw7 = ConvBNAct(channels, channels, 7, groups=channels, act=False)
        self.dw3 = ConvBNAct(channels, channels, 3, groups=channels, act=False)
        self.pw = nn.Sequential(
            ConvBNAct(channels, channels * 4, 1),
            GRN(channels * 4) if use_grn else nn.Identity(),
            ConvBNAct(channels * 4, channels, 1, act=False),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.dw7(x) + self.dw3(x)
        if self.dw13 is not None:
            branch = branch + self.dw13(x)
        branch = branch + x
        out = self.pw(branch)
        return x + self.drop_path(out)


class SharedKVAttention(nn.Module):
    """CBS-MoSA style shared-KV attention with optional cascaded groups."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cascade: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cascade = cascade

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, self.head_dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        q = self.q(x).view(bsz, num_tokens, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(x).view(bsz, num_tokens, 2, self.head_dim)
        k = kv[:, :, 0].unsqueeze(1)
        v = kv[:, :, 1].unsqueeze(1)

        outputs = []
        carry = None
        attn_maps = []
        for head_idx in range(self.num_heads):
            qi = q[:, head_idx]
            if self.cascade and carry is not None:
                qi = qi + carry
            attn = (qi.unsqueeze(1) @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).squeeze(1)
            carry = out
            outputs.append(out)
            if not self.training:
                attn_maps.append(attn.detach())

        out = torch.cat(outputs, dim=-1)
        out = self.proj_drop(self.proj(out))
        if attn_maps:
            self.last_attn = torch.cat(attn_maps, dim=1)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).view(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        out = self.proj_drop(self.proj(out))
        if not self.training:
            self.last_attn = attn.detach()
        return out


class MoFFN(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: float = 4.0,
        drop: float = 0.0,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(int(channels * expansion))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            GRN(hidden) if use_grn else nn.Identity(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, channels, 1),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenMixBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention: str = "cbs",
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention == "mha":
            self.attn = MultiHeadSelfAttention(dim, num_heads, proj_drop=drop)
        elif attention == "none":
            self.attn = nn.Identity()
        elif attention == "cbs":
            self.attn = SharedKVAttention(dim, num_heads, proj_drop=drop, cascade=True)
        else:
            raise ValueError(f"Unknown attention type: {attention}")
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MoFFN(dim, expansion=4.0, drop=drop, use_grn=use_grn)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        if not isinstance(self.attn, nn.Identity):
            tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        bsz, num_tokens, channels = tokens.shape
        h, w = grid_hw
        ffn_in = self.norm2(tokens).transpose(1, 2).reshape(bsz, channels, h, w)
        ffn_out = self.ffn(ffn_in).flatten(2).transpose(1, 2)
        return tokens + self.drop_path(ffn_out)


class GatedMobileViTBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int = 2,
        num_heads: int = 5,
        attention: str = "cbs",
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_large_kernel: bool = True,
        use_grn: bool = True,
        use_gate: bool = True,
        transformer_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_gate = use_gate
        self.transformer_enabled = transformer_enabled
        local_kernel = 7 if use_large_kernel else 3
        self.local = nn.Sequential(
            ConvBNAct(channels, channels, local_kernel, groups=channels),
            ConvBNAct(channels, channels, 1),
            GRN(channels) if use_grn else nn.Identity(),
        )

        patch_dim = channels * patch_size * patch_size
        self.patch_proj_in = nn.Linear(patch_dim, channels)
        self.token_block = TokenMixBlock(
            channels,
            num_heads,
            attention=attention if transformer_enabled else "none",
            drop=drop,
            drop_path=drop_path,
            use_grn=use_grn,
        )
        self.patch_proj_out = nn.Linear(channels, patch_dim)
        self.fuse = ConvBNAct(channels, channels, 1, act=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )
        self.drop_path = DropPath(drop_path)
        self.last_gate: Optional[torch.Tensor] = None

    def _pad_to_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _unfold_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        x, pad = self._pad_to_patch(x)
        bsz, channels, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        tokens = self.patch_proj_in(patches)
        return tokens, (h // self.patch_size, w // self.patch_size), pad

    def _fold_tokens(
        self,
        tokens: torch.Tensor,
        grid_hw: Tuple[int, int],
        pad: Tuple[int, int],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, num_tokens, _ = tokens.shape
        patch_vectors = self.patch_proj_out(tokens).transpose(1, 2)
        h_grid, w_grid = grid_hw
        out = F.fold(
            patch_vectors,
            output_size=(h_grid * self.patch_size, w_grid * self.patch_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        pad_h, pad_w = pad
        if pad_h or pad_w:
            out = out[:, :, : out_hw[0], : out_hw[1]]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local = self.local(x)
        if self.transformer_enabled:
            tokens, grid_hw, pad = self._unfold_tokens(x)
            tokens = self.token_block(tokens, grid_hw)
            global_feat = self._fold_tokens(tokens, grid_hw, pad, x.shape[-2:])
        else:
            global_feat = local

        if self.use_gate:
            gate = self.gate(torch.cat([local, global_feat], dim=1))
            fused = gate * global_feat + (1.0 - gate) * local
            if not self.training:
                self.last_gate = gate.detach()
        else:
            fused = 0.5 * (global_feat + local)
        fused = self.fuse(fused)
        return residual + self.drop_path(fused)


class GlobalRefinementBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        attention: str = "cbs",
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_large_kernel: bool = True,
        use_grn: bool = True,
        transformer_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.local = RepLargeKernelBlock(
            channels,
            drop_path=drop_path,
            use_large_kernel=use_large_kernel,
            use_grn=use_grn,
        )
        self.norm = nn.LayerNorm(channels)
        if transformer_enabled:
            if attention == "mha":
                self.attn = MultiHeadSelfAttention(channels, num_heads, proj_drop=drop)
            elif attention == "cbs":
                self.attn = SharedKVAttention(channels, num_heads, proj_drop=drop, cascade=True)
            elif attention == "none":
                self.attn = nn.Identity()
            else:
                raise ValueError(f"Unknown attention type: {attention}")
        else:
            self.attn = nn.Identity()
        self.ffn = MoFFN(channels, expansion=4.0, drop=drop, use_grn=use_grn)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        bsz, channels, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if not isinstance(self.attn, nn.Identity):
            tokens = tokens + self.drop_path(self.attn(self.norm(tokens)))
        x = tokens.transpose(1, 2).reshape(bsz, channels, h, w)
        x = x + self.drop_path(self.ffn(x))
        return x


class MultiScaleAttentionHead(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        head_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(ch, head_dim) for ch in in_channels])
        self.query = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)
        self.norm = nn.LayerNorm(head_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_dim, num_classes)
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feat, proj in zip(features, self.proj):
            vec = feat.mean(dim=(2, 3))
            pooled.append(proj(vec))
        tokens = torch.stack(pooled, dim=1)
        query = self.query.expand(tokens.shape[0], -1, -1)
        weights = (query @ tokens.transpose(1, 2)) / math.sqrt(tokens.shape[-1])
        weights = weights.softmax(dim=-1)
        if not self.training:
            self.last_weights = weights.detach()
        out = (weights @ tokens).squeeze(1)
        out = self.dropout(self.norm(out))
        return self.fc(out)


@dataclass
class ModelConfig:
    channels: Tuple[int, int, int, int, int]
    stage_depths: Tuple[int, int, int, int]
    heads_stage3: int
    heads_stage4: int
    drop_path: float
    dropout: float


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "glf_tiny": ModelConfig((24, 32, 64, 96, 160), (1, 2, 2, 2), 4, 5, 0.08, 0.0),
    "glf_small": ModelConfig((32, 48, 96, 128, 224), (2, 3, 3, 2), 4, 7, 0.12, 0.05),
    "glf_base": ModelConfig((32, 48, 96, 160, 256), (2, 3, 4, 3), 5, 8, 0.18, 0.1),
}


class GLFMobileViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        variant: str = "glf_base",
        attention: str = "cbs",
        use_large_kernel: bool = True,
        use_grn: bool = True,
        use_gate: bool = True,
        transformer_enabled: bool = True,
        drop_path: Optional[float] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant {variant}. Available: {sorted(MODEL_CONFIGS)}")
        cfg = MODEL_CONFIGS[variant]
        c0, c1, c2, c3, c4 = cfg.channels
        d1, d2, d3, d4 = cfg.stage_depths
        dp_rate = cfg.drop_path if drop_path is None else drop_path
        drop = cfg.dropout if dropout is None else dropout
        total_blocks = d1 + d2 + d3 + d4
        dp_rates = torch.linspace(0, dp_rate, total_blocks).tolist()
        dp_iter = iter(dp_rates)

        self.stem = nn.Sequential(
            ConvBNAct(3, c0, 3),
            ConvBNAct(c0, c0, 3),
        )

        stage1 = [UIBConvBlock(c0, c1, stride=2, drop_path=next(dp_iter))]
        for _ in range(d1 - 1):
            stage1.append(UIBConvBlock(c1, c1, stride=1, drop_path=next(dp_iter)))
        self.stage1 = nn.Sequential(*stage1)

        stage2: List[nn.Module] = [ConvBNAct(c1, c2, 3, stride=2)]
        for _ in range(d2):
            stage2.append(
                RepLargeKernelBlock(
                    c2,
                    drop_path=next(dp_iter),
                    use_large_kernel=use_large_kernel,
                    use_grn=use_grn,
                )
            )
        self.stage2 = nn.Sequential(*stage2)

        stage3: List[nn.Module] = [ConvBNAct(c2, c3, 3, stride=2)]
        for _ in range(d3):
            stage3.append(
                GatedMobileViTBlock(
                    c3,
                    patch_size=2,
                    num_heads=cfg.heads_stage3,
                    attention=attention,
                    drop=drop,
                    drop_path=next(dp_iter),
                    use_large_kernel=use_large_kernel,
                    use_grn=use_grn,
                    use_gate=use_gate,
                    transformer_enabled=transformer_enabled,
                )
            )
        self.stage3 = nn.Sequential(*stage3)

        stage4: List[nn.Module] = [ConvBNAct(c3, c4, 3, stride=2)]
        for _ in range(d4):
            stage4.append(
                GlobalRefinementBlock(
                    c4,
                    num_heads=cfg.heads_stage4,
                    attention=attention,
                    drop=drop,
                    drop_path=next(dp_iter),
                    use_large_kernel=use_large_kernel,
                    use_grn=use_grn,
                    transformer_enabled=transformer_enabled,
                )
            )
        self.stage4 = nn.Sequential(*stage4)
        self.head = MultiScaleAttentionHead((c2, c3, c4), c4, num_classes, dropout=drop)
        self.num_classes = num_classes
        self.variant = variant
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        aux = {
            "stage2": x2,
            "stage3": x3,
            "stage4": x4,
        }
        return [x2, x3, x4], aux

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features, aux = self.forward_features(x)
        logits = self.head(features)
        if return_aux:
            return logits, aux
        return logits


def list_models() -> List[str]:
    names = list(MODEL_CONFIGS)
    names.extend(["cnn_only", "mobilevit_lite", "mocovit_lite"])
    return names


def create_model(
    name: str,
    num_classes: int = 100,
    attention: Optional[str] = None,
    use_large_kernel: Optional[bool] = None,
    use_grn: Optional[bool] = None,
    use_gate: Optional[bool] = None,
    drop_path: Optional[float] = None,
    dropout: Optional[float] = None,
) -> GLFMobileViT:
    if name in MODEL_CONFIGS:
        return GLFMobileViT(
            num_classes=num_classes,
            variant=name,
            attention=attention or "cbs",
            use_large_kernel=True if use_large_kernel is None else use_large_kernel,
            use_grn=True if use_grn is None else use_grn,
            use_gate=True if use_gate is None else use_gate,
            transformer_enabled=True,
            drop_path=drop_path,
            dropout=dropout,
        )
    if name == "cnn_only":
        return GLFMobileViT(
            num_classes=num_classes,
            variant="glf_base",
            attention="none",
            use_large_kernel=True if use_large_kernel is None else use_large_kernel,
            use_grn=True if use_grn is None else use_grn,
            use_gate=False,
            transformer_enabled=False,
            drop_path=drop_path,
            dropout=dropout,
        )
    if name == "mobilevit_lite":
        return GLFMobileViT(
            num_classes=num_classes,
            variant="glf_small",
            attention=attention or "mha",
            use_large_kernel=False if use_large_kernel is None else use_large_kernel,
            use_grn=False if use_grn is None else use_grn,
            use_gate=True if use_gate is None else use_gate,
            transformer_enabled=True,
            drop_path=drop_path,
            dropout=dropout,
        )
    if name == "mocovit_lite":
        return GLFMobileViT(
            num_classes=num_classes,
            variant="glf_small",
            attention=attention or "cbs",
            use_large_kernel=False if use_large_kernel is None else use_large_kernel,
            use_grn=False if use_grn is None else use_grn,
            use_gate=False if use_gate is None else use_gate,
            transformer_enabled=True,
            drop_path=drop_path,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model {name}. Available: {list_models()}")
