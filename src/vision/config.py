"""Model configurations and architecture presets."""

from dataclasses import dataclass, replace
from typing import Dict

import jax.numpy as jnp


@dataclass(frozen=True)
class VisionConfig:
    arch: str = "algebraic"      # "algebraic", "standard", or "resnet"
    image_size: int = 224
    patch: int = 16
    num_classes: int = 1000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1536
    eps: float = 1e-5
    alu_beta: float = 1.2
    attn_n: int = 16
    attn_c: float = 8.0
    sink_init: float = 0.0
    head_n: int = 8
    head_c: float = 2.0
    init_std: float = 0.02
    out_init_scale: float = 1.0
    dtype: str = "bfloat16"
    remat: bool = True
    attn_impl: str = "dense"     # "dense" or "flash" (Pallas kernels)
    flash_block: int = 256
    # ResNet fields (unused by ViT arms).
    resnet_depth: int = 50       # 18 or 50
    resnet_width: int = 64

    @property
    def grid(self):
        g = self.image_size // self.patch
        return (g, g)

    @property
    def n_patches(self):
        g = self.image_size // self.patch
        return g * g

    @property
    def n_tokens(self):
        return self.n_patches + 1  # patches plus CLS

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def compute_dtype(self):
        return jnp.dtype(self.dtype)

    @property
    def algebraic_path(self) -> bool:
        return self.arch == "algebraic"


SIZES: Dict[str, Dict[str, int]] = {
    # ~5.5M params at 224px (CIFAR-100 sanity tier uses 32px inputs).
    "Ti": dict(d_model=192, n_layers=12, n_heads=3, d_ff=768),
    # ~22M params at 224px.
    "S": dict(d_model=384, n_layers=12, n_heads=6, d_ff=1536),
    # ~86M params at 224px.
    "B": dict(d_model=768, n_layers=12, n_heads=12, d_ff=3072),
}


def algebraic_vit(size: str = "S", **overrides) -> VisionConfig:
    """RMS norms, QK normalization, 2D Cayley code, rho(s/8)^16 attention
    with learned per-head sink, ALU-GLU with beta=1.2, algebraic head.

    The gated feed-forward uses two thirds of the standard width so both
    arms match in parameters (three matrices vs two)."""
    dims = dict(SIZES[size])
    dims["d_ff"] = dims["d_ff"] * 2 // 3
    cfg = VisionConfig(arch="algebraic", **dims)
    return replace(cfg, **overrides)


def standard_vit(size: str = "S", **overrides) -> VisionConfig:
    """LayerNorm, learned absolute positions, softmax attention, GELU MLP,
    softmax cross-entropy head."""
    cfg = VisionConfig(arch="standard", **SIZES[size])
    return replace(cfg, **overrides)


def resnet(depth: int = 50, **overrides) -> VisionConfig:
    """Convolutional baseline with batch normalization and ReLU."""
    if depth not in (18, 50):
        raise ValueError("resnet depth must be 18 or 50")
    cfg = VisionConfig(arch="resnet", resnet_depth=depth)
    return replace(cfg, **overrides)


PRESETS = {"algebraic": algebraic_vit, "standard": standard_vit, "resnet": resnet}
