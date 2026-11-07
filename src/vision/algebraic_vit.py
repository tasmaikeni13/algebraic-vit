"""Algebraic Vision Transformer: patch classifier without transcendental ops.

Architecture: convolutional patch embedding, CLS token, pre-norm blocks
(RMS norm, QK-normalized rho-power attention with learned per-head sink
and 2D Cayley rotary code, RMS norm, ALU-GLU feed-forward), final RMS
norm, and the algebraic maximum-likelihood head with the exact log-loss
gradient. The training objective and its gradient use arithmetic,
comparisons, and reciprocal square roots only.
"""

from typing import Any, Dict

import jax
from jax import lax
import jax.numpy as jnp

from vision import ops as O
from vision.config import VisionConfig

F32 = jnp.float32


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def init_params(cfg: VisionConfig, key: jax.Array) -> Dict[str, Any]:
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    keys = jax.random.split(key, 10)
    L, d, f, C, H, D = (cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.num_classes,
                        cfg.n_heads, cfg.head_dim)
    p = cfg.patch

    def normal(k, shape, scale=1.0):
        return jax.random.normal(k, shape, F32) * (cfg.init_std * scale)

    params: Dict[str, Any] = {
        # Patch embedding stored as a dense matrix over flattened patches.
        "patch_w": normal(keys[0], (3 * p * p, d)),
        "patch_b": jnp.zeros((d,), F32),
        "cls": normal(keys[1], (d,)),
        "head_w": normal(keys[2], (d, C)),
        "head_b": jnp.zeros((C,), F32),
        "final_gain": jnp.ones((d,), F32),
        "layers": {
            "w_q": normal(keys[3], (L, d, d)),
            "w_k": normal(keys[4], (L, d, d)),
            "w_v": normal(keys[5], (L, d, d)),
            "w_o": normal(keys[6], (L, d, d), cfg.out_init_scale),
            "w_g": normal(keys[7], (L, d, f)),
            "w_u": normal(keys[8], (L, d, f)),
            "w_d": normal(keys[9], (L, f, d), cfg.out_init_scale),
            "ln1": jnp.ones((L, d), F32),
            "ln2": jnp.ones((L, d), F32),
            "q_gain": jnp.ones((L, D), F32),
            "k_gain": jnp.ones((L, D), F32),
            "sink": jnp.full((L, H), cfg.sink_init, F32),
        },
    }
    return params


def decay_mask(params) -> Any:
    """Decay projection matrices; not gains, biases, CLS, or sink logits."""

    def rule(path, _):
        name = getattr(path[-1], "key", str(path[-1]))
        return name.startswith("w_") or name in ("patch_w", "head_w")

    return jax.tree_util.tree_map_with_path(rule, params)


def position_tables(cfg: VisionConfig):
    return O.cayley_table_2d(cfg.head_dim, *cfg.grid)


def patchify(images: jax.Array, cfg: VisionConfig) -> jax.Array:
    """(B, H, W, 3) images to (B, T, 3*patch*patch) flattened patches."""
    B, H, W, C = images.shape
    p = cfg.patch
    gh, gw = H // p, W // p
    x = images.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, gh * gw, p * p * C)


def _attention(q, k, v, sink_logit, cfg: VisionConfig):
    if cfg.attn_impl == "flash":
        from vision.kernels.flash import flash_attention
        t = q.shape[-2]
        block = cfg.flash_block if t % cfg.flash_block == 0 else t
        return flash_attention(q, k, v, sink_logit, kind="asinh", n=cfg.attn_n,
                               c=cfg.attn_c, block=block, causal=False)
    return O.asinh_attention(q, k, v, n=cfg.attn_n, c=cfg.attn_c, sink_logit=sink_logit)


def _block(cfg: VisionConfig, tables, x, lp):
    B, T, d = x.shape
    H, D = cfg.n_heads, cfg.head_dim
    dt = x.dtype
    h = O.rms_norm(x, lp["ln1"], cfg.eps)
    q = (h @ lp["w_q"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = (h @ lp["w_k"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    v = (h @ lp["w_v"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    q = O.rms_norm(q, lp["q_gain"], cfg.eps)
    k = O.rms_norm(k, lp["k_gain"], cfg.eps)
    q = O.apply_rotary_2d(q, tables, cfg.grid)
    k = O.apply_rotary_2d(k, tables, cfg.grid)
    sink = lp["sink"].astype(F32)[None, :, None, None]
    a = _attention(q, k, v, sink, cfg)
    a = a.transpose(0, 2, 1, 3).reshape(B, T, d)
    x = x + a @ lp["w_o"].astype(dt)
    h2 = O.rms_norm(x, lp["ln2"], cfg.eps)
    gate = h2 @ lp["w_g"].astype(dt)
    up = h2 @ lp["w_u"].astype(dt)
    return x + (gate * O.alu(up, cfg.alu_beta)) @ lp["w_d"].astype(dt)


def forward_tokens(params, images: jax.Array, cfg: VisionConfig, tables) -> jax.Array:
    """All token states (B, 1+T, d) before the final norm, compute dtype."""
    dt = cfg.compute_dtype
    patches = patchify(images, cfg).astype(dt)
    tok = patches @ params["patch_w"].astype(dt) + params["patch_b"].astype(dt)
    cls = jnp.broadcast_to(params["cls"].astype(dt), (tok.shape[0], 1, tok.shape[2]))
    x = jnp.concatenate([cls, tok], axis=1)

    def body(carry, lp):
        return _block(cfg, tables, carry, lp), None

    if cfg.remat:
        body = jax.checkpoint(body, prevent_cse=False)
    x, _ = lax.scan(body, x, params["layers"])
    return x


def forward_logits(params, images: jax.Array, cfg: VisionConfig, tables) -> jax.Array:
    """CLS-pooled classification logits (B, C) in float32."""
    x = forward_tokens(params, images, cfg, tables)
    h = O.rms_norm(x[:, 0], params["final_gain"], cfg.eps)
    return (h @ params["head_w"].astype(h.dtype) + params["head_b"].astype(h.dtype)).astype(F32)


def head_loss(params, images: jax.Array, targets: jax.Array, cfg: VisionConfig, tables):
    """Training objective and stop-gradient per-example target probability."""
    logits = forward_logits(params, images, cfg, tables)
    obj, pt = O.algebraic_likelihood(logits, targets, cfg.head_n, cfg.head_c)
    return obj, lax.stop_gradient(pt)
