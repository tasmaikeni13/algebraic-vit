"""Standard Vision Transformer baseline.

Architecture: convolutional patch embedding, CLS token, learned absolute
positional embeddings, pre-norm blocks (LayerNorm, softmax attention,
LayerNorm, GELU MLP), final LayerNorm, and a softmax cross-entropy head.
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
    L, d, f, C = cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.num_classes
    T, p = cfg.n_tokens, cfg.patch

    def normal(k, shape, scale=1.0):
        return jax.random.normal(k, shape, F32) * (cfg.init_std * scale)

    return {
        "patch_w": normal(keys[0], (3 * p * p, d)),
        "patch_b": jnp.zeros((d,), F32),
        "cls": normal(keys[1], (d,)),
        "pos": normal(keys[2], (T, d)),
        "head_w": normal(keys[3], (d, C)),
        "head_b": jnp.zeros((C,), F32),
        "final_gain": jnp.ones((d,), F32),
        "final_bias": jnp.zeros((d,), F32),
        "layers": {
            "w_qkv": normal(keys[4], (L, d, 3 * d)),
            "w_o": normal(keys[5], (L, d, d), cfg.out_init_scale),
            "w_1": normal(keys[6], (L, d, f)),
            "b_1": jnp.zeros((L, f), F32),
            "w_2": normal(keys[7], (L, f, d), cfg.out_init_scale),
            "b_2": jnp.zeros((L, d), F32),
            "ln1_gain": jnp.ones((L, d), F32),
            "ln1_bias": jnp.zeros((L, d), F32),
            "ln2_gain": jnp.ones((L, d), F32),
            "ln2_bias": jnp.zeros((L, d), F32),
        },
    }


def decay_mask(params) -> Any:
    """Decay projection matrices; not gains, biases, CLS, or positions."""

    def rule(path, _):
        name = getattr(path[-1], "key", str(path[-1]))
        return name.startswith("w_") or name in ("patch_w", "head_w")

    return jax.tree_util.tree_map_with_path(rule, params)


def patchify(images: jax.Array, cfg: VisionConfig) -> jax.Array:
    B, H, W, C = images.shape
    p = cfg.patch
    gh, gw = H // p, W // p
    x = images.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, gh * gw, p * p * C)


def _attention(q, k, v, cfg: VisionConfig):
    if cfg.attn_impl == "flash":
        from vision.kernels.flash import flash_attention
        t = q.shape[-2]
        block = cfg.flash_block if t % cfg.flash_block == 0 else t
        return flash_attention(q, k, v, None, kind="softmax", block=block,
                               causal=False)
    return O.softmax_attention(q, k, v)


def _block(cfg: VisionConfig, x, lp):
    B, T, d = x.shape
    H, D = cfg.n_heads, cfg.head_dim
    dt = x.dtype
    h = O.layer_norm(x, lp["ln1_gain"], lp["ln1_bias"], cfg.eps)
    qkv = (h @ lp["w_qkv"].astype(dt)).reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    a = _attention(q, k, v, cfg)
    a = a.transpose(0, 2, 1, 3).reshape(B, T, d)
    x = x + a @ lp["w_o"].astype(dt)
    h2 = O.layer_norm(x, lp["ln2_gain"], lp["ln2_bias"], cfg.eps)
    h2 = O.gelu(h2 @ lp["w_1"].astype(dt) + lp["b_1"].astype(dt))
    return x + h2 @ lp["w_2"].astype(dt) + lp["b_2"].astype(dt)


def forward_tokens(params, images: jax.Array, cfg: VisionConfig) -> jax.Array:
    dt = cfg.compute_dtype
    patches = patchify(images, cfg).astype(dt)
    tok = patches @ params["patch_w"].astype(dt) + params["patch_b"].astype(dt)
    cls = jnp.broadcast_to(params["cls"].astype(dt), (tok.shape[0], 1, tok.shape[2]))
    x = jnp.concatenate([cls, tok], axis=1) + params["pos"].astype(dt)

    def body(carry, lp):
        return _block(cfg, carry, lp), None

    if cfg.remat:
        body = jax.checkpoint(body, prevent_cse=False)
    x, _ = lax.scan(body, x, params["layers"])
    return x


def forward_logits(params, images: jax.Array, cfg: VisionConfig) -> jax.Array:
    x = forward_tokens(params, images, cfg)
    h = O.layer_norm(x[:, 0], params["final_gain"], params["final_bias"], cfg.eps)
    return (h @ params["head_w"].astype(h.dtype) + params["head_b"].astype(h.dtype)).astype(F32)


def head_loss(params, images: jax.Array, targets: jax.Array, cfg: VisionConfig):
    """Mean cross-entropy and stop-gradient per-example NLL."""
    logits = forward_logits(params, images, cfg)
    return O.cross_entropy(logits, targets)
