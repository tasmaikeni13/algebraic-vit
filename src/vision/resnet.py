"""ResNet-18/50 convolutional baseline with batch normalization.

Stem: 7x7 convolution, batch norm, ReLU, 3x3 max pool. Stages follow the
canonical BasicBlock (ResNet-18: layers 2-2-2-2) and Bottleneck
(ResNet-50: layers 3-4-6-3) layouts with projection shortcuts, global
average pooling, and a softmax cross-entropy head. Parameters and batch
statistics are stored separately; the trainer updates the exponential
moving averages outside the optimizer.
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from vision import ops as O
from vision.config import VisionConfig

F32 = jnp.float32
BN_EPS = 1e-5
BN_MOMENTUM = 0.9


def lax_conv(x, w, stride=1):
    import jax.lax as lax
    return lax.conv_general_dilated(
        x, w, window_strides=(stride, stride), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32)


def _bn(x, stats, training: bool):
    gamma, beta, mean, var = stats
    if training:
        mu = jnp.mean(x, axis=(0, 1, 2))
        va = jnp.var(x, axis=(0, 1, 2))
        new_mean = BN_MOMENTUM * mean + (1 - BN_MOMENTUM) * mu
        new_var = BN_MOMENTUM * var + (1 - BN_MOMENTUM) * va
        y = (x - mu) * jax.lax.rsqrt(va + BN_EPS) * gamma + beta
        return y.astype(x.dtype), (gamma, beta, new_mean, new_var)
    y = (x - mean) * jax.lax.rsqrt(var + BN_EPS) * gamma + beta
    return y.astype(x.dtype), stats


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))


def _he(key, shape):
    fan_in = shape[0] * shape[1] * shape[2] if len(shape) == 4 else shape[0]
    return jax.random.normal(key, shape, F32) * (2.0 / fan_in) ** 0.5


def _bn_init(width):
    return (jnp.ones((width,), F32), jnp.zeros((width,), F32),
            jnp.zeros((width,), F32), jnp.ones((width,), F32))


def init_params(cfg: VisionConfig, key: jax.Array) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    depth = cfg.resnet_depth
    if depth == 18:
        widths = (64, 128, 256, 512)
        blocks = (2, 2, 2, 2)
        bottleneck = False
    elif depth == 50:
        widths = (256, 512, 1024, 2048)
        blocks = (3, 4, 6, 3)
        bottleneck = True
    else:
        raise ValueError("resnet depth must be 18 or 50")
    n_conv = 1 + sum(blocks) * (3 if bottleneck else 2)
    keys = jax.random.split(key, n_conv + 2)
    ki = 0

    def take(shape):
        nonlocal ki
        k = keys[ki]
        ki += 1
        return _he(k, shape)

    small = cfg.image_size < 64
    params: Dict[str, Any] = {"stem_w": take((3, 3, 3, 64) if small else (7, 7, 3, 64))}
    stats: Dict[str, Any] = {"stem_bn": _bn_init(64)}
    in_w = 64
    for si, (w, nb) in enumerate(zip(widths, blocks)):
        for bi in range(nb):
            stride = 2 if (bi == 0 and si > 0) else 1
            tag = f"s{si}b{bi}"
            if bottleneck:
                mid = w // 4
                params[tag + "_w1"] = take((1, 1, in_w, mid))
                params[tag + "_w2"] = take((3, 3, mid, mid))
                params[tag + "_w3"] = take((1, 1, mid, w))
                stats[tag + "_bn1"] = _bn_init(mid)
                stats[tag + "_bn2"] = _bn_init(mid)
                stats[tag + "_bn3"] = _bn_init(w)
            else:
                params[tag + "_w1"] = take((3, 3, in_w, w))
                params[tag + "_w2"] = take((3, 3, w, w))
                stats[tag + "_bn1"] = _bn_init(w)
                stats[tag + "_bn2"] = _bn_init(w)
            if stride != 1 or in_w != w:
                params[tag + "_wp"] = take((1, 1, in_w, w))
                stats[tag + "_bnp"] = _bn_init(w)
            in_w = w
    params["head_w"] = _he(keys[-2], (in_w, cfg.num_classes))
    params["head_b"] = jnp.zeros((cfg.num_classes,), F32)
    return params, stats


def decay_mask(params) -> Any:
    """Decay convolution and head matrices; not BN scales or biases."""

    def rule(path, _):
        name = getattr(path[-1], "key", str(path[-1]))
        return name.endswith("_w") or name == "stem_w"

    return jax.tree_util.tree_map_with_path(rule, params)


def _block(params, stats, x, tag, stride, bottleneck, training):
    new_stats = dict(stats)
    if bottleneck:
        y, s = _bn(lax_conv(x, params[tag + "_w1"]), stats[tag + "_bn1"], training)
        new_stats[tag + "_bn1"] = s
        y = jnp.maximum(y, 0)
        y, s = _bn(lax_conv(y, params[tag + "_w2"], stride), stats[tag + "_bn2"], training)
        new_stats[tag + "_bn2"] = s
        y = jnp.maximum(y, 0)
        y, s = _bn(lax_conv(y, params[tag + "_w3"]), stats[tag + "_bn3"], training)
        new_stats[tag + "_bn3"] = s
    else:
        y, s = _bn(lax_conv(x, params[tag + "_w1"], stride), stats[tag + "_bn1"], training)
        new_stats[tag + "_bn1"] = s
        y = jnp.maximum(y, 0)
        y, s = _bn(lax_conv(y, params[tag + "_w2"]), stats[tag + "_bn2"], training)
        new_stats[tag + "_bn2"] = s
    shortcut = x
    if tag + "_wp" in params:
        shortcut, s = _bn(lax_conv(x, params[tag + "_wp"], stride), stats[tag + "_bnp"], training)
        new_stats[tag + "_bnp"] = s
    return jnp.maximum(y + shortcut, 0).astype(x.dtype), new_stats


def forward_logits(params, stats, images: jax.Array, cfg: VisionConfig,
                   training: bool):
    """Logits in float32 and updated batch statistics."""
    import jax.lax as lax
    depth = cfg.resnet_depth
    bottleneck = depth == 50
    widths_blocks = ((64, 2), (128, 2), (256, 2), (512, 2)) if depth == 18 else \
        ((256, 3), (512, 4), (1024, 6), (2048, 3))
    stats = dict(stats)
    x = images.astype(jnp.float32)
    small = cfg.image_size < 64
    y, s = _bn(lax_conv(x, params["stem_w"], 1 if small else 2), stats["stem_bn"], training)
    stats["stem_bn"] = s
    y = jnp.maximum(y, 0)
    if not small:
        y = lax.reduce_window(y, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME")
    for si, (_, nb) in enumerate(widths_blocks):
        for bi in range(nb):
            stride = 2 if (bi == 0 and si > 0) else 1
            y, stats = _block(params, stats, y, f"s{si}b{bi}", stride, bottleneck, training)
    y = jnp.mean(y, axis=(1, 2))
    logits = (y @ params["head_w"].astype(y.dtype) + params["head_b"].astype(y.dtype))
    return logits.astype(F32), stats


def head_loss(params, stats, images, targets, cfg: VisionConfig, training: bool):
    logits, new_stats = forward_logits(params, stats, images, cfg, training)
    obj, nll = O.cross_entropy(logits, targets)
    return (obj, (nll, new_stats))
