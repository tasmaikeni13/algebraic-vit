"""Shared tensor operations for the vision models.

Algebraic path (``rho``, ``alu``, Cayley rotations, RMS normalization,
asinh-power attention, algebraic likelihood): arithmetic, comparisons,
and ``lax.rsqrt`` only. Standard path (``gelu``, ``softmax_attention``,
``cross_entropy``) uses transcendental functions and is the reference
baseline. Layer normalization is algebraic and shared by ResNet-style
designs where noted.
"""

import math
from functools import partial
from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp

F32 = jnp.float32


def _acc(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, F32))


# ---------------------------------------------------------------------------
# Positive link and integer powers
# ---------------------------------------------------------------------------

def rho(x: jax.Array) -> jax.Array:
    """``x + sqrt(1 + x^2)`` without cancellation for negative ``x``."""
    rad = 1.0 + x * x
    t = jnp.abs(x) + rad * lax.rsqrt(rad)
    return jnp.where(x >= 0, t, 1.0 / t)


def int_pow(x: jax.Array, n: int) -> jax.Array:
    """``x**n`` for a static positive integer by binary exponentiation."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    result = None
    base = x
    while n:
        if n & 1:
            result = base if result is None else result * base
        n >>= 1
        if n:
            base = base * base
    return result


# ---------------------------------------------------------------------------
# Normalization and activation
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, gain: Optional[jax.Array], eps: float = 1e-5) -> jax.Array:
    """RMS normalization with an optional learned gain."""
    xf = _acc(x)
    tau = lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    y = xf * tau
    if gain is not None:
        y = y * gain.astype(y.dtype)
    return y.astype(x.dtype)


def layer_norm(x: jax.Array, gain: Optional[jax.Array], bias: Optional[jax.Array],
               eps: float = 1e-5) -> jax.Array:
    """Layer normalization with optional learned gain and bias."""
    xf = _acc(x)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * lax.rsqrt(var + eps)
    if gain is not None:
        y = y * gain.astype(y.dtype)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y.astype(x.dtype)


def _alu_parts(x, beta):
    z = _acc(x)
    bz = beta * z
    r = lax.rsqrt(1.0 + bz * bz)
    u = bz * r
    one_plus_u = jnp.where(bz < 0, (r * r) / (1.0 - u), 1.0 + u)
    return (0.5 * z * one_plus_u).astype(x.dtype), u, r


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def alu(x: jax.Array, beta: float = 1.2) -> jax.Array:
    """Algebraic linear unit ``x/2 (1 + bx / sqrt(1 + b^2 x^2))``.

    ``beta = 1.2`` matches GELU's mean-field forward energy and gradient
    gain at unit variance (see ``analysis/vit_primitives_analysis.py``).
    """
    return _alu_parts(x, beta)[0]


def _alu_fwd(x, beta):
    y, u, r = _alu_parts(x, beta)
    return y, (u, r)


def _alu_bwd(beta, cache, g):
    u, r = cache
    derivative = 0.5 * (1.0 + u) + 0.5 * u * (r * r)
    return ((g.astype(derivative.dtype) * derivative).astype(g.dtype),)


alu.defvjp(_alu_fwd, _alu_bwd)


def gelu(x: jax.Array) -> jax.Array:
    """Exact GELU for the standard baseline."""
    return 0.5 * x * (1.0 + jax.scipy.special.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Two-dimensional factorized Cayley rotary code
# ---------------------------------------------------------------------------

def cayley_powers(w: jax.Array, length: int) -> Tuple[jax.Array, jax.Array]:
    """Rows of ``((1 + i w)^2 / (1 + w^2))^m`` for ``m < length``.

    Binary exponentiation over the bits of ``m`` with reciprocal-root
    renormalization; uses no trigonometric functions.
    """
    w = jnp.asarray(w, F32)
    w2 = w * w
    c = (1.0 - w2) / (1.0 + w2)
    s = (2.0 * w) / (1.0 + w2)
    positions = jnp.arange(length, dtype=jnp.int32)
    out_c = jnp.ones((length, w.shape[0]), F32)
    out_s = jnp.zeros((length, w.shape[0]), F32)
    bit = 0
    while (1 << bit) < length:
        take = ((positions >> bit) & 1).astype(bool)[:, None]
        new_c = out_c * c - out_s * s
        new_s = out_c * s + out_s * c
        out_c = jnp.where(take, new_c, out_c)
        out_s = jnp.where(take, new_s, out_s)
        c, s = c * c - s * s, 2.0 * c * s
        inv = lax.rsqrt(c * c + s * s)
        c, s = c * inv, s * inv
        bit += 1
    inv = lax.rsqrt(out_c * out_c + out_s * out_s)
    return out_c * inv, out_s * inv


def cayley_table_2d(head_dim: int, rows: int, cols: int, base: float = 10000.0,
                    w0: float = 0.5):
    """Row and column rotation tables for a patch grid.

    Returns ``(row_cos, row_sin, col_cos, col_sin)`` with ``head_dim // 4``
    frequencies each. ``head_dim`` must be divisible by 4 with
    ``head_dim // 4`` a power of two.
    """
    q = head_dim // 4
    if head_dim % 4:
        raise ValueError("2D Cayley code needs head_dim divisible by 4")
    levels = int(round(math.log2(q)))
    if 1 << levels != q:
        raise ValueError("2D Cayley code needs head_dim/4 to be a power of two")
    root = float(base)
    for _ in range(levels):
        root = math.sqrt(root)
    ratio = 1.0 / root
    w = jnp.asarray([w0 * ratio ** i for i in range(q)], F32)
    return cayley_powers(w, rows) + cayley_powers(w, cols)


def apply_rotary_2d(x: jax.Array, tables, grid: Tuple[int, int]) -> jax.Array:
    """Rotate (B, H, 1+T, D) patch tokens; the CLS token (index 0) is fixed.

    The head channels split into four groups; row tables rotate the first
    pair by row index and column tables rotate the second pair by column
    index, giving exact joint-shift invariance of pairwise scores.
    """
    rows, cols = grid
    row_cos, row_sin, col_cos, col_sin = tables
    cls, patches = x[:, :, :1, :], x[:, :, 1:, :]
    q = x.shape[-1] // 4
    r1, r2, c1, c2 = jnp.split(patches, 4, axis=-1)
    rr = jnp.repeat(jnp.arange(rows), cols)
    cc = jnp.tile(jnp.arange(cols), rows)
    rc = row_cos[rr].astype(x.dtype)[None, None]
    rs = row_sin[rr].astype(x.dtype)[None, None]
    cc_ = col_cos[cc].astype(x.dtype)[None, None]
    cs = col_sin[cc].astype(x.dtype)[None, None]
    out = jnp.concatenate([r1 * rc - r2 * rs, r1 * rs + r2 * rc,
                           c1 * cc_ - c2 * cs, c1 * cs + c2 * cc_], axis=-1)
    return jnp.concatenate([cls, out], axis=2)


# ---------------------------------------------------------------------------
# Bidirectional attention
# ---------------------------------------------------------------------------

def asinh_power_weights(scores: jax.Array, n: int, c: float,
                        sink_logit: Optional[jax.Array]) -> jax.Array:
    """Normalized bidirectional ``rho(s/c)^n`` weights with max shift.

    ``scores`` (..., Tq, Tk) float32; ``sink_logit`` broadcastable to
    (..., Tq, 1) or None. The sink carries zero value, so visible
    weights may sum to less than one.
    """
    x = scores / c
    m = jnp.max(x, axis=-1, keepdims=True)
    if sink_logit is not None:
        xb = sink_logit / c
        m = jnp.maximum(m, xb)
    m = lax.stop_gradient(m)
    shift = rho(-m)
    w = jnp.where(jnp.isfinite(x), int_pow(rho(x) * shift, n), 0.0)
    denom = jnp.sum(w, axis=-1, keepdims=True)
    if sink_logit is not None:
        denom = denom + int_pow(rho(xb) * shift, n)
    return w / denom


def asinh_attention(q, k, v, *, n: int, c: float, sink_logit=None):
    """Dense bidirectional asinh-power attention on (B, H, T, D)."""
    d = q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=F32)
    scores = scores * (1.0 / math.sqrt(d))
    p = asinh_power_weights(scores, n, c, sink_logit)
    out = jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v, preferred_element_type=F32)
    return out.astype(q.dtype)


def softmax_attention(q, k, v):
    """Dense bidirectional softmax attention on (B, H, T, D)."""
    d = q.shape[-1]
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=F32)
    s = s * (1.0 / math.sqrt(d))
    p = jax.nn.softmax(s, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v, preferred_element_type=F32)
    return out.astype(q.dtype)


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------

def algebraic_probs(z: jax.Array, n: int, c: float) -> jax.Array:
    """``p_j ~ rho(z_j / c)^n`` over the last axis, max shifted."""
    x = z / c
    m = lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
    w = int_pow(rho(x) * rho(-m), n)
    return w / jnp.sum(w, axis=-1, keepdims=True)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def algebraic_likelihood(logits: jax.Array, targets: jax.Array, n: int, c: float):
    """Maximum-likelihood head for ``p_j ~ rho(z_j / c)^n``.

    Returns ``(objective, p_target)``. The negative log-likelihood has
    the algebraic logit gradient ``(n / c)(1 + (z_j / c)^2)^(-1/2)
    (p_j - y_j)``, applied by the custom VJP. The objective is the mean
    target-miss probability (an algebraic monotone stand-in); the
    per-example target probability is stop-gradient telemetry converted
    to NLL by the trainer outside the model.
    """
    p = algebraic_probs(_acc(logits), n, c)
    pt = jnp.take_along_axis(p, targets[:, None], axis=-1)[:, 0]
    return jnp.mean(1.0 - pt), pt


def _alg_lik_fwd(logits, targets, n, c):
    z = _acc(logits)
    p = algebraic_probs(z, n, c)
    pt = jnp.take_along_axis(p, targets[:, None], axis=-1)[:, 0]
    return (jnp.mean(1.0 - pt), pt), (p, z, targets)


def _alg_lik_bwd(n, c, cache, g):
    p, z, targets = cache
    g_obj = g[0]
    x = z / c
    r = lax.rsqrt(1.0 + x * x)
    y = (jnp.arange(p.shape[-1])[None, :] == targets[:, None]).astype(p.dtype)
    grad = (g_obj * (n / c) / p.shape[0]) * r * (p - y)
    return grad, None


algebraic_likelihood.defvjp(_alg_lik_fwd, _alg_lik_bwd)


def cross_entropy(logits: jax.Array, targets: jax.Array):
    """Mean cross-entropy and stop-gradient per-example NLL."""
    z = logits.astype(F32)
    lse = jax.nn.logsumexp(z, axis=-1)
    zt = jnp.take_along_axis(z, targets[:, None], axis=-1)[:, 0]
    nll = lse - zt
    return jnp.mean(nll), lax.stop_gradient(nll)
