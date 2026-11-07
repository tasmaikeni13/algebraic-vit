"""Fused batch-normalization + ReLU Pallas kernel.

The memory-bound epilogue of every ResNet block: per-channel normalize
over the batch and spatial dims, affine transform, ReLU, in minimal
passes. Convolutions themselves dispatch to XLA (compute-bound optimal);
this kernel fuses the surrounding traffic. Reductions use XLA (optimal);
the elementwise apply runs in Pallas with an exact fused VJP.

The kernels use only portable Pallas constructs, so the logic is
verifiable on CPU with ``interpret=True``; on TPU they run natively.

Training-mode contract: ``mean``/``var`` must be the batch statistics of
``x`` (see ``bn_relu_train``). The VJP folds their ``x``-dependence into
``dx`` analytically and takes no cotangents for them. For inference with
running statistics, differentiate the elementwise path instead.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

F32 = jnp.float32


def ref_bn_relu(x, mean, var, gamma, beta, eps=1e-5):
    """Dense reference: fused BN+ReLU forward over the first axis."""
    xhat = (x - mean) * lax.rsqrt(var + eps)
    return jnp.maximum(xhat * gamma + beta, 0).astype(x.dtype), xhat


def _apply_kernel(x_ref, mean_ref, var_ref, g_ref, b_ref, y_ref, xh_ref, *, eps):
    m = pl.program_id(0)
    c = pl.program_id(1)
    x = x_ref[:].astype(F32)
    mean = mean_ref[:].astype(F32)
    var = var_ref[:].astype(F32)
    gamma = g_ref[:].astype(F32)
    beta = b_ref[:].astype(F32)
    xhat = (x - mean) * lax.rsqrt(var + eps)
    xh_ref[:] = xhat.astype(xh_ref.dtype)
    y_ref[:] = jnp.maximum(xhat * gamma + beta, 0).astype(y_ref.dtype)


def _bwd_kernel(dy_ref, xh_ref, var_ref, g_ref, mdy_ref, mdyxh_ref, dx_ref, *, eps):
    m = pl.program_id(0)
    c = pl.program_id(1)
    dy = dy_ref[:].astype(F32)
    xhat = xh_ref[:].astype(F32)
    var = var_ref[:].astype(F32)
    gamma = g_ref[:].astype(F32)
    # The caller passes dy already ReLU-masked (dy * (y > 0)) plus the
    # global XLA-reduced means; this kernel applies the BN input VJP.
    mean_dy = mdy_ref[:].astype(F32)
    mean_dy_xh = mdyxh_ref[:].astype(F32)
    rstd = lax.rsqrt(var + eps)
    dx = (dy - mean_dy - xhat * mean_dy_xh) * (rstd * gamma)
    dx_ref[:] = dx.astype(dx_ref.dtype)


def bn_relu(x, mean, var, gamma, beta, *, eps=1e-5, block_m=1024, block_c=128,
            interpret=False):
    """Fused BN+ReLU over axis 0 of ``x`` (M, C) with per-channel stats."""
    M, C = x.shape
    if M % block_m or C % block_c:
        raise ValueError(f"shape {(M, C)} must divide blocks {(block_m, block_c)}")
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    xh_shape = jax.ShapeDtypeStruct(x.shape, F32)
    grid = (M // block_m, C // block_c)
    y, xhat = pl.pallas_call(
        partial(_apply_kernel, eps=eps),
        grid_spec=pl.GridSpec(
            grid=grid,
            in_specs=[pl.BlockSpec((block_m, block_c), lambda m, c: (m, c)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,))],
            out_specs=[pl.BlockSpec((block_m, block_c), lambda m, c: (m, c)),
                       pl.BlockSpec((block_m, block_c), lambda m, c: (m, c))],
        ),
        out_shape=[out_shape, xh_shape],
        interpret=interpret,
        name="bn_relu_fwd",
    )(x, mean, var, gamma, beta)
    return y, xhat


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def bn_relu_vjp(x, mean, var, gamma, beta, eps=1e-5, block_m=1024, block_c=128,
                interpret=False):
    """BN+ReLU with the exact fused vector-Jacobian product."""
    y, _ = bn_relu(x, mean, var, gamma, beta, eps=eps, block_m=block_m,
                   block_c=block_c, interpret=interpret)
    return y


def _bn_relu_fwd(x, mean, var, gamma, beta, eps, block_m, block_c, interpret):
    y, xhat = bn_relu(x, mean, var, gamma, beta, eps=eps, block_m=block_m,
                      block_c=block_c, interpret=interpret)
    return y, (xhat, var, gamma, y)


def _bn_relu_bwd(eps, block_m, block_c, interpret, res, dy):
    xhat, var, gamma, y = res
    mask = (y > 0).astype(F32)
    dym = dy.astype(F32) * mask
    M, C = xhat.shape
    dgamma = jnp.sum(dym * xhat, axis=0)
    dbeta = jnp.sum(dym, axis=0)
    mean_dy = jnp.mean(dym, axis=0)
    mean_dy_xh = jnp.mean(dym * xhat, axis=0)
    dx = pl.pallas_call(
        partial(_bwd_kernel, eps=eps),
        grid_spec=pl.GridSpec(
            grid=(M // block_m, C // block_c),
            in_specs=[pl.BlockSpec((block_m, block_c), lambda m, c: (m, c)),
                      pl.BlockSpec((block_m, block_c), lambda m, c: (m, c)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,)),
                      pl.BlockSpec((block_c,), lambda m, c: (c,))],
            out_specs=pl.BlockSpec((block_m, block_c), lambda m, c: (m, c)),
        ),
        out_shape=jax.ShapeDtypeStruct(xhat.shape, F32),
        interpret=interpret,
        name="bn_relu_bwd",
    )(dym, xhat, var, gamma, mean_dy, mean_dy_xh)
    return dx.astype(dy.dtype), None, None, dgamma.astype(gamma.dtype), dbeta.astype(dy.dtype)


bn_relu_vjp.defvjp(_bn_relu_fwd, _bn_relu_bwd)


def bn_relu_train(x, gamma, beta, *, eps=1e-5, block_m=1024, block_c=128,
                  interpret=False):
    """Training-mode fused BN+ReLU: stats computed from ``x`` internally."""
    mean = jnp.mean(x, axis=0)
    var = jnp.var(x, axis=0)
    return bn_relu_vjp(x, mean, var, gamma, beta, eps, block_m, block_c, interpret)
