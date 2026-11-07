"""Kernel verification: CPU-interpreted Pallas logic plus TPU execution tests.

The fused BN+ReLU kernels use portable constructs and are verified on CPU
with ``interpret=True``. Flash attention needs TPU execution; those tests
skip without it and run on the pod.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vision.kernels import bn_relu as B

T = jnp.float32


def test_bn_relu_forward_matches_reference():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (512, 128), T) * 2
    mean = jnp.zeros((128,), T)
    var = jnp.ones((128,), T)
    g = jax.random.normal(key, (128,), T) + 1.0
    b = jax.random.normal(key, (128,), T) * 0.5
    y, _ = B.bn_relu(x, mean, var, g, b, block_m=256, block_c=64, interpret=True)
    yr, _ = B.ref_bn_relu(x, mean, var, g, b)
    np.testing.assert_allclose(y, yr, rtol=1e-5, atol=1e-6)


def test_bn_relu_train_vjp_matches_autodiff():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (512, 128), T) * 2
    g = jax.random.normal(key, (128,), T) + 1.0
    b = jax.random.normal(key, (128,), T) * 0.5

    def f(xx, gg, bb):
        return B.bn_relu_train(xx, gg, bb, block_m=256, block_c=64, interpret=True).sum()

    def fr(xx, gg, bb):
        mm = xx.mean(0)
        vv = xx.var(0)
        xh = (xx - mm) * jax.lax.rsqrt(vv + 1e-5)
        return jnp.maximum(xh * gg + bb, 0).sum()

    gk = jax.grad(f, (0, 1, 2))(x, g, b)
    gr = jax.grad(fr, (0, 1, 2))(x, g, b)
    for a, c in zip(gk, gr):
        np.testing.assert_allclose(a, c, rtol=1e-4, atol=1e-5)


def _need_tpu():
    return pytest.mark.skipif(
        not any(d.platform == "tpu" for d in jax.devices()),
        reason="needs TPU execution")


@_need_tpu()
@pytest.mark.parametrize("kind", ["softmax", "asinh"])
def test_flash_bidirectional_matches_dense(kind):
    from vision import ops as O
    from vision.kernels.flash import flash_attention
    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    Bsz, H, Tlen, D = 2, 3, 197, 64
    q = jax.random.normal(k1, (Bsz, H, Tlen, D), jnp.bfloat16)
    k = jax.random.normal(k2, (Bsz, H, Tlen, D), jnp.bfloat16)
    v = jax.random.normal(k3, (Bsz, H, Tlen, D), jnp.bfloat16)
    sink = jnp.zeros((H,), jnp.float32)
    kw = dict(kind=kind, block=Tlen, causal=False)
    if kind == "asinh":
        kw.update(n=16, c=8.0)
        ref = O.asinh_attention(q, k, v, n=16, c=8.0,
                                sink_logit=sink[None, :, None, None])
        got = flash_attention(q, k, v, sink, **kw)
    else:
        ref = O.softmax_attention(q, k, v)
        got = flash_attention(q, k, v, None, **kw)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(ref, np.float32),
                               rtol=2e-2, atol=2e-2)


@_need_tpu()
def test_flash_bidirectional_vjp_matches_dense():
    from vision import ops as O
    from vision.kernels.flash import flash_attention
    key = jax.random.PRNGKey(3)
    k1, k2, k3 = jax.random.split(key, 3)
    Bsz, H, Tlen, D = 1, 2, 197, 64
    q = jax.random.normal(k1, (Bsz, H, Tlen, D), jnp.bfloat16)
    k = jax.random.normal(k2, (Bsz, H, Tlen, D), jnp.bfloat16)
    v = jax.random.normal(k3, (Bsz, H, Tlen, D), jnp.bfloat16)

    def f(qq, kk, vv):
        return flash_attention(qq, kk, vv, None, kind="softmax", block=Tlen,
                               causal=False).sum()

    def fr(qq, kk, vv):
        return O.softmax_attention(qq, kk, vv).sum().astype(jnp.float32)

    gk = jax.grad(f, (0, 1, 2))(q, k, v)
    gr = jax.grad(fr, (0, 1, 2))(q, k, v)
    for a, c in zip(gk, gr):
        np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(c, np.float32),
                                   rtol=5e-2, atol=5e-2)
