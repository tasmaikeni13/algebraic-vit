"""Numerical checks of the vision components (CPU, float64/float32)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vision import algebraic_vit as A
from vision import config as C
from vision import ops as O
from vision import resnet as R
from vision import standard_vit as S

jax.config.update("jax_enable_x64", True)


def test_rho_identities():
    x = jnp.linspace(-50, 50, 1001, dtype=jnp.float64)
    y = jnp.linspace(-3, 7, 1001, dtype=jnp.float64)
    np.testing.assert_allclose(O.rho(x), np.exp(np.arcsinh(np.asarray(x))), rtol=1e-12)
    np.testing.assert_allclose(O.rho(x) * O.rho(-x), 1.0, rtol=1e-12)
    oplus = x * jnp.sqrt(1 + y * y) + y * jnp.sqrt(1 + x * x)
    np.testing.assert_allclose(O.rho(x) * O.rho(y), O.rho(oplus), rtol=1e-10)


@pytest.mark.parametrize("n,c", [(8, 8.0), (16, 8.0), (24, 8.0)])
def test_bidirectional_weights_equal_softmax_of_compressed_logits(n, c):
    key = jax.random.PRNGKey(0)
    s = jax.random.normal(key, (2, 3, 65, 65), jnp.float64) * 5
    b = jnp.full((1, 3, 1, 1), 0.7, jnp.float64)
    w = O.asinh_power_weights(s, n, c, b)
    logits = n * jnp.arcsinh(s / c)
    tail = jnp.broadcast_to(n * jnp.arcsinh(b / c), s.shape[:-1] + (1,))
    full = jnp.concatenate([logits, tail], -1)
    ref = jax.nn.softmax(full, -1)[..., :-1]
    np.testing.assert_allclose(w, ref, rtol=1e-10, atol=1e-14)
    big = O.asinh_power_weights(s * 1e6, n, c, None)
    assert np.all(np.isfinite(big)) and np.allclose(big.sum(-1), 1.0)


def test_algebraic_likelihood_gradient_is_exact_log_loss_gradient():
    key = jax.random.PRNGKey(1)
    z = jax.random.normal(key, (6, 11), jnp.float64) * 4
    t = jnp.array([0, 3, 10, 5, 5, 1])

    def true_nll(z):
        lp = 8 * jnp.arcsinh(z / 2.0)
        lp = lp - jax.nn.logsumexp(lp, axis=-1, keepdims=True)
        return -jnp.mean(jnp.take_along_axis(lp, t[:, None], axis=-1)[:, 0])

    _, vjp_fn = jax.vjp(lambda zz: O.algebraic_likelihood(zz, t, 8, 2.0)[0], z)
    (got,) = vjp_fn(jnp.ones((), jnp.float64))
    (want,) = jax.grad(true_nll)(z),
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


def test_cayley_2d_unit_norm_and_shift_invariance():
    rc, rs, cc, cs = O.cayley_table_2d(64, 8, 8)
    np.testing.assert_allclose(np.asarray(rc) ** 2 + np.asarray(rs) ** 2, 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(cc) ** 2 + np.asarray(cs) ** 2, 1.0, atol=1e-6)
    key = jax.random.PRNGKey(2)
    v1 = jax.random.normal(key, (64,), jnp.float32)
    v2 = jax.random.normal(jax.random.split(key)[1], (64,), jnp.float32)

    def rot(v, m, n):
        a1, a2, b1, b2 = jnp.split(v, 4)
        return jnp.concatenate([a1 * rc[m] - a2 * rs[m], a1 * rs[m] + a2 * rc[m],
                                b1 * cc[n] - b2 * cs[n], b1 * cs[n] + b2 * cc[n]])

    s0 = rot(v1, 1, 2) @ rot(v2, 3, 4)
    s1 = rot(v1, 2, 3) @ rot(v2, 4, 5)
    np.testing.assert_allclose(s0, s1, atol=1e-5)


def test_alu_matches_gelu_statistics():
    import math
    x, w = np.polynomial.hermite_e.hermegauss(161)
    w = w / math.sqrt(2 * math.pi)

    def alu(z):
        return 0.5 * z * (1 + 1.2 * z / np.sqrt(1 + (1.2 * z) ** 2))

    def alu_d(z):
        u = 1.2 * z / np.sqrt(1 + (1.2 * z) ** 2)
        return 0.5 * (1 + u) + 0.5 * u / (1 + (1.2 * z) ** 2)

    from scipy import special
    gelu = lambda z: 0.5 * z * (1 + special.erf(z / math.sqrt(2)))
    g_alu = float(np.sum(w * alu_d(x) ** 2))
    g_gelu = float(np.sum(w * (0.5 * (1 + special.erf(x / math.sqrt(2)))
                               + x * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)) ** 2))
    assert abs(g_alu / g_gelu - 1) < 0.05
    l2 = math.sqrt(float(np.sum(w * (alu(2 * x) - gelu(2 * x)) ** 2)))
    assert l2 < 0.05


@pytest.mark.parametrize("arch", ["algebraic", "standard"])
def test_vit_forward_shapes_and_finite(arch):
    cfg = C.algebraic_vit("Ti", image_size=32, patch=4, num_classes=100,
                          remat=False, dtype="float32")
    if arch == "standard":
        cfg = C.standard_vit("Ti", image_size=32, patch=4, num_classes=100,
                             remat=False, dtype="float32")
    model = A if arch == "algebraic" else S
    key = jax.random.PRNGKey(3)
    params = model.init_params(cfg, key)
    x = jax.random.normal(key, (2, 32, 32, 3), jnp.float32)
    if arch == "algebraic":
        logits = model.forward_logits(params, x, cfg, model.position_tables(cfg))
    else:
        logits = model.forward_logits(params, x, cfg)
    assert logits.shape == (2, 100) and np.all(np.isfinite(np.asarray(logits)))


def test_resnet_forward_train_eval():
    cfg = C.resnet(18, image_size=32, num_classes=100)
    key = jax.random.PRNGKey(4)
    params, stats = R.init_params(cfg, key)
    x = jax.random.normal(key, (4, 32, 32, 3), jnp.float32)
    logits, stats2 = R.forward_logits(params, stats, x, cfg, True)
    assert logits.shape == (4, 100) and np.all(np.isfinite(np.asarray(logits)))
    logits_e, _ = R.forward_logits(params, stats2, x, cfg, False)
    assert np.all(np.isfinite(np.asarray(logits_e)))


def test_vit_gradients_finite():
    cfg = C.algebraic_vit("Ti", image_size=32, patch=4, num_classes=100,
                          remat=False, dtype="float32")
    key = jax.random.PRNGKey(5)
    params = A.init_params(cfg, key)
    tables = A.position_tables(cfg)
    x = jax.random.normal(key, (2, 32, 32, 3), jnp.float32)
    y = jnp.array([1, 2])

    def obj(p):
        o, _ = A.head_loss(p, x, y, cfg, tables)
        return o

    g = jax.grad(obj)(params)
    for leaf in jax.tree_util.tree_leaves(g):
        assert np.all(np.isfinite(np.asarray(leaf)))
