#!/usr/bin/env python3
"""Trace the algebraic training objective and reject transcendental ops.

Covers ``algebraic_vit.head_loss`` and its gradient, including the dense
attention and likelihood paths. Rejects exp, log, trigonometric,
hyperbolic, sigmoid, erf, and general-power primitives. Initialization,
the cosine schedule, telemetry, and evaluation run outside this claim.

Usage:
  python scripts/audit_algebraic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# NOTE: "integer_pow" (repeated multiplication fused by XLA) is algebraic and
# allowed; only the general real power "pow" is transcendental.
REJECTED = ("exp", "log", "sin", "cos", "tan", "atan", "sinh", "cosh", "tanh",
            "asinh", "acosh", "atanh", "sigmoid", "logistic", "erf", "erfc",
            "pow")


def main():
    import jax
    import jax.numpy as jnp
    from vision import algebraic_vit as A
    from vision import config as C

    cfg = C.algebraic_vit("Ti", image_size=32, patch=4, num_classes=100,
                          remat=False, dtype="float32")
    key = jax.random.PRNGKey(0)
    params = A.init_params(cfg, key)
    tables = A.position_tables(cfg)
    x = jax.random.normal(key, (2, 32, 32, 3), jnp.float32)
    y = jnp.array([3, 7])

    def objective(p):
        obj, _ = A.head_loss(p, x, y, cfg, tables)
        return obj

    jaxpr = jax.make_jaxpr(objective)(params)
    prims = set()

    def visit(eqn):
        prims.add(str(eqn.primitive))
        for value in eqn.params.values():
            subs = value if isinstance(value, (list, tuple)) else [value]
            for sub in subs:
                inner = getattr(sub, "jaxpr", sub)
                for s in getattr(inner, "eqns", ()):
                    visit(s)

    for eqn in jaxpr.eqns:
        visit(eqn)

    grad_fn = jax.grad(objective)
    gjaxpr = jax.make_jaxpr(grad_fn)(params)
    for eqn in gjaxpr.eqns:
        visit(eqn)

    bad = sorted(p for p in prims
                 if any(r == p or (r in p and "integer_pow" not in p) for r in REJECTED))
    print("primitives:", sorted(prims))
    if bad:
        raise SystemExit(f"REJECTED primitives in algebraic path: {bad}")
    print("algebraic objective + gradient: no transcendental primitives")


if __name__ == "__main__":
    main()
