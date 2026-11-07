#!/usr/bin/env python3
"""Numerical certificates for the algebraic vision operators (float64).

1. Link identity: bidirectional rho(s/c)^n weights with sink equal
   softmax(n*arcsinh(s/c)) over keys+sink to 1e-10.
2. Global bound: rho(x) <= 2|x|+1 on a wide grid, plus rho(x)rho(-x)=1
   and the group law rho(x)rho(y)=rho(x*sqrt(1+y^2)+y*sqrt(1+x^2)).
3. Max-shift finiteness: weights stay finite with row-sum 1 at 1e6 scale.
4. Cayley composition: C(m+n) = C(m)C(n) - S(m)S(n) for the 2D tables,
   and unit norm C^2+S^2 = 1.
5. Head gradient: the algebraic logit gradient (n/c)(1+x^2)^-1/2 (p-y)
   matches complex-step differentiation of exact log loss to 1e-10.

Writes results/analysis/vit_certificates.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rho(x):
    t = np.abs(x) + np.sqrt(1.0 + x * x)
    return np.where(x >= 0, t, 1.0 / t)


def alg_weights(s, n, c, sink_b):
    x = s / c
    m = np.maximum(x.max(-1, keepdims=True), sink_b / c)
    sh = rho(-m)
    w = (rho(x) * sh) ** n
    sw = (rho(np.asarray(sink_b) / c) * sh) ** n
    return w / (w.sum(-1, keepdims=True) + sw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "analysis" / "vit_certificates.json")
    args = p.parse_args()
    certs = {}
    rng = np.random.default_rng(0)

    # 1. Link identity (bidirectional, with sink).
    worst = 0.0
    for (n, c) in [(8, 8.0), (16, 8.0), (24, 8.0), (8, 2.0)]:
        s = rng.standard_normal((4, 197)) * 5
        b = 0.7
        w = alg_weights(s, n, c, b)
        logits = np.concatenate([n * np.arcsinh(s / c),
                                 np.full((4, 1), n * np.arcsinh(b / c))], -1)
        ref = np.exp(logits - logits.max(-1, keepdims=True))
        ref = ref / ref.sum(-1, keepdims=True)
        worst = max(worst, float(np.abs(w - ref[..., :-1]).max()))
    certs["link_identity_max_abs_err"] = worst

    # 2. Rho identities and global bound.
    x = np.linspace(-1e6, 1e6, 20001)
    certs["rho_bound_max_violation"] = float(np.maximum(rho(x) - (2 * np.abs(x) + 1), 0).max())
    certs["reciprocal_max_err"] = float(np.abs(rho(x) * rho(-x) - 1).max())
    xx = np.linspace(-50, 50, 2001)
    yy = np.linspace(-7, 7, 2001)
    oplus = xx * np.sqrt(1 + yy ** 2) + yy * np.sqrt(1 + xx ** 2)
    num = np.abs(rho(xx) * rho(yy) - rho(oplus))
    den = np.abs(rho(oplus))
    certs["group_law_max_rel_err"] = float((num / np.maximum(den, 1e-300)).max())

    # 3. Max-shift finiteness.
    big = rng.standard_normal((4, 197)) * 1e6
    wb = alg_weights(big, 16, 8.0, 0.0)
    certs["extreme_finite"] = bool(np.all(np.isfinite(wb)))
    certs["extreme_rowsum_err"] = float(np.abs(wb.sum(-1) - 1).max())

    # 4. Cayley composition for 2D row/col tables.
    w = 0.5 * (1.0 / math.sqrt(math.sqrt(10000.0))) ** np.arange(16)
    c0 = (1 - w * w) / (1 + w * w)
    s0 = (2 * w) / (1 + w * w)
    C = np.ones((14, 16))
    S = np.zeros((14, 16))
    for m in range(1, 14):
        C[m] = C[m - 1] * c0 - S[m - 1] * s0
        S[m] = C[m - 1] * s0 + S[m - 1] * c0
    comp_err, norm_err = 0.0, float(np.abs(C * C + S * S - 1).max())
    for a in range(14):
        for b_ in range(14 - a):
            comp_err = max(comp_err, float(np.abs(C[a + b_] - (C[a] * C[b_] - S[a] * S[b_])).max()))
            comp_err = max(comp_err, float(np.abs(S[a + b_] - (S[a] * C[b_] + C[a] * S[b_])).max()))
    certs["cayley_composition_err"] = comp_err
    certs["cayley_unit_norm_err"] = norm_err

    # 5. Head gradient vs complex-step exact log loss.
    n, c = 8, 2.0
    z = rng.standard_normal(11) * 4
    t = 3
    h = 1e-20

    # Complex-step the scalar exact NLL.
    def nll(zz):
        lp = n * np.arcsinh(zz / c)
        m = np.max(lp.real) if np.iscomplexobj(lp) else lp.max()
        return -(lp[t] - m - np.log(np.sum(np.exp(lp - m))))
    g_cs = np.array([float(np.imag(nll(z.astype(complex) + 1j * h * np.eye(11)[j])) / h)
                     for j in range(11)])
    x = z / c
    w8 = rho(x) ** n
    pr = w8 / w8.sum()
    y = np.zeros(11)
    y[t] = 1.0
    g_alg = (n / c) / np.sqrt(1 + x * x) * (pr - y)
    certs["head_grad_max_abs_err"] = float(np.abs(g_cs - g_alg).max())

    worst_err = max(certs["link_identity_max_abs_err"], certs["head_grad_max_abs_err"],
                    certs["cayley_composition_err"], certs["cayley_unit_norm_err"])
    report = {
        "schema": 1,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "float64 numerical certificates; Lean proofs are the deductive counterpart",
        "certificates": certs,
        "gates": {"max_identity_err": worst_err,
                  "verdict": "pass" if worst_err < 1e-8 and certs["extreme_finite"]
                  and certs["rho_bound_max_violation"] == 0.0 else "FAIL"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
