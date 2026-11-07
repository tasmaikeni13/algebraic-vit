#!/usr/bin/env python3
"""Attention-link shootout for bidirectional vision transformers (no training).

Compares four routing kernels on synthetic ViT score rows (T patches,
head dim 64, unit-sphere q/k with gain sweeps):
  C1 rho-power + sink : w ~ rho(s/8)^16, sink logit b, row mass <= 1
  C2 clipped binomial : w ~ clip(1+s/4, 1e-3, 8)^8, exact zeros, no sink
  C3 Cauchy gap       : w ~ (1+((m-s)/4)^2)^-4, row-max relative, no sink
  S0 softmax          : w ~ exp(s), no sink (Standard ViT reference)

Metrics per link and gain: weight entropy, top-1 mass, effective
candidates (exp of entropy), Jacobian spectral proxy (max row gradient
norm of the top weight w.r.t. scores), and finiteness under extreme
scores. Falsifier curves: sink-mass-vs-shift t for C1, zero fraction
for C2, top1/top2 log-ratio-vs-gap slope for C3 vs S0.

Writes results/analysis/vit_link_analysis.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"


def rho(x):
    t = np.abs(x) + np.sqrt(1.0 + x * x)
    return np.where(x >= 0, t, 1.0 / t)


def weights(name, s, sink_b=0.0):
    """Normalized weights for one score matrix (..., T). Returns (w, aux)."""
    if name == "C1":
        x = s / 8.0
        m = np.maximum(x.max(-1, keepdims=True), sink_b / 8.0)
        sh = rho(-m)
        w = (rho(x) * sh) ** 16
        denom = w.sum(-1, keepdims=True) + (rho(sink_b / 8.0) * sh) ** 16
        return w / denom, {"sink_mass": float(((rho(sink_b / 8.0) * sh) ** 16 / denom).mean())}
    if name == "C2":
        w = np.clip(1.0 + s / 4.0, 1e-3, 8.0) ** 8
        zf = float((w <= 1e-3 ** 8 * 1.01).mean())
        return w / w.sum(-1, keepdims=True), {"zero_fraction": zf}
    if name == "C3":
        m = s.max(-1, keepdims=True)
        w = (1.0 + ((m - s) / 4.0) ** 2) ** -4
        return w / w.sum(-1, keepdims=True), {}
    if name == "S0":
        e = np.exp(s - s.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True), {}
    raise ValueError(name)


def row_stats(w):
    ent = -(w * np.log(np.clip(w, 1e-45, 1.0))).sum(-1)
    top1 = w.max(-1)
    # Jacobian proxy: d(top weight)/d(scores) has max entry <= max w(1-w)*slope;
    # report max over rows of w*(1-w) as the concentration-modulated sensitivity.
    sens = (w * (1.0 - w)).max(-1)
    return ent, top1, sens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--patches", type=int, default=196)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--gains", nargs="+", type=float, default=[1.0, 1.5, 2.0, 4.0])
    p.add_argument("--out", type=Path, default=OUT / "vit_link_analysis.json")
    args = p.parse_args()
    T, d = args.patches, args.dim
    links = ["C1", "C2", "C3", "S0"]
    report = {
        "schema": 1,
        "links": links,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "static bidirectional routing on synthetic ViT scores; not trained retrieval",
        "seeds": list(range(args.seeds)),
        "gains": args.gains,
        "trials": [],
        "falsifiers": {},
    }
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((args.rows, d))
        k = rng.standard_normal((args.rows, T, d))
        q *= math.sqrt(d) / np.linalg.norm(q, axis=-1, keepdims=True)
        k *= math.sqrt(d) / np.linalg.norm(k, axis=-1, keepdims=True)
        unit = np.einsum("bd,btd->bt", q, k) / math.sqrt(d)
        for scale in args.gains:
            s = unit * scale
            row = {"seed": seed, "gain": scale, "links": {}}
            for name in links:
                w, aux = weights(name, s)
                ent, top1, sens = row_stats(w)
                cell = {
                    "entropy_mean": float(ent.mean()),
                    "entropy_std": float(ent.std()),
                    "top1_mean": float(top1.mean()),
                    "eff_candidates": float(np.exp(ent).mean()),
                    "sensitivity_mean": float(sens.mean()),
                    "finite": bool(np.all(np.isfinite(w))),
                }
                cell.update(aux)
                row["links"][name] = cell
            report["trials"].append(row)

    # Falsifier 1: C1 sink mass falls as a uniform shift t is added to a row.
    rng = np.random.default_rng(1234)
    s0 = rng.standard_normal((512, T))
    shifts = [0.0, 2.0, 4.0, 8.0]
    sink_curve, soft_invariant = [], True
    for t in shifts:
        w_c1, aux = weights("C1", s0 + t)
        sink_curve.append(aux["sink_mass"])
        w_s0, _ = weights("S0", s0 + t)
        w_s0b, _ = weights("S0", s0)
        soft_invariant &= bool(np.allclose(w_s0, w_s0b, atol=1e-6))
    report["falsifiers"]["C1_sink_mass_vs_shift"] = {
        "shifts": shifts, "sink_mass": sink_curve,
        "softmax_row_invariant": soft_invariant,
        "verdict": "pass" if sink_curve[-1] < sink_curve[0] and soft_invariant else "FAIL",
    }
    # Falsifier 2: C2 exact-zero fraction on realistically scaled ViT scores.
    w_c2, aux2 = weights("C2", s0 * 2.0)
    report["falsifiers"]["C2_zero_fraction"] = {
        "zero_fraction": aux2["zero_fraction"],
        "verdict": "pass" if aux2["zero_fraction"] > 0 else "FAIL",
    }
    # Falsifier 3: C3 top1/top2 log-ratio slope rises then falls with the gap
    # (d/dg 4 log(1+(g/4)^2) peaks at g=4); S0 slope is constant 1.
    gaps = [2.0, 4.0, 6.0, 10.0]
    c3_ratios, s0_ratios = [], []
    for g in gaps:
        row = np.zeros((1, 8))
        row[0, 0] = g
        c3, _ = weights("C3", row)
        s0w, _ = weights("S0", row)
        c3_ratios.append(float(np.log(c3[0, 0] / c3[0, 1])))
        s0_ratios.append(float(np.log(s0w[0, 0] / s0w[0, 1])))
    slopes_c3 = (np.diff(c3_ratios) / np.diff(gaps)).tolist()
    slopes_s0 = (np.diff(s0_ratios) / np.diff(gaps)).tolist()
    report["falsifiers"]["C3_slope_shape_vs_S0_linear"] = {
        "gaps": gaps, "C3_log_ratios": c3_ratios, "S0_log_ratios": s0_ratios,
        "C3_slopes": slopes_c3, "S0_slopes": slopes_s0,
        "verdict": "pass" if slopes_c3[-1] < slopes_c3[0]
        and abs(slopes_s0[-1] - 1.0) < 1e-9 else "FAIL",
    }
    # Falsifier 4: extreme-score finiteness (1e6 scale) for every link.
    big = s0 * 1e6
    fin = {}
    for name in links:
        w, _ = weights(name, big)
        fin[name] = bool(np.all(np.isfinite(w)))
    report["falsifiers"]["extreme_finiteness"] = fin

    # Aggregate verdict: mean entropy gap C1 vs S0 and sharpness control.
    import collections
    agg = collections.defaultdict(list)
    for row in report["trials"]:
        for name in links:
            agg[(name, "entropy_mean")].append(row["links"][name]["entropy_mean"])
            agg[(name, "top1_mean")].append(row["links"][name]["top1_mean"])
    report["aggregate"] = {f"{n}.{m}": float(np.mean(v)) for (n, m), v in agg.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))
    print(json.dumps(report["falsifiers"], indent=2))


if __name__ == "__main__":
    main()
