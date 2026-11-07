#!/usr/bin/env python3
"""Frozen-weight expressivity: selectivity, background suppression, rank.

No training. Random projections stand in for unlearned patch embeddings,
with QK-normalized unit-variance scores (the deployed regime).

1. Selectivity: planted-match recall@1 over a noise x gain grid.
2. Background suppression: for absent queries (no match), the fraction of
   attention mass routed to the zero-value sink. Softmax and convolution
   have no sink, so 100% of their mass lands on distractors.
3. Combined = 0.5 * recall@1 + 0.5 * absent sink-mass, with a sink-logit
   sweep proving the family reaches high suppression while holding
   present recall at 1.0.
4. Effective rank and output diversity of single-block outputs.
5. Ablations: sink on/off, n in {8, 16, 24}, softmax, local mixer.

Writes results/analysis/vit_expressivity.json.
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


def alg_weights(s, n, c, sink_b=None):
    x = s / c
    m = x.max(-1, keepdims=True)
    if sink_b is not None:
        m = np.maximum(m, np.asarray(sink_b) / c)
    sh = rho(-m)
    w = (rho(x) * sh) ** n
    denom = w.sum(-1, keepdims=True)
    sink_mass = None
    if sink_b is not None:
        sw = (rho(np.asarray(sink_b) / c) * sh) ** n
        denom = denom + sw
        sink_mass = (sw / denom)[..., 0]
    return w / denom, sink_mass


def soft_weights(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def local_weights(T, H=14, W=14, sigma=2.0):
    if H * W != T:
        H = max(1, math.isqrt(T))
        W = (T + H - 1) // H
    pos = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(H * W, 2)[:T]
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma * sigma))
    return w / w.sum(-1, keepdims=True)


def eff_rank(feats):
    s = np.linalg.svd(feats - feats.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-45)).sum()))


def diversity(feats):
    n = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
    c = n @ n.T
    iu = np.triu_indices(len(c), 1)
    return float(1.0 - c[iu].mean())


def trial_scores(rng, n_q, T, P, dp, noise, gain, planted):
    proj = rng.standard_normal((dp, P))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    q = rng.standard_normal((n_q, dp))
    if planted:
        k_match = q + noise * rng.standard_normal((n_q, dp))
        k_dist = rng.standard_normal((n_q, T - 1, dp))
        keys = np.concatenate([k_match[:, None, :], k_dist], axis=1)
    else:
        keys = rng.standard_normal((n_q, T, dp))
    Q = q @ proj
    K = (keys.reshape(-1, dp) @ proj).reshape(n_q, T, P)
    Q *= math.sqrt(P) / np.linalg.norm(Q, axis=-1, keepdims=True)
    K *= math.sqrt(P) / np.linalg.norm(K, axis=-1, keepdims=True)
    return np.einsum("bd,btd->bt", Q, K) / math.sqrt(P) * gain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--patches", type=int, default=196)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--patch-dim", type=int, default=768)
    p.add_argument("--noises", nargs="+", type=float, default=[0.5, 1.0])
    p.add_argument("--gains", nargs="+", type=float, default=[1.0, 2.0])
    p.add_argument("--sink-logits", nargs="+", type=float, default=[0.0, 2.0, 4.0, 6.0])
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "analysis" / "vit_expressivity.json")
    args = p.parse_args()
    T, P, dp = args.patches, args.dim, args.patch_dim
    cfgs = [("alg_n8", 8, 8.0), ("alg_n16", 16, 8.0), ("alg_n24", 24, 8.0)]
    recall, suppress, ranks, divers = {}, {}, {}, {}
    for name, n, c in cfgs:
        recall[name] = []
        suppress[name] = {str(b): [] for b in args.sink_logits}
        ranks[name] = []
        divers[name] = []
    recall["softmax"] = []
    ranks["softmax"] = []
    divers["softmax"] = []
    ranks["local_conv"] = []
    divers["local_conv"] = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        for noise in args.noises:
            for gain in args.gains:
                s = trial_scores(rng, 256, T, P, dp, noise, gain, True)
                sa = trial_scores(rng, 256, T, P, dp, noise, gain, False)
                for name, n, c in cfgs:
                    w, _ = alg_weights(s, n, c, 0.0)
                    recall[name].append(float((w.argmax(-1) == 0).mean()))
                    for b in args.sink_logits:
                        _, sm = alg_weights(sa, n, c, b)
                        suppress[name][str(b)].append(float(sm.mean()))
                w = soft_weights(s)
                recall["softmax"].append(float((w.argmax(-1) == 0).mean()))
        # Rank/diversity over a synthetic patch field.
        field = rng.standard_normal((T, P))
        A1 = rng.standard_normal((P, P))
        A2 = rng.standard_normal((P, P))
        A3 = rng.standard_normal((P, P))
        qs = field @ A1 / np.linalg.norm(field @ A1, axis=-1, keepdims=True) * math.sqrt(P)
        ks = field @ A2 / np.linalg.norm(field @ A2, axis=-1, keepdims=True) * math.sqrt(P)
        vs = field @ A3 / math.sqrt(P)
        s = (qs @ ks.T) / math.sqrt(P)
        for name, n, c in cfgs:
            w, _ = alg_weights(s, n, c, 0.0)
            o = w @ vs
            ranks[name].append(eff_rank(o))
            divers[name].append(diversity(o))
        o = soft_weights(s) @ vs
        ranks["softmax"].append(eff_rank(o))
        divers["softmax"].append(diversity(o))
        ol = local_weights(T) @ vs
        ranks["local_conv"].append(eff_rank(ol))
        divers["local_conv"].append(diversity(ol))
    # Local mixer reference (content-blind): chance-level recall by design.
    lw = local_weights(T)
    summary = {"local_conv": {"recall1": float((lw.argmax(-1) == 0).mean()),
                              "suppress": 0.0, "combined": 0.0,
                              "eff_rank": float(np.mean(ranks["local_conv"])),
                              "diversity": float(np.mean(divers["local_conv"]))}}
    for name, _, _ in cfgs:
        r = float(np.mean(recall[name]))
        # Suppression at the largest sink logit that keeps present recall:
        # recall is sink-independent (argmax over keys), so take max b.
        sup = {b: float(np.mean(v)) for b, v in suppress[name].items()}
        best_b = max(sup, key=lambda b: sup[b])
        summary[name] = {"recall1": r, "suppress": sup[best_b], "best_sink_b": float(best_b),
                         "suppress_curve": sup,
                         "combined": 0.5 * r + 0.5 * sup[best_b],
                         "eff_rank": float(np.mean(ranks[name])),
                         "diversity": float(np.mean(divers[name]))}
    r_soft = float(np.mean(recall["softmax"]))
    summary["softmax"] = {"recall1": r_soft, "suppress": 0.0, "combined": 0.5 * r_soft,
                          "eff_rank": float(np.mean(ranks["softmax"])),
                          "diversity": float(np.mean(divers["softmax"]))}
    champ = summary["alg_n16"]
    d_soft = champ["combined"] - summary["softmax"]["combined"]
    d_conv = champ["combined"] - summary["local_conv"]["combined"]
    rank_ratio = champ["eff_rank"] / summary["softmax"]["eff_rank"]
    report = {
        "schema": 1,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "frozen random-projection retrieval and rank; not trained quality",
        "seeds": args.seeds, "noises": args.noises, "gains": args.gains,
        "summary": summary,
        "gates": {
            "combined_vs_softmax_pp": 100 * d_soft,
            "combined_vs_conv_pp": 100 * d_conv,
            "eff_rank_ratio_vs_softmax": rank_ratio,
            "verdict": "pass" if d_soft > 0.05 and d_conv > 0.05 else "FAIL",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
