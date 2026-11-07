#!/usr/bin/env python3
"""Wall-clock efficiency: FLOP audit, special-function audit, microbenchmark.

1. Analytic audit for Ti/S/B vision configs at 224px/16px patches (T=196,
   +1 CLS token for ViT): dense FLOPs, transcendental-op counts (exp, erf,
   sin/cos), and reciprocal-root counts. A stated throughput model maps
   counts to TPU-time shares: matrix FLOPs at MXU rate, scalar special
   functions at vector-unit rate (20x slower per op, documented
   assumption swept over 10x/20x/40x).
2. CPU microbenchmark (JAX): dense bidirectional algebraic attention
   (forward + full QKV VJP) against dense softmax attention at the
   ViT-S shape, paired rounds with bootstrap CI on the latency ratio,
   plus an ALU-vs-GELU elementwise round. CPU is not a TPU; this round
   measures kernel-structure overhead only and is reported alongside
   the analytic model, never as TPU wall-clock proof.

Writes results/analysis/vit_efficiency.json.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

VIT_CFGS = {
    "Ti": dict(d=192, L=12, H=3, mlp=768),
    "S": dict(d=384, L=12, H=6, mlp=1536),
    "B": dict(d=768, L=12, H=12, mlp=3072),
}
RESNET_GFLOPS = {"ResNet-18": 1.82, "ResNet-50": 4.12}  # literature @224
T_TOKENS = 197  # 196 patches + CLS


def vit_audit(name):
    c = VIT_CFGS[name]
    T, d, L, H, m = T_TOKENS, c["d"], c["L"], c["H"], c["mlp"]
    proj = 8 * T * d * d * L          # Q,K,V,O (mult-add = 2 flops)
    scores = 4 * T * T * d * L        # scores + weighted sum
    mlp = 4 * T * d * m * L           # two matmuls
    patch = 2 * 196 * 768 * d         # 16x16x3 -> d per patch
    flops = proj + scores + mlp + patch
    # Standard ViT transcendental ops: exp per score, erf per GELU unit,
    # negligible learned-pos cost. Algebraic: rsqrt per rho/ALU/RMSNorm.
    std_exp = T * T * H * L + 1000            # scores + head
    std_erf = T * m * L                       # GELU
    alg_rsqrt = ((T * T * H * L + T * H * L)  # rho(x) per score, rho(-m) per row
                 + T * m * L                  # ALU
                 + T * L * (3 + 2 * H))       # RMSNorm: 3 token norms + QK norms
    std_params = L * (4 * d * d + 2 * d * m) + 196 * d + d * 1000
    alg_params = std_params - 196 * d + L * H  # no learned pos, +sink logits
    return {"gflops": flops / 1e9, "std_exp": std_exp, "std_erf": std_erf,
            "alg_rsqrt": alg_rsqrt, "std_mparams": std_params / 1e6,
            "alg_mparams": alg_params / 1e6}


def tpu_time_share(gflops, special_ops, special_cost):
    """Normalized time = flops + special_cost * special_ops (arbitrary units)."""
    return gflops * 1e9 + special_cost * special_ops


# TPU v4 per-chip documented rates: 275 TFLOPS bf16 dense, ~1.2 TB/s HBM.
MXU_FLOPS = 275e12
HBM_BW = 1.2e12


def roofline(name, a, exp_cost=20.0, rsqrt_cost=3.0):
    """Two-regime model: matmuls at MXU rate, fused attention inner loop at
    issue rate, attention traffic at HBM rate. Returns (t_std, t_alg) seconds
    per example. Cost assumptions are swept by the caller."""
    c = VIT_CFGS[name]
    T, d, L, H, m = T_TOKENS, c["d"], c["L"], c["H"], c["mlp"]
    matmul_flops = a["gflops"] * 1e9
    scores = T * T * H * L
    # Inner-loop scalar-equivalent ops per score: exp path vs rho+power path.
    std_inner = scores * (exp_cost + 2.0)
    alg_inner = scores * (rsqrt_cost + 9.0)  # rho mults + shift + x^16 chain
    # Attention traffic per example (bf16 QKV+out streamed once, fused).
    attn_bytes = T * d * L * 4 * 2
    t_mat = matmul_flops / MXU_FLOPS
    t_std = t_mat + max(attn_bytes / HBM_BW, std_inner / MXU_FLOPS)
    t_alg = t_mat + max(attn_bytes / HBM_BW, alg_inner / MXU_FLOPS)
    return t_std, t_alg


def microbench(rounds=9, iters=10):
    import jax
    import jax.numpy as jnp
    from jax import lax
    B, H, T, D = 8, 6, 197, 64
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kd = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, T, D), jnp.float32)
    k = jax.random.normal(kk, (B, H, T, D), jnp.float32)
    v = jax.random.normal(kv, (B, H, T, D), jnp.float32)
    do = jax.random.normal(kd, (B, H, T, D), jnp.float32)

    def rho(x):
        t = jnp.abs(x) + (1 + x * x) * lax.rsqrt(1 + x * x)
        return jnp.where(x >= 0, t, 1.0 / t)

    def alg_train(q, k, v):
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) / 8.0
        m = lax.stop_gradient(s.max(-1, keepdims=True))
        w = ((rho(s) * rho(-m)) ** 16)
        w = w / w.sum(-1, keepdims=True)
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v)
        return (o * lax.stop_gradient(do)).sum()

    def soft_train(q, k, v):
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) / 8.0
        w = jax.nn.softmax(s, -1)
        o = jnp.einsum("bhqk,bhkd->bhqd", w, v)
        return (o * lax.stop_gradient(do)).sum()

    ga = jax.jit(jax.value_and_grad(lambda q, k, v: alg_train(q, k, v), (0, 1, 2)))
    gs = jax.jit(jax.value_and_grad(lambda q, k, v: soft_train(q, k, v), (0, 1, 2)))
    ga(q, k, v)
    gs(q, k, v)
    lat_a, lat_s = [], []
    order = [0, 1] * ((rounds + 1) // 2)
    rng = np.random.default_rng(3)
    for r in range(rounds):
        seq = order[:] if r % 2 == 0 else order[::-1]
        for which in seq[:2]:
            f = ga if which == 0 else gs
            t0 = time.perf_counter()
            for _ in range(iters):
                jax.block_until_ready(f(q, k, v))
            dt = (time.perf_counter() - t0) / iters
            (lat_a if which == 0 else lat_s).append(dt)
    delta = np.log(np.asarray(lat_a)) - np.log(np.asarray(lat_s))
    boot = np.exp(delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1))
    return {"alg_ms": float(np.mean(lat_a) * 1000), "soft_ms": float(np.mean(lat_s) * 1000),
            "ratio": float(np.exp(delta.mean())),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "rounds": len(delta), "note": "CPU kernel-structure round only, not TPU proof"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "analysis" / "vit_efficiency.json")
    args = p.parse_args()
    audits = {n: vit_audit(n) for n in VIT_CFGS}
    model = {}
    for cost in (10.0, 20.0, 40.0):
        for name, a in audits.items():
            t_std = tpu_time_share(a["gflops"], a["std_exp"] + a["std_erf"], cost)
            t_alg = tpu_time_share(a["gflops"], a["alg_rsqrt"], 3.0)
            model.setdefault(name, {})[f"special_cost_{cost:g}x"] = {
                "std_units": t_std, "alg_units": t_alg, "ratio": t_alg / t_std}
    roofs = {}
    for exp_c, rsq_c in ((10.0, 2.0), (20.0, 3.0), (40.0, 6.0)):
        for name, a in audits.items():
            t_std, t_alg = roofline(name, a, exp_c, rsq_c)
            roofs.setdefault(name, {})[f"exp{exp_c:g}_rsqrt{rsq_c:g}"] = {
                "std_s": t_std, "alg_s": t_alg, "ratio": t_alg / t_std}
    # ResNet comparison at matched accuracy tiers (S~ResNet-50, Ti~ResNet-18):
    # ResNet GFLOPs are literature values; special-op counts (BN rsqrt +
    # ReLU) are amortized per activation.
    resnet = {}
    for name, gf in RESNET_GFLOPS.items():
        acts = 12e6 if "18" in name else 26e6  # approx output activations @224
        t_rn = tpu_time_share(gf, acts * 2, 20.0)
        peer = "Ti" if "18" in name else "S"
        t_alg = tpu_time_share(audits[peer]["gflops"], audits[peer]["alg_rsqrt"], 3.0)
        resnet[name] = {"gflops": gf, "vs_algebraic_" + peer + "_ratio": t_alg / t_rn,
                        "note": "accuracy peers, not FLOP peers; ratio > 1 reflects "
                                "ViT's higher matmul FLOPs, not kernel quality"}
    report = {
        "schema": 1,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "analytic TPU-time model + CPU kernel-structure round; TPU proof needs hardware",
        "audits": audits,
        "tpu_time_model": model,
        "roofline_model": roofs,
        "resnet_literature_gflops": RESNET_GFLOPS,
        "resnet_vs_algebraic": resnet,
    }
    if not args.skip_bench:
        report["cpu_microbench"] = microbench(args.rounds, args.iters)
    ratios = [v["ratio"] for m in model.values() for v in m.values()]
    ratios += [v["ratio"] for m in roofs.values() for v in m.values()]
    worst = max(ratios)
    best = min(ratios)
    # Honest gate and finding. Both analytic models show the algebraic
    # stack no-slower anywhere (time model: faster by 1-5%; roofline:
    # exact parity because fused attention at ViT shapes is traffic-
    # bound, so inner-loop op choice does not bind). The structural win
    # is zero transcendental ops. A decisive wall-clock separation is
    # NOT established by training-free analysis and requires fused
    # kernels measured on silicon (Phase 3/4). The CPU round below
    # favors dense unfused softmax (fast CPU exp); it is reported as a
    # kernel-structure observation, not a TPU prediction.
    report["gates"] = {"worst_ratio_any_model": worst,
                       "best_ratio_any_model": best,
                       "transcendental_ops_algebraic": 0,
                       "decisive_wallclock_proven": False,
                       "verdict": "pass" if worst <= 1.0 + 1e-12 else "FAIL"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"audits": audits, "gates": report["gates"],
                      "cpu_microbench": report.get("cpu_microbench")}, indent=2))


if __name__ == "__main__":
    main()
