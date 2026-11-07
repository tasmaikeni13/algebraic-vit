#!/usr/bin/env python3
"""CIFAR-100 sanity tier: Algebraic ViT-Ti, Standard ViT-Ti, ResNet-18.

Runs the three architectures end to end and writes a comparison report.
Short step budgets smoke-test the pipeline on CPU; the full tier trains
100-200 epochs (see ``configs/sanity_cifar100.json``) on TPU hardware.

Usage (smoke):
  python scripts/sanity_cifar100.py --data-dir ~/data/cifar100 --steps 20 --out runs/sanity
Usage (full tier):
  python scripts/sanity_cifar100.py --data-dir ~/data/cifar100 --config configs/sanity_cifar100.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

ARCHS = ("algebraic", "standard", "resnet")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="runs/sanity")
    p.add_argument("--steps", type=int, default=0, help="short smoke budget (0: use --config)")
    p.add_argument("--config", default=str(ROOT / "configs" / "sanity_cifar100.json"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plan", action="store_true", help="print the launch plan without running")
    args = p.parse_args()
    out = Path(args.out)
    results = {}
    for arch in ARCHS:
        cmd = [sys.executable, "-m", "vision.train", "--arch", arch, "--dataset", "cifar100",
               "--data-dir", args.data_dir, "--batch", str(args.batch),
               "--seed", str(args.seed), "--out", str(out / arch)]
        if arch == "resnet":
            cmd += ["--set", "resnet_depth=18"]
        if args.steps:
            cmd += ["--steps", str(args.steps), "--log-every", "5"]
        else:
            tier = json.loads(Path(args.config).read_text())
            t = tier[arch]
            cmd += ["--epochs", str(t["epochs"]), "--lr", str(t["lr"]),
                    "--wd", str(t["wd"]), "--label-smoothing", str(t.get("label_smoothing", 0.1)),
                    "--eval-every", "10", "--ckpt-every", "25"]
            if arch == "resnet":
                cmd += ["--size", "Ti"]
        print("+ " + " ".join(cmd), flush=True)
        if args.plan:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT, env=ENV)
        results[arch] = json.loads((out / arch / "result.json").read_text())
    if args.plan:
        print(f"{len(ARCHS)} arch runs (plan only; no results)", flush=True)
        return
    summary = {a: {k: r.get(k) for k in ("n_params", "final_eval_acc", "final_eval_nll",
                                         "train_seconds", "mean_img_per_s")}
               for a, r in results.items()}
    (out / "sanity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
