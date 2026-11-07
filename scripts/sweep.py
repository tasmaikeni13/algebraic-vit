#!/usr/bin/env python3
"""Hyperparameter sweep driver for the Small and Base main runs.

Reads a sweep config, launches one trial per entry (one worker each on
a v4-32 pod; see run_pod.py orchestration), and writes the winning trial
per architecture by validation accuracy. Literature-standard budget: a
small learning-rate x weight-decay grid at reduced epochs, then
full-length confirmation is handled by the main-run scripts.

Usage:
  python scripts/sweep.py --config configs/sweep_small.json --data-dir ~/data/imagenet
"""

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def expand_trials(sweep):
    trials = []
    for arch, grid in sweep["grids"].items():
        keys = sorted(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            trials.append((arch, dict(zip(keys, combo))))
    return trials


def select_trials(trials, worker, num_workers):
    """Trials owned by one worker (round-robin shard)."""
    return [t for i, t in enumerate(trials) if i % num_workers == worker]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--worker", type=int, default=None,
                   help="run only this worker's shard (pod sharding)")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--plan", action="store_true", help="print the trial plan without running")
    p.add_argument("--aggregate", action="store_true",
                   help="collect winners from existing out/*/result.json without launching trials")
    args = p.parse_args()
    if args.num_workers < 1:
        p.error("--num-workers must be >= 1")
    if args.worker is not None and not 0 <= args.worker < args.num_workers:
        p.error("--worker must be in [0, --num-workers)")
    sweep = json.loads(Path(args.config).read_text())
    out = Path(args.out or sweep["out"])
    trials = expand_trials(sweep)
    indexed = list(enumerate(trials))
    if args.aggregate:
        results = []
        for i, (arch, combo) in indexed:
            name = f"{arch}_trial{i}"
            res_path = out / name / "result.json"
            if not res_path.exists():
                print(f"skip {name} (no result.json)", flush=True)
                continue
            res = json.loads(res_path.read_text())
            results.append({"trial": name, "arch": arch, "hparams": combo,
                            "val_acc": res["final_eval_acc"], "val_nll": res["final_eval_nll"]})
        winners = {}
        for arch in sweep["grids"]:
            cands = [r for r in results if r["arch"] == arch]
            winners[arch] = max(cands, key=lambda r: r["val_acc"]) if cands else None
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep_results.json").write_text(json.dumps(
            {"trials": results, "winners": winners}, indent=2))
        print(json.dumps(winners, indent=2))
        return
    if args.worker is not None:
        indexed = select_trials(indexed, args.worker, args.num_workers)
        print("warning: sharded run; winners are shard-local, not global",
              file=sys.stderr, flush=True)
    results = []
    for i, (arch, combo) in indexed:
        name = f"{arch}_trial{i}"
        cmd = [sys.executable, "-m", "vision.train", "--arch", arch, "--dataset", "imagenet",
               "--data-dir", args.data_dir, "--size", sweep["size"],
               "--batch", str(sweep["batch"]), "--epochs", str(sweep["epochs"]),
               "--seed", str(sweep["seed"]), "--eval-every", str(sweep["epochs"]),
               "--out", str(out / name)]
        for k, v in combo.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        for extra in sweep.get("extra_args", {}).get(arch, []):
            cmd += shlex.split(extra) if isinstance(extra, str) else list(extra)
        print("+ " + " ".join(cmd), flush=True)
        if args.plan:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT, env=ENV)
        res = json.loads((out / name / "result.json").read_text())
        results.append({"trial": name, "arch": arch, "hparams": combo,
                        "val_acc": res["final_eval_acc"], "val_nll": res["final_eval_nll"]})
    if args.plan:
        print(f"{len(indexed)} trials (plan only; no results)", flush=True)
        return
    winners = {}
    for arch in sweep["grids"]:
        cands = [r for r in results if r["arch"] == arch]
        winners[arch] = max(cands, key=lambda r: r["val_acc"]) if cands else None
    (out / "sweep_results.json").write_text(json.dumps(
        {"trials": results, "winners": winners}, indent=2))
    print(json.dumps(winners, indent=2))


if __name__ == "__main__":
    main()
