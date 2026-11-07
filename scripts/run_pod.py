#!/usr/bin/env python3
"""TPU v4-32 pod orchestration: sweeps and full-pod main runs.

A v4-32 pod holds 16 TPU v4 chips across 4 workers (w0-w3, 4 chips each).

Sweep mode assigns one hyperparameter trial per worker: each worker runs
a single trial across its local chips, covering one wave of 4 trials at a
time. Main-run mode launches one synchronized 16-chip
distributed training per (architecture, seed) via ``torchrun``-style
multi-host JAX (``vision.train --distributed``), reading winners from the
sweep results.

Usage (from w0, after scripts/setup_workers.sh):
  python scripts/run_pod.py --mode sweep --config configs/sweep_small.json --hosts w0,w1,w2,w3
  python scripts/run_pod.py --mode main --config configs/main_small.json --hosts w0,w1,w2,w3
  python scripts/run_pod.py --mode main --config configs/main_small.json --plan   # print only
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHIPS_PER_WORKER = 4


def sweep_trials(sweep):
    trials = []
    for arch, grid in sweep["grids"].items():
        keys = sorted(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            trials.append((arch, dict(zip(keys, combo))))
    return trials


def assign_waves(trials, hosts):
    """Group trials into waves with one trial per worker."""
    waves = []
    for wave_start in range(0, len(trials), len(hosts)):
        wave = trials[wave_start:wave_start + len(hosts)]
        waves.append([(hosts[k], trial) for k, trial in enumerate(wave)])
    return waves


def run_sweep(args, hosts):
    sweep = json.loads(Path(args.config).read_text())
    trials = sweep_trials(sweep)
    waves = assign_waves(trials, hosts)
    print(f"{len(trials)} trials, one per worker ({len(hosts)} per wave)", flush=True)
    for w, wave in enumerate(waves):
        procs = []
        for k, (host, (arch, combo)) in enumerate(wave):
            name = f"{arch}_trial{w * len(hosts) + k}"
            cmd = [sys.executable, "-m", "vision.train",
                   "--arch", arch, "--dataset", "imagenet", "--data-dir", args.data_dir,
                   "--size", sweep["size"], "--batch", str(sweep["batch"]),
                   "--epochs", str(sweep["epochs"]), "--seed", str(sweep["seed"]),
                   "--out", str(Path(sweep["out"]) / name)]
            for key, val in combo.items():
                cmd += [f"--{key.replace('_', '-')}", str(val)]
            print(f"[{host}] {name} {combo}", flush=True)
            if args.plan:
                print("  " + " ".join(cmd), flush=True)
                continue
            remote = ["ssh", host, f"cd {ROOT} && PYTHONPATH={ROOT}/src"] + cmd
            procs.append(subprocess.Popen(remote, cwd=ROOT))
        for p in procs:
            if p.wait() != 0:
                raise SystemExit("a sweep trial failed; fix and re-run "
                                 "(completed trials keep their outputs)")
    print("sweep wave complete; trial outputs are on each worker's local disk.")
    print("mirror them to this host first (trial names are globally unique, so merging is safe):")
    for host in hosts[1:]:
        print(f"  rsync -a {host}:{ROOT / sweep['out']}/ {ROOT / sweep['out']}/", flush=True)
    print("then aggregate with scripts/sweep.py --config "
          f"{args.config} --data-dir {args.data_dir} --aggregate (winners by val_acc)")


def run_main(args, hosts):
    main = json.loads(Path(args.config).read_text())
    try:
        winners = json.loads(Path(main["hparams_from"]).read_text())["winners"]
    except (FileNotFoundError, KeyError, ValueError):
        if not args.plan:
            raise
        winners = {a: {"hparams": {"lr": "<sweep>", "wd": "<sweep>"}}
                   for a in main["architectures"]}
    coordinator = f"{hosts[0]}:8476"
    for arch in main["architectures"]:
        entry = winners.get(arch) if isinstance(winners, dict) else None
        if not entry or "hparams" not in entry:
            raise SystemExit(
                f"no winner for arch {arch!r} in {main['hparams_from']}; "
                "re-run the sweep or aggregate completed trials first")
        hp = entry["hparams"]
        for seed in main["seeds"]:
            name = f"{arch}_s{seed}"
            out = str(Path(main["out"]) / name)
            base = [sys.executable, "-m", "vision.train", "--arch", arch,
                    "--dataset", "imagenet", "--data-dir", args.data_dir,
                    "--size", main["size"],
                    "--batch", str(main["batch"]),
                    "--epochs", str(main["epochs"]), "--seed", str(seed),
                    "--mixup", str(main.get("mixup", 0.0)),
                    "--cutmix", str(main.get("cutmix", 0.0)),
                    "--eval-every", "10", "--ckpt-every", "25", "--save-params",
                    "--distributed", "--out", out]
            if args.resume:
                base += ["--resume"]
            for key, val in hp.items():
                base += [f"--{key.replace('_', '-')}", str(val)]
            print(f"main run {name} on {len(hosts)} workers x {CHIPS_PER_WORKER} chips", flush=True)
            if args.plan:
                print("  " + " ".join(base), flush=True)
                continue
            procs = []
            for i, host in enumerate(hosts):
                env = (f"cd {ROOT} && PYTHONPATH={ROOT}/src "
                       f"JAX_COORDINATOR_ADDRESS={coordinator} JAX_NUM_PROCESSES={len(hosts)} "
                       f"JAX_PROCESS_INDEX={i}")
                procs.append(subprocess.Popen(["ssh", host, env] + base, cwd=ROOT))
            failed = [p.wait() for p in procs]
            if any(failed):
                raise SystemExit(f"main run {name} failed on a worker; resume with --resume")
    print("all main runs complete")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("sweep", "main"))
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--plan", action="store_true", help="print the launch plan without running")
    p.add_argument("--resume", action="store_true",
                   help="pass --resume to vision.train for main runs")
    args = p.parse_args()
    hosts = args.hosts.split(",")
    if args.mode == "sweep":
        run_sweep(args, hosts)
    else:
        run_main(args, hosts)


if __name__ == "__main__":
    main()
