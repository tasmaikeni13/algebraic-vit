"""Sweep mapping: one trial per worker with complete, disjoint coverage."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assign_waves_one_trial_per_worker():
    run_pod = _load("run_pod")
    trials = [(f"arch{i % 2}", {"lr": 0.001 * i}) for i in range(10)]
    hosts = ["w0", "w1", "w2", "w3"]
    waves = run_pod.assign_waves(trials, hosts)
    assert len(waves) == 3
    for wave in waves:
        assert len(wave) == len({host for host, _ in wave})  # one trial per worker
    flat = [t for wave in waves for _, t in wave]
    assert sorted(map(str, flat)) == sorted(map(str, trials))  # complete, no dupes
    assert [host for host, _ in waves[0]] == hosts  # wave order follows hosts


def test_sweep_trials_match_between_drivers():
    run_pod = _load("run_pod")
    sweep = _load("sweep")
    cfg = {"grids": {"algebraic": {"lr": [0.001, 0.002], "wd": [0.05]},
                     "standard": {"lr": [0.003]}}}
    assert run_pod.sweep_trials(cfg) == sweep.expand_trials(cfg)


def test_worker_shard_partition_is_disjoint_and_complete():
    sweep = _load("sweep")
    trials = [(f"a{i}", {"lr": i}) for i in range(15)]
    parts = [sweep.select_trials(trials, w, 4) for w in range(4)]
    flat = [t for part in parts for t in part]
    assert len(flat) == len(trials)
    assert sorted(map(str, flat)) == sorted(map(str, trials))
