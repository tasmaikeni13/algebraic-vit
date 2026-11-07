"""End-to-end training smoke on synthetic data (CPU, tiny, deterministic).

Guards eval-metric correctness when the validation set does not divide the
eval batch: identical samples must give identical metrics with and without
a partial last batch (padding must not leak into the averages).
"""

import json

import numpy as np


def _cache(directory, n_valid):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (32, 32, 3), np.uint8)
    train_x = rng.integers(0, 256, (16, 32, 32, 3), np.uint8)
    train_y = rng.integers(0, 100, (16,), np.int64)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "train.npy", train_x)
    np.save(directory / "train_labels.npy", train_y)
    np.save(directory / "test.npy", np.stack([img] * n_valid))
    np.save(directory / "test_labels.npy", np.full((n_valid,), 7, np.int64))


def _run(data_dir, out_dir):
    from vision.train import main
    rc = main(["--arch", "standard", "--size", "Ti",
               "--set", "n_layers=1", "--set", "d_model=16", "--set", "n_heads=2",
               "--set", "d_ff=32", "--set", "remat=0",
               "--dataset", "cifar100", "--data-dir", str(data_dir),
               "--batch", "8", "--micro", "8", "--steps", "1",
               "--eval-batch", "4", "--seed", "0", "--out", str(out_dir)])
    assert rc == 0
    return json.loads((out_dir / "result.json").read_text())


def test_partial_eval_batch_matches_full_batches(tmp_path):
    _cache(tmp_path / "n4", 4)
    _cache(tmp_path / "n10", 10)
    m4 = _run(tmp_path / "n4", tmp_path / "out4")
    m10 = _run(tmp_path / "n10", tmp_path / "out10")
    assert m4["completed_steps"] == m10["completed_steps"] == 1
    np.testing.assert_allclose(m10["final_eval_nll"], m4["final_eval_nll"],
                               rtol=1e-6, atol=0)
    assert m10["final_eval_acc"] == m4["final_eval_acc"]
