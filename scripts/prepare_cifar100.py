#!/usr/bin/env python3
"""Cache CIFAR-100 locally (download + parse once).

Usage:
  python scripts/prepare_cifar100.py --data-dir ~/data/cifar100
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    from vision import data as datalib
    out = datalib.ensure_cifar100(args.data_dir)
    import numpy as np
    x = np.load(out / "train.npy", mmap_mode="r")
    t = np.load(out / "test.npy", mmap_mode="r")
    print(f"train {x.shape} test {t.shape} in {out}")


if __name__ == "__main__":
    main()
