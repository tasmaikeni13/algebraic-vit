#!/usr/bin/env python3
"""Convert extracted ILSVRC JPEGs into uint8 NumPy shards.

Expects ``<src>/train/<class>/*.JPEG`` and ``<src>/val/<class>/*.JPEG``
with WordNet-id class directories, plus a ``classes.txt`` mapping each
id to a 0-999 index (one id per line, in order). Training images are
resized to 256px on the short side and packed into 128 shards of about
10k images; validation images are resized to 256x256 center crops in a
single ``val.npz``.

Usage:
  python scripts/prepare_imagenet.py --src ~/data/ilsvrc --out ~/data/imagenet
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_resize(path, short=256):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
    return np.asarray(img)


def _center_crop(arr, size=256):
    h, w, _ = arr.shape
    top, left = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(arr[top:top + size, left:left + size])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, default=128)
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    classes = (src / "classes.txt").read_text().split()
    if len(classes) != 1000:
        raise ValueError("classes.txt must list 1000 WordNet ids in index order")
    index = {c: i for i, c in enumerate(classes)}
    out.mkdir(parents=True, exist_ok=True)

    def collect(split):
        items = []
        for cls in sorted((src / split).iterdir()):
            if cls.name not in index:
                continue
            for img in sorted(cls.glob("*.JPEG")):
                items.append((img, index[cls.name]))
        return items

    train = collect("train")
    rng = np.random.default_rng(0)
    rng.shuffle(train)
    per = (len(train) + args.shards - 1) // args.shards
    for s in range(args.shards):
        chunk = train[s * per:(s + 1) * per]
        if not chunk:
            continue
        xs = np.stack([_center_crop(_load_resize(p)) for p, _ in chunk])
        ys = np.array([y for _, y in chunk], np.int64)
        np.savez_compressed(out / f"train_shard_{s:03d}.npz", images=xs, labels=ys)
        print(f"shard {s}: {len(chunk)} images", flush=True)
    val = collect("val")
    if not val:
        raise ValueError(f"no validation images found under {src / 'val'}")
    xs = np.stack([_center_crop(_load_resize(p)) for p, _ in val])
    ys = np.array([y for _, y in val], np.int64)
    np.savez_compressed(out / "val.npz", images=xs, labels=ys)
    print(f"val: {len(val)} images in {out}")


if __name__ == "__main__":
    main()
