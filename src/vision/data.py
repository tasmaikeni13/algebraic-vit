"""Dataset loading, augmentation, and batch sampling (NumPy, prefetched).

CIFAR-100: downloaded once from the official archive, parsed from the
Python batches, and cached as ``train.npy``/``test.npy`` uint8 arrays
with label vectors. ImageNet-1k: prepared by
``scripts/prepare_imagenet.py`` into ``train_shard_*.npz``/``val.npz``
uint8 files; this module streams them. Augmentation follows standard
practice: random crop with padding + horizontal flip for CIFAR-100,
random resized crop + flip for ImageNet, with optional MixUp/CutMix
applied by the sampler.
"""

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_MEAN = np.array([0.5071, 0.4867, 0.4408], np.float32)
CIFAR_STD = np.array([0.2675, 0.2565, 0.2761], np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def ensure_cifar100(directory) -> Path:
    """Download and parse CIFAR-100 once; return the cache directory."""
    directory = Path(directory)
    train_npy, test_npy = directory / "train.npy", directory / "test.npy"
    if train_npy.exists() and test_npy.exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "cifar-100-python.tar.gz"
    if not archive.exists():
        print(f"downloading {CIFAR_URL}", flush=True)
        urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(directory, filter="data")
        except TypeError:  # python < 3.10.12 without the filter kwarg
            tar.extractall(directory)

    def load(names):
        xs, ys = [], []
        for name in names:
            with open(directory / "cifar-100-python" / name, "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.asarray(d[b"fine_labels"]))
        return np.concatenate(xs).astype(np.uint8), np.concatenate(ys).astype(np.int64)

    train_x, train_y = load(["train"])
    test_x, test_y = load(["test"])
    np.save(train_npy, train_x)
    np.save(directory / "train_labels.npy", train_y)
    np.save(test_npy, test_x)
    np.save(directory / "test_labels.npy", test_y)
    return directory


def _flip_crop_cifar(batch_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 32x32 crop from 4px reflection padding + horizontal flip."""
    padded = np.pad(batch_x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    top = rng.integers(0, 9, size=len(batch_x))
    left = rng.integers(0, 9, size=len(batch_x))
    out = np.stack([padded[i, t:t + 32, l:l + 32] for i, (t, l) in enumerate(zip(top, left))])
    flips = rng.random(len(batch_x)) < 0.5
    out[flips] = out[flips, :, ::-1, :]
    return np.ascontiguousarray(out)


def _resized_crop_imagenet(batch_x: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Random resized crop from 256px inputs + horizontal flip (bilinear)."""
    from PIL import Image
    out = np.empty((len(batch_x), size, size, 3), np.uint8)
    for i, img in enumerate(batch_x):
        area = rng.uniform(0.08, 1.0) * 256 * 256
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        w, h = int(round(area ** 0.5 * ratio ** 0.5)), int(round(area ** 0.5 / ratio ** 0.5))
        w, h = min(w, 256), min(h, 256)
        left, top = rng.integers(0, 257 - w), rng.integers(0, 257 - h)
        crop = Image.fromarray(img[top:top + h, left:left + w]).resize((size, size), Image.BILINEAR)
        arr = np.asarray(crop)
        if rng.random() < 0.5:
            arr = arr[:, ::-1, :]
        out[i] = arr
    return np.ascontiguousarray(out)


class ClassificationSampler:
    """Deterministic per-(seed, step) sampler with optional MixUp/CutMix.

    Batches are float32 normalized images and int64 labels, or MixUp/CutMix
    triples (images, labels_a, labels_b, lam) when enabled.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, global_batch: int,
                 seed: int, process_index: int, process_count: int, *, image_size: int,
                 dataset: str, augment: bool, mixup: float = 0.0, cutmix: float = 0.0):
        if len(images) < global_batch:
            raise ValueError("dataset smaller than one global batch")
        if global_batch % process_count:
            raise ValueError("global batch must divide across processes")
        self.images = images
        self.labels = np.asarray(labels)
        self.B = global_batch
        self.seed = seed
        self.local = global_batch // process_count
        self.offset = process_index * self.local
        self.size = image_size
        self.dataset = dataset
        self.augment = augment
        self.mixup = mixup
        self.cutmix = cutmix
        if dataset == "cifar100":
            self.mean, self.std = CIFAR_MEAN, CIFAR_STD
        elif dataset == "imagenet":
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        else:
            raise ValueError(f"unknown dataset {dataset!r}")

    def batch(self, step):
        rng = np.random.default_rng([self.seed, step])
        idx = rng.integers(0, len(self.images), size=self.B)
        x = np.asarray(self.images[idx])
        y = self.labels[idx]
        if self.augment:
            if self.dataset == "cifar100":
                x = _flip_crop_cifar(x, rng)
            else:
                x = _resized_crop_imagenet(x, rng, self.size)
        elif self.dataset == "imagenet" and x.shape[1] != self.size:
            from PIL import Image
            x = np.stack([np.asarray(Image.fromarray(im).resize((self.size, self.size),
                                                              Image.BILINEAR)) for im in x])
        x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
        lam, ya, yb = 1.0, y, y
        use_mix = self.mixup > 0 and rng.random() < 0.5
        use_cut = not use_mix and self.cutmix > 0 and rng.random() < 0.5
        if use_mix or use_cut:
            lam = float(rng.beta(self.mixup or self.cutmix, self.mixup or self.cutmix))
            perm = rng.permutation(self.B)
            if use_mix:
                x = lam * x + (1 - lam) * x[perm]
            else:
                cut = int(round(self.size * (1 - lam) ** 0.5))
                cy, cx = rng.integers(0, self.size, size=2)
                y1, y2 = max(0, cy - cut // 2), min(self.size, cy + cut // 2)
                x1, x2 = max(0, cx - cut // 2), min(self.size, cx + cut // 2)
                x[:, y1:y2, x1:x2] = x[perm][:, y1:y2, x1:x2]
                lam = 1.0 - ((y2 - y1) * (x2 - x1) / self.size ** 2)
            ya, yb = y, y[perm]
        s = slice(self.offset, self.offset + self.local)
        return (np.ascontiguousarray(x[s]).astype(np.float32),
                np.ascontiguousarray(ya[s]).astype(np.int64),
                np.ascontiguousarray(yb[s]).astype(np.int64), np.float32(lam))


def load_imagenet_val(directory):
    path = Path(directory) / "val.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_imagenet.py first")
    d = np.load(path)
    return d["images"], d["labels"]


def load_imagenet_train_shards(directory):
    shards = sorted(Path(directory).glob("train_shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no train shards in {directory}")
    return shards
