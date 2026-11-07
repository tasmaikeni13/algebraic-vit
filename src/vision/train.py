#!/usr/bin/env python3
"""Train one vision model on one chip, one host, or a multi-host TPU slice.

Data parallel: parameters and optimizer moments are replicated, each device
runs its share of the global batch inside ``shard_map``, and gradients are
averaged with ``pmean``. Vision Transformers train with AdamW and a cosine
schedule; ResNet trains with SGD with momentum and a cosine schedule.
Logs wall-clock time, throughput, parameter count, and accuracy.

Example (single chip, CIFAR-100 smoke):
  python -m vision.train --arch algebraic --size Ti --dataset cifar100 \\
      --data-dir ~/data/cifar100 --epochs 1 --batch 128 --out runs/alg
"""

import argparse
from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def parse_overrides(pairs):
    from vision.config import VisionConfig
    types = {f.name: f.type for f in fields(VisionConfig)}
    out = {}
    for item in pairs or []:
        key, _, value = item.partition("=")
        if key not in types:
            raise ValueError(f"unknown config field {key!r}")
        kind = types[key]
        kind = kind if isinstance(kind, type) else eval(kind)
        if kind is bool:
            out[key] = value.lower() in ("1", "true", "yes")
        else:
            out[key] = kind(value)
    return out


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="algebraic", choices=("algebraic", "standard", "resnet"))
    p.add_argument("--size", default="Ti", help="Ti/S/B for ViT (ignored for ResNet)")
    p.add_argument("--set", action="append", default=[], help="config override key=value")
    p.add_argument("--dataset", default="cifar100", choices=("cifar100", "imagenet"))
    p.add_argument("--data-dir", default=os.path.expanduser("~/data/cifar100"))
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)
    p.add_argument("--batch", type=int, default=128, help="global batch in images")
    p.add_argument("--micro", type=int, default=32, help="per-device micro-batch in images")
    p.add_argument("--epochs", type=float, default=0)
    p.add_argument("--steps", type=int, default=0, help="optimizer steps (0: use --epochs)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction (<1) or steps (>=1)")
    p.add_argument("--optimizer", default=None,
                   help="adamw/sgd (default: adamw for ViT, sgd for ResNet)")
    p.add_argument("--wd", type=float, default=None,
                   help="weight decay (default 0.05 ViT, 1e-4 ResNet)")
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=0, help="epochs between evals (0: end only)")
    p.add_argument("--eval-batch", type=int, default=64, help="per-device eval images")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--save-params", action="store_true")
    p.add_argument("--distributed", action="store_true", help="call jax.distributed.initialize")
    p.add_argument("--profile-steps", type=int, default=0, help="time N steps then exit")
    p.add_argument("--ckpt-every", type=int, default=0, help="epochs between checkpoints (0: off)")
    p.add_argument("--keep-ckpts", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config/mesh/batch math and exit before loading data")
    return p.parse_args(argv)


class Prefetcher:
    def __init__(self, sampler, start, stop, depth=4):
        self.q = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, args=(sampler, start, stop), daemon=True)
        self.thread.start()

    def _run(self, sampler, start, stop):
        for step in range(start, stop):
            self.q.put(sampler.batch(step))

    def get(self):
        return self.q.get()


def save_checkpoint(directory: Path, step: int, payload: dict, keep: int) -> None:
    import pickle
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:07d}.pkl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    for old in sorted(directory.glob("step-*.pkl"))[:-keep]:
        old.unlink()


def _host(a):
    return np.asarray(a.addressable_data(0))


def latest_checkpoint(directory: Path):
    import pickle
    found = sorted(Path(directory).glob("step-*.pkl"))
    if not found:
        return None
    with found[-1].open("rb") as fh:
        return pickle.load(fh)


def lr_schedule(args, total_steps):
    import jax.numpy as jnp
    peak = args.lr
    warm = args.warmup if args.warmup >= 1 else max(1, int(round(args.warmup * total_steps)))
    floor = peak * args.min_lr_ratio
    T = float(total_steps)

    def fn(step):
        t = jnp.asarray(step, jnp.float32) + 1.0
        warm_f = jnp.minimum(1.0, t / warm)
        prog = jnp.clip((t - warm) / jnp.maximum(1.0, T - warm), 0.0, 1.0)
        lr = floor + (peak - floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * prog))
        return jnp.where(t < warm, peak * warm_f, lr)

    return fn, warm


def build_config(args):
    from vision import config as cfgs
    if args.arch == "resnet":
        cfg = cfgs.resnet()
    elif args.arch == "standard":
        cfg = cfgs.standard_vit(args.size)
    else:
        cfg = cfgs.algebraic_vit(args.size)
    kw = {}
    if args.dataset == "cifar100":
        kw["num_classes"] = 100
        kw["image_size"] = args.image_size or 32
        kw["patch"] = args.patch or 4
    else:
        kw["num_classes"] = 1000
        kw["image_size"] = args.image_size or 224
        kw["patch"] = args.patch or 16
    cfg = replace(cfg, **kw)
    overrides = parse_overrides(args.set)
    return replace(cfg, **overrides)


def main(argv=None):
    args = get_args(argv)
    if (args.steps < 0 or args.epochs < 0
            or (not (args.steps or args.epochs) and not args.dry_run)
            or min(args.batch, args.micro, args.eval_batch, args.log_every) < 1
            or args.keep_ckpts < 1 or args.ckpt_every < 0):
        raise ValueError("specify a positive budget, batch sizes, and checkpoint retention")
    import jax
    if args.distributed:
        jax.distributed.initialize()
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    from vision import algebraic_vit, data as datalib, resnet, standard_vit

    cfg = build_config(args)
    is_resnet = cfg.arch == "resnet"
    model = resnet if is_resnet else (standard_vit if cfg.arch == "standard" else algebraic_vit)
    optimizer = args.optimizer or ("sgd" if is_resnet else "adamw")
    wd = args.wd if args.wd is not None else (1e-4 if is_resnet else 0.05)

    devices = jax.devices()
    n_dev = len(devices)
    pidx, pcount = jax.process_index(), jax.process_count()
    is_main = pidx == 0
    mesh = Mesh(np.array(devices), ("data",))

    if args.dry_run:
        if args.batch % (n_dev * args.micro):
            raise ValueError(
                f"batch {args.batch} not divisible by devices*micro {n_dev}*{args.micro}")
        plan = {"arch": cfg.arch, "dataset": args.dataset, "devices": n_dev,
                "processes": pcount, "batch": args.batch, "micro": args.micro,
                "accum": args.batch // (n_dev * args.micro), "out": args.out}
        if is_main:
            print(json.dumps(plan, indent=2), flush=True)
        return 0

    # Training data.
    data_dir = Path(args.data_dir)
    if args.dataset == "cifar100":
        data_dir = datalib.ensure_cifar100(data_dir)
        train_x = np.load(data_dir / "train.npy", mmap_mode="r")
        train_y = np.load(data_dir / "train_labels.npy")
        valid_x = np.load(data_dir / "test.npy", mmap_mode="r")
        valid_y = np.load(data_dir / "test_labels.npy")
    else:
        shards = datalib.load_imagenet_train_shards(data_dir)
        train_x = np.concatenate([np.load(s)["images"] for s in shards])
        train_y = np.concatenate([np.load(s)["labels"] for s in shards])
        valid_x, valid_y = datalib.load_imagenet_val(data_dir)
    n_train = len(train_x)
    steps_per_epoch = math.ceil(n_train / args.batch)
    total_steps = args.steps or math.ceil(args.epochs * steps_per_epoch)
    if args.batch % (n_dev * args.micro):
        raise ValueError(f"batch {args.batch} not divisible by devices*micro {n_dev}*{args.micro}")
    accum = args.batch // (n_dev * args.micro)
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    def log(rec):
        if is_main:
            with open(out / "log.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    key = jax.random.PRNGKey(args.seed)
    if is_resnet:
        params, batch_stats = model.init_params(cfg, key)
        tables = None
        n_params = model.count_params(params) + model.count_params(batch_stats)
    else:
        tables = (model.position_tables(cfg) if cfg.arch == "algebraic" else None)
        params = model.init_params(cfg, key)
        batch_stats = None
        n_params = model.count_params(params)
    mask = model.decay_mask(params)
    sched, warm = lr_schedule(args, total_steps)
    is_prob = cfg.arch == "algebraic"
    smooth = args.label_smoothing
    n_cls = cfg.num_classes

    if optimizer == "adamw":
        opt_state = {"count": jnp.zeros([], jnp.int32),
                     "m": jax.tree_util.tree_map(jnp.zeros_like, params),
                     "v": jax.tree_util.tree_map(jnp.zeros_like, params)}
    else:
        opt_state = {"count": jnp.zeros([], jnp.int32),
                     "m": jax.tree_util.tree_map(jnp.zeros_like, params)}

    from vision import ops as O

    def single_obj(logits, targets):
        if is_prob:
            p = O.algebraic_probs(logits.astype(jnp.float32), cfg.head_n, cfg.head_c)
            pt = jnp.take_along_axis(p, targets[:, None], axis=-1)[:, 0]
            return jnp.mean(1.0 - pt), pt
        z = logits.astype(jnp.float32)
        lse = jax.nn.logsumexp(z, axis=-1)
        zt = jnp.take_along_axis(z, targets[:, None], axis=-1)[:, 0]
        nll = lse - zt
        return jnp.mean(nll), nll

    def smooth_obj(logits, targets):
        z = logits.astype(jnp.float32)
        logp = z - jax.nn.logsumexp(z, axis=-1, keepdims=True)
        tgt = (jnp.arange(n_cls)[None, :] == targets[:, None]).astype(jnp.float32)
        tgt = (1 - smooth) * tgt + smooth / n_cls
        obj = jnp.mean(-jnp.sum(tgt * logp, axis=-1))
        tel = -jnp.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
        return obj, tel

    def make_local_step(mode):
        """Build the device step for objective ``mode`` (a Python string).

        Modes: "single" (exact head VJP for the algebraic arm), "smooth"
        (label smoothing, standard arms), "mix" (MixUp/CutMix stand-in).
        """

        def local_step(params, batch_stats, opt_state, xs, yas, ybs, lam):
            def micro_fn(carry, batch):
                gsum, lsum, asum, ssum, stats = carry
                x, ya, yb = batch

                def loss_fn(p):
                    if is_resnet:
                        logits, new_stats = model.forward_logits(p, stats, x, cfg, True)
                    elif cfg.arch == "algebraic":
                        new_stats = stats
                        if mode == "single":
                            obj, tel = model.head_loss(p, x, ya, cfg, tables)
                            logits = model.forward_logits(p, x, cfg, tables)
                            pred = jnp.argmax(logits, axis=-1)
                            acc = jnp.mean((pred == ya).astype(jnp.float32))
                            return obj, (tel, acc, new_stats)
                        logits = model.forward_logits(p, x, cfg, tables)
                    else:
                        new_stats = stats
                        logits = model.forward_logits(p, x, cfg)
                    if mode == "mix":
                        la, ta = single_obj(logits, ya)
                        lb, _ = single_obj(logits, yb)
                        obj = lam * la + (1 - lam) * lb
                        tel = ta
                    elif mode == "smooth":
                        obj, tel = smooth_obj(logits, ya)
                    else:
                        obj, tel = single_obj(logits, ya)
                    pred = jnp.argmax(logits.astype(jnp.float32), axis=-1)
                    acc = jnp.mean((pred == ya).astype(jnp.float32))
                    return obj, (tel, acc, new_stats)

                (obj, (tel, acc, new_stats)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
                nll = -jnp.log(jnp.maximum(tel, 1e-30)) if is_prob else tel
                gsum = jax.tree_util.tree_map(lambda a, b: a + b.astype(jnp.float32), gsum, g)
                aux = (gsum, lsum + jnp.mean(nll), asum + acc, ssum + obj, new_stats)
                return aux, None

            zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
            init = (zeros, jnp.zeros([], jnp.float32), jnp.zeros([], jnp.float32),
                    jnp.zeros([], jnp.float32), batch_stats)
            (g, lsum, asum, ssum, new_stats), _ = lax.scan(micro_fn, init, (xs, yas, ybs))
            g = jax.tree_util.tree_map(lambda a: lax.pmean(a / accum, "data"), g)
            loss = lax.pmean(lsum / accum, "data")
            acc = lax.pmean(asum / accum, "data")
            obj = lax.pmean(ssum / accum, "data")
            batch_stats = new_stats
            if is_resnet:
                batch_stats = jax.tree_util.tree_map(lambda a: lax.pmean(a, "data"), new_stats)
            gnorm = jnp.sqrt(sum(jnp.sum(a * a) for a in jax.tree_util.tree_leaves(g)))
            scale = jnp.minimum(1.0, args.clip / jnp.maximum(gnorm, 1e-12))
            count = opt_state["count"] + 1
            lr = sched(count - 1)
            if optimizer == "adamw":
                b1, b2 = args.b1, args.b2
                c1 = 1.0 - b1 ** count.astype(jnp.float32)
                c2 = 1.0 - b2 ** count.astype(jnp.float32)

                def upd(p, gi, m, v, dmask):
                    gi = gi * scale
                    m = b1 * m + (1 - b1) * gi
                    v = b2 * v + (1 - b2) * gi * gi
                    u = (m / c1) / (jnp.sqrt(v / c2) + args.adam_eps)
                    if dmask:
                        u = u + wd * p
                    return p - lr * u, m, v

                flat_p, tdef = jax.tree_util.tree_flatten(params)
                new = [upd(*z) for z in zip(flat_p, tdef.flatten_up_to(g),
                                            tdef.flatten_up_to(opt_state["m"]),
                                            tdef.flatten_up_to(opt_state["v"]),
                                            tdef.flatten_up_to(mask))]
                params = tdef.unflatten([a for a, _, _ in new])
                opt_state = {"count": count, "m": tdef.unflatten([b for _, b, _ in new]),
                             "v": tdef.unflatten([c for _, _, c in new])}
            else:
                mu = args.momentum

                def upd_sgd(p, gi, m, dmask):
                    gi = gi * scale
                    if dmask:
                        gi = gi + wd * p
                    m = mu * m + gi
                    return p - lr * m, m

                flat_p, tdef = jax.tree_util.tree_flatten(params)
                new = [upd_sgd(*z) for z in zip(flat_p, tdef.flatten_up_to(g),
                                                tdef.flatten_up_to(opt_state["m"]),
                                                tdef.flatten_up_to(mask))]
                params = tdef.unflatten([a for a, _ in new])
                opt_state = {"count": count, "m": tdef.unflatten([b for _, b in new])}
            rep_out = {"loss": loss, "acc": acc, "obj": obj, "gnorm": gnorm, "lr": lr}
            return params, batch_stats, opt_state, rep_out

        return local_step

    rep = P()
    img_spec = P(None, "data", None, None, None)
    lbl_spec = P(None, "data")

    def make_step_fn(mode):
        local_step = make_local_step(mode)
        if is_resnet:
            return jax.jit(
                shard_map(local_step, mesh=mesh,
                          in_specs=(rep, rep, rep, img_spec, lbl_spec, lbl_spec, rep),
                          out_specs=(rep, rep, rep, rep), check_rep=False),
                donate_argnums=(0, 1, 2))

        def single(p, o, xs, yas, ybs, lam):
            out = local_step(p, None, o, xs, yas, ybs, lam)
            return out[0], out[2], out[3]

        return jax.jit(
            shard_map(single, mesh=mesh,
                      in_specs=(rep, rep, img_spec, lbl_spec, lbl_spec, rep),
                      out_specs=(rep, rep, rep), check_rep=False),
            donate_argnums=(0, 1))

    need_mix = args.mixup > 0 or args.cutmix > 0
    smooth_mode = smooth > 0 and not is_prob
    if cfg.arch == "algebraic" and smooth > 0 and is_main:
        print("note: label smoothing is ignored by the algebraic exact-likelihood arm", flush=True)
    step_fns = {m: make_step_fn(m) for m in ("single", "smooth", "mix")}

    def local_eval(params, batch_stats, x, y):
        if is_resnet:
            logits, _ = model.forward_logits(params, batch_stats, x, cfg, False)
        elif cfg.arch == "algebraic":
            logits = model.forward_logits(params, x, cfg, tables)
        else:
            logits = model.forward_logits(params, x, cfg)
        z = logits.astype(jnp.float32)
        pred = jnp.argmax(z, axis=-1)
        lse = jax.nn.logsumexp(z, axis=-1)
        zt = jnp.take_along_axis(z, y[:, None], axis=-1)[:, 0]
        nll = jnp.sum(lse - zt)
        hits = jnp.sum((pred == y).astype(jnp.float32))
        n = jnp.asarray(x.shape[0], jnp.float32)
        return lax.psum(nll, "data"), lax.psum(hits, "data"), lax.psum(n, "data")

    if is_resnet:
        eval_fn = jax.jit(shard_map(local_eval, mesh=mesh,
                                    in_specs=(rep, rep, P("data", None, None, None), P("data",)),
                                    out_specs=(rep, rep, rep), check_rep=False))
    else:
        def local_eval_nostat(p, x, y):
            return local_eval(p, None, x, y)

        eval_fn = jax.jit(shard_map(local_eval_nostat, mesh=mesh,
                                    in_specs=(rep, P("data", None, None, None), P("data",)),
                                    out_specs=(rep, rep, rep), check_rep=False))

    rep_sharding = NamedSharding(mesh, rep)
    img_sharding = NamedSharding(mesh, img_spec)
    lbl_sharding = NamedSharding(mesh, lbl_spec)
    params = jax.device_put(params, rep_sharding)
    if is_resnet:
        batch_stats = jax.device_put(batch_stats, rep_sharding)
    opt_state = jax.device_put(opt_state, rep_sharding)

    def put_global(local_arr, sharding, global_shape):
        return jax.make_array_from_process_local_data(sharding, local_arr, global_shape)

    def run_eval():
        tot_nll, tot_hits, tot_n = 0.0, 0.0, 0
        per_proc = args.eval_batch * n_dev // pcount
        bs = args.eval_batch * n_dev
        mean = datalib.CIFAR_MEAN if args.dataset == "cifar100" else datalib.IMAGENET_MEAN
        std = datalib.CIFAR_STD if args.dataset == "cifar100" else datalib.IMAGENET_STD

        def eval_global(idxs):
            sel = idxs[pidx * per_proc:(pidx + 1) * per_proc]
            raw = np.asarray(valid_x[sel])
            if args.dataset == "imagenet" and raw.shape[1] != cfg.image_size:
                from PIL import Image
                raw = np.stack([np.asarray(Image.fromarray(im).resize(
                    (cfg.image_size, cfg.image_size), Image.BILINEAR)) for im in raw])
            xb = (raw.astype(np.float32) / 255.0 - mean) / std
            yb = np.asarray(valid_y[sel]).astype(np.int64)
            x = put_global(np.ascontiguousarray(xb).astype(np.float32),
                           NamedSharding(mesh, P("data", None, None, None)),
                           (bs,) + xb.shape[1:])
            y = put_global(np.ascontiguousarray(yb), NamedSharding(mesh, P("data",)), (bs,))
            if is_resnet:
                return eval_fn(params, batch_stats, x, y)
            return eval_fn(params, x, y)

        for b0 in range(0, len(valid_x), bs):
            idx = np.arange(b0, min(b0 + bs, len(valid_x)))
            n_real = len(idx)
            if n_real < bs:
                # Pad to the sharded shape, then subtract the duplicates'
                # contribution (eval is deterministic, so a uniform batch of
                # the duplicated sample measures it exactly).
                n_pad = bs - n_real
                s, h, _ = eval_global(np.concatenate([idx, np.full(n_pad, idx[0])]))
                s0, h0, _ = eval_global(np.full(bs, idx[0]))
                s, h = s - s0 * (n_pad / bs), h - h0 * (n_pad / bs)
            else:
                s, h, _ = eval_global(idx)
            tot_nll += float(s)
            tot_hits += float(h)
            tot_n += n_real
        return tot_nll / tot_n, tot_hits / tot_n

    start_step, history, elapsed_before = 0, [], 0.0
    ckpt_dir = out / "ckpt"
    if args.resume:
        saved = latest_checkpoint(ckpt_dir)
        if saved is not None:
            if saved["config"] != asdict(cfg) or saved["total_steps"] != total_steps:
                raise ValueError("checkpoint configuration does not match this run")
            params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
            if is_resnet:
                batch_stats = jax.tree_util.tree_map(jnp.asarray, saved["batch_stats"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, saved["opt_state"])
            start_step = int(saved["step"])
            history = saved["history"]
            elapsed_before = float(saved["elapsed"])
            if is_main:
                print(f"resumed from step {start_step}", flush=True)
    params = jax.device_put(params, rep_sharding)
    opt_state = jax.device_put(opt_state, rep_sharding)

    meta = {"config": asdict(cfg), "args": vars(args), "n_params": n_params,
            "devices": n_dev, "processes": pcount, "accum": accum,
            "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
            "optimizer": optimizer, "wd": wd, "warmup_steps": warm}
    if is_main:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in
                          ("n_params", "devices", "accum", "total_steps")}), flush=True)

    sampler = datalib.ClassificationSampler(
        train_x, train_y, args.batch, args.data_seed if args.data_seed is not None else args.seed,
        pidx, pcount, image_size=cfg.image_size, dataset=args.dataset, augment=True,
        mixup=args.mixup, cutmix=args.cutmix)
    prefetch = Prefetcher(sampler, start_step, total_steps)
    local_rows = args.batch // pcount
    C = cfg.image_size
    ch = 3
    global_img = (accum, args.batch // accum, C, C, ch)
    global_lbl = (accum, args.batch // accum)

    t_start = time.time() - elapsed_before
    t_last = time.time()
    compile_s = None
    ckpt_every_steps = (args.ckpt_every * steps_per_epoch) if args.ckpt_every else 0
    eval_every_steps = (args.eval_every * steps_per_epoch) if args.eval_every else 0
    for step in range(start_step, total_steps):
        x_np, ya_np, yb_np, lam_np = prefetch.get()
        x_np = x_np.reshape(local_rows // accum, accum, C, C, ch).transpose(1, 0, 2, 3, 4)
        ya_np = ya_np.reshape(local_rows // accum, accum).transpose(1, 0)
        yb_np = yb_np.reshape(local_rows // accum, accum).transpose(1, 0)
        x = put_global(np.ascontiguousarray(x_np), img_sharding, global_img)
        ya = put_global(np.ascontiguousarray(ya_np), lbl_sharding, global_lbl)
        yb = put_global(np.ascontiguousarray(yb_np), lbl_sharding, global_lbl)
        lam = jnp.asarray(float(lam_np), jnp.float32)
        if need_mix and lam_np < 1 - 1e-9:
            mode = "mix"
        elif smooth_mode:
            mode = "smooth"
        else:
            mode = "single"
        step_fn = step_fns[mode]
        if is_resnet:
            params, batch_stats, opt_state, metrics = step_fn(
                params, batch_stats, opt_state, x, ya, yb, lam)
        else:
            params, opt_state, metrics = step_fn(params, opt_state, x, ya, yb, lam)
        if step == start_step:
            jax.block_until_ready(metrics["loss"])
            compile_s = time.time() - t_last
            t_last = time.time()
        if args.profile_steps and step + 1 == start_step + args.profile_steps:
            jax.block_until_ready(params)
            dt = (time.time() - t_last) / args.profile_steps
            res = {"step_time": dt, "images_per_s": args.batch / dt, "compile_s": compile_s}
            if is_main:
                print(json.dumps(res), flush=True)
                (out / "profile.json").write_text(json.dumps(res))
            return 0
        if (step + 1) % args.log_every == 0 or step + 1 == total_steps:
            m = {k: float(v) for k, v in metrics.items()}
            now = time.time()
            span = max(1, min(args.log_every, step - start_step + 1))
            rate = args.batch * span / max(now - t_last, 1e-9)
            t_last = now
            rec = {"step": step + 1, "epoch": (step + 1) / steps_per_epoch, **m,
                   "img_per_s": rate, "elapsed": now - t_start}
            history.append(rec)
            log(rec)
            if is_main:
                print(f"step {step+1}/{total_steps} loss {m['loss']:.4f} acc {m['acc']:.4f} "
                      f"lr {m['lr']:.2e} {rate:.0f} img/s", flush=True)
            if not all(math.isfinite(v) for v in m.values()):
                log({"error": "non-finite training metrics", "step": step + 1})
                raise FloatingPointError(f"non-finite training at step {step+1}")
        if ckpt_every_steps and (step + 1) % ckpt_every_steps == 0 and step + 1 < total_steps:
            payload = {"config": asdict(cfg), "total_steps": total_steps, "step": step + 1,
                       "params": jax.tree_util.tree_map(_host, params),
                       "opt_state": jax.tree_util.tree_map(_host, opt_state),
                       "history": history, "elapsed": time.time() - t_start}
            if is_resnet:
                payload["batch_stats"] = jax.tree_util.tree_map(
                    lambda a: np.asarray(a.addressable_data(0)), batch_stats)
            save_checkpoint(ckpt_dir, step + 1, payload, args.keep_ckpts)
        if eval_every_steps and (step + 1) % eval_every_steps == 0 and step + 1 < total_steps:
            nll, acc = run_eval()
            log({"step": step + 1, "eval_nll": nll, "eval_acc": acc})
            if is_main:
                print(f"  eval nll {nll:.4f} acc {acc:.4f}", flush=True)

    jax.block_until_ready(params)
    train_s = time.time() - t_start
    nll, acc = run_eval()
    if not (math.isfinite(nll) and math.isfinite(acc)):
        raise FloatingPointError("non-finite final evaluation")
    result = {**meta, "completed_steps": total_steps, "final_eval_nll": nll,
              "final_eval_acc": acc, "train_seconds": train_s, "compile_seconds": compile_s,
              "mean_img_per_s": total_steps * args.batch / max(train_s - (compile_s or 0), 1e-9),
              "final_train_loss": history[-1]["loss"] if history else None}
    log({"step": total_steps, "eval_nll": nll, "eval_acc": acc, "final": True})
    if is_main:
        if args.save_params:
            host = jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), params)
            flat, _ = jax.tree_util.tree_flatten_with_path(host)
            pending = out / "params.npz.tmp"
            with pending.open("wb") as stream:
                np.savez(stream, **{jax.tree_util.keystr(k): v for k, v in flat})
            pending.replace(out / "params.npz")
        pending = out / "result.json.tmp"
        pending.write_text(json.dumps(result, indent=2))
        pending.replace(out / "result.json")
        print(f"FINAL eval acc {acc:.4f} nll {nll:.4f} ({train_s/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
