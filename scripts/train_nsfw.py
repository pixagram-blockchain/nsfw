#!/usr/bin/env python
"""
train_nsfw.py — fine-tune a small MobileNet backbone (timm) into a multi-class
NSFW classifier for fast in-browser inference via onnxruntime-web.

DATASET LAYOUT (ImageFolder: one subdir per class). The community
nsfw_data_scraper produces exactly this:
    data/
      drawings/   hentai/   neutral/   porn/   sexy/
Pass --data-dir data/. Optionally pass --val-dir for a held-out set; otherwise a
stratified split is carved out of --data-dir.

TRAIN/SERVE PARITY: training augments with RandomResizedCrop, but VALIDATION uses
a plain resize-to-NxN — because the browser preprocesses with a plain canvas
resize. The model is thus evaluated the way it will actually be served, which
matters more than squeezing out the last ImageNet-style accuracy point.

OUTPUTS (consumed by export_mnv4.py and scripts/embed-assets.mjs):
    model/nsfw_mnv4.pt      best checkpoint (by macro-F1) incl. backbone+labels+preprocess
    model/labels.json       class order (index -> name), normalized to your 5-class scheme
    model/preprocess.json   plain resize-to-NxN params for the JS side

INSTALL:
    pip install "timm>=1.0.0" torch torchvision pillow numpy

RUN:
    python scripts/train_nsfw.py --data-dir /path/to/nsfw_data --epochs 15
    # faster/smaller backbone:
    python scripts/train_nsfw.py --data-dir ... --backbone mobilenetv4_conv_small_050

RESPONSIBLE-USE NOTE: NSFW data scraped from the web can contain illegal
material, including CSAM. Use a reputable, screened source, run perceptual-hash
matching against known-bad sets BEFORE training, and comply with your local law.
You are legally responsible for data you hold.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# nsfw_data_scraper folder names -> your existing 5-class scheme. The adapter
# gates by NAME on porn/hentai/sexy, so index order is irrelevant; normalizing
# the two differing names just keeps labels.json consistent with the old model.
NORMALIZE = {"drawings": "drawing", "neutral": "safe"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="ImageFolder root (subdir per class)")
    p.add_argument("--val-dir", default=None, help="optional held-out ImageFolder root")
    p.add_argument("--backbone", default="mobilenetv4_conv_small",
                   help="timm model: mobilenetv4_conv_small | mobilenetv4_conv_small_050 | mobilenetv3_small_100")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1, help="used only if --val-dir not given")
    p.add_argument("--limit-per-class", type=int, default=None, help="cap images per class (quick runs)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="model")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def stratified_split(targets, val_frac, seed):
    rng = random.Random(seed)
    by_cls = {}
    for idx, t in enumerate(targets):
        by_cls.setdefault(int(t), []).append(idx)
    train_idx, val_idx = [], []
    for _, idxs in by_cls.items():
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * val_frac)))
        val_idx += idxs[:k]
        train_idx += idxs[k:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def cap_per_class(targets, limit, seed):
    if not limit:
        return list(range(len(targets)))
    rng = random.Random(seed)
    by_cls = {}
    for idx, t in enumerate(targets):
        by_cls.setdefault(int(t), []).append(idx)
    keep = []
    for _, idxs in by_cls.items():
        rng.shuffle(idxs)
        keep += idxs[:limit]
    return sorted(keep)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(ce(logits, y)) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1).cpu().numpy()
        for t, pr in zip(y.cpu().numpy(), pred):
            cm[t, pr] += 1
    # per-class precision/recall/F1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    acc = tp.sum() / max(1, cm.sum())
    return {"loss": loss_sum / max(1, n), "acc": float(acc),
            "prec": prec, "rec": rec, "f1": f1, "macro_f1": float(f1.mean()), "cm": cm}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  backbone={args.backbone}")

    # Build the model first so we can read its native mean/std/size, then build
    # transforms that match (and that the browser will reproduce at serve time).
    probe = timm.create_model(args.backbone, pretrained=True, num_classes=2)
    dcfg = timm.data.resolve_model_data_config(probe)
    size = int(dcfg["input_size"][-1])
    mean = [float(m) for m in dcfg["mean"]]
    std = [float(s) for s in dcfg["std"]]
    del probe
    print(f"[train] input size={size}  mean={mean}  std={std}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),  # plain resize == the browser's canvas resize
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Two ImageFolders over the same root index identically; we Subset them so
    # train and val get different transforms.
    base = datasets.ImageFolder(args.data_dir)
    classes = base.classes
    labels = [NORMALIZE.get(c.lower(), c.lower()) for c in classes]
    num_classes = len(classes)
    print(f"[train] classes (folder -> label): " +
          ", ".join(f"{c}->{l}" for c, l in zip(classes, labels)))
    for key in ("porn", "hentai", "sexy"):
        if key not in labels:
            print(f"[train] WARNING: gating class '{key}' not present in dataset labels!")

    targets = [t for _, t in base.samples]
    kept = cap_per_class(targets, args.limit_per_class, args.seed)

    if args.val_dir:
        train_full = datasets.ImageFolder(args.data_dir, transform=train_tf)
        val_full = datasets.ImageFolder(args.val_dir, transform=val_tf)
        train_ds = Subset(train_full, kept) if args.limit_per_class else train_full
        val_ds = val_full
        train_targets = [targets[i] for i in kept] if args.limit_per_class else targets
    else:
        kept_targets = [targets[i] for i in kept]
        tr_local, va_local = stratified_split(kept_targets, args.val_frac, args.seed)
        train_idx = [kept[i] for i in tr_local]
        val_idx = [kept[i] for i in va_local]
        train_ds = Subset(datasets.ImageFolder(args.data_dir, transform=train_tf), train_idx)
        val_ds = Subset(datasets.ImageFolder(args.data_dir, transform=val_tf), val_idx)
        train_targets = [targets[i] for i in train_idx]
        print(f"[train] split: {len(train_idx)} train / {len(val_idx)} val")

    # Inverse-frequency class weights to counter imbalance (porn/neutral dominate).
    counts = np.bincount(np.array(train_targets), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    print("[train] class counts: " +
          ", ".join(f"{l}={int(c)}" for l, c in zip(labels, counts)))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, "nsfw_mnv4.pt")
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            gstep += 1
            running += float(loss)
            if bi % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  e{epoch} step {bi}/{steps_per_epoch} loss={float(loss):.4f} lr={lr_now:.2e}")

        m = evaluate(model, val_loader, device, num_classes)
        per_f1 = ", ".join(f"{l}={f:.3f}" for l, f in zip(labels, m["f1"]))
        print(f"[val] epoch {epoch}: loss={m['loss']:.4f} acc={m['acc']:.4f} "
              f"macroF1={m['macro_f1']:.4f} | F1[{per_f1}]")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": args.backbone,
                "labels": labels,
                "size": size,
                "mean": mean,
                "std": std,
                "macro_f1": best_f1,
            }, best_path)
            print(f"[train] saved best (macroF1={best_f1:.4f}) -> {best_path}")

    # Emit labels + preprocess for the export/JS side (plain resize; no center
    # crop; ToTensor scale 1/255; normalize by mean/std; NOT EfficientNet include_top).
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    preprocess = {
        "size": size,
        "cropSize": None,
        "doCenterCrop": False,
        "rescaleFactor": 1.0 / 255.0,
        "rescaleOffset": False,
        "doNormalize": True,
        "mean": mean,
        "std": std,
        "includeTop": False,
    }
    with open(os.path.join(args.out_dir, "preprocess.json"), "w") as f:
        json.dump(preprocess, f, indent=2)

    print(f"\n[train] done. best macroF1={best_f1:.4f}")
    print("[train] next:  python scripts/export_mnv4.py --calib-data " + args.data_dir)
    print("[train] then:  npm run embed-model && npm run build")


if __name__ == "__main__":
    main()
