"""13_train_loss.py — Train CMPD300 encoders with 4 losses, everything else fixed (Stage 3).

The ONLY variable is the loss: {ce, arcface, supcon, triplet}. Backbone (ResNet-50,
ImageNet init), strong augmentation, PK sampler, optimizer/schedule, epochs and dataset
are identical across conditions. The encoder is then frozen and evaluated by clustering on
Zenodo (scripts/14_eval_losses.py).

The checkpoint is saved as a STANDARD ResNet-50 state_dict (head goes in the fc slot, which
the extractor discards), so all four load identically via src/reid/encoders.resnet50_checkpoint
and evaluation uses the 2048-d backbone feature for every condition (brief guardrail #3).

Usage:
    python scripts/13_train_loss.py --train-dir ~/data/cmpd300/Baseline/train --loss supcon \
        --epochs 80 --seed 0 --out outputs/checkpoints/cmpd300_supcon_s0.pt
    python scripts/13_train_loss.py --train-dir ... --loss ce --smoke     # quick pipeline check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from src.dataset import MuzzleDataset
from src.models import build_model
from src.reid.metric_losses import ProjectionHead, SupConLoss, TripletBatchHard
from src.reid.reid_dataset import entries_from_folders
from src.reid.sampler import PKSampler
from src.transforms import build_strong_train_transform
from src.utils import get_device, get_logger, save_json, set_seed

# ArcFace head lives in the existing script; import to avoid duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
ArcMarginProduct = import_module("07_train_arcface").ArcMarginProduct

LOSSES = ("ce", "arcface", "supcon", "triplet")


class Encoder(nn.Module):
    """ResNet-50 backbone (fc→Identity, 2048-d feat) + a per-loss head."""

    def __init__(self, loss: str, num_classes: int, pretrained: bool = True,
                 arc_s: float = 30.0, arc_m: float = 0.5):
        super().__init__()
        base = build_model("resnet50", num_classes=num_classes, freeze_backbone=False,
                           pretrained=pretrained)
        self.feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.loss = loss
        if loss == "ce":
            self.head = nn.Linear(self.feat_dim, num_classes)
        elif loss == "arcface":
            self.head = ArcMarginProduct(self.feat_dim, num_classes, s=arc_s, m=arc_m)
        elif loss == "supcon":
            self.head = ProjectionHead(self.feat_dim, self.feat_dim, 128)
        elif loss == "triplet":
            self.head = None
        else:
            raise ValueError(f"unknown loss {loss}")

    def forward(self, x, labels=None):
        feat = self.backbone(x)                       # [B, 2048]
        if self.loss == "ce":
            return self.head(feat)
        if self.loss == "arcface":
            return self.head(feat, labels)
        if self.loss == "supcon":
            return self.head(feat)                    # [B, 128] normalized
        return F.normalize(feat, dim=1)               # triplet: normalized backbone feat

    def export_state_dict(self, num_classes: int) -> dict:
        """Standard ResNet-50 state_dict (backbone trained; fc slot = dummy/head weight)."""
        export = build_model("resnet50", num_classes=num_classes, freeze_backbone=False,
                             pretrained=False)
        sd = export.state_dict()
        sd.update(self.backbone.state_dict())         # trained conv/bn (backbone.fc=Identity)
        if self.loss in ("ce", "arcface"):
            w = self.head.weight.detach().cpu().clone()
            sd["fc.weight"] = w
            sd["fc.bias"] = (self.head.bias.detach().cpu().clone()
                             if getattr(self.head, "bias", None) is not None
                             else torch.zeros(num_classes))
        # supcon/triplet: keep the fresh dummy fc (discarded by the extractor anyway)
        return sd


def compute_loss(model, criterion, imgs, labels):
    if model.loss in ("ce", "arcface"):
        logits = model(imgs, labels)
        return criterion(logits, labels), (logits.argmax(1) == labels).float().mean().item()
    out = model(imgs)                                 # supcon: projected; triplet: normalized feat
    return criterion(out, labels), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--loss", required=True, choices=LOSSES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--P", type=int, default=16, help="identities per batch")
    ap.add_argument("--K", type=int, default=4, help="images per identity per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE_S2)
    ap.add_argument("--arc-s", type=float, default=30.0)
    ap.add_argument("--arc-m", type=float, default=0.5)
    ap.add_argument("--supcon-t", type=float, default=0.07)
    ap.add_argument("--triplet-margin", type=float, default=None,
                    help="None → soft-margin (softplus); float → hard margin")
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    config.ensure_output_dirs()
    log = get_logger(f"train.{args.loss}")
    device = get_device()

    train_dir = Path(args.train_dir).expanduser()
    entries, id_map = entries_from_folders(train_dir)
    num_classes = len(id_map)
    P, K = (min(args.P, num_classes - 1), args.K)
    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        entries = entries[:400]
        # recount ids present in the subset (labels are global; sampler needs present ones)
    labels = [e["label"] for e in entries]
    use_norm = config.USE_IMAGENET_NORM_S2
    log.info(f"loss={args.loss} | device={device} | ids={num_classes} | imgs={len(entries)} | "
             f"P={P} K={K} | epochs={epochs} | image_size={args.image_size} | norm={use_norm}")

    tf = build_strong_train_transform(args.image_size, use_norm)
    ds = MuzzleDataset(entries, transform=tf, data_dir=train_dir)
    sampler = PKSampler(labels, P=P, K=K, seed=args.seed)
    loader = DataLoader(ds, batch_size=P * K, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True)

    model = Encoder(args.loss, num_classes, pretrained=not args.smoke,
                    arc_s=args.arc_s, arc_m=args.arc_m).to(device)
    if args.loss in ("ce", "arcface"):
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "supcon":
        criterion = SupConLoss(args.supcon_t)
    else:
        criterion = TripletBatchHard(args.triplet_margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - args.warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup])

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run, acc_sum, nb = 0.0, 0.0, 0
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            loss, acc = compute_loss(model, criterion, imgs, labs)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run += loss.item(); nb += 1
            if acc is not None:
                acc_sum += acc
        sched.step()
        msg = f"ep {epoch:02d}/{epochs} | loss {run / max(1, nb):.4f}"
        if args.loss in ("ce", "arcface"):
            msg += f" | train acc {acc_sum / max(1, nb):.4f}"
        msg += f" | lr {optimizer.param_groups[0]['lr']:.2e}"
        log.info(msg)

    out = Path(args.out) if args.out else config.CHECKPOINTS_DIR / f"cmpd300_{args.loss}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.export_state_dict(num_classes), "model_name": "resnet50",
                "num_classes": num_classes,
                "run_config": {"image_size": args.image_size, "use_imagenet_norm": use_norm},
                "method": args.loss, "seed": args.seed}, out)
    summary = {"loss": args.loss, "seed": args.seed, "num_classes": num_classes,
               "epochs": epochs, "P": P, "K": K, "image_size": args.image_size,
               "checkpoint": str(out), "elapsed_sec": round(time.time() - t0, 1)}
    save_json(summary, config.RESULTS_DIR / f"13_train_{args.loss}_s{args.seed}.json")
    log.info(f"saved {args.loss} encoder → {out} | {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
