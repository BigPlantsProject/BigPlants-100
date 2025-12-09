from __future__ import annotations

import os
import json
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # images fixed size => faster

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def list_images_direct(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and is_image_file(p)]

# ---------------------------
# Dataset curation (hand/leaf/flower/fruit + top-up available, cap<=100)
# ---------------------------
PARTS = ["hand", "leaf", "flower", "fruit"]

@dataclass
class SpeciesSelection:
    species: str
    paths: List[Path]
    used_available: int
    sub_parts_count: int
    available_count: int
    final_count: int

def build_selection_for_species(species_dir: Path, cap: int = 100) -> SpeciesSelection:
    species = species_dir.name
    sub_paths: List[Path] = []
    for part in PARTS:
        part_dir = species_dir / part
        if part_dir.exists():
            sub_paths.extend([p for p in part_dir.rglob("*") if p.is_file() and is_image_file(p)])
    # unique + shuffle
    sub_paths = list(dict.fromkeys(sub_paths))
    random.shuffle(sub_paths)

    used_available = 0
    avail_total = 0
    if len(sub_paths) >= cap:
        selected = sub_paths[:cap]
    else:
        avail_paths = list_images_direct(species_dir)
        avail_total = len(avail_paths)
        random.shuffle(avail_paths)
        need = cap - len(sub_paths)
        take = avail_paths[:need]
        selected = sub_paths + take
        used_available = len(take)

    return SpeciesSelection(
        species=species,
        paths=selected,
        used_available=used_available,
        sub_parts_count=len(sub_paths),
        available_count=avail_total,
        final_count=len(selected),
    )

def scan_dataset(data_root: Path, cap: int = 100, out_dir: Path | None = None) -> pd.DataFrame:
    species_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    species_dirs.sort(key=lambda p: p.name)

    rows = []
    label_map: Dict[str, int] = {d.name: i for i, d in enumerate(species_dirs)}
    for d in tqdm(species_dirs, desc=f"Scanning dataset & selecting images per class (cap={cap})"):
        sel = build_selection_for_species(d, cap=cap)
        for p in sel.paths:
            # record origin 'sub' (part) vs 'available'
            part_name = "available"
            src = "available"
            for part in PARTS:
                if (d / part) in p.parents:
                    part_name = part; src = "sub"; break
            rows.append({
                "path": str(p.resolve()),
                "species": d.name,
                "label_id": label_map[d.name],
                "source": src,
                "part": part_name,
            })

    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "dataset_selected.csv", index=False)
        with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Selected images: {len(df)}; classes: {df['species'].nunique()}")
    return df

# ---------------------------
# Dataset & Transforms
# ---------------------------
class PlantImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_id"])

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    fill_val = tuple(int(x * 255) for x in (0.485, 0.456, 0.406))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15, fill=fill_val),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

# ---------------------------
# Model Factory: EfficientNetV2-S (prefer timm, fallback torchvision)
# ---------------------------
def build_efficientnetv2_s(num_classes: int) -> Tuple[nn.Module, str]:
    # Prefer timm
    try:
        import timm
        for name in ["efficientnetv2_s", "tf_efficientnetv2_s", "tf_efficientnetv2_s_in21k"]:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return model, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass
    # Fallback: torchvision
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    tv_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # replace classifier last Linear
    if hasattr(tv_model, "classifier") and isinstance(tv_model.classifier, nn.Sequential):
        last = tv_model.classifier[-1]
        if isinstance(last, nn.Linear):
            in_f = last.in_features
            tv_model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            # search in classifier
            for i in reversed(range(len(tv_model.classifier))):
                if isinstance(tv_model.classifier[i], nn.Linear):
                    in_f = tv_model.classifier[i].in_features
                    tv_model.classifier[i] = nn.Linear(in_f, num_classes)
                    break
    else:
        # extreme fallback: search any last Linear
        replaced = False
        for name, mod in reversed(list(tv_model.named_modules())):
            if isinstance(mod, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                last_name = name.split(".")[-1]
                parent = tv_model.get_submodule(parent_name) if parent_name else tv_model
                setattr(parent, last_name, nn.Linear(mod.in_features, num_classes))
                replaced = True
                break
        if not replaced:
            raise RuntimeError("Cannot replace final Linear for torchvision efficientnet_v2_s.")
    return tv_model, "torchvision"

# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        run_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        ys.append(labels.detach().cpu().numpy())
        ps.append(preds.detach().cpu().numpy())
    avg_loss = run_loss / max(total, 1)
    acc = correct / max(total, 1)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return avg_loss, acc, y_true, y_pred

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> Tuple[float, float]:
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return run_loss / max(total, 1), correct / max(total, 1)

# ---------------------------
# One fold (outer test + inner val)
# ---------------------------
def run_one_fold(fold_id: int, df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                 args, device: torch.device, out_dir: Path) -> Dict:
    # Outer split
    train_df = df.iloc[train_idx].reset_index(drop=True)   # for inner train/val
    test_df  = df.iloc[test_idx].reset_index(drop=True)    # CV holdout

    # Inner split
    inner_train_df, inner_val_df = train_test_split(
        train_df,
        test_size=args.val_ratio,
        stratify=train_df["label_id"],
        random_state=args.seed + fold_id,
    )

    # Datasets
    train_tf, eval_tf = get_transforms(args.img_size)
    ds_train = PlantImageDataset(inner_train_df, transform=train_tf)
    ds_val   = PlantImageDataset(inner_val_df,   transform=eval_tf)
    ds_test  = PlantImageDataset(test_df,        transform=eval_tf)

    # Loaders
    if args.use_weighted_sampler:
        counts = inner_train_df["label_id"].value_counts().sort_index().values.astype(np.float64)
        weights_per_class = 1.0 / np.maximum(counts, 1)
        mapping = {i: w for i, w in enumerate(weights_per_class)}
        sample_w = inner_train_df["label_id"].map(mapping).values
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            replacement=True
        )
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model & optim
    num_classes = df["label_id"].nunique()
    model, backend = build_efficientnetv2_s(num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n========== Fold {fold_id+1} / {args.kfolds} | backend={backend} ==========")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[Fold {fold_id+1}] Epoch {epoch}/{args.epochs}")
        print(f"[Fold {fold_id+1}] Train | loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[Fold {fold_id+1}] Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "val_acc": float(best_val_acc),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "backend": backend,
            }
            fold_dir = out_dir / f"fold_{fold_id+1}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, fold_dir / f"best_model_fold{fold_id+1}.pt")
            print(f"✅ [Fold {fold_id+1}] Saved best checkpoint → {fold_dir / f'best_model_fold{fold_id+1}.pt'} (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[Fold {fold_id+1}] Early stopping triggered (patience={args.patience}).")
                break

    # Load best & evaluate on held-out TEST
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        print(f"[Fold {fold_id+1}] Loaded best checkpoint (val_acc={best_val_acc:.4f})")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"[Fold {fold_id+1}] Test (CV holdout) | loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Save classification report on TEST fold
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    fold_dir = out_dir / f"fold_{fold_id+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(fold_dir / f"test_classification_report_fold{fold_id+1}.csv")

    return {
        "fold": fold_id + 1,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }

# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EfficientNetV2-S Pure K-Fold Cross-Validation (no fixed test set)")
    p.add_argument("--data_root", type=str, required=True, help="Root folder containing class subfolders")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for logs/checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.1, help="Inner validation ratio (for early stopping)")
    p.add_argument("--cap_per_class", type=int, default=100)
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience (#epochs without val improvement)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scan dataset (apply selection rule & cap)
    df = scan_dataset(data_root, cap=args.cap_per_class, out_dir=out_dir)

    # 2) Stratified K-Fold (pure CV)
    labels = df["label_id"].values
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        set_seed(args.seed + fold_id)
        m = run_one_fold(
            fold_id=fold_id, df=df, train_idx=train_idx, test_idx=test_idx,
            args=args, device=device, out_dir=out_dir
        )
        metrics.append(m)

    # 3) Summaries
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "kfold_metrics.csv", index=False)

    mean_val_acc = float(metrics_df["best_val_acc"].mean())
    std_val_acc  = float(metrics_df["best_val_acc"].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    mean_test_acc = float(metrics_df["test_acc"].mean())
    std_test_acc  = float(metrics_df["test_acc"].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    mean_test_loss = float(metrics_df["test_loss"].mean())
    std_test_loss  = float(metrics_df["test_loss"].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    summary = {
        "kfolds": int(args.kfolds),
        "val_acc_inner": {"mean": mean_val_acc, "std": std_val_acc},
        "test_acc_cv_holdout": {"mean": mean_test_acc, "std": std_test_acc},
        "test_loss_cv_holdout": {"mean": mean_test_loss, "std": std_test_loss},
    }
    with open(out_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== PURE K-FOLD SUMMARY =====")
    print(f"Val Acc (inner):        mean={mean_val_acc:.4f}  std={std_val_acc:.4f}")
    print(f"Test Acc (CV holdout):  mean={mean_test_acc:.4f}  std={std_test_acc:.4f}")
    print(f"Test Loss (CV holdout): mean={mean_test_loss:.4f}  std={std_test_loss:.4f}")
    print(f"Saved: {out_dir / 'kfold_metrics.csv'} and {out_dir / 'kfold_summary.json'}")

if __name__ == "__main__":
    main()
