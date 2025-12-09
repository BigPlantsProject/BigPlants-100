from __future__ import annotations
import os
import argparse
import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms

try:
    import timm
    _HAVE_TIMM = True
except Exception:
    _HAVE_TIMM = False

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ------------------------------
# Utils & Reproducibility
# ------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # images are fixed size → ok


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def list_images_direct(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and is_image_file(p)]


# ------------------------------
# Dataset curation
# ------------------------------

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

    # 1) collect from subfolders (hand/leaf/flower/fruit)
    sub_paths: List[Path] = []
    for part in PARTS:
        part_dir = species_dir / part
        if part_dir.exists():
            sub_paths.extend([p for p in part_dir.rglob("*") if p.is_file() and is_image_file(p)])

    # unique & shuffle
    sub_paths = list(dict.fromkeys(sub_paths))
    random.shuffle(sub_paths)

    used_available = 0
    avail_total = 0

    if len(sub_paths) >= cap:
        selected = sub_paths[:cap]
    else:
        # need to top-up from images directly under species_dir (available)
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
    label_map: Dict[str, int] = {}
    for idx, d in enumerate(species_dirs):
        label_map[d.name] = idx

    selections: List[SpeciesSelection] = []
    for d in tqdm(species_dirs, desc="Scanning dataset & selecting images per class (cap=%d)" % cap):
        sel = build_selection_for_species(d, cap=cap)
        selections.append(sel)
        for p in sel.paths:
            # detect origin (sub vs available) for analysis only
            part_name = "available"
            src = "available"
            for part in PARTS:
                if (d / part) in p.parents:
                    part_name = part
                    src = "sub"
                    break
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


# ------------------------------
# Torch Dataset & Transforms
# ------------------------------

class PlantImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label_id"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    # fill color to avoid black corners after rotation
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


# ------------------------------
# Model builder
# ------------------------------

def build_mobilenetv3_large(num_classes: int) -> Tuple[nn.Module, str]:
    backend = "timm" if _HAVE_TIMM else "torchvision"
    if _HAVE_TIMM:
        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
    else:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model, backend


# ------------------------------
# Train / Eval loops
# ------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    ys = []
    ps = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        ys.append(labels.detach().cpu().numpy())
        ps.append(preds.detach().cpu().numpy())
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return avg_loss, acc, y_true, y_pred


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ------------------------------
# Fold runner (PURE CV)
# ------------------------------

def run_one_fold(
    fold_id: int,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    device: torch.device,
    out_dir: Path,
):
    # Inner split (train/val) for early stopping on the training portion only
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)  # held-out fold test

    inner_train_df, inner_val_df = train_test_split(
        train_df,
        test_size=args.val_ratio,
        stratify=train_df["label_id"],
        random_state=args.seed + fold_id,
    )

    # Datasets & loaders
    train_tf, eval_tf = get_transforms(args.img_size)
    ds_train = PlantImageDataset(inner_train_df, transform=train_tf)
    ds_val = PlantImageDataset(inner_val_df, transform=eval_tf)
    ds_test = PlantImageDataset(test_df, transform=eval_tf)

    if args.use_weighted_sampler:
        # Compute inverse-frequency weights on inner_train only
        class_counts = inner_train_df["label_id"].value_counts().sort_index().values.astype(np.float64)
        weights_per_class = 1.0 / np.maximum(class_counts, 1)
        sample_weights = inner_train_df["label_id"].map({i: w for i, w in enumerate(weights_per_class)}).values
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(),
                                        num_samples=len(sample_weights), replacement=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=shuffle_flag,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model / Optim / Sched
    num_classes = df["label_id"].nunique()
    model, backend = build_mobilenetv3_large(num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n========== Fold {fold_id+1} / {args.kfolds} | backend={backend} ==========")

    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        # Val
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[Fold {fold_id+1}] Epoch {epoch}/{args.epochs}")
        print(f"[Fold {fold_id+1}] Train | loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[Fold {fold_id+1}] Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Early stopping on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save best weights in memory (and to disk)
            best_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'epoch': epoch,
                'backend': backend,
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

    # Load best & evaluate on the held-out test fold
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        print(f"[Fold {fold_id+1}] Loaded best checkpoint (val_acc={best_val_acc:.4f})")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"[Fold {fold_id+1}] Test (CV holdout) | loss={test_loss:.4f}, acc={test_acc:.4f}")

    # classification report per fold
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).transpose()
    fold_dir = out_dir / f"fold_{fold_id+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(fold_dir / f"test_classification_report_fold{fold_id+1}.csv")

    return {
        'fold': fold_id + 1,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
    }


# ------------------------------
# Main
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MobileNetV3-Large Pure K-Fold Cross-Validation (no fixed test set)")
    p.add_argument('--data_root', type=str, required=True, help='Root folder containing class subfolders')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory for logs/checkpoints')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--kfolds', type=int, default=5)
    p.add_argument('--val_ratio', type=float, default=0.1, help='Inner validation ratio used only for early stopping within each fold')
    p.add_argument('--cap_per_class', type=int, default=100)
    p.add_argument('--use_weighted_sampler', action='store_true')
    p.add_argument('--patience', type=int, default=7, help='Early stopping patience (epochs without val acc improvement)')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scan & select images per class (cap to 100)
    df = scan_dataset(data_root, cap=args.cap_per_class, out_dir=out_dir)

    # 2) Pure Stratified K-Fold on the entire dataset (no fixed test)
    labels = df["label_id"].values
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_seed = args.seed + fold_id
        set_seed(fold_seed)
        m = run_one_fold(
            fold_id=fold_id,
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        metrics.append(m)

    # 3) Summaries
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / 'kfold_metrics.csv', index=False)

    mean_val = float(metrics_df['best_val_acc'].mean())
    std_val = float(metrics_df['best_val_acc'].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    mean_test = float(metrics_df['test_acc'].mean())
    std_test = float(metrics_df['test_acc'].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    mean_test_loss = float(metrics_df['test_loss'].mean())
    std_test_loss  = float(metrics_df['test_loss'].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    summary = {
        'kfolds': args.kfolds,
        'val_acc': {'mean': mean_val, 'std': std_val},
        'test_acc_cv_holdout': {'mean': mean_test, 'std': std_test},
        'test_loss_cv_holdout': {'mean': mean_test_loss, 'std': std_test_loss},
    }
    with open(out_dir / 'kfold_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== PURE K-FOLD SUMMARY =====")
    print(f"Val Acc (inner):          mean={mean_val:.4f}  std={std_val:.4f}")
    print(f"Test Acc (CV holdout):    mean={mean_test:.4f}  std={std_test:.4f}")
    print(f"Test Loss (CV holdout):   mean={mean_test_loss:.4f}  std={std_test_loss:.4f}")
    print(f"Saved: {out_dir / 'kfold_metrics.csv'} and {out_dir / 'kfold_summary.json'}")


if __name__ == '__main__':
    main()
