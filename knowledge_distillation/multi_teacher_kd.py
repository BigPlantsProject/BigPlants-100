"""
Multi-Class Classification (100 classes) with Multi-Teacher Knowledge Distillation

Teachers:   ResNet-50 + ConvNeXtV2-S + EfficientNetV2-S
Student:    MobileNetV3-Large

Dataset rule per class (folder):
  - Prioritize images in subfolders: hand, leaf, flower, fruit (visible parts).
  - If total visible < cap (default 100), top up from images directly inside the class folder ("available").
  - If still < cap, accept smaller; if > cap, trim to cap.
  - Ignore seed/root/stem for training/validation/test.

Pipeline stages (controlled by --stage):
  1) 'teachers': train each teacher individually on train/val; save best checkpoints.
  2) 'distill' : load teacher checkpoints, freeze them; train student with KD on train/val; test on test.
  3) 'all'     : do (1) then (2).

Final artifacts:
  - Best checkpoints under: out_dir/teachers/<model>/best.pt and out_dir/student/best_student_kd.pt
  - CSV/JSON metrics and classification reports.

Usage example:
  python kd_multiteacher_mobilenetv3_resnet.py \
    --data_root /path/to/dataset_root \
    --out_dir /path/to/outputs \
    --epochs 30 --batch_size 32 --lr 3e-4 --num_workers 8 \
    --stage all --test_ratio 0.2 --val_ratio 0.1 --cap_per_class 100 --use_weighted_sampler
"""
from __future__ import annotations

import os
import json
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed-size imgs → faster

# ---------------------------
# Dataset scanning & selection logic
# ---------------------------
VISIBLE_PARTS = ["hand", "leaf", "flower", "fruit"]

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def list_images_direct(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and is_image_file(p)]

@dataclass
class ClassSelection:
    species: str
    selected_paths: List[Path]
    visible_count: int
    available_used: int
    final_count: int

def build_selection_for_class(class_dir: Path, cap: int = 100) -> ClassSelection:
    species = class_dir.name
    vis_paths: List[Path] = []
    for part in VISIBLE_PARTS:
        part_dir = class_dir / part
        if part_dir.exists():
            vis_paths.extend([p for p in part_dir.rglob("*") if p.is_file() and is_image_file(p)])
    vis_paths = list(dict.fromkeys(vis_paths))
    random.shuffle(vis_paths)

    selected = []
    available_used = 0
    if len(vis_paths) >= cap:
        selected = vis_paths[:cap]
    else:
        selected = list(vis_paths)
        need = cap - len(vis_paths)
        avail = list_images_direct(class_dir)
        random.shuffle(avail)
        take = avail[:need]
        available_used = len(take)
        selected.extend(take)

    return ClassSelection(
        species=species,
        selected_paths=selected,
        visible_count=len(vis_paths),
        available_used=available_used,
        final_count=len(selected),
    )

def scan_dataset(data_root: Path, cap: int = 100, out_dir: Optional[Path] = None) -> pd.DataFrame:
    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda p: p.name)

    label_map: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    rows = []
    for d in tqdm(class_dirs, desc=f"Selecting images per class (cap={cap})"):
        sel = build_selection_for_class(d, cap)
        for p in sel.selected_paths:
            src = "available"
            part_name = "available"
            for part in VISIBLE_PARTS:
                if (d / part) in p.parents:
                    src = "sub"
                    part_name = part
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
            json.dump({k: int(v) for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)
    print(f"Selected images: {len(df)}; classes: {df['species'].nunique()}")
    return df

# ---------------------------
# Dataset & transforms
# ---------------------------
class PlantDataset(Dataset):
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
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15, fill=(123, 116, 103)),
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
# Model builders
# ---------------------------
def build_resnet50(num_classes: int) -> Tuple[nn.Module, str]:
    try:
        import timm
        for name in ["resnet50", "resnet50.a1_in1k"]:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return m, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass
    from torchvision.models import resnet50, ResNet50_Weights
    # Use IMAGENET1K_V2 if available, fallback handled by torchvision
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m, "torchvision:resnet50"

def build_convnextv2_s(num_classes: int) -> Tuple[nn.Module, str]:
    try:
        import timm
        for name in ["convnextv2_small", "convnextv2_small.fcmae_ft_in22k_in1k"]:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return m, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
    m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, num_classes)
    return m, "torchvision:convnext_small (fallback)"

def build_efficientnetv2_s(num_classes: int) -> Tuple[nn.Module, str]:
    try:
        import timm
        for name in ["efficientnetv2_s", "tf_efficientnetv2_s", "tf_efficientnetv2_s_in21k"]:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return m, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # replace final linear safely
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
        replaced = False
        for i in reversed(range(len(m.classifier))):
            if isinstance(m.classifier[i], nn.Linear):
                in_f = m.classifier[i].in_features
                m.classifier[i] = nn.Linear(in_f, num_classes)
                replaced = True
                break
        if not replaced:
            raise RuntimeError("Could not replace classifier Linear in torchvision efficientnet_v2_s")
    else:
        raise RuntimeError("Unexpected EfficientNetV2-S structure in torchvision")
    return m, "torchvision:efficientnet_v2_s"

def build_mobilenetv3_large(num_classes: int) -> Tuple[nn.Module, str]:
    try:
        import timm
        for name in ["mobilenetv3_large_100", "mobilenetv3_large_100.ra_in1k"]:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return m, f"timm:{name}"
            except Exception:
                continue
    except Exception:
        pass
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    # replace last linear robustly
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
        for i in reversed(range(len(m.classifier))):
            if isinstance(m.classifier[i], nn.Linear):
                in_f = m.classifier[i].in_features
                m.classifier[i] = nn.Linear(in_f, num_classes)
                break
    else:
        for name, mod in reversed(list(m.named_modules())):
            if isinstance(mod, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                last_name = name.split(".")[-1]
                parent = m.get_submodule(parent_name) if parent_name else m
                setattr(parent, last_name, nn.Linear(mod.in_features, num_classes))
                break
    return m, "torchvision:mobilenet_v3_large"

# ---------------------------
# Distillation losses
# ---------------------------
def kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, T: float) -> torch.Tensor:
    """
    KL(student || teacher) where teacher_probs are soft targets (already softmax(T)).
    Use standard KD scaling by T^2.
    """
    log_p_s = torch.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T * T)
    return loss

def ensemble_teacher_probs(teacher_logits: List[torch.Tensor], T: float, weights: List[float]) -> torch.Tensor:
    """
    Combine teacher logits by weighted average in probability space with temperature T.
    Returns probs shape [B, C].
    """
    assert len(teacher_logits) == len(weights)
    w = torch.tensor(weights, device=teacher_logits[0].device, dtype=teacher_logits[0].dtype)
    w = w / w.sum()
    probs = None
    for i, z in enumerate(teacher_logits):
        p = torch.softmax(z / T, dim=1)
        probs = p * w[i] if probs is None else probs + p * w[i]
    return probs.clamp(min=1e-8, max=1.0)

# ---------------------------
# Train & evaluate helpers
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total, correct, run_loss = 0, 0, 0.0
    ys, ps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        run_loss += float(loss.item()) * labels.size(0)
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


def train_one_epoch_ce(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> Tuple[float, float]:
    model.train()
    total, correct, run_loss = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return run_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch_kd(student: nn.Module, teachers: List[nn.Module], loader: DataLoader,
                       ce_criterion: nn.Module, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler,
                       device: torch.device, T: float, lambda_kd: float, t_weights: List[float]) -> Tuple[float, float]:
    student.train()
    for t in teachers:
        t.eval()
    total, correct, run_loss = 0, 0, 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            with torch.no_grad():
                t_logits = [t(imgs) for t in teachers]
                t_probs = ensemble_teacher_probs(t_logits, T=T, weights=t_weights)
            s_logits = student(imgs)
            loss_ce = ce_criterion(s_logits, labels)
            loss_kd = kd_loss(s_logits, t_probs, T=T)
            loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += float(loss.item()) * labels.size(0)
        preds = s_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return run_loss / max(total, 1), correct / max(total, 1)

# ---------------------------
# High-level training routines
# ---------------------------
def train_teacher(model: nn.Module, name: str,
                  train_loader: DataLoader, val_loader: DataLoader,
                  device: torch.device, epochs: int, lr: float, weight_decay: float,
                  out_dir: Path, patience: int = 7) -> Dict:
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    subdir = out_dir / name
    subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Train {name} =====")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_ce(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[{name}] Epoch {epoch}/{epochs}")
        print(f"[{name}] Train | loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[{name}] Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "val_acc": float(best_val_acc),
                "model_state": model.state_dict(),
            }
            torch.save(best_state, subdir / "best.pt")
            print(f"✅ [{name}] Saved best → {subdir/'best.pt'} (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{name}] Early stopping (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        print(f"[{name}] Loaded best (val_acc={best_val_acc:.4f})")

    return {"name": name, "model": model, "best_val_acc": float(best_val_acc)}


def train_student_kd(student: nn.Module, teachers: List[nn.Module], teacher_names: List[str],
                     train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                     device: torch.device, epochs: int, lr: float, weight_decay: float,
                     out_dir: Path, T: float, lambda_kd: float, t_weights: List[float],
                     patience: int = 7) -> Dict:
    student = student.to(device)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)

    for t in teachers:
        for p in t.parameters():
            p.requires_grad = False
        t.eval()
        t.to(device)

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    sdir = out_dir / "student"
    sdir.mkdir(parents=True, exist_ok=True)

    print("\n===== Train Student KD (MobileNetV3-Large) =====")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_kd(student, teachers, train_loader,
                                             ce_criterion, optimizer, scaler, device,
                                             T=T, lambda_kd=lambda_kd, t_weights=t_weights)
        val_loss, val_acc, _, _ = evaluate(student, val_loader, ce_criterion, device)
        scheduler.step()

        print(f"[StudentKD] Epoch {epoch}/{epochs}")
        print(f"[StudentKD] Train | loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[StudentKD] Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "val_acc": float(best_val_acc),
                "model_state": student.state_dict(),
                "teacher_names": teacher_names,
                "T": float(T),
                "lambda_kd": float(lambda_kd),
                "teacher_weights": [float(w) for w in t_weights],
            }
            torch.save(best_state, sdir / "best_student_kd.pt")
            print(f"✅ [StudentKD] Saved best → {sdir/'best_student_kd.pt'} (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[StudentKD] Early stopping (patience={patience})")
                break

    if best_state is not None:
        student.load_state_dict(best_state["model_state"])
        print(f"[StudentKD] Loaded best (val_acc={best_val_acc:.4f})")

    test_loss, test_acc, y_true, y_pred = evaluate(student, test_loader, ce_criterion, device)
    print(f"[StudentKD] Test | loss={test_loss:.4f}, acc={test_acc:.4f}")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(sdir / "student_test_report.csv")
    cm = confusion_matrix(y_true, y_pred)
    np.save(sdir / "student_test_confusion_matrix.npy", cm)

    return {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }

# ---------------------------
# Data split helpers
# ---------------------------
def make_dataloaders(df: pd.DataFrame, img_size: int, batch_size: int, num_workers: int,
                     val_ratio: float, test_ratio: float, use_weighted_sampler: bool, seed: int):
    trainval_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label_id"], random_state=seed
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_ratio, stratify=trainval_df["label_id"], random_state=seed
    )
    print(f"Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    train_tf, eval_tf = get_transforms(img_size)
    ds_train = PlantDataset(train_df, transform=train_tf)
    ds_val   = PlantDataset(val_df,   transform=eval_tf)
    ds_test  = PlantDataset(test_df,  transform=eval_tf)

    if use_weighted_sampler:
        counts = train_df["label_id"].value_counts().sort_index().values.astype(np.float64)
        weights_per_class = 1.0 / np.maximum(counts, 1)
        mapping = {i: w for i, w in enumerate(weights_per_class)}
        sample_w = train_df["label_id"].map(mapping).values
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            replacement=True
        )
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ---------------------------
# CLI & main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-Teacher KD → MobileNetV3-Large (100-class plants)")
    p.add_argument("--data_root", type=str, required=True, help="Root dir: 100 class folders")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--cap_per_class", type=int, default=100)
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--patience", type=int, default=7)

    # KD hyper-parameters
    p.add_argument("--kd_T", type=float, default=3.0, help="KD temperature")
    p.add_argument("--kd_lambda", type=float, default=0.5, help="KD weight (0..1)")
    p.add_argument("--teacher_weights", type=str, default="", help="Comma separated weights for [ResNet, ConvNeXtV2-S, EfficientNetV2-S]")

    # Stage control
    p.add_argument("--stage", type=str, choices=["teachers", "distill", "all"], default="all")
    # Optionally provide teacher ckpts to skip training
    p.add_argument("--resnet_ckpt", type=str, default="")
    p.add_argument("--convnextv2s_ckpt", type=str, default="")
    p.add_argument("--effnetv2s_ckpt", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(data_root, cap=args.cap_per_class, out_dir=out_dir)
    num_classes = df["label_id"].nunique()

    train_loader, val_loader, test_loader = make_dataloaders(
        df, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        use_weighted_sampler=args.use_weighted_sampler, seed=args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.teacher_weights.strip():
        try:
            t_weights = [float(x) for x in args.teacher_weights.split(",")]
        except Exception:
            raise ValueError("--teacher_weights must be comma-separated floats, e.g. '0.34,0.33,0.33'")
        if len(t_weights) != 3 or any(w < 0 for w in t_weights):
            raise ValueError("--teacher_weights must have 3 non-negative numbers")
    else:
        t_weights = [1/3, 1/3, 1/3]

    # 3) Build models
    # Teachers: ResNet-50, ConvNeXtV2-S, EfficientNetV2-S
    resnet, resnet_backend = build_resnet50(num_classes)
    conv2, conv2_backend = build_convnextv2_s(num_classes)
    eff2, eff2_backend = build_efficientnetv2_s(num_classes)
    teacher_backends = [resnet_backend, conv2_backend, eff2_backend]
    print("Teacher backends:", teacher_backends)
    # Student
    student, student_backend = build_mobilenetv3_large(num_classes)
    print("Student backend:", student_backend)

    # 4) Train teachers (if needed)
    teachers_dir = out_dir / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpts = {
        "resnet": args.resnet_ckpt or str(teachers_dir / "resnet50" / "best.pt"),
        "conv2s": args.convnextv2s_ckpt or str(teachers_dir / "convnextv2_s" / "best.pt"),
        "eff2s": args.effnetv2s_ckpt or str(teachers_dir / "efficientnetv2_s" / "best.pt"),
    }

    if args.stage in ("teachers", "all"):
        if not args.resnet_ckpt or not Path(args.resnet_ckpt).exists():
            _ = train_teacher(resnet, "resnet50", train_loader, val_loader, device,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              out_dir=teachers_dir, patience=args.patience)
        else:
            print(f"[ResNet-50] Skipping training, using provided ckpt: {args.resnet_ckpt}")

        if not args.convnextv2s_ckpt or not Path(args.convnextv2s_ckpt).exists():
            _ = train_teacher(conv2, "convnextv2_s", train_loader, val_loader, device,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              out_dir=teachers_dir, patience=args.patience)
        else:
            print(f"[ConvNeXtV2-S] Skipping training, using provided ckpt: {args.convnextv2s_ckpt}")

        if not args.effnetv2s_ckpt or not Path(args.effnetv2s_ckpt).exists():
            _ = train_teacher(eff2, "efficientnetv2_s", train_loader, val_loader, device,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              out_dir=teachers_dir, patience=args.patience)
        else:
            print(f"[EffNetV2-S] Skipping training, using provided ckpt: {args.effnetv2s_ckpt}")

    # 5) Load teacher weights (from provided or from trained)
    def load_ckpt(model: nn.Module, path: str, tag: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found for {tag}: {path}")
        state = torch.load(p, map_location="cpu")
        key = "model_state" if isinstance(state, dict) and "model_state" in state else None
        if key is None:
            model.load_state_dict(state)
        else:
            model.load_state_dict(state[key])
        print(f"[{tag}] Loaded checkpoint: {path}")

    # reload fresh teacher instances
    resnet_t, _ = build_resnet50(num_classes)
    conv2_t, _ = build_convnextv2_s(num_classes)
    eff2_t, _ = build_efficientnetv2_s(num_classes)

    if args.stage in ("distill", "all"):
        load_ckpt(resnet_t, teacher_ckpts["resnet"], "ResNet-50")
        load_ckpt(conv2_t, teacher_ckpts["conv2s"], "ConvNeXtV2-S")
        load_ckpt(eff2_t, teacher_ckpts["eff2s"], "EfficientNetV2-S")

        # 6) Train student with KD
        kd_results = train_student_kd(
            student=student, teachers=[resnet_t, conv2_t, eff2_t], teacher_names=["resnet50","convnextv2_s","efficientnetv2_s"],
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            out_dir=out_dir, T=args.kd_T, lambda_kd=args.kd_lambda, t_weights=t_weights,
            patience=args.patience
        )

        with open(out_dir / "student_kd_summary.json", "w", encoding="utf-8") as f:
            json.dump(kd_results, f, ensure_ascii=False, indent=2)

    print("🎉🎉🎉Done!")


if __name__ == "__main__":
    main()
