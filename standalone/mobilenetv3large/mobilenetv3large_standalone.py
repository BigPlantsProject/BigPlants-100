import os
import json
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}


def list_images_direct(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if is_image_file(p)]


# -----------------------------
# Dataset curation theo yêu cầu
# -----------------------------

def build_selection_for_species(species_dir: Path, parts_keep=("hand","leaf","flower","fruit"), per_class_cap=100, seed=42):
    rng = random.Random(seed)
    sub_images: List[Path] = []

    for part in parts_keep:
        part_dir = species_dir / part
        if part_dir.exists() and part_dir.is_dir():
            imgs = [p for p in part_dir.rglob("*") if is_image_file(p)]
            sub_images.extend(imgs)

    # unique + shuffle
    sub_images = list(dict.fromkeys(sub_images))
    rng.shuffle(sub_images)

    # Nếu >= cap → cắt đúng cap, ngược lại bù từ available
    if len(sub_images) >= per_class_cap:
        chosen = sub_images[:per_class_cap]
        return chosen

    available_images = list_images_direct(species_dir)
    rng.shuffle(available_images)
    need = per_class_cap - len(sub_images)
    chosen = sub_images + available_images[:need]
    return chosen


def scan_dataset(data_root: Path, parts_keep=("hand","leaf","flower","fruit"), per_class_cap=100, seed=42) -> pd.DataFrame:
    species_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    species_names = [p.name for p in species_dirs]
    species_to_id = {name: i for i, name in enumerate(species_names)}

    rows = []
    for species_dir in species_dirs:
        species = species_dir.name
        selected_paths = build_selection_for_species(species_dir, parts_keep, per_class_cap, seed)
        for img in selected_paths:
            # xác định nguồn để tham khảo
            part_val = None
            for anc in img.parents:
                if anc == species_dir:
                    break
                if anc.name in parts_keep:
                    part_val = anc.name
                    break
            source = "sub" if part_val is not None else "available"
            rows.append({
                "path": str(img.resolve()),
                "species": species,
                "label_id": species_to_id[species],
                "source": source,
                "part": part_val if part_val is not None else "available",
            })
    return pd.DataFrame(rows)


# -----------------------------
# Torch Dataset & Transforms
# -----------------------------

class PlantImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_id"])


def get_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15, fill=tuple(int(x*255) for x in mean)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms


# -----------------------------
# Model
# -----------------------------

def build_mobilenetv3_large(num_classes: int):
    try:
        import timm
        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=num_classes)
        return model, "timm"
    except Exception:
        pass
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    tv_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = tv_model.classifier[3].in_features
    tv_model.classifier[3] = nn.Linear(in_features, num_classes)
    return tv_model, "torchvision"


# -----------------------------
# Train / Eval
# -----------------------------

from torch.amp import autocast, GradScaler


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and torch.cuda.is_available():
            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    return avg_loss, acc, y_true, y_pred


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_mnv3_cap100_70_10_20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_weighted_sampler", action="store_true")
    # Quan trọng: để đúng 70:10:20, set mặc định test=0.20, val=0.10
    parser.add_argument("--val_ratio", type=float, default=0.10, help="tỉ lệ validation trên toàn bộ dữ liệu")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="tỉ lệ test trên toàn bộ dữ liệu")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    assert 0.0 < args.test_ratio < 0.5, "test_ratio nên hợp lý (ví dụ 0.2 cho 20%)"
    assert 0.0 < args.val_ratio < 0.5,  "val_ratio nên hợp lý (ví dụ 0.1 cho 10%)"
    assert args.val_ratio + args.test_ratio < 1.0, "val_ratio + test_ratio phải < 1.0"

    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset & selecting images per class (cap=100)...")
    df = scan_dataset(data_root=data_root, per_class_cap=100, seed=args.seed)
    assert len(df) > 0, "No valid images found!"
    df.to_csv(out_dir / "dataset_selected.csv", index=False)

    species_list = sorted(df["species"].unique().tolist())
    label_map = {s: int(df[df["species"] == s]["label_id"].iloc[0]) for s in species_list}
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"Selected images: {len(df)}; classes: {len(species_list)}")

    # ---- Split 70:10:20 ----
    print("Splitting train/val/test (target ≈ 70/10/20)...")
    df_trainval, df_test = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=df["species"],
    )
    # Để val chiếm đúng 10% tổng, cần tách từ phần 80% còn lại theo tỉ lệ 0.10 / 0.80 = 0.125
    val_ratio_adj = args.val_ratio / (1.0 - args.test_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adj,
        random_state=args.seed,
        stratify=df_trainval["species"],
    )

    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        d.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} images")

    total_n = len(df)
    print(
        f"Actual ratios → train: {len(df_train)/total_n:.3f}, val: {len(df_val)/total_n:.3f}, test: {len(df_test)/total_n:.3f}"
    )

    # ---- Datasets & Loaders ----
    num_classes = len(species_list)
    train_tfms, eval_tfms = get_transforms(img_size=args.img_size)

    ds_train = PlantImageDataset(df_train, transform=train_tfms)
    ds_val   = PlantImageDataset(df_val,   transform=eval_tfms)
    ds_test  = PlantImageDataset(df_test,  transform=eval_tfms)

    if args.use_weighted_sampler:
        class_counts = df_train["label_id"].value_counts().sort_index().values
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = df_train["label_id"].map(lambda x: class_weights[x]).values
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backend = build_mobilenetv3_large(num_classes=num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    print(f"Start training on {device} | backend={backend}")
    best_val_acc = 0.0
    best_ckpt = out_dir / "best_model.pt"
    patience = 7
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")
        scheduler.step()
        print(f"Train | loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val   | loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "label_map": label_map,
                "backend": backend,
            }
            torch.save(state, best_ckpt)
            print(f"✅ Saved best checkpoint to {best_ckpt} (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (val_acc={ckpt.get('val_acc', -1):.4f})")

    # ---- Final Test ----
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\nTest  | loss={test_loss:.4f}, acc={test_acc:.4f}")

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=[s for s in species_list],
        zero_division=0,
        output_dict=True,
    )
    rep_df = pd.DataFrame(report).transpose()
    rep_df.to_csv(out_dir / "test_classification_report.csv")

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "best_val_acc": float(best_val_acc),
            "splits": {
                "train": len(df_train),
                "val": len(df_val),
                "test": len(df_test),
            }
        }, f, indent=2)

    print(f"Saved test report to {out_dir / 'test_classification_report.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
