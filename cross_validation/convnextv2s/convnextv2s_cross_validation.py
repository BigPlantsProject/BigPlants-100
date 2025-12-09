import os
import random
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import time
from contextlib import nullcontext

# -------------------------- Defaults (tuned) --------------------------
DATA_ROOT_DEFAULT = r"D:\Homework\NC\bigplants_dataset_100_resized"
OUT_DIR_DEFAULT = "./output_convnextv2_cv_resume_improved"
PARTS = {"hand","leaf","flower","fruit"}
MAX_PER_CLASS_DEFAULT = 100
RANDOM_SEED_DEFAULT = 42
TEST_SIZE_DEFAULT = 0.15
K_FOLDS_DEFAULT = 5
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# tuned defaults:
IMG_SIZE_DEFAULT = 224
EPOCHS_DEFAULT = 40
BATCH_SIZE_DEFAULT = 16
LR_DEFAULT = 2e-4
WEIGHT_DECAY_DEFAULT = 1e-2
NUM_WORKERS_DEFAULT = 3
MODEL_NAME_DEFAULT = "convnextv2_tiny.fcmae_ft_in22k_in1k"
USE_ONECYCLE_DEFAULT = True
WARMUP_EPOCHS_DEFAULT = 3
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def list_images_direct(d: Path):
    if not d.exists() or not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and is_image_file(p)]

def build_manifest(root_dir, out_csv, parts_priority=PARTS, max_per_class=100, seed=42):
    random.seed(seed)
    rows = []
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {root_dir}")
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for cls_dir in tqdm(class_dirs, desc="Building manifest"):
        class_name = cls_dir.name
        part_files = []
        available_files = []
        for item in cls_dir.iterdir():
            if item.is_dir():
                part_name = item.name.lower()
                if part_name in parts_priority:
                    for f in item.rglob("*"):
                        if f.is_file() and f.suffix in IMG_EXT:
                            part_files.append((str(f), part_name))
            elif item.is_file() and item.suffix in IMG_EXT:
                available_files.append(str(item))
        seen = set()
        unique_part_files = []
        for p, tag in part_files:
            if p not in seen:
                seen.add(p)
                unique_part_files.append((p, tag))
        part_paths = [p for p,_ in unique_part_files]
        selected = []
        if len(part_paths) >= max_per_class:
            selected = random.sample(part_paths, max_per_class)
        else:
            selected = list(part_paths)
            need = max_per_class - len(selected)
            if available_files:
                random.shuffle(available_files)
                selected.extend(available_files[:need])
        part_map = {p:tag for p,tag in unique_part_files}
        for p in selected:
            part_tag = part_map.get(p, "available")
            rows.append({"filepath": p, "class": class_name, "part": part_tag})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[manifest] saved {len(df)} entries -> {out_csv}")
    return df

def create_kfold_manifest(manifest_df, out_csv, n_splits=5, test_size=0.15, seed=42):
    trainval_df, test_df = train_test_split(manifest_df, test_size=test_size, stratify=manifest_df['class'], random_state=seed)
    test_df = test_df.copy()
    test_df['split'] = 'test'
    test_df['fold'] = -1
    trainval_df = trainval_df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trainval_df['fold'] = -1
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(trainval_df, trainval_df['class'])):
        trainval_df.loc[val_idx, 'fold'] = fold_id
    trainval_df['split'] = 'trainval'
    final_df = pd.concat([trainval_df, test_df]).reset_index(drop=True)
    final_df = skshuffle(final_df, random_state=seed).reset_index(drop=True)
    final_df.to_csv(out_csv, index=False)
    print(f"[split] Saved K-fold splits -> {out_csv}")
    return final_df

class PlantDataset(Dataset):
    def __init__(self, manifest_df, split="train", fold_id=None, transform=None, class_order=None):
        if split == 'train':
            self.df = manifest_df[(manifest_df['split'] == 'trainval') & (manifest_df['fold'] != fold_id)].reset_index(drop=True)
        elif split == 'val':
            self.df = manifest_df[(manifest_df['split'] == 'trainval') & (manifest_df['fold'] == fold_id)].reset_index(drop=True)
        elif split == 'trainval_all':
            self.df = manifest_df[manifest_df['split'] == 'trainval'].reset_index(drop=True)
        else:
            self.df = manifest_df[manifest_df['split'] == 'test'].reset_index(drop=True)
        self.transform = transform
        self.class_order = class_order if class_order is not None else sorted(manifest_df['class'].unique())
        self.class2idx = {c:i for i,c in enumerate(self.class_order)}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class2idx[row['class']]
        return img, label

def compute_class_weights_from_df(df, class_order, device):
    counts = df['class'].value_counts().to_dict()
    weights = []
    for c in class_order:
        cnt = counts.get(c, 0)
        weights.append(1.0 / max(1.0, cnt))
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum() * len(w)
    return torch.tensor(w, dtype=torch.float, device=device)

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def accuracy_top1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        return correct, total

# ---------------- Main training w/ improvements ----------------
def train_and_evaluate_cv(full_manifest_df, out_dir, device, k_folds=5,
                          model_name=MODEL_NAME_DEFAULT,
                          epochs=EPOCHS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
                          lr=LR_DEFAULT, weight_decay=WEIGHT_DECAY_DEFAULT,
                          num_workers=NUM_WORKERS_DEFAULT,
                          resume_path=None,
                          use_onecycle=USE_ONECYCLE_DEFAULT,
                          warmup_epochs=WARMUP_EPOCHS_DEFAULT,
                          use_class_weights=False,
                          use_weighted_sampler=False,
                          use_mixup=False,
                          mixup_alpha=0.2,
                          label_smoothing=0.0,
                          img_size=IMG_SIZE_DEFAULT):
    os.makedirs(out_dir, exist_ok=True)
    device_cuda = (device == "cuda")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class_order = sorted(full_manifest_df['class'].unique())
    num_classes = len(class_order)
    print(f"Total classes: {num_classes}")

    start_fold = 0
    all_fold_val_accs = []
    checkpoint = None
    if resume_path and os.path.exists(resume_path):
        print(f"--- Resuming training from checkpoint: {resume_path} ---")
        checkpoint = torch.load(resume_path, map_location=device)
        start_fold = checkpoint.get('fold_id', 0)
        all_fold_val_accs = checkpoint.get('all_fold_val_accs', [])
        if checkpoint.get('epoch', 0) >= epochs:
            start_fold = checkpoint.get('fold_id', 0) + 1
            start_epoch_for_current_fold = 1
        else:
            start_epoch_for_current_fold = checkpoint.get('epoch', 0) + 1
    else:
        start_epoch_for_current_fold = 1

    pin_memory = True if device_cuda else False

    for fold_id in range(start_fold, k_folds):
        print(f"\n{'='*20} FOLD {fold_id+1}/{k_folds} {'='*20}")

        train_ds = PlantDataset(full_manifest_df, split='train', fold_id=fold_id, transform=train_transform, class_order=class_order)
        val_ds = PlantDataset(full_manifest_df, split='val', fold_id=fold_id, transform=val_transform, class_order=class_order)
        print(f"Fold {fold_id+1} -> Train/Val sizes: {len(train_ds)} / {len(val_ds)}")

        if use_weighted_sampler:
            counts = train_ds.df['class'].value_counts().sort_index()
            mapping = {c: 1.0/counts[c] for c in counts.index}
            sample_w = train_ds.df['class'].map(mapping).values.astype(np.float32)
            sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_w).float(), num_samples=len(sample_w), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        if use_class_weights:
            weight_tensor = compute_class_weights_from_df(train_ds.df, class_order, device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        steps_per_epoch = max(1, len(train_loader))
        scheduler = None
        if use_onecycle:
            try:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    total_steps=epochs * steps_per_epoch,
                    pct_start=0.1,
                    anneal_strategy="cos",
                    div_factor=10.0,
                    final_div_factor=1e4
                )
                use_onecycle_local = True
                print("[LR] Using OneCycleLR")
            except Exception as e:
                print("[LR] OneCycleLR failed -> fallback to CosineAnnealingLR", e)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                use_onecycle_local = False
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            use_onecycle_local = False

        scaler = torch.cuda.amp.GradScaler() if device_cuda else None
        autocast_cm = torch.cuda.amp.autocast if device_cuda else nullcontext

        best_val_acc = 0.0
        if checkpoint and fold_id == checkpoint.get('fold_id', -1):
            print(f"Loading resume states for fold {fold_id+1}")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print("Warning restoring scheduler:", e)
            best_val_acc = checkpoint.get('best_val_acc_this_fold', 0.0)
            start_epoch_for_current_fold = checkpoint.get('epoch', 0) + 1
            checkpoint = None
        else:
            start_epoch_for_current_fold = 1

        for epoch in range(start_epoch_for_current_fold, epochs + 1):
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Fold {fold_id+1} Epoch {epoch}/{epochs} train", leave=False)
            for imgs, labels in pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                if use_mixup:
                    imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
                    with autocast_cm():
                        outputs = model(imgs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    with autocast_cm():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                c, t = accuracy_top1(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                running_correct += c
                running_total += t
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{running_loss/max(1,running_total):.4f}", acc=f"{running_correct/max(1,running_total):.4f}", lr=f"{current_lr:.2e}")

                if use_onecycle_local:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
            
            if not use_onecycle_local:
                try:
                    scheduler.step()
                except Exception:
                    pass

            # validation
            model.eval()
            val_correct, val_total = 0, 0
            val_loss_total = 0.0  # <--- THÊM VÀO
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast_cm():
                        outputs = model(imgs)
                        val_loss = criterion(outputs, labels) # <--- THÊM VÀO
                    c, t = accuracy_top1(outputs, labels)
                    val_loss_total += val_loss.item() * imgs.size(0) # <--- THÊM VÀO
                    val_correct += c; val_total += t
            
            val_acc = val_correct / max(1, val_total)
            val_loss_avg = val_loss_total / max(1, val_total) # <--- THÊM VÀO
            print(f"Fold {fold_id+1} Epoch {epoch} -> val_loss: {val_loss_avg:.4f}, val_acc: {val_acc:.4f} (best_acc {best_val_acc:.4f})") # <--- SỬA ĐỔI

            # save best model
            fold_dir = os.path.join(out_dir, f"fold_{fold_id+1}")
            os.makedirs(fold_dir, exist_ok=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(fold_dir, f"best_model_fold_{fold_id+1}.pth"))
                print(f" -> New best model saved for fold {fold_id+1}")

            # save resume checkpoint every epoch
            resume_checkpoint = {
                'fold_id': fold_id,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'all_fold_val_accs': all_fold_val_accs,
                'best_val_acc_this_fold': best_val_acc,
            }
            torch.save(resume_checkpoint, os.path.join(out_dir, "resume_checkpoint.pth"))

        # --- THÊM VÀO: Báo cáo classification report cho fold này dùng model tốt nhất ---
        fold_dir = os.path.join(out_dir, f"fold_{fold_id+1}")
        best_model_path = os.path.join(fold_dir, f"best_model_fold_{fold_id+1}.pth")
        if os.path.exists(best_model_path):
            print(f"Fold {fold_id+1} - Tải model tốt nhất để tạo validation report...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            all_labels = []; all_preds = []
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Fold {fold_id+1} Val Report", leave=False):
                    imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                    with autocast_cm():
                        outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1)
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
            
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            target_names = [str(c) for c in class_order]
            report = classification_report(y_true, y_pred, target_names=target_names, labels=list(range(len(target_names))), zero_division=0, output_dict=True)
            report_path = os.path.join(fold_dir, "validation_classification_report.csv")
            pd.DataFrame(report).transpose().to_csv(report_path)
            print(f"Fold {fold_id+1} validation report đã lưu -> {report_path}")
        else:
            print(f"Fold {fold_id+1} - Không tìm thấy best model, bỏ qua validation report.")
        # --- KẾT THÚC PHẦN THÊM VÀO ---

        print(f"Fold {fold_id+1} finished. Best val_acc: {best_val_acc:.4f}")
        all_fold_val_accs.append(best_val_acc)

    mean_acc = np.mean(all_fold_val_accs) if all_fold_val_accs else 0.0
    std_acc = np.std(all_fold_val_accs) if all_fold_val_accs else 0.0
    print(f"\nCV summary: val_accs per fold = {[f'{a:.4f}' for a in all_fold_val_accs]}")
    print(f"Mean Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # --- ĐÃ XÓA PHẦN FINAL TRAINING VÀ FINAL TEST EVALUATION ---

    return mean_acc

# --------------------- CLI ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--max_per_class", type=int, default=MAX_PER_CLASS_DEFAULT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--k_folds", type=int, default=K_FOLDS_DEFAULT)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT)
    parser.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--resume_path", type=str, default=OUT_DIR_DEFAULT + r"/resume_checkpoint.pth")
    parser.add_argument("--use_onecycle", action="store_true", help="Use OneCycleLR schedule")
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE_DEFAULT)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_csv = os.path.join(args.out_dir, "manifest_selected.csv")
    split_csv = os.path.join(args.out_dir, "manifest_kfold_splits.csv")

    if not os.path.exists(split_csv):
        print("Building initial manifest ...")
        manifest_df = build_manifest(args.data_root, manifest_csv, max_per_class=args.max_per_class, seed=args.seed)
        create_kfold_manifest(manifest_df, split_csv, n_splits=args.k_folds, test_size=args.test_size, seed=args.seed)
    else:
        print("Found existing splits file, loading it...")

    full_manifest_df = pd.read_csv(split_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_and_evaluate_cv(
        full_manifest_df=full_manifest_df,
        out_dir=args.out_dir,
        device=device,
        k_folds=args.k_folds,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        resume_path=args.resume_path,
        use_onecycle=args.use_onecycle,
        warmup_epochs=args.warmup_epochs,
        use_class_weights=args.use_class_weights,
        use_weighted_sampler=args.use_weighted_sampler,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()
  
