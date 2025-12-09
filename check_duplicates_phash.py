# pip install pillow imagehash pandas tqdm
import argparse
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash
import pandas as pd
from tqdm import tqdm
import shutil
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def all_image_paths(root: Path):
    print(f"[LOG] Đang duyệt thư mục: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def compute_phash(path: Path, hash_size=8):
    try:
        return str(imagehash.phash(Image.open(path), hash_size=hash_size))
    except (OSError, UnidentifiedImageError):
        return None

def hamming_distance_hex(hex1: str, hex2: str):
    i1 = int(hex1, 16)
    i2 = int(hex2, 16)
    return (i1 ^ i2).bit_count()

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def main(args):
    root = Path(args.root)
    if not root.exists():
        print("Root path not found:", root)
        return

    # ============================
    # 1. FIND ALL IMAGE FILES
    # ============================
    print("\n======================")
    print("[LOG] BẮT ĐẦU QUÉT ẢNH")
    print("======================")
    files = list(all_image_paths(root))
    print(f"[LOG] Tổng số ảnh tìm thấy: {len(files)}")

    # ============================
    # 2. HASHING
    # ============================
    print("\n======================")
    print("[LOG] BẮT ĐẦU HASHING (phash)")
    print("======================")
    phash_map = {}
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(compute_phash, p, args.hash_size): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Hashing"):
            p = futures[fut]
            try:
                h = fut.result()
                if h is None:
                    errors.append(str(p))
                else:
                    phash_map[str(p)] = h
            except Exception as e:
                errors.append(f"{p}: {e}")

    print(f"[LOG] Hash thành công: {len(phash_map)} | Lỗi: {len(errors)}")

    # ============================
    # 3. EXACT DUPLICATES
    # ============================
    print("\n======================")
    print("[LOG] KIỂM TRA TRÙNG KHỚP PHASH EXACT")
    print("======================")

    hash_to_paths = {}
    for path, h in phash_map.items():
        hash_to_paths.setdefault(h, []).append(path)

    duplicates_records = []
    kept = set()

    for h, paths in hash_to_paths.items():
        if len(paths) > 1:
            print(f"[LOG] → exact duplicate group tìm thấy ({len(paths)} ảnh)")
            paths_sorted = sorted(paths)
            canonical = paths_sorted[0]
            kept.add(canonical)
            for dup in paths_sorted[1:]:
                print(f"   [EXACT] {dup}  →  {canonical}")
                duplicates_records.append((canonical, dup, 0, "exact"))
        else:
            kept.add(paths[0])

    # ============================
    # 4. NEAR DUPLICATES (phash distance)
    # ============================
    print("\n======================")
    print("[LOG] KIỂM TRA TRÙNG GẦN GIỐNG (NEAR-DUPLICATES)")
    print(f"[LOG] Threshold distance = {args.threshold}")
    print("======================")

    items = [(p, int(h, 16), h) for p, h in phash_map.items()]
    hash_bits = args.hash_size * args.hash_size
    prefix_bits = min(16, hash_bits)
    shift = hash_bits - prefix_bits

    buckets = {}
    for p, i, h in items:
        buckets.setdefault(i >> shift, []).append((p, i, h))

    seen_dup = set([r[1] for r in duplicates_records])

    for key, group in tqdm(buckets.items(), desc="Near-dup buckets"):
        n = len(group)
        if n <= 1:
            continue

        for i in range(n):
            p1, int1, hex1 = group[i]
            for j in range(i+1, n):
                p2, int2, hex2 = group[j]
                if p1 in seen_dup or p2 in seen_dup:
                    continue

                dist = (int1 ^ int2).bit_count()
                if dist <= args.threshold:
                    canonical, duplicate = (p1, p2) if p1 < p2 else (p2, p1)
                    print(f"[NEAR] dist={dist} → {duplicate}  →  {canonical}")
                    duplicates_records.append((canonical, duplicate, dist, "near"))
                    seen_dup.add(duplicate)

    # ============================
    # 5. EXPORT CSV
    # ============================
    print("\n======================")
    print("[LOG] XUẤT FILE CSV")
    print("======================")

    df = pd.DataFrame(duplicates_records, columns=["original", "duplicate", "distance", "method"])
    out_csv = Path(args.out)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)

    print(f"[LOG] Report CSV: {out_csv}")
    print(f"[LOG] Tổng duplicate tìm thấy: {len(df)}")

    # ============================
    # 6. MOVE DUPLICATES
    # ============================
    if args.move:
        print("\n======================")
        print("[LOG] BẮT ĐẦU MOVE FILE DUPLICATE")
        print("======================")

        dupdir = Path(args.dupdir)
        ensure_dir(dupdir)
        moved = 0

        for orig, dup, dist, method in df.itertuples(index=False):
            dup_p = Path(dup)

            try:
                rel = dup_p.relative_to(root)
                species = rel.parts[0] if len(rel.parts) >= 1 else "unknown"
            except Exception:
                species = "unknown"

            target_dir = dupdir / species
            ensure_dir(target_dir)
            target_path = target_dir / dup_p.name

            if args.dry_run:
                print(f"[dry-run] would move {dup_p}  →  {target_path}")
            else:
                if target_path.exists():
                    base = target_path.stem
                    ext = target_path.suffix
                    k = 1
                    while True:
                        alt = target_dir / f"{base}_{k}{ext}"
                        if not alt.exists():
                            target_path = alt
                            break
                        k += 1

                print(f"[MOVE] {dup_p}  →  {target_path}")
                shutil.move(str(dup_p), str(target_path))
                moved += 1

        print(f"[LOG] Tổng số file đã move: {moved} (dry-run={args.dry_run})")

    # ============================
    # 7. LOG ERRORS
    # ============================
    if errors:
        errf = out_csv.parent / "hash_errors.txt"
        with open(errf, "w", encoding="utf8") as f:
            for e in errors:
                f.write(e + "\n")
        print(f"[LOG] Ghi lỗi vào file: {errf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Root folder chứa ảnh")
    p.add_argument("--out", default="duplicates_report.csv", help="CSV report path")
    p.add_argument("--dupdir", default="duplicates", help="Folder để move duplicates (nếu bật --move)")
    p.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold")
    p.add_argument("--hash-size", type=int, default=8, help="phash hash_size")
    p.add_argument("--move", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    main(args)
