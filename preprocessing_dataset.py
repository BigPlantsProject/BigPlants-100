import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_size(s: str) -> Tuple[int, int]:
    s = s.lower().replace(" ", "")
    if "x" in s:
        w, h = s.split("x")
        return int(w), int(h)
    n = int(s)
    return n, n

def parse_bg(s: str) -> Optional[Tuple[int, int, int]]:
    if s.lower() == "edge":
        return None
    s = s.strip()
    if s.startswith("#") and len(s) in (4, 7):
        if len(s) == 4:  # #RGB -> #RRGGBB
            s = "#" + "".join([c*2 for c in s[1:]])
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (r, g, b)
    raise ValueError("Invalid --bg. Use 'edge' or hex color like #FFFFFF.")

def edge_mean_color(img: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    bw = max(1, min(h, w) // 20)  # ~5% of the shorter edge
    top = arr[:bw, :, :]
    bottom = arr[h - bw :, :, :]
    left = arr[:, :bw, :]
    right = arr[:, w - bw :, :]
    pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    m = pixels.mean(axis=0)
    return tuple(int(round(x)) for x in m)

def ensure_rgb(img: Image.Image, bg_color=(0, 0, 0)) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        base = Image.new("RGB", img.size, bg_color)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        base.paste(img, mask=img.split()[-1])
        return base
    if img.mode == "CMYK":
        return img.convert("RGB")
    return img.convert("RGB")

def resize_pad(img: Image.Image, target: Tuple[int, int], bg: Optional[Tuple[int, int, int]]) -> Image.Image:
    tw, th = target
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_res = img.resize((nw, nh), resample=Image.LANCZOS)
    bg_color = edge_mean_color(img_res) if bg is None else bg
    canvas = Image.new("RGB", (tw, th), bg_color)
    left = (tw - nw) // 2
    top = (th - nh) // 2
    canvas.paste(img_res, (left, top))
    return canvas

def resize_center_crop(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    tw, th = target
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_res = img.resize((nw, nh), resample=Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img_res.crop((left, top, left + tw, top + th))

def resize_stretch(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    return img.resize(target, resample=Image.LANCZOS)

def out_path_for_renamed(
    dst_root: Path,
    rel_path: Path,
    fmt: str,
    new_stem: str
) -> Path:
    """
    Build destination path:
    - Keep the same subdirectory structure (rel_path.parent)
    - Filename becomes `new_stem` + extension
    - Extension: if fmt == 'keep' -> keep original suffix; else use '.jpg' or '.png'
    """
    sub = rel_path.parent
    if fmt == "keep":
        ext = rel_path.suffix.lower() or ".jpg"
    else:
        ext = ".jpg" if fmt == "jpg" else ".png"
    return dst_root / sub / f"{new_stem}{ext}"

def process_one(
    src_path: Path,
    in_root: Path,
    out_root: Path,
    size: Tuple[int, int],
    mode: str,
    bg: Optional[Tuple[int, int, int]],
    fmt: str,
    quality: int,
    new_stem: str,
) -> Tuple[str, str, Tuple[int, int], Tuple[int, int], str]:
    rel = src_path.relative_to(in_root)
    try:
        with Image.open(src_path) as im:
            base_bg = (0, 0, 0) if bg is None else bg
            im = ensure_rgb(im, base_bg)
            orig_size = im.size

            if mode == "pad":
                out_img = resize_pad(im, size, bg)
            elif mode == "crop":
                out_img = resize_center_crop(im, size)
            elif mode == "stretch":
                out_img = resize_stretch(im, size)
            else:
                raise ValueError("Invalid mode")

            dst_path = out_path_for_renamed(out_root, rel, fmt, new_stem)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            if dst_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs.update(dict(quality=quality, optimize=True, subsampling=1))
            out_img.save(dst_path, **save_kwargs)
            return (str(rel), str(dst_path.relative_to(out_root)), orig_size, out_img.size, "ok")
    except (UnidentifiedImageError, OSError) as e:
        return (str(rel), "", (0, 0), (0, 0), f"error: {type(e).__name__}: {e}")

def scan_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

def build_index_map(in_root: Path, files) -> Dict[Path, str]:
    """
    Assign per-class (top-level folder) incremental indices starting at 1.
    The new stem is `ClassName_index`.
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for p in files:
        rel = p.relative_to(in_root)
        # class folder is the first path component under in_root
        class_name = rel.parts[0] if len(rel.parts) > 0 else "unknown"
        by_class[class_name].append(p)

    index_map: Dict[Path, str] = {}
    for cls, lst in by_class.items():
        # stable order for reproducibility
        lst_sorted = sorted(lst, key=lambda x: str(x.relative_to(in_root)))
        for i, sp in enumerate(lst_sorted, start=1):
            index_map[sp] = f"{cls}_{i}"
    return index_map

def main():
    parser = argparse.ArgumentParser(description="Resize dataset to a uniform size with pad/crop/stretch, and rename outputs as FolderName_index.")
    parser.add_argument("--input", required=True, help="Thư mục gốc chứa 100 folders (mỗi lớp 1 folder).")
    parser.add_argument("--output", required=True, help="Thư mục xuất ảnh đã chuẩn hoá.")
    parser.add_argument("--size", default="224", help="Ví dụ: 224 hoặc 384x384 (mặc định: 224).")
    parser.add_argument("--mode", choices=["pad", "crop", "stretch"], default="pad",
                        help="pad (khuyến nghị), crop (center-crop), hoặc stretch (không khuyến nghị).")
    parser.add_argument("--bg", default="edge", help="Nền cho pad: 'edge' hoặc mã hex như #FFFFFF.")
    parser.add_argument("--format", choices=["keep", "jpg", "png"], default="jpg",
                        help="Định dạng lưu: giữ nguyên, ép JPG, hoặc ép PNG (mặc định: jpg).")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (95 mặc định).")
    parser.add_argument("--workers", type=int, default=8, help="Số luồng xử lý song song.")
    args = parser.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    size = parse_size(args.size)
    bg = parse_bg(args.bg)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"
    errors_path = out_root / "errors.log"

    files = list(scan_images(in_root))
    if not files:
        print("Không tìm thấy ảnh hợp lệ trong --input.", file=sys.stderr)
        sys.exit(1)

    # Build deterministic per-class index map BEFORE threading
    index_map = build_index_map(in_root, files)

    print(f"Found {len(files)} images. Processing to {size[0]}x{size[1]} ({args.mode}), format={args.format} ...")
    print("Renaming outputs as: <FolderName>_<index> (index starts at 1 per top-level folder)")

    ok_count = 0
    err_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(manifest_path, "w", newline="", encoding="utf-8") as mf, \
         open(errors_path, "w", encoding="utf-8") as ef:
        writer = csv.writer(mf)
        writer.writerow(["rel_path", "out_rel_path", "class", "new_stem", "orig_w", "orig_h", "new_w", "new_h", "mode", "note"])
        futures = [
            ex.submit(
                process_one,
                p,
                in_root,
                out_root,
                size,
                args.mode,
                bg,
                args.format,
                args.quality,
                index_map[p],
            )
            for p in files
        ]
        for fut in as_completed(futures):
            rel, out_rel, orig, new, note = fut.result()
            # derive class and new_stem for manifest
            rel_path = Path(rel)
            cls = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
            new_stem = index_map[in_root / rel_path]
            if note == "ok":
                ok_count += 1
            else:
                err_count += 1
                ef.write(f"{rel}\t{note}\n")
            writer.writerow([rel, out_rel, cls, new_stem, orig[0], orig[1], new[0], new[1], args.mode, note])

    print(f"Done. OK: {ok_count}, Errors: {err_count}")
    print(f"Manifest: {manifest_path}")
    if err_count:
        print(f"Errors: {errors_path}")

if __name__ == "__main__":
    main()
