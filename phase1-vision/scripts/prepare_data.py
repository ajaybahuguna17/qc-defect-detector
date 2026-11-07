#!/usr/bin/env python3
"""
Industrial Defect Detection - Benchmark Dataset Preparation Script (Phase 1)
Part of the Real-Time Industrial Defect Detection & Quality-Control Assistant project.

Automates the entire data acquisition, annotation conversion, and directory setup:
1. Automatically downloads the benchmark NEU Surface Defect Database (NEU-DET)
   (falls back to kagglehub; never to synthetic data unless --synthetic is given).
2. Parses PASCAL VOC XML annotations and converts them to normalized YOLO format.
3. Performs a stratified, seeded train/val split (default 80/20, seed 42).
   The output images/ and labels/ train+val dirs are CLEARED first, and the script
   fails if any image stem ends up in both train and val.
4. Writes data.yaml WITHOUT an absolute path (Ultralytics resolves it from the
   yaml's own folder), so the file is portable between Windows, Colab and Linux.
5. Runs dataset integrity checks and prints final counts / class distribution.

The synthetic generator is only used when --synthetic is passed explicitly. If the
real NEU-DET cannot be found or downloaded the script raises.

Runs 100% programmatically with zero mandatory third-party dependencies (pure standard library).
"""

import os
import sys
import glob
import math
import time
import shutil
import random
import struct
import argparse
import urllib.request
import urllib.error
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Standard benchmark defect classes for NEU Surface Defect Database (NEU-DET)
DEFAULT_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Canonical mapping for class aliases and naming variations found in public mirrors
CLASS_ALIASES: Dict[str, str] = {
    "crazing": "crazing",
    "craze": "crazing",
    "cr": "crazing",
    "inclusion": "inclusion",
    "inclusions": "inclusion",
    "in": "inclusion",
    "patches": "patches",
    "patch": "patches",
    "pa": "patches",
    "pitted_surface": "pitted_surface",
    "pitted-surface": "pitted_surface",
    "pitted surface": "pitted_surface",
    "pitting": "pitted_surface",
    "ps": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "rolled_in_scale": "rolled-in_scale",
    "rolled-in scale": "rolled-in_scale",
    "rolledin_scale": "rolled-in_scale",
    "scale": "rolled-in_scale",
    "rs": "rolled-in_scale",
    "scratches": "scratches",
    "scratch": "scratches",
    "sc": "scratches",
}

PHASE1_DIR = Path(__file__).resolve().parents[1]

# Primary official direct download URL for NEU-DET (Google Drive usercontent direct link)
PRIMARY_NEU_DET_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k&export=download&confirm=t"
)


# ==============================================================================
# Image Metadata & Dimension Utilities (Pure Standard Library)
# ==============================================================================

def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Extract (width, height) of an image using pure standard library binary parsing.
    Falls back to PIL.Image if installed, or default (200, 200) for standard NEU-DET.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        pass

    # Pure Python header parser for JPEG, PNG, and BMP
    try:
        with open(image_path, "rb") as f:
            header = f.read(32)

            # PNG detection
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                w, h = struct.unpack(">II", header[16:24])
                return int(w), int(h)

            # BMP detection
            if header.startswith(b"BM"):
                f.seek(18)
                b_info = f.read(8)
                w, h = struct.unpack("<ii", b_info)
                return abs(int(w)), abs(int(h))

            # JPEG detection
            if header.startswith(b"\xff\xd8"):
                f.seek(2)
                byte = f.read(1)
                while byte:
                    while byte != b"\xff":
                        byte = f.read(1)
                        if not byte:
                            break
                    while byte == b"\xff":
                        byte = f.read(1)
                    if not byte:
                        break
                    marker = byte[0]
                    # Markers SOF0, SOF1, SOF2 (baseline, extended, progressive DCT)
                    if marker in (0xC0, 0xC1, 0xC2):
                        f.read(3)  # skip length & precision
                        h, w = struct.unpack(">HH", f.read(4))
                        return int(w), int(h)
                    else:
                        length_bytes = f.read(2)
                        if len(length_bytes) < 2:
                            break
                        seg_len = struct.unpack(">H", length_bytes)[0]
                        f.seek(seg_len - 2, os.SEEK_CUR)
                    byte = f.read(1)
    except Exception:
        pass

    # Standard default dimensions for NEU Surface Defect images (200x200 grayscale)
    return 200, 200


# ==============================================================================
# Download & Extraction Subsystem
# ==============================================================================

def download_file_with_progress(url: str, output_path: Path, timeout: int = 120) -> bool:
    """Downloads a file from url to output_path with a live console progress bar."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    print(f"[INFO] Connecting to benchmark repository...")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(output_path, "wb") as out_file:
            total_size_hdr = response.headers.get("Content-Length")
            total_size = int(total_size_hdr.strip()) if total_size_hdr else 0
            block_size = 1024 * 64
            downloaded = 0
            start_time = time.time()

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                elapsed = max(time.time() - start_time, 1e-4)
                speed = (downloaded / (1024 * 1024)) / elapsed

                if total_size > 0:
                    percent = downloaded / total_size * 100
                    done_mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\r[DOWNLOADING] {done_mb:.1f}/{tot_mb:.1f} MB [{percent:5.1f}%] - {speed:.2f} MB/s"
                    )
                else:
                    done_mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r[DOWNLOADING] {done_mb:.1f} MB downloaded - {speed:.2f} MB/s")
                sys.stdout.flush()

        print("\n[SUCCESS] Download complete!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def create_synthetic_defect_benchmark(
    raw_images_dir: Path,
    raw_xmls_dir: Path,
    num_samples_per_class: int = 15,
    img_size: int = 200,
) -> bool:
    """
    Generates a realistic synthetic industrial defect dataset for offline/air-gapped testing.
    Produces valid 200x200 BMP images simulating steel surface grain with bounding-box annotations
    across all 6 classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
    """
    print("\n[NOTICE] Generating synthetic industrial surface defect benchmark...")
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_xmls_dir.mkdir(parents=True, exist_ok=True)

    def write_bmp_image(file_path: Path, width: int, height: int, draw_defect_type: str):
        """Creates an uncompressed 24-bit BMP image with procedural steel grain & defect."""
        row_size = (width * 3 + 3) & ~3
        image_size = row_size * height
        file_size = 54 + image_size

        # BMP 54-byte header
        header = bytearray(54)
        header[0:2] = b"BM"
        struct.pack_into("<I", header, 2, file_size)
        struct.pack_into("<I", header, 10, 54)
        struct.pack_into("<I", header, 14, 40)
        struct.pack_into("<i", header, 18, width)
        struct.pack_into("<i", header, 22, height)
        struct.pack_into("<H", header, 26, 1)
        struct.pack_into("<H", header, 28, 24)
        struct.pack_into("<I", header, 34, image_size)

        pixels = bytearray(image_size)
        boxes = []

        # Generate base metallic steel texture with subtle noise
        for y in range(height):
            for x in range(width):
                base_lum = int(140 + 20 * math.sin(x * 0.05) + random.randint(-15, 15))
                base_lum = max(50, min(230, base_lum))
                idx = y * row_size + x * 3
                pixels[idx] = base_lum
                pixels[idx + 1] = base_lum
                pixels[idx + 2] = base_lum

        # Synthesize defect features and calculate bounding box
        if draw_defect_type == "crazing":
            # Network of fine crack lines
            bx1, by1 = random.randint(20, 60), random.randint(20, 60)
            bw, bh = random.randint(70, 110), random.randint(70, 110)
            bx2, by2 = bx1 + bw, by1 + bh
            for _ in range(35):
                cx, cy = random.randint(bx1, bx2), random.randint(by1, by2)
                for step in range(random.randint(5, 15)):
                    nx = max(0, min(width - 1, cx + random.randint(-2, 2)))
                    ny = max(0, min(height - 1, cy + random.randint(-2, 2)))
                    idx = ny * row_size + nx * 3
                    pixels[idx] = max(10, pixels[idx] - 80)
                    pixels[idx + 1] = pixels[idx]
                    pixels[idx + 2] = pixels[idx]
                    cx, cy = nx, ny
            boxes.append((bx1, by1, bx2, by2))

        elif draw_defect_type == "inclusion":
            # Dark impurity spots
            for _ in range(random.randint(1, 3)):
                cx, cy = random.randint(30, 170), random.randint(30, 170)
                r = random.randint(5, 14)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            nx, ny = max(0, min(width - 1, cx + dx)), max(0, min(height - 1, cy + dy))
                            idx = ny * row_size + nx * 3
                            pixels[idx] = 25
                            pixels[idx + 1] = 25
                            pixels[idx + 2] = 25
                boxes.append((max(0, cx - r - 2), max(0, cy - r - 2), min(width, cx + r + 2), min(height, cy + r + 2)))

        elif draw_defect_type == "patches":
            # Irregular lighter/darker oxidation patch
            bx1, by1 = random.randint(20, 70), random.randint(20, 70)
            bw, bh = random.randint(60, 110), random.randint(60, 110)
            bx2, by2 = min(width - 5, bx1 + bw), min(height - 5, by1 + bh)
            for py in range(by1, by2):
                for px in range(bx1, bx2):
                    idx = py * row_size + px * 3
                    pixels[idx] = min(250, pixels[idx] + 65)
                    pixels[idx + 1] = pixels[idx]
                    pixels[idx + 2] = pixels[idx]
            boxes.append((bx1, by1, bx2, by2))

        elif draw_defect_type == "pitted_surface":
            # Cluster of small crater dots
            bx1, by1 = random.randint(20, 80), random.randint(20, 80)
            bw, bh = random.randint(50, 100), random.randint(50, 100)
            bx2, by2 = bx1 + bw, by1 + bh
            for _ in range(40):
                px, py = random.randint(bx1, bx2), random.randint(by1, by2)
                r = random.randint(1, 3)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = max(0, min(width - 1, px + dx)), max(0, min(height - 1, py + dy))
                        idx = ny * row_size + nx * 3
                        pixels[idx] = 15
                        pixels[idx + 1] = 15
                        pixels[idx + 2] = 15
            boxes.append((bx1, by1, bx2, by2))

        elif draw_defect_type == "rolled-in_scale":
            # Long dark streaks aligned with rolling direction
            by1 = random.randint(30, 140)
            bh = random.randint(15, 35)
            by2 = by1 + bh
            bx1 = random.randint(10, 30)
            bx2 = random.randint(150, 190)
            for py in range(by1, by2):
                for px in range(bx1, bx2):
                    idx = py * row_size + px * 3
                    pixels[idx] = max(10, pixels[idx] - 70)
                    pixels[idx + 1] = pixels[idx]
                    pixels[idx + 2] = pixels[idx]
            boxes.append((bx1, by1, bx2, by2))

        elif draw_defect_type == "scratches":
            # Thin high-contrast diagonal scratch line
            bx1, by1 = random.randint(10, 60), random.randint(10, 60)
            bx2, by2 = random.randint(140, 190), random.randint(140, 190)
            steps = 150
            for s in range(steps):
                t = s / float(steps)
                px = int(bx1 + t * (bx2 - bx1))
                py = int(by1 + t * (by2 - by1))
                for w_off in (-1, 0, 1):
                    nx = max(0, min(width - 1, px + w_off))
                    ny = max(0, min(height - 1, py + w_off))
                    idx = ny * row_size + nx * 3
                    pixels[idx] = 240
                    pixels[idx + 1] = 240
                    pixels[idx + 2] = 240
            boxes.append((max(0, min(bx1, bx2) - 2), max(0, min(by1, by2) - 2), min(width, max(bx1, bx2) + 2), min(height, max(by1, by2) + 2)))

        with open(file_path, "wb") as f:
            f.write(header)
            f.write(pixels)

        return boxes

    def write_pascal_xml(xml_path: Path, filename: str, width: int, height: int, defect_type: str, boxes: List[Tuple[int, int, int, int]]):
        """Writes valid PASCAL VOC XML annotation file."""
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "IMAGES"
        ET.SubElement(root, "filename").text = filename

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "1"

        for box in boxes:
            x1, y1, x2, y2 = box
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = defect_type
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x1)
            ET.SubElement(bndbox, "ymin").text = str(y1)
            ET.SubElement(bndbox, "xmax").text = str(x2)
            ET.SubElement(bndbox, "ymax").text = str(y2)

        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    count = 0
    for cls_name in DEFAULT_CLASSES:
        for i in range(1, num_samples_per_class + 1):
            stem = f"{cls_name}_{i}"
            img_file = raw_images_dir / f"{stem}.bmp"
            xml_file = raw_xmls_dir / f"{stem}.xml"
            boxes = write_bmp_image(img_file, img_size, img_size, cls_name)
            write_pascal_xml(xml_file, f"{stem}.bmp", img_size, img_size, cls_name, boxes)
            count += 1

    print(f"[SUCCESS] Synthesized {count} benchmark images & Pascal VOC XMLs across {len(DEFAULT_CLASSES)} classes.")
    return True


def ensure_raw_dataset(raw_dir: Path, force_download: bool = False, allow_synthetic: bool = False) -> Tuple[Path, Path]:
    """
    Ensures that raw defect images and VOC XML annotations exist.
    If not present or force_download is True, automatically downloads and extracts NEU-DET.
    Returns (raw_images_dir, raw_annotations_dir).
    """
    raw_images = raw_dir / "IMAGES"
    raw_annotations = raw_dir / "ANNOTATIONS"

    # Check if raw files already exist and are non-empty
    has_images = raw_images.exists() and any(raw_images.glob("*.*"))
    has_annotations = raw_annotations.exists() and any(raw_annotations.glob("*.xml"))

    if has_images and has_annotations and not force_download:
        print(f"[INFO] Raw dataset already exists in: {raw_dir}")
        return raw_images, raw_annotations

    print("==================================================================")
    print(" Benchmark Defect Dataset Acquisition")
    print(" Target: NEU Surface Defect Database (NEU-DET)")
    print("==================================================================")

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_target = raw_dir / "NEU-DET.zip"

    # Step 1: Attempt download from primary mirror
    download_success = False
    if not download_success:
        print("[INFO] Attempting download from primary benchmark mirror...")
        download_success = download_file_with_progress(PRIMARY_NEU_DET_URL, zip_target)

    # Step 2: Attempt extraction if zip file exists
    if download_success and zip_target.exists():
        print(f"[INFO] Extracting {zip_target.name}...")
        try:
            with zipfile.ZipFile(zip_target, "r") as zf:
                zf.extractall(raw_dir)
            print("[INFO] Extraction complete!")

            # Clean up zip archive to save disk space
            try:
                zip_target.unlink()
            except Exception:
                pass

            # Detect directory layout after extraction
            # Standard NEU-DET archive unpacks into NEU-DET/IMAGES and NEU-DET/ANNOTATIONS
            nested_candidate = raw_dir / "NEU-DET"
            if nested_candidate.exists():
                nested_images = nested_candidate / "IMAGES"
                nested_xmls = nested_candidate / "ANNOTATIONS"
                if nested_images.exists() and nested_xmls.exists():
                    raw_images.mkdir(parents=True, exist_ok=True)
                    raw_annotations.mkdir(parents=True, exist_ok=True)
                    for f in nested_images.iterdir():
                        if f.is_file() and not (raw_images / f.name).exists():
                            shutil.move(str(f), str(raw_images / f.name))
                    for f in nested_xmls.iterdir():
                        if f.is_file() and not (raw_annotations / f.name).exists():
                            shutil.move(str(f), str(raw_annotations / f.name))
                    shutil.rmtree(nested_candidate, ignore_errors=True)

            # Re-verify directories
            if raw_images.exists() and any(raw_images.glob("*.*")):
                print(f"[SUCCESS] Dataset unpacked into {raw_images} and {raw_annotations}")
                return raw_images, raw_annotations
        except Exception as e:
            print(f"[ERROR] Failed to extract archive: {e}")

    # Step 3: Kagglehub fallback (if user has kagglehub installed)
    try:
        import kagglehub
        print("[INFO] Attempting download via kagglehub...")
        k_path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
        if k_path and Path(k_path).exists():
            print(f"[INFO] Found Kaggle dataset at: {k_path}")
            k_root = Path(k_path)
            for img in k_root.rglob("*.jpg"):
                raw_images.mkdir(parents=True, exist_ok=True)
                if not (raw_images / img.name).exists():
                    shutil.copy2(img, raw_images / img.name)
            for xml in k_root.rglob("*.xml"):
                raw_annotations.mkdir(parents=True, exist_ok=True)
                if not (raw_annotations / xml.name).exists():
                    shutil.copy2(xml, raw_annotations / xml.name)
            if any(raw_images.glob("*.*")):
                return raw_images, raw_annotations
    except Exception as e:
        print(f"[WARN] kagglehub download not available/failed: {e}")

    # Step 4: Synthetic data ONLY when explicitly requested (--synthetic). Never a silent fallback.
    if allow_synthetic:
        create_synthetic_defect_benchmark(raw_images, raw_annotations, num_samples_per_class=20)
        return raw_images, raw_annotations

    raise RuntimeError(
        "Could not download or find industrial defect dataset. "
        "Please check your internet connection or supply --raw-dir / --synthetic."
    )


# ==============================================================================
# VOC Annotation Parsing & YOLO Bounding Box Conversion
# ==============================================================================

def voc_to_yolo_bbox(
    size: Tuple[int, int],
    box: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Converts PASCAL VOC bounding box (xmin, ymin, xmax, ymax)
    to normalized YOLO format (x_center, y_center, width, height).
    Strictly clips all coordinates to image boundaries.
    """
    width, height = size
    if width <= 0 or height <= 0:
        width, height = 200, 200

    xmin, ymin, xmax, ymax = box

    # Swap inverted coordinates if necessary
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    # Clip coordinates to image boundary
    xmin = max(0.0, min(float(xmin), float(width)))
    xmax = max(0.0, min(float(xmax), float(width)))
    ymin = max(0.0, min(float(ymin), float(height)))
    ymax = max(0.0, min(float(ymax), float(height)))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0.0 or bh <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    dw = 1.0 / width
    dh = 1.0 / height

    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    w = bw * dw
    h = bh * dh

    return round(x_center, 6), round(y_center, 6), round(w, 6), round(h, 6)


def normalize_class_name(raw_name: str, allowed_classes: List[str]) -> Optional[str]:
    """Normalizes class names, handling spaces, underscores, hyphens, and aliases."""
    clean = raw_name.strip().lower()
    if clean in CLASS_ALIASES:
        canonical = CLASS_ALIASES[clean]
        if canonical in allowed_classes:
            return canonical

    clean_var = clean.replace(" ", "_").replace("-", "_")
    for cls in allowed_classes:
        if cls.replace("-", "_") == clean_var or cls == clean:
            return cls

    return None


def parse_voc_xml(
    xml_path: Path,
    class_to_id: Dict[str, int],
    fallback_dimensions: Tuple[int, int] = (200, 200)
) -> List[Tuple[int, float, float, float, float]]:
    """Parses a PASCAL VOC XML annotation file and returns list of (class_id, xc, yc, w, h)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[WARN] Failed to parse XML {xml_path.name}: {e}")
        return []

    # Extract dimensions
    size_tag = root.find("size")
    if size_tag is not None:
        try:
            img_width = int(size_tag.find("width").text)
            img_height = int(size_tag.find("height").text)
        except Exception:
            img_width, img_height = fallback_dimensions
    else:
        img_width, img_height = fallback_dimensions

    if img_width <= 0 or img_height <= 0:
        img_width, img_height = fallback_dimensions

    records = []
    allowed_classes = list(class_to_id.keys())

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue

        matched_class = normalize_class_name(name_tag.text, allowed_classes)
        if matched_class is None:
            continue

        class_id = class_to_id[matched_class]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            continue

        xc, yc, w, h = voc_to_yolo_bbox((img_width, img_height), (xmin, ymin, xmax, ymax))
        if w > 0.0 and h > 0.0:
            records.append((class_id, xc, yc, w, h))

    return records


# ==============================================================================
# Train / Validation Splitter & YOLO Dataset Builder
# ==============================================================================

def clear_dir_keep_gitkeep(d: Path) -> None:
    """Delete every file/subdir in d (except .gitkeep), creating d if needed."""
    d.mkdir(parents=True, exist_ok=True)
    for item in d.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def assert_no_overlap(train_pairs: List[Tuple[Path, Path]], val_pairs: List[Tuple[Path, Path]]) -> None:
    """Fail loudly if any image (by stem, case-insensitive) is in both splits."""
    train_stems = {img.stem.lower() for img, _ in train_pairs}
    val_stems = {img.stem.lower() for img, _ in val_pairs}
    dup_train = len(train_stems) != len(train_pairs)
    dup_val = len(val_stems) != len(val_pairs)
    overlap = sorted(train_stems & val_stems)
    if overlap or dup_train or dup_val:
        raise RuntimeError(
            f"Train/val split is not clean: {len(overlap)} stems in both splits "
            f"(e.g. {overlap[:10]}), duplicate stems within train={dup_train}, within val={dup_val}."
        )


def verify_dataset_on_disk(dataset_dir: Path) -> Dict[str, int]:
    """Re-read the written dataset and fail if splits overlap or images/labels mismatch."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out: Dict[str, int] = {}
    stems = {}
    for split in ("train", "val"):
        imgs = [p for p in (dataset_dir / "images" / split).iterdir() if p.suffix.lower() in exts]
        lbls = list((dataset_dir / "labels" / split).glob("*.txt"))
        img_stems = {p.stem.lower() for p in imgs}
        lbl_stems = {p.stem.lower() for p in lbls}
        if img_stems != lbl_stems:
            raise RuntimeError(
                f"{split}: image/label mismatch, images without labels={sorted(img_stems - lbl_stems)[:10]}, "
                f"labels without images={sorted(lbl_stems - img_stems)[:10]}"
            )
        stems[split] = img_stems
        out[f"{split}_images"] = len(imgs)
        out[f"{split}_labels"] = len(lbls)
    overlap = sorted(stems["train"] & stems["val"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} images are in BOTH train and val on disk, e.g. {overlap[:10]}")
    return out


def write_data_yaml(yaml_path: Path, classes: List[str]):
    """Generates a portable Ultralytics data.yaml (no absolute paths).

    `path` is intentionally omitted: Ultralytics then uses the folder that contains
    this yaml as the dataset root. Always pass the yaml path itself to YOLO.
    """
    names_yaml = "\n".join([f"  {idx}: {cls_name}" for idx, cls_name in enumerate(classes)])
    content = f"""# Ultralytics YOLO Industrial Defect Detection Dataset
# Standard Benchmark: NEU Surface Defect Database (NEU-DET)
# Auto-generated by prepare_data.py
# No `path:` key on purpose -> dataset root = the folder containing this file.
train: images/train
val: images/val
test: # optional

# Defect class count
nc: {len(classes)}

# Defect class mapping
names:
{names_yaml}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def prepare_dataset(
    raw_dir: Path,
    dataset_dir: Path,
    classes: Optional[List[str]] = None,
    split_ratio: float = 0.8,
    seed: int = 42,
    force_download: bool = False,
    allow_synthetic: bool = False,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main programmatic orchestrator:
    1. Ensures raw dataset exists (downloading or creating synthetic benchmark if needed).
    2. Matches images with corresponding VOC XML annotations.
    3. Performs stratified train/val split.
    4. Writes normalized YOLO label files and images.
    5. Auto-generates data.yaml.
    6. Returns execution summary dictionary.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    rng = random.Random(seed)  # private, seeded RNG -> identical split on every machine
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

    # 1. Acquire raw data
    raw_images_dir, raw_xmls_dir = ensure_raw_dataset(
        raw_dir=raw_dir,
        force_download=force_download,
        allow_synthetic=allow_synthetic,
    )

    # 2. Output directory setup
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"

    # Clear previous outputs so stale files from older runs can never leak across splits.
    for d in [images_train, images_val, labels_train, labels_val]:
        clear_dir_keep_gitkeep(d)
    for cache in (dataset_dir / "labels").glob("*.cache"):  # Ultralytics label caches
        cache.unlink()

    # 3. Match raw images with XML annotations
    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in supported_exts:
        image_files.extend(list(raw_images_dir.glob(ext)))
        image_files.extend(list(raw_images_dir.glob(ext.upper())))

    # Deduplicate in case of case-insensitive filesystem overlaps
    image_files = sorted(list({f.resolve(): f for f in image_files}.values()), key=lambda p: p.name)

    if not image_files:
        raise FileNotFoundError(f"No image files found in {raw_images_dir}")

    # Build XML lookup map for fast stem matching
    xml_map = {}
    for xf in raw_xmls_dir.glob("*.xml"):
        xml_map[xf.stem.lower()] = xf
    for xf in raw_xmls_dir.glob("*.XML"):
        xml_map[xf.stem.lower()] = xf

    valid_pairs = []
    for img_path in image_files:
        xml_path = xml_map.get(img_path.stem.lower())
        if xml_path and xml_path.exists():
            valid_pairs.append((img_path, xml_path))

    if sample_limit and sample_limit > 0:
        valid_pairs = valid_pairs[:sample_limit]

    print(f"\n[INFO] Matched {len(valid_pairs)} image-annotation pairs.")

    # 4. Stratified Split by Defect Class (based on filename prefix or annotation)
    class_groups: Dict[str, List[Tuple[Path, Path]]] = {c: [] for c in classes}
    unclassified: List[Tuple[Path, Path]] = []

    for img, xml in valid_pairs:
        assigned = False
        stem_lower = img.stem.lower()
        for c in classes:
            c_prefix = c.replace("-", "_").lower()
            if stem_lower.startswith(c_prefix) or stem_lower.startswith(c.lower()):
                class_groups[c].append((img, xml))
                assigned = True
                break
        if not assigned:
            unclassified.append((img, xml))

    train_pairs: List[Tuple[Path, Path]] = []
    val_pairs: List[Tuple[Path, Path]] = []

    for c, pairs in class_groups.items():
        rng.shuffle(pairs)
        s_idx = int(len(pairs) * split_ratio)
        train_pairs.extend(pairs[:s_idx])
        val_pairs.extend(pairs[s_idx:])

    if unclassified:
        rng.shuffle(unclassified)
        s_idx = int(len(unclassified) * split_ratio)
        train_pairs.extend(unclassified[:s_idx])
        val_pairs.extend(unclassified[s_idx:])

    # Shuffle final train and val sets
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    assert_no_overlap(train_pairs, val_pairs)

    print(f"[INFO] Partitioned into: {len(train_pairs)} Train images | {len(val_pairs)} Validation images")

    # 5. Process Partitions
    def process_partition(
        pairs: List[Tuple[Path, Path]],
        img_dest: Path,
        lbl_dest: Path,
        part_name: str
    ) -> Tuple[int, Dict[str, int]]:
        box_counts = {c: 0 for c in classes}
        total_boxes = 0

        for img_path, xml_path in pairs:
            # Get actual image dimensions
            dims = get_image_dimensions(img_path)

            # Parse VOC XML
            yolo_records = parse_voc_xml(xml_path, class_to_id, fallback_dimensions=dims)

            # Copy image file
            target_img = img_dest / img_path.name
            shutil.copy2(img_path, target_img)

            # Write YOLO TXT file
            target_lbl = lbl_dest / f"{img_path.stem}.txt"
            with open(target_lbl, "w", encoding="utf-8") as f:
                for rec in yolo_records:
                    cls_id, xc, yc, w, h = rec
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    box_counts[classes[cls_id]] += 1
                    total_boxes += 1

        return total_boxes, box_counts

    train_total_boxes, train_box_counts = process_partition(train_pairs, images_train, labels_train, "Train")
    val_total_boxes, val_box_counts = process_partition(val_pairs, images_val, labels_val, "Validation")

    # 6. Auto-generate data.yaml
    yaml_path = dataset_dir / "data.yaml"
    write_data_yaml(yaml_path, classes)

    # 6b. Verify what is actually on disk (catches anything the in-memory check could miss)
    disk = verify_dataset_on_disk(dataset_dir)

    # 7. Print Terminal Diagnostic Report
    print("\n" + "=" * 62)
    print(" DATASET PREPARATION SUMMARY & CLASS DISTRIBUTION")
    print("=" * 62)
    print(f"{'Defect Class':<20} | {'Train Boxes':<12} | {'Val Boxes':<12} | {'Total':<8}")
    print("-" * 62)
    for cls_name in classes:
        t_c = train_box_counts[cls_name]
        v_c = val_box_counts[cls_name]
        print(f"{cls_name:<20} | {t_c:<12} | {v_c:<12} | {t_c + v_c:<8}")
    print("-" * 62)
    print(f"{'TOTAL DEFECT BOXES':<20} | {train_total_boxes:<12} | {val_total_boxes:<12} | {train_total_boxes + val_total_boxes:<8}")
    print(f"{'TOTAL IMAGES':<20} | {len(train_pairs):<12} | {len(val_pairs):<12} | {len(train_pairs) + len(val_pairs):<8}")
    print("=" * 62)
    print(f"[CHECK] On disk: train {disk['train_images']} images / {disk['train_labels']} labels, "
          f"val {disk['val_images']} images / {disk['val_labels']} labels, train/val overlap = 0")
    print(f"[CHECK] Split seed = {seed}")
    print(f"[SUCCESS] YOLO configuration generated at: {yaml_path}")
    print(f"[SUCCESS] Dataset root ready for training at: {dataset_dir.resolve()}")
    print("=" * 62 + "\n")

    return {
        "status": "success",
        "seed": seed,
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "classes": classes,
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "train_boxes": train_total_boxes,
        "val_boxes": val_total_boxes,
        "train_distribution": train_box_counts,
        "val_distribution": val_box_counts,
    }


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Industrial Defect Detection - Dataset Preparation & Converter (Phase 1)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PHASE1_DIR / "data" / "raw"),
        help="Directory to store or locate raw images and VOC annotations"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PHASE1_DIR / "data" / "dataset"),
        help="Target directory for processed YOLO dataset"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8 for 80/20 train/val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified train/val split"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force re-download of benchmark dataset even if raw directory exists"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a SYNTHETIC dataset (pipeline smoke test only; never use for training results)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (for quick CI/smoke test)"
    )

    args = parser.parse_args()

    # Resolve paths relative to working directory
    raw_dir = Path(args.raw_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    print("==================================================================")
    print(" Real-Time Industrial Defect Detection - Dataset Pipeline")
    print("==================================================================")
    print(f"Raw Directory:       {raw_dir}")
    print(f"Output Dataset Dir:  {dataset_dir}")
    print(f"Train/Val Ratio:     {args.split_ratio:.0%} / {(1.0 - args.split_ratio):.0%}")
    print(f"Random Seed:         {args.seed}")
    print(f"Force Download:      {args.download}")
    print(f"Synthetic Mode:      {args.synthetic}")
    print("==================================================================")

    if args.synthetic:
        raw_images = raw_dir / "IMAGES"
        raw_annotations = raw_dir / "ANNOTATIONS"
        create_synthetic_defect_benchmark(raw_images, raw_annotations, num_samples_per_class=20)

    prepare_dataset(
        raw_dir=raw_dir,
        dataset_dir=dataset_dir,
        classes=DEFAULT_CLASSES,
        split_ratio=args.split_ratio,
        seed=args.seed,
        force_download=args.download,
        allow_synthetic=args.synthetic,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
