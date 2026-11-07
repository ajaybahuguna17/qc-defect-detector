#!/usr/bin/env python3
"""
Phase 1.1: build the image-classifier dataset from the SAME split as the detector.

Reads the detector dataset written by prepare_data.py (data/dataset/images/{train,val,test})
and copies every image into an Ultralytics/torchvision ImageFolder layout:

    data/cls/train/<class>/*.jpg
    data/cls/val/<class>/*.jpg
    data/cls/test/<class>/*.jpg

The class of an image is its NEU-DET category, taken from the file name prefix
(crazing_12.jpg -> crazing, rolled-in_scale_7.jpg -> rolled-in_scale). The YOLO label
files are not used: a few NEU images also carry a box of a second class, but the image
category is the filename one.

Strict: the output dir is cleared first; the script raises if a file name has no known
class prefix, if a split is missing or empty, if any image is in two splits, or if the
written tree does not match the source split one to one.

Pure standard library.

Usage (from the repo root):
    python phase1-vision/scripts/build_cls_dataset.py            # data/dataset -> data/cls
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PHASE1_DIR = Path(__file__).resolve().parents[1]

NEU_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def class_from_filename(name: str, classes: Sequence[str] = NEU_CLASSES) -> str:
    """NEU category from a file name: the longest class name followed by '_' that prefixes it."""
    low = Path(name).name.lower()
    for c in sorted(classes, key=len, reverse=True):
        if low.startswith(c.lower() + "_"):
            return c
    raise ValueError(f"cannot get the class from the file name {name!r} (expected '<class>_<n>.jpg')")


def list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        raise FileNotFoundError(f"split folder not found: {d} (run prepare_data.py first)")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def build(dataset_dir: Path, out_dir: Path, classes: Sequence[str] = NEU_CLASSES) -> Dict:
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir).resolve()
    # out_dir is deleted before writing, so it must not be, contain, or sit inside the detector dataset.
    if out_dir == dataset_dir or out_dir in dataset_dir.parents or dataset_dir in out_dir.parents:
        raise ValueError(f"--out-dir {out_dir} overlaps the detector dataset {dataset_dir}; pick a separate folder")

    src = {sp: list_images(dataset_dir / "images" / sp) for sp in SPLITS}
    for sp, imgs in src.items():
        if not imgs:
            raise RuntimeError(f"split '{sp}' is empty in {dataset_dir / 'images' / sp}; "
                               "run prepare_data.py with the default --split 70/15/15")

    stems = {sp: {p.stem.lower() for p in imgs} for sp, imgs in src.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = sorted(stems[a] & stems[b])
            if overlap:
                raise RuntimeError(f"{len(overlap)} images in both {a} and {b}, e.g. {overlap[:10]}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: Dict[str, Dict[str, int]] = {}
    for sp in SPLITS:
        counts[sp] = {c: 0 for c in classes}
        for c in classes:
            (out_dir / sp / c).mkdir(parents=True, exist_ok=True)
        for img in src[sp]:
            c = class_from_filename(img.name, classes)
            shutil.copy2(img, out_dir / sp / c / img.name)
            counts[sp][c] += 1

    # Verify the written tree against the source split, one to one.
    for sp in SPLITS:
        written = {p.name for c in classes for p in list_images(out_dir / sp / c)}
        expected = {p.name for p in src[sp]}
        if written != expected:
            raise RuntimeError(f"{sp}: written files differ from the source split "
                               f"(missing {sorted(expected - written)[:5]}, extra {sorted(written - expected)[:5]})")
        for c in classes:
            for p in list_images(out_dir / sp / c):
                if class_from_filename(p.name, classes) != c:
                    raise RuntimeError(f"{p} is in the wrong class folder")

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "classes": list(classes),
        "counts": counts,
        "totals": {sp: sum(v.values()) for sp, v in counts.items()},
    }
    (out_dir / "cls_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'class':18s} " + " ".join(f"{sp:>6s}" for sp in SPLITS))
    for c in classes:
        print(f"{c:18s} " + " ".join(f"{counts[sp][c]:6d}" for sp in SPLITS))
    print(f"{'TOTAL':18s} " + " ".join(f"{summary['totals'][sp]:6d}" for sp in SPLITS))
    print(f"[CHECK] pairwise overlap = 0; written tree matches the detector split -> {out_dir}")
    return summary


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=PHASE1_DIR / "data" / "dataset",
                    help="detector dataset from prepare_data.py (needs images/train, val, test)")
    ap.add_argument("--out-dir", type=Path, default=PHASE1_DIR / "data" / "cls",
                    help="ImageFolder output root (cleared first)")
    args = ap.parse_args(argv)
    return build(args.dataset_dir, args.out_dir)


if __name__ == "__main__":
    main()
