#!/usr/bin/env python3
"""
Phase 1.1: export the TRAINED NEU-DET image classifier (YOLOv8n-cls) to TFLite FP32 and/or INT8.

Strict, like export_tflite.py:
  * The weights must exist and be a 6-class classifier with the NEU class order. No fallback to
    the ImageNet yolov8n-cls.pt, no cached files. Any failure raises.
  * INT8 is calibrated on the TRAIN split only (a private calibration folder whose `val` entry
    points at the train images, so val/test stay unseen).
  * Every exported file is loaded with LiteRT and checked: output must be exactly [1, 6], and on
    a real train image the 6 outputs must be probabilities (each in 0..1, sum 1 +- 0.01), i.e. the
    softmax is inside the model. Input shape / layout / dtype and the pre-processing are printed
    and written to model_io.json.

Runs on Linux x86_64 or macOS (Ultralytics LiteRT export requirement), i.e. Colab.

Usage (from the repo root):
    python phase1-vision/scripts/export_cls_tflite.py \
        --weights /path/to/cls/best.pt --data phase1-vision/data/cls --imgsz 224 --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_utils import (  # noqa: E402
    NEU_CLASSES, assert_classifier_model, describe_cls_io, load_interpreter, run_cls_tflite,
)

PHASE1_DIR = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def make_calibration_root(cls_root: Path, work: Path) -> Path:
    """Folder with train/ and val/ that BOTH hold the train images (symlinked per class), so
    whichever split Ultralytics reads for INT8 calibration, it only ever sees train images."""
    train = cls_root / "train"
    if not train.is_dir():
        raise FileNotFoundError(f"{train} not found; run build_cls_dataset.py first")
    calib = work / "calib_cls"
    for sp in ("train", "val"):
        for cdir in sorted(p for p in train.iterdir() if p.is_dir()):
            dst = calib / sp / cdir.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(cdir.resolve(), dst, target_is_directory=True)
    return calib


def export_one(model_path: Path, imgsz: int, int8: bool, calib_root: Optional[Path], fraction: float) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kwargs = dict(format="litert", imgsz=imgsz, batch=1)
    if int8:
        if calib_root is None:
            raise ValueError("INT8 export needs calibration data (--data)")
        kwargs.update(quantize=8, data=str(calib_root), fraction=fraction)
    print(f"[export] YOLO.export({kwargs})")
    out = model.export(**kwargs)
    out = Path(out) if out else None
    if out is None or not out.is_file():
        raise RuntimeError(f"Ultralytics export returned {out!r}; no .tflite produced")
    return out


def describe_transforms(weights: Path, imgsz: int) -> Dict:
    """Pre-processing the .pt was trained/evaluated with (from the checkpoint when stored)."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    tf = getattr(m.model, "transforms", None)
    mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    if tf is not None:
        for t in getattr(tf, "transforms", []):
            if type(t).__name__ == "Normalize":
                mean, std = [float(v) for v in t.mean], [float(v) for v in t.std]
    return {
        "checkpoint_transforms": str(tf) if tf is not None else None,
        "colour_order": "RGB",
        "resize": f"shorter side -> {imgsz} (bilinear), centre crop {imgsz}x{imgsz}",
        "scale": "pixel / 255.0 -> 0..1",
        "mean": mean,
        "std": std,
        "reference_code": "phase1-vision/scripts/tflite_utils.py: cls_preprocess + cls_input_tensor",
    }


def check_tflite(path: Path, n_classes: int, probe_image: Path) -> Dict:
    interp = load_interpreter(path)
    io = describe_cls_io(interp)
    assert_classifier_model(io, n_classes)
    probs = run_cls_tflite(interp, io, probe_image)
    s = float(probs.sum())
    if probs.min() < -1e-4 or probs.max() > 1 + 1e-4 or abs(s - 1.0) > 0.01:
        raise AssertionError(f"{path.name}: output on {probe_image.name} is not a probability vector "
                             f"(min {probs.min():.4f}, max {probs.max():.4f}, sum {s:.4f}); softmax missing?")
    print(f"[check] {path.name}")
    print(f"        input : shape={io.input_shape} layout={io.input_layout} dtype={io.input_dtype} quant={io.input_quant}")
    print(f"        output: shape={io.output_shape} dtype={io.output_dtype} quant={io.output_quant} "
          f"(softmax inside: sum on probe = {s:.4f})")
    print(f"        size={path.stat().st_size / 1e6:.2f} MB")
    d = io.to_dict()
    d.update(file=path.name, size_bytes=path.stat().st_size, output_is_softmax=True,
             probe={"image": probe_image.name, "probs": [round(float(v), 6) for v in probs]})
    return d


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained classifier checkpoint (best.pt)")
    ap.add_argument("--data", required=True, help="ImageFolder root from build_cls_dataset.py (train/ is used)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--out-dir", default=str(PHASE1_DIR / "models" / "exported" / "classifier"))
    ap.add_argument("--precisions", default="fp32,int8", help="comma list of fp32,int8")
    ap.add_argument("--calib-fraction", type=float, default=0.3, help="fraction of train used for INT8 calibration")
    args = ap.parse_args(argv)

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained classifier not found: {weights}. Train first; there is no fallback model.")
    cls_root = Path(args.data).resolve()
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    if not precisions or any(p not in ("fp32", "int8") for p in precisions):
        raise ValueError(f"--precisions must be fp32 and/or int8, got {args.precisions}")

    from ultralytics import YOLO

    m = YOLO(str(weights))
    if m.task != "classify":
        raise AssertionError(f"{weights} is a '{m.task}' model, not a classifier")
    class_list = [m.names[i] for i in sorted(m.names)]
    if class_list != NEU_CLASSES:
        raise AssertionError(f"checkpoint classes {class_list} != expected NEU-DET classes {NEU_CLASSES}")

    probe = next((p for p in sorted((cls_root / "train" / NEU_CLASSES[0]).iterdir()) if p.suffix.lower() in IMG_EXT), None)
    if probe is None:
        raise FileNotFoundError(f"no train image found under {cls_root / 'train' / NEU_CLASSES[0]}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_export_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    src = work / "defect_cls_yolov8n.pt"
    shutil.copy2(weights, src)
    calib_root = make_calibration_root(cls_root, work) if "int8" in precisions else None

    report = {"task": "classify", "weights": str(weights), "imgsz": args.imgsz, "classes": class_list,
              "preprocessing": describe_transforms(weights, args.imgsz), "models": {}}
    for prec in precisions:
        produced = export_one(src, args.imgsz, prec == "int8", calib_root, args.calib_fraction)
        final = out_dir / f"defect_cls_yolov8n_{args.imgsz}_{prec}.tflite"
        shutil.copy2(produced, final)
        report["models"][prec] = check_tflite(final, len(class_list), probe)
        if prec == "int8":
            report["models"][prec]["calibration"] = {"split": "train", "fraction": args.calib_fraction}

    shutil.rmtree(work)
    (out_dir / "labels.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")
    (out_dir / "model_io.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[preprocessing]", json.dumps(report["preprocessing"], indent=2))
    print(f"[done] wrote {', '.join(p.name for p in sorted(out_dir.glob('*.tflite')))} + labels.txt + model_io.json in {out_dir}")
    return report


if __name__ == "__main__":
    main()
