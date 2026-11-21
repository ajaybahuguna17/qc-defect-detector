#!/usr/bin/env python3
"""
Phase 1.1 / 1.2: export the TRAINED NEU-DET image classifier (YOLOv8n-cls) to TFLite FP32 and/or INT8.

  * --num-classes 6 (default, Phase 1.1): the 6 NEU classes.
    --num-classes 7 (Phase 1.2, classifier v2): the 6 NEU classes in the same order, then "not_steel".
    The checkpoint class list must match exactly and the TFLite output must be [1, num_classes].

Strict, like export_tflite.py:
  * The weights must exist and be a 6-class classifier with the NEU class order. No fallback to
    the ImageNet yolov8n-cls.pt, no cached files.
  * INT8 is calibrated on the TRAIN split only (a private calibration folder whose `val` entry
    points at the train images, so val/test stay unseen).
  * Every .tflite is loaded with LiteRT and checked: output must be exactly [1, 6], and on a real
    train image the 6 outputs must be probabilities (each in 0..1, sum 1 +- 0.01), i.e. the softmax
    is inside the model. Input shape / layout / dtype and the pre-processing are printed and written
    to model_io.json.
  * Each precision is exported in a child process (export_child.py) with a timeout
    (--timeout-min, default 15). FP32 failure or timeout is always a hard error. INT8 failure or
    timeout is a hard error UNLESS --allow-skip-int8 is given: then model_io.json records int8 as
    {"status": "skipped", "reason", "timestamp"} and a large warning is printed. Never silent.
  * --precisions fp32 --allow-skip-int8 (no --reuse-existing): INT8 is NOT attempted at all (so it
    cannot hang) and model_io.json records it as skipped ("not requested").
  * --reuse-existing: no export; check the .tflite files already in --out-dir and (re)write
    model_io.json from them. With --allow-skip-int8 --int8-skipped-reason "...", INT8 is recorded as
    skipped with that reason (used to resume the 2026-11-07 Colab run, whose INT8 calibration hung).

Runs on Linux x86_64 or macOS (Ultralytics LiteRT export requirement), i.e. Colab.

Usage (from the repo root):
    python phase1-vision/scripts/export_cls_tflite.py \\
        --weights /path/to/cls/best.pt --data phase1-vision/data/cls --imgsz 224 --out-dir out/
    # rebuild model_io.json from an existing FP32 file, INT8 explicitly skipped:
    python phase1-vision/scripts/export_cls_tflite.py --weights best.pt --data phase1-vision/data/cls \\
        --out-dir out/ --reuse-existing --precisions fp32 --allow-skip-int8 --int8-skipped-reason "why"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_utils import (  # noqa: E402
    CLS7_CLASSES, NEU_CLASSES, assert_classifier_model, describe_cls_io, load_interpreter, run_cls_tflite,
)
from export_child import (  # noqa: E402
    DEFAULT_TIMEOUT_MIN, print_skip_warning, resolve_outcome, run_export, skipped_entry,
)

PHASE1_DIR = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_LISTS = {6: NEU_CLASSES, 7: CLS7_CLASSES}
INT8_NOT_REQUESTED = "INT8 not requested (--precisions fp32 --allow-skip-int8); FP32 only"


def tflite_name(imgsz: int, prec: str) -> str:
    return f"defect_cls_yolov8n_{imgsz}_{prec}.tflite"


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
    d.update(status="ok", file=path.name, size_bytes=path.stat().st_size,
             sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output_is_softmax=True,
             probe={"image": probe_image.name, "probs": [round(float(v), 6) for v in probs]})
    return d


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained classifier checkpoint (best.pt)")
    ap.add_argument("--data", required=True, help="ImageFolder root from build_cls_dataset.py (train/ is used)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--num-classes", type=int, choices=sorted(CLASS_LISTS), default=6,
                    help="6 = NEU classes (Phase 1.1); 7 = NEU classes + not_steel (Phase 1.2)")
    ap.add_argument("--out-dir", default=str(PHASE1_DIR / "models" / "exported" / "classifier"))
    ap.add_argument("--precisions", default="fp32,int8", help="comma list of fp32,int8 (fp32 required)")
    ap.add_argument("--calib-fraction", type=float, default=0.3, help="fraction of train used for INT8 calibration")
    ap.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN,
                    help="per-precision export timeout in minutes; the child process is killed")
    ap.add_argument("--allow-skip-int8", action="store_true",
                    help="an INT8 failure/timeout is recorded as skipped (with a loud warning) instead of raising")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="no export: check the .tflite files already in --out-dir and write model_io.json")
    ap.add_argument("--int8-skipped-reason", default=None,
                    help="with --reuse-existing --allow-skip-int8 and --precisions fp32: record INT8 as skipped")
    args = ap.parse_args(argv)

    args.precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    if not args.precisions or any(p not in ("fp32", "int8") for p in args.precisions):
        ap.error(f"--precisions must be fp32 and/or int8, got {args.precisions}")
    if "fp32" not in args.precisions:
        ap.error("fp32 is required in --precisions")
    if args.timeout_min <= 0:
        ap.error("--timeout-min must be > 0")
    if args.int8_skipped_reason is not None:
        if not (args.reuse_existing and args.allow_skip_int8):
            ap.error("--int8-skipped-reason needs --reuse-existing and --allow-skip-int8")
        if "int8" in args.precisions:
            ap.error("--int8-skipped-reason conflicts with int8 in --precisions")
        if not args.int8_skipped_reason.strip():
            ap.error("--int8-skipped-reason must not be empty")
    if args.reuse_existing and "int8" not in args.precisions and args.int8_skipped_reason is None:
        ap.error("--reuse-existing without int8 needs --allow-skip-int8 and --int8-skipped-reason")
    return args


def main(argv: Optional[List[str]] = None) -> Dict:
    args = parse_args(argv)
    precisions = args.precisions

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained classifier not found: {weights}. Train first; there is no fallback model.")
    cls_root = Path(args.data).resolve()

    from ultralytics import YOLO

    m = YOLO(str(weights))
    if m.task != "classify":
        raise AssertionError(f"{weights} is a '{m.task}' model, not a classifier")
    class_list = [m.names[i] for i in sorted(m.names)]
    expected = CLASS_LISTS[args.num_classes]
    if class_list != expected:
        raise AssertionError(f"checkpoint classes {class_list} != expected {args.num_classes}-class list {expected}")

    probe = next((p for p in sorted((cls_root / "train" / NEU_CLASSES[0]).iterdir()) if p.suffix.lower() in IMG_EXT), None)
    if probe is None:
        raise FileNotFoundError(f"no train image found under {cls_root / 'train' / NEU_CLASSES[0]}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"task": "classify", "weights": str(weights), "imgsz": args.imgsz, "classes": class_list,
              "num_classes": len(class_list),
              "preprocessing": describe_transforms(weights, args.imgsz), "models": {}}

    if args.reuse_existing:
        report["source"] = "reused existing .tflite files (--reuse-existing), not exported in this call"
        for prec in precisions:
            f = out_dir / tflite_name(args.imgsz, prec)
            if not f.is_file():
                raise FileNotFoundError(f"--reuse-existing: {f} not found")
            report["models"][prec] = check_tflite(f, len(class_list), probe)
            if prec == "int8":
                report["models"][prec]["calibration"] = {"split": "train", "fraction": args.calib_fraction}
        if "int8" not in precisions:
            print_skip_warning("classifier", "int8", args.int8_skipped_reason)
            report["models"]["int8"] = skipped_entry(args.int8_skipped_reason)
    else:
        work = out_dir / "_export_work"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        src = work / "defect_cls_yolov8n.pt"
        shutil.copy2(weights, src)
        calib_root = make_calibration_root(cls_root, work) if "int8" in precisions else None
        for prec in precisions:
            outcome = run_export(src, args.imgsz, prec, calib_root, args.calib_fraction, args.timeout_min,
                                 work, "classifier")
            skipped = resolve_outcome(outcome, allow_skip_int8=args.allow_skip_int8, label="classifier")
            if skipped is not None:
                report["models"][prec] = skipped
                continue
            final = out_dir / tflite_name(args.imgsz, prec)
            shutil.copy2(outcome.output, final)
            report["models"][prec] = check_tflite(final, len(class_list), probe)
            report["models"][prec]["export_seconds"] = outcome.seconds
            if prec == "int8":
                report["models"][prec]["calibration"] = {"split": "train", "fraction": args.calib_fraction}
        shutil.rmtree(work)
        if "int8" not in precisions and args.allow_skip_int8:
            print_skip_warning("classifier", "int8", INT8_NOT_REQUESTED)
            report["models"]["int8"] = skipped_entry(INT8_NOT_REQUESTED)

    (out_dir / "labels.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")
    (out_dir / "model_io.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[preprocessing]", json.dumps(report["preprocessing"], indent=2))
    status = ", ".join(f"{p}={v.get('status')}" for p, v in report["models"].items())
    print(f"[done] {status}; labels.txt + model_io.json in {out_dir}")
    return report


if __name__ == "__main__":
    main()
