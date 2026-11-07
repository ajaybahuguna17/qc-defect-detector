#!/usr/bin/env python3
"""
Phase 1: export the TRAINED NEU-DET YOLOv8n checkpoint to TFLite (FP32 and/or INT8).

Strict by design:
  * The weights file must exist and must be a 6-class model. There is no fallback to
    COCO yolov8n.pt and no "copy a cached .tflite" fallback. Any failure raises.
  * Every exported file is loaded with LiteRT and checked: output must be [1, 10, N]
    (4 box rows + 6 class rows). The real input shape / layout / dtype are printed and
    written to model_io.json.

Runs on Linux x86_64 or macOS only (Ultralytics LiteRT export requirement), i.e. Colab.
Pinned toolchain: phase1-vision/requirements.txt.

Usage (from the repo root):
    python phase1-vision/scripts/export_tflite.py \
        --weights /path/to/best.pt --data /path/to/data.yaml --imgsz 320 --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_utils import NEU_CLASSES, assert_defect_model, describe_io, load_interpreter  # noqa: E402

PHASE1_DIR = Path(__file__).resolve().parents[1]


def write_calibration_yaml(data_yaml: Path, out_path: Path, split: str) -> Path:
    """Write a copy of data.yaml whose `val` entry points at `split`, so Ultralytics'
    INT8 calibration loader (which reads the `val` key) uses that split's images."""
    import yaml

    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path") or Path(data_yaml).resolve().parent)
    if not root.is_absolute():
        root = (Path(data_yaml).resolve().parent / root).resolve()
    data["path"] = str(root)
    data["val"] = data[split]
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def export_one(model_path: Path, imgsz: int, int8: bool, calib_yaml: Optional[Path], calib_fraction: float) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kwargs = dict(format="litert", imgsz=imgsz, batch=1)
    if int8:
        if calib_yaml is None:
            raise ValueError("INT8 export needs calibration data (--data)")
        kwargs.update(quantize=8, data=str(calib_yaml), fraction=calib_fraction)
    print(f"[export] YOLO.export({kwargs})")
    out = model.export(**kwargs)
    out = Path(out) if out else None
    if out is None or not out.is_file():
        raise RuntimeError(f"Ultralytics export returned {out!r}; no .tflite produced")
    return out


def check_tflite(path: Path, n_classes: int) -> Dict:
    interp = load_interpreter(path)
    io = describe_io(interp)
    assert_defect_model(io, n_classes)
    print(f"[check] {path.name}")
    print(f"        input : shape={io.input_shape} layout={io.input_layout} dtype={io.input_dtype} quant={io.input_quant}")
    print(f"        output: shape={io.output_shape} dtype={io.output_dtype} quant={io.output_quant}")
    print(f"        classes={io.num_classes} anchors={io.num_anchors} size={path.stat().st_size / 1e6:.2f} MB")
    d = io.to_dict()
    d["file"] = path.name
    d["size_bytes"] = path.stat().st_size
    return d


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained NEU-DET checkpoint (best.pt)")
    ap.add_argument("--data", required=True, help="data.yaml of the prepared dataset (INT8 calibration)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out-dir", default=str(PHASE1_DIR / "models" / "exported"))
    ap.add_argument("--precisions", default="fp32,int8", help="comma list of fp32,int8")
    ap.add_argument("--calib-split", default="train", choices=["train", "val"],
                    help="images used for INT8 calibration (default train, so val stays untouched for evaluation)")
    ap.add_argument("--calib-fraction", type=float, default=0.3,
                    help="fraction of the calibration split to use (0.3 of train ~= 430 images)")
    args = ap.parse_args(argv)

    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}. Train first; there is no fallback model.")
    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    if not precisions or any(p not in ("fp32", "int8") for p in precisions):
        raise ValueError(f"--precisions must be fp32 and/or int8, got {args.precisions}")

    from ultralytics import YOLO

    names = YOLO(str(weights)).names
    class_list = [names[i] for i in sorted(names)]
    if class_list != NEU_CLASSES:
        raise AssertionError(f"checkpoint classes {class_list} != expected NEU-DET classes {NEU_CLASSES}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export from a private copy so the two exports never overwrite each other or the source dir.
    work = out_dir / "_export_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    src = work / "defect_yolov8n.pt"
    shutil.copy2(weights, src)

    calib_yaml = None
    if "int8" in precisions:
        calib_yaml = write_calibration_yaml(data_yaml, work / "calib_data.yaml", args.calib_split)

    report = {"weights": str(weights), "imgsz": args.imgsz, "classes": class_list, "models": {}}
    for prec in precisions:
        produced = export_one(src, args.imgsz, prec == "int8", calib_yaml, args.calib_fraction)
        final = out_dir / f"defect_yolov8n_{args.imgsz}_{prec}.tflite"
        shutil.copy2(produced, final)
        report["models"][prec] = check_tflite(final, len(class_list))
        if prec == "int8":
            report["models"][prec]["calibration"] = {"split": args.calib_split, "fraction": args.calib_fraction}

    shutil.rmtree(work)
    (out_dir / "labels.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")
    (out_dir / "model_io.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {', '.join(p.name for p in sorted(out_dir.glob('*.tflite')))} + labels.txt + model_io.json in {out_dir}")
    return report


if __name__ == "__main__":
    main()
