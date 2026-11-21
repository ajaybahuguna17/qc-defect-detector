"""
Shared core of the Phase 1 demo (app.py, predict.py): one image -> detector + classifier TFLite ->
fusion + severity -> one result. TFLite only (no .pt, no ultralytics), CPU, offline.

All model logic stays in the reference code the phone app must match: scripts/tflite_utils.py
(pre-processing, decoder, NMS), fusion.py (FUSION_CONFIG) and severity.py. Nothing is re-implemented
here, and missing or wrong model files raise: no other model is ever used instead.

Classifier: the Phase 1.2 7-class v2 (6 NEU classes + not_steel) from models/exported/phase1_2/classifier/
when its files are there, else the Phase 1.1 6-class v1. The detector is always Phase 1.1. With v2 a
non-steel photo gets decision RETAKE (fusion.py not_steel gate).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PHASE1_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE1_DIR), str(PHASE1_DIR / "scripts")]

try:
    import cv2  # noqa: F401  (tflite_utils.letterbox uses it; without it the resize silently changes to PIL)
except ImportError as e:
    raise ImportError("opencv is required: pip install -r phase1-vision/requirements-demo.txt") from e

from fusion import CLS7_CLASSES, FUSION_CONFIG, NEU_CLASSES, classifier_classes, classifier_top1, fuse  # noqa: E402
from tflite_utils import (  # noqa: E402
    assert_classifier_model, assert_defect_model, describe_cls_io, describe_io, load_interpreter, read_rgb,
    run_cls_tflite, run_tflite_on_image,
)

MODELS_DIR = PHASE1_DIR / "models" / "exported" / "phase1_1"
DRIVE_DIR = "MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/{detector,classifier}/"
DETECTOR_FILES = {"fp32": "defect_yolov8n_320_fp32.tflite", "int8": "defect_yolov8n_320_int8.tflite"}
CLASSIFIER_FILE = "defect_cls_yolov8n_224_fp32.tflite"
DRIVE_DIR_V2 = "MyDrive/qc-defect-detector/phase1/p12_cls7_notsteel_s42/models/classifier/"


def classifier_dir() -> Path:
    """phase1_2/classifier (v2) next to MODELS_DIR when its model + labels exist, else MODELS_DIR/classifier (v1)."""
    v2 = MODELS_DIR.parent / "phase1_2" / "classifier"
    if (v2 / CLASSIFIER_FILE).is_file() and (v2 / "labels.txt").is_file():
        return v2
    return MODELS_DIR / "classifier"

# Same decode as the Phase 1.1 evaluation (notebook BASE_CONF, NMS_IOU). It must stay below every
# threshold in FUSION_CONFIG: fuse() filters per class afterwards (crazing keeps boxes down to 0.10).
DECODE_CONF, DECODE_IOU = 0.05, 0.45

CLASS_COLOURS = {
    "crazing": "#E69F00",
    "inclusion": "#56B4E9",
    "patches": "#009E73",
    "pitted_surface": "#F0E442",
    "rolled-in_scale": "#CC79A7",
    "scratches": "#D55E00",
}
MIN_ANNOTATED_SIDE = 640  # NEU images are 200x200: too small for readable labels


class ModelFilesError(RuntimeError):
    """A required model file is missing or does not match the Phase 1.1 contract."""


def required_files(detectors: Sequence[str]) -> List[Path]:
    det, cls = MODELS_DIR / "detector", classifier_dir()
    return [det / DETECTOR_FILES[d] for d in detectors] + [det / "labels.txt", cls / CLASSIFIER_FILE, cls / "labels.txt"]


def check_model_files(detectors: Sequence[str]) -> None:
    missing = [p for p in required_files(detectors) if not p.is_file()]
    if missing:
        listing = "\n".join(f"  {p}" for p in missing)
        raise ModelFilesError(
            f"Phase 1.1 model file(s) missing:\n{listing}\n"
            f"Download them from Google Drive: {DRIVE_DIR}\n"
            f"and copy them to: {MODELS_DIR}\\{{detector,classifier}}\\ (hashes in phase1-vision/MODEL_IO.md).\n"
            "The demo does not use any other model instead."
        )
    for labels in (p for p in required_files(detectors) if p.name == "labels.txt"):
        names = labels.read_text(encoding="utf-8").split()
        allowed = [NEU_CLASSES] if labels.parent.name == "detector" else [NEU_CLASSES, CLS7_CLASSES]
        if names not in allowed:
            raise ModelFilesError(f"{labels} lists {names}, expected one of {allowed}")


class _TimedInterpreter:
    """Lets the reference run_* functions run unchanged while timing only the interpreter call."""

    def __init__(self, interp: Any) -> None:
        self._interp = interp
        self.last_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._interp, name)

    def invoke(self) -> None:
        start = time.perf_counter()
        self._interp.invoke()
        self.last_ms = (time.perf_counter() - start) * 1000.0


@dataclass
class DemoResult:
    image_name: str
    detector: str                     # "fp32" | "int8"
    alert: bool
    final_class: str
    decided_by: str                   # "classifier" | "detector" | "review" | "none"
    classifier_class: Optional[str]
    classifier_confidence: Optional[float]
    num_boxes: int                    # kept boxes, any class (the boxes that are drawn)
    num_final_boxes: int              # kept boxes of final_class (the boxes severity is computed on)
    largest_area_pct: Optional[float]  # largest final_class box; None when there is no such box
    severity_level: Optional[str]     # Low | Medium | High | Critical | Review | None (no alert)
    reason: str
    detector_ms: float
    classifier_ms: float
    boxes: List[Dict[str, Any]] = field(default_factory=list)
    decision: Optional[str] = None    # "RETAKE" (not_steel gate) or None
    classifier_version: str = "v1"    # "v1" (6 classes) | "v2" (7 classes, not_steel)


def line_decision(r: "DemoResult") -> str:
    """Operator decision, same mapping as the app's QualityDecision.kt (fusion rule only; the app's
    extra image checks and unsure-HOLD gate are not in this demo)."""
    if r.decision:
        return r.decision
    if not r.alert:
        return "PASS"
    return {"low": "ACCEPT", "medium": "REWORK", "high": "REJECT",
            "critical": "REJECT + STOP LINE"}.get((r.severity_level or "").lower(), "HOLD")


def to_result(fused: Any, image_name: str, detector: str, detector_ms: float, classifier_ms: float) -> DemoResult:
    """FusionResult -> DemoResult. "review" = fusion's Review level (alert, but no box to size)."""
    review = fused.severity_level == FUSION_CONFIG["review_level"]
    return DemoResult(
        image_name=image_name,
        detector=detector,
        alert=fused.alert,
        final_class=fused.final_class,
        decided_by="review" if review else fused.source,
        classifier_class=fused.classifier_class,
        classifier_confidence=fused.classifier_confidence,
        num_boxes=len(fused.kept_detections),
        num_final_boxes=len(fused.final_class_detections),
        largest_area_pct=fused.severity["max_area_ratio"] * 100.0 if fused.severity else None,
        severity_level=fused.severity_level,
        reason=fused.reason,
        detector_ms=detector_ms,
        classifier_ms=classifier_ms,
        boxes=fused.kept_detections,
        decision=fused.decision,
    )


class DefectPipeline:
    def __init__(self, detectors: Sequence[str] = ("fp32", "int8")) -> None:
        check_model_files(detectors)
        self._detectors = {}
        for name in detectors:
            interp = _TimedInterpreter(load_interpreter(MODELS_DIR / "detector" / DETECTOR_FILES[name]))
            io = describe_io(interp)
            assert_defect_model(io)
            self._detectors[name] = (interp, io)
        cls_dir = classifier_dir()
        self.classifier_classes = (cls_dir / "labels.txt").read_text(encoding="utf-8").split()
        self.classifier_version = "v2" if self.classifier_classes == CLS7_CLASSES else "v1"
        self._classifier = _TimedInterpreter(load_interpreter(cls_dir / CLASSIFIER_FILE))
        self._classifier_io = describe_cls_io(self._classifier)
        assert_classifier_model(self._classifier_io, len(self.classifier_classes))

    def analyze(self, image_path: Path, detector: str) -> DemoResult:
        image_path = Path(image_path)
        interp, io = self._detectors[detector]
        height, width = read_rgb(image_path).shape[:2]
        detections, _ = run_tflite_on_image(interp, io, image_path, NEU_CLASSES, DECODE_CONF, DECODE_IOU)
        probs = run_cls_tflite(self._classifier, self._classifier_io, image_path)
        top = classifier_top1(probs.tolist(), classifier_classes(FUSION_CONFIG, len(probs)))
        fused = fuse(detections, top, width, height)
        r = to_result(fused, image_path.name, detector, interp.last_ms, self._classifier.last_ms)
        r.classifier_version = self.classifier_version
        return r


def annotate(image_path: Path, boxes: Sequence[Dict[str, Any]]) -> Image.Image:
    """The image with one box + "class confidence" label per kept detection, colour by class."""
    rgb = read_rgb(Path(image_path))
    height, width = rgb.shape[:2]
    scale = max(1.0, MIN_ANNOTATED_SIDE / max(width, height))
    image = Image.fromarray(np.asarray(rgb)).resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    longest = max(image.size)
    font = ImageFont.load_default(size=max(14, longest // 40))
    line = max(2, longest // 320)
    for box in reversed(boxes):  # lowest confidence first, so the strongest box ends up on top
        x1, y1, x2, y2 = (v * scale for v in box["bbox_xyxy"])
        colour = CLASS_COLOURS[box["class_name"]]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line)
        label = f"{box['class_name']} {box['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left + 8, bottom - top + 6
        text_x = min(max(0.0, x1), max(0, image.width - text_w))
        text_y = y1 - text_h if y1 >= text_h else y1
        draw.rectangle((text_x, text_y, text_x + text_w, text_y + text_h), fill=colour)
        draw.text((text_x + 4 - left, text_y + 3 - top), label, fill="#000000", font=font)
    return image


def format_card(r: DemoResult) -> str:
    """The result card as plain ASCII text (CLI)."""
    if r.classifier_confidence is None:
        classifier = "not run"
    else:
        classifier = f"{r.classifier_class}, confidence {r.classifier_confidence:.3f}"
    area = "n/a (no box of the final type)" if r.largest_area_pct is None else f"{r.largest_area_pct:.1f} %"
    rows = [
        ("Decision", line_decision(r)),
        ("Alert", "YES" if r.alert else "no"),
        ("Final defect type", r.final_class),
        ("Decided by", r.decided_by),
        ("Classifier", classifier),
        ("Boxes", f"{r.num_boxes} kept ({r.num_final_boxes} of the final type)"),
        ("Largest area", area),
        ("Severity", r.severity_level or "none"),
        ("Reason", r.reason),
        ("Inference", f"detector {r.detector_ms:.1f} ms, classifier {r.classifier_ms:.1f} ms"),
    ]
    header = f"=== {r.image_name} | detector {r.detector.upper()} + classifier {r.classifier_version} FP32 ==="
    return "\n".join([header] + [f"{name + ':':<19}{value}" for name, value in rows])
