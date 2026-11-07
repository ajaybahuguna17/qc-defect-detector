"""
Deterministic fusion of the defect DETECTOR (YOLOv8n) and the image CLASSIFIER (YOLOv8n-cls)
for ONE image (Phase 1.1).

Inputs per image
----------------
* detector output after decode + NMS: list of ``{class_name, confidence, bbox_xyxy}``
  (original image pixels, see MODEL_IO.md),
* classifier top-1: ``{class_name, confidence}`` (use ``classifier_top1(probs)`` on the
  6 softmax outputs),
* the original image width and height.

The rule
--------
1. **Kept detections** = detections with ``confidence >= det_class_min_confidence[class]``
   (``det_min_confidence`` for a class not in the dict).
2. **final_class**
   * the classifier class, if ``classifier confidence >= cls_threshold``           (source "classifier")
   * else the class of the highest-confidence kept detection                       (source "detector")
     (ties: lower class index first, so the result never depends on list order)
   * else ``"no defect found"``                                                     (source "none")
3. **alert** = any kept detection OR ``classifier confidence >= cls_threshold``.
4. **severity**
   * ``severity.py`` (``assess_severity``) on the kept boxes of ``final_class`` only
     -> Low / Medium / High / Critical (count escalation counts only those boxes);
   * if alert but there is no kept box of ``final_class`` (classifier fired, detector found
     nothing of that class) -> level ``"Review"``: the operator must check the part, the reason
     says why;
   * no alert -> level ``None``.

Every threshold lives in ``FUSION_CONFIG`` (the detector thresholds, the classifier threshold
and, under ``"severity"``, the full severity config). The detector thresholds override the
severity config's min confidences, so the box filter is the same in both places.

**Threshold status:** ``cls_threshold`` and ``det_class_min_confidence`` below are
PLACEHOLDERS (the Phase 1 detector values; classifier 0.50) until the Phase 1.1 Colab run picks
them on the val split (notebook ``train_colab_phase1_1.ipynb``, step "threshold sweep"). Copy
the chosen values from ``results/phase1_1/fusion_config.json`` into this dict.

Pure standard library (plus ``severity.py``), so it ports line for line to Kotlin.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from severity import SEVERITY_CONFIG, assess_severity

NEU_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

NO_DEFECT = "no defect found"
REVIEW = "Review"

FUSION_CONFIG: Dict[str, Any] = {
    # Class order of both models (labels.txt). Detections / classifier classes outside it raise.
    "classes": list(NEU_CLASSES),
    # Classifier top-1 is trusted when its softmax confidence >= this.  PLACEHOLDER until the val sweep.
    "cls_threshold": 0.50,
    # A detection is "kept" when confidence >= its class threshold.  PLACEHOLDER (Phase 1 values)
    # until the val sweep. Classes not listed use det_min_confidence.
    "det_min_confidence": 0.50,
    "det_class_min_confidence": {
        "crazing": 0.25,
        "inclusion": 0.50,
        "patches": 0.50,
        "pitted_surface": 0.50,
        "rolled-in_scale": 0.50,
        "scratches": 0.50,
    },
    # Area / count rule for the kept boxes of final_class (see severity.py). Its own min-confidence
    # entries are replaced by the detector thresholds above at run time.
    "severity": copy.deepcopy(SEVERITY_CONFIG),
    "no_defect_label": NO_DEFECT,
    "review_level": REVIEW,
}


@dataclass
class FusionResult:
    final_class: str                      # a class name, or config["no_defect_label"]
    source: str                           # "classifier" | "detector" | "none"
    alert: bool
    severity_level: Optional[str]         # Low | Medium | High | Critical | "Review" | None (no alert)
    reason: str
    classifier_class: Optional[str]
    classifier_confidence: Optional[float]
    kept_detections: List[Dict[str, Any]] = field(default_factory=list)       # all kept, any class
    final_class_detections: List[Dict[str, Any]] = field(default_factory=list)  # kept boxes of final_class
    severity: Optional[Dict[str, Any]] = None  # SeverityResult.to_dict() when severity.py ran

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _check_conf(value: Any, what: str) -> float:
    v = float(value)
    if math.isnan(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{what} must be in [0, 1], got {value!r}")
    return v


def det_threshold(class_name: str, config: Mapping[str, Any]) -> float:
    return float(config["det_class_min_confidence"].get(class_name, config["det_min_confidence"]))


def classifier_top1(probs: Sequence[float], classes: Sequence[str] = NEU_CLASSES) -> Dict[str, Any]:
    """Top-1 of the classifier's softmax output. Ties -> lower class index."""
    if len(probs) != len(classes):
        raise ValueError(f"classifier output has {len(probs)} values, expected {len(classes)}")
    vals = [_check_conf(p, f"classifier prob[{i}]") for i, p in enumerate(probs)]
    i = max(range(len(vals)), key=lambda k: (vals[k], -k))
    return {"class_name": classes[i], "class_id": i, "confidence": vals[i]}


def _severity_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    sev = copy.deepcopy(dict(config["severity"]))
    sev["min_confidence"] = float(config["det_min_confidence"])
    sev["class_min_confidence"] = {k: float(v) for k, v in config["det_class_min_confidence"].items()}
    return sev


def fuse(
    detections: Iterable[Mapping[str, Any]],
    classifier: Optional[Mapping[str, Any]],
    image_width: float,
    image_height: float,
    config: Optional[Mapping[str, Any]] = None,
) -> FusionResult:
    """Fuse detector + classifier output for ONE image. See the module docstring for the rule.

    ``classifier`` is ``{class_name, confidence}`` (e.g. from ``classifier_top1``). Passing
    ``None`` means the classifier was not run; it is then treated as below threshold.
    """
    cfg = FUSION_CONFIG if config is None else config
    classes = list(cfg["classes"])
    class_idx = {c: i for i, c in enumerate(classes)}
    cls_thr = _check_conf(cfg["cls_threshold"], "cls_threshold")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")

    # 1. kept detections
    kept: List[Dict[str, Any]] = []
    for d in detections:
        name = str(d["class_name"])
        if name not in class_idx:
            raise ValueError(f"detection class {name!r} is not one of {classes}")
        conf = _check_conf(d["confidence"], f"confidence of a {name} detection")
        if conf >= det_threshold(name, cfg):
            kept.append({"class_name": name, "confidence": conf, "bbox_xyxy": [float(v) for v in d["bbox_xyxy"]]})
    # canonical order: confidence desc, class index, box -> output independent of input order
    kept.sort(key=lambda k: (-k["confidence"], class_idx[k["class_name"]], k["bbox_xyxy"]))

    # classifier
    c_name: Optional[str] = None
    c_conf: Optional[float] = None
    if classifier is not None:
        c_name = str(classifier["class_name"])
        if c_name not in class_idx:
            raise ValueError(f"classifier class {c_name!r} is not one of {classes}")
        c_conf = _check_conf(classifier["confidence"], "classifier confidence")
    cls_fires = c_conf is not None and c_conf >= cls_thr

    # 2. final class
    if cls_fires:
        final_class, source = c_name, "classifier"
        why = f"classifier {c_name} {c_conf:.3f} >= {cls_thr:.2f}"
    elif kept:
        final_class, source = kept[0]["class_name"], "detector"
        why = (f"classifier {'not run' if c_conf is None else f'{c_name} {c_conf:.3f} < {cls_thr:.2f}'}; "
               f"top kept detection {final_class} {kept[0]['confidence']:.3f}")
    else:
        return FusionResult(
            final_class=str(cfg["no_defect_label"]), source="none", alert=False, severity_level=None,
            reason=(f"no detection at or above its class threshold and classifier "
                    f"{'not run' if c_conf is None else f'{c_name} {c_conf:.3f} < {cls_thr:.2f}'}"),
            classifier_class=c_name, classifier_confidence=c_conf, kept_detections=[],
        )

    # 3. alert (true on both remaining branches)
    alert = True

    # 4. severity on the kept boxes of final_class
    final_dets = [k for k in kept if k["class_name"] == final_class]
    if not final_dets:
        others = sorted({k["class_name"] for k in kept}, key=class_idx.get)
        detail = (f"detector kept boxes only of {others}" if others else "detector kept no box at all")
        return FusionResult(
            final_class=final_class, source=source, alert=alert, severity_level=str(cfg["review_level"]),
            reason=f"{why}; no kept {final_class} box ({detail}) -> {cfg['review_level']}: operator must check the part",
            classifier_class=c_name, classifier_confidence=c_conf, kept_detections=kept,
        )

    sev = assess_severity(final_dets, image_width, image_height, config=_severity_config(cfg))
    if sev.level is None or sev.defect_count != len(final_dets):  # same filter on both sides -> cannot happen
        raise AssertionError(f"severity filter disagrees with the fusion filter: {sev.defect_count} vs {len(final_dets)}")
    return FusionResult(
        final_class=final_class, source=source, alert=alert, severity_level=sev.level,
        reason=f"{why}; severity on {len(final_dets)} kept {final_class} box(es): {sev.reason}",
        classifier_class=c_name, classifier_confidence=c_conf, kept_detections=kept,
        final_class_detections=final_dets, severity=sev.to_dict(),
    )
