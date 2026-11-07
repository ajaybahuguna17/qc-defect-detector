"""
Deterministic defect severity rule (Phase 1, vision side).

One rule, used everywhere. Given the detections for ONE image it returns one of
``Low``, ``Medium``, ``High``, ``Critical`` (or ``None`` when no detection passes
the confidence filter). The LLM in Phase 2 may *explain* this severity, but it
must never decide it.

The rule
--------
1. Drop every detection with ``confidence < class_min_confidence[class_name]``
   (0.25 for crazing, 0.50 for every other class; classes not in the dict use
   ``min_confidence`` = 0.50). Crazing gets a lower bar because the trained model
   is under-confident on it (e.g. 0.39 FP32 / 0.299 INT8 on val image crazing_90.jpg).
   0.25 was chosen from the val sweep in results/phase1_checks.json.
2. For each remaining detection compute
       area_ratio = (bbox area, clipped to the image) / (image width * image height)
   and multiply it by ``class_area_weight[class_name]`` (all 1.0 by default, so the
   rule is class-neutral until a process engineer gives per-class weights).
3. Map the weighted area ratio to a level (defaults):
       ratio <  0.02            -> Low
       0.02 <= ratio <  0.05    -> Medium
       0.05 <= ratio <= 0.10    -> High
       ratio >  0.10            -> Critical
4. The image level is the highest per-detection level.
5. If ``defect_count >= count_escalation_threshold`` (default 3), raise the image
   level by one step (capped at Critical).

All thresholds live in ``SEVERITY_CONFIG``. Pass a modified copy to
``assess_severity(..., config=...)`` to experiment; do not hard-code numbers
anywhere else.

Pure standard library, no third-party imports, so the same logic can be ported
line for line to Kotlin for the Android app.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

LEVELS: tuple = ("Low", "Medium", "High", "Critical")

SEVERITY_CONFIG: Dict[str, Any] = {
    # Detections below their class's min confidence are ignored entirely (not counted, not sized).
    # Default for any class not listed in class_min_confidence.
    "min_confidence": 0.50,
    # Per-class min confidence. crazing is lower because the model is under-confident on it
    # (crazing_90.jpg: 0.39 FP32, 0.299 INT8). 0.25 chosen from the 360-image val sweep
    # (results/phase1_checks.json): crazing images flagged FP32 56/60, INT8 51/60 (vs 25/21 at 0.50),
    # precision no worse than at 0.30, and 0 crazing detections on the 300 non-crazing images.
    "class_min_confidence": {
        "crazing": 0.25,
        "inclusion": 0.50,
        "patches": 0.50,
        "pitted_surface": 0.50,
        "rolled-in_scale": 0.50,
        "scratches": 0.50,
    },
    # Upper bounds (exclusive for Low/Medium, inclusive for High) of the weighted area ratio.
    #   ratio < low_max -> Low ; ratio < medium_max -> Medium ; ratio <= high_max -> High ; else Critical
    "area_thresholds": {
        "low_max": 0.02,
        "medium_max": 0.05,
        "high_max": 0.10,
    },
    # Number of valid detections in one image that raises the level by one step.
    "count_escalation_threshold": 3,
    # Per-class multiplier applied to the area ratio before thresholding.
    # 1.0 everywhere = class-neutral. Unknown classes use 1.0.
    "class_area_weight": {
        "crazing": 1.0,
        "inclusion": 1.0,
        "patches": 1.0,
        "pitted_surface": 1.0,
        "rolled-in_scale": 1.0,
        "scratches": 1.0,
    },
}


@dataclass
class DetectionSeverity:
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    area_ratio: float
    weighted_area_ratio: float
    level: str


@dataclass
class SeverityResult:
    level: Optional[str]                 # "Low" | "Medium" | "High" | "Critical" | None (no valid defect)
    defect_count: int                    # detections with confidence >= their class min confidence
    max_area_ratio: float                # largest weighted area ratio among valid detections
    escalated: bool                      # True if the count rule raised the level
    reason: str
    detections: List[DetectionSeverity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _area_level(ratio: float, thresholds: Mapping[str, float]) -> str:
    if ratio < thresholds["low_max"]:
        return "Low"
    if ratio < thresholds["medium_max"]:
        return "Medium"
    if ratio <= thresholds["high_max"]:
        return "High"
    return "Critical"


def _bbox_area_ratio(bbox: Sequence[float], width: float, height: float) -> float:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    return ((x2 - x1) * (y2 - y1)) / (width * height)


def assess_severity(
    detections: Iterable[Mapping[str, Any]],
    image_width: float,
    image_height: float,
    config: Optional[Mapping[str, Any]] = None,
) -> SeverityResult:
    """Return the severity of ONE image.

    Args:
        detections: iterable of dicts with keys
            ``class_name`` (str), ``confidence`` (float 0-1) and
            ``bbox_xyxy`` ([x1, y1, x2, y2] in pixels of the original image).
        image_width, image_height: original image size in pixels.
        config: optional override of ``SEVERITY_CONFIG`` (full dict).
    """
    cfg = SEVERITY_CONFIG if config is None else config
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")

    min_conf = float(cfg["min_confidence"])
    class_min_conf = cfg.get("class_min_confidence", {})
    thresholds = cfg["area_thresholds"]
    weights = cfg.get("class_area_weight", {})

    scored: List[DetectionSeverity] = []
    for det in detections:
        conf = float(det["confidence"])
        name = str(det["class_name"])
        if conf < float(class_min_conf.get(name, min_conf)):
            continue
        ratio = _bbox_area_ratio(det["bbox_xyxy"], image_width, image_height)
        weighted = ratio * float(weights.get(name, 1.0))
        scored.append(
            DetectionSeverity(
                class_name=name,
                confidence=conf,
                bbox_xyxy=[float(v) for v in det["bbox_xyxy"]],
                area_ratio=ratio,
                weighted_area_ratio=weighted,
                level=_area_level(weighted, thresholds),
            )
        )

    if not scored:
        return SeverityResult(
            level=None,
            defect_count=0,
            max_area_ratio=0.0,
            escalated=False,
            reason="no detection at or above its class min confidence",
        )

    worst = max(scored, key=lambda d: LEVELS.index(d.level))
    level_idx = LEVELS.index(worst.level)
    max_ratio = max(d.weighted_area_ratio for d in scored)
    count = len(scored)

    escalated = False
    reason = f"largest-severity defect '{worst.class_name}' area ratio {worst.weighted_area_ratio:.4f} -> {worst.level}"
    if count >= int(cfg["count_escalation_threshold"]) and level_idx < len(LEVELS) - 1:
        level_idx += 1
        escalated = True
        reason += f"; {count} defects >= {cfg['count_escalation_threshold']} -> raised to {LEVELS[level_idx]}"

    return SeverityResult(
        level=LEVELS[level_idx],
        defect_count=count,
        max_area_ratio=max_ratio,
        escalated=escalated,
        reason=reason,
        detections=scored,
    )
