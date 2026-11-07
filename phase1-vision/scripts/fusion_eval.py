"""
Phase 1.1: threshold selection (VAL only) and image-level evaluation of the fusion rule.

Pure Python (no numpy), deterministic. Used by notebooks/train_colab_phase1_1.ipynb.

A *record* is one image with the raw outputs of one detector + one classifier:

    {"image": "crazing_12.jpg", "true_class": "crazing", "width": 200, "height": 200,
     "detections": [{class_name, confidence, bbox_xyxy}, ...],   # decoded at a LOW conf (0.05) + NMS
     "probs": [p0, ..., p5]}                                      # classifier softmax, class order

Detections are decoded once at a low confidence; any higher threshold is applied afterwards.
That is exact: a box above threshold t can only be suppressed by NMS through a box with a higher
score, which is also above t.

Threshold selection (both on VAL, summed over the FP32 and INT8 record sets so one threshold
serves both models):
  1. detector, per class c, grid DET_GRID: maximise  correct_c - wrong_c, where
       correct_c = val images of class c with >= 1 kept box of class c,
       wrong_c   = val images of another class with >= 1 kept box of class c;
  2. classifier, grid CLS_GRID, with the detector thresholds from 1 fixed: maximise the fused
     image-level accuracy (final_class == image class).
  Ties: the middle value of the longest run of grid values that reach the maximum (a plateau
  centre is less sensitive to val noise than its edge). Lower value wins if the run is even.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import FUSION_CONFIG, NEU_CLASSES, classifier_top1, fuse  # noqa: E402

DET_GRID: List[float] = [round(0.05 * k, 2) for k in range(1, 20)]      # 0.05 .. 0.95
CLS_GRID: List[float] = [round(0.01 * k, 2) for k in range(30, 100)]    # 0.30 .. 0.99


def pick_plateau_centre(grid: Sequence[float], scores: Sequence[float]) -> float:
    """Middle of the longest contiguous run of grid points at the max score (ties -> earliest run)."""
    if not grid or len(grid) != len(scores):
        raise ValueError("grid and scores must be non-empty and the same length")
    best = max(scores)
    runs, start = [], None
    for i, s in enumerate(scores):
        if s == best and start is None:
            start = i
        if s != best and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(scores) - 1))
    a, b = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    return grid[(a + b) // 2]


def det_class_scores(record_sets: Sequence[Sequence[Mapping[str, Any]]], cls: str,
                     grid: Sequence[float] = DET_GRID) -> List[Dict[str, int]]:
    out = []
    for t in grid:
        correct = wrong = 0
        for records in record_sets:
            for r in records:
                hit = any(d["class_name"] == cls and d["confidence"] >= t for d in r["detections"])
                if hit and r["true_class"] == cls:
                    correct += 1
                elif hit:
                    wrong += 1
        out.append({"threshold": t, "correct": correct, "wrong": wrong, "score": correct - wrong})
    return out


def choose_det_thresholds(record_sets, classes: Sequence[str] = NEU_CLASSES,
                          grid: Sequence[float] = DET_GRID) -> Dict[str, Any]:
    chosen, table = {}, {}
    for c in classes:
        rows = det_class_scores(record_sets, c, grid)
        chosen[c] = pick_plateau_centre([r["threshold"] for r in rows], [r["score"] for r in rows])
        table[c] = rows
    return {"thresholds": chosen, "sweep": table}


def make_config(det_thresholds: Mapping[str, float], cls_threshold: float,
                base: Mapping[str, Any] = FUSION_CONFIG) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base))
    cfg["det_class_min_confidence"] = {c: float(t) for c, t in det_thresholds.items()}
    cfg["cls_threshold"] = float(cls_threshold)
    return cfg


def evaluate(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
             use_classifier: bool = True) -> Dict[str, Any]:
    """Fused image-level accuracy, overall and per class, plus how each image was decided."""
    classes = list(config["classes"])
    per = {c: {"images": 0, "correct": 0} for c in classes}
    sources = {"classifier": 0, "detector": 0, "none": 0}
    levels: Dict[str, int] = {}
    confusion = {c: {} for c in classes}          # true -> predicted -> count
    errors = []
    for r in sorted(records, key=lambda x: x["image"]):
        clf = classifier_top1(r["probs"], classes) if use_classifier else None
        res = fuse(r["detections"], clf, r["width"], r["height"], config)
        t = r["true_class"]
        per[t]["images"] += 1
        ok = res.final_class == t
        per[t]["correct"] += int(ok)
        sources[res.source] += 1
        levels[str(res.severity_level)] = levels.get(str(res.severity_level), 0) + 1
        confusion[t][res.final_class] = confusion[t].get(res.final_class, 0) + 1
        if not ok:
            errors.append({"image": r["image"], "true_class": t, "final_class": res.final_class,
                           "source": res.source, "classifier": clf, "reason": res.reason})
    for v in per.values():
        v["accuracy"] = round(v["correct"] / v["images"], 4) if v["images"] else None
    n = sum(v["images"] for v in per.values())
    k = sum(v["correct"] for v in per.values())
    return {"images": n, "correct": k, "accuracy": round(k / n, 4) if n else None,
            "alerts": n - sources["none"], "decided_by": sources, "severity_levels": levels,
            "review": levels.get(str(config["review_level"]), 0),
            "per_class": per, "confusion": confusion, "errors": errors}


def choose_cls_threshold(record_sets, det_thresholds: Mapping[str, float],
                         grid: Sequence[float] = CLS_GRID,
                         base: Mapping[str, Any] = FUSION_CONFIG) -> Dict[str, Any]:
    rows = []
    for t in grid:
        cfg = make_config(det_thresholds, t, base)
        evs = [evaluate(rs, cfg) for rs in record_sets]
        rows.append({"threshold": t, "correct": sum(e["correct"] for e in evs),
                     "review": sum(e["review"] for e in evs)})
    chosen = pick_plateau_centre([r["threshold"] for r in rows], [r["correct"] for r in rows])
    return {"threshold": chosen, "sweep": rows}


def classifier_only_accuracy(records: Sequence[Mapping[str, Any]], classes: Sequence[str] = NEU_CLASSES) -> Dict[str, Any]:
    per = {c: {"images": 0, "correct": 0} for c in classes}
    for r in records:
        top = classifier_top1(r["probs"], classes)["class_name"]
        per[r["true_class"]]["images"] += 1
        per[r["true_class"]]["correct"] += int(top == r["true_class"])
    n = sum(v["images"] for v in per.values())
    k = sum(v["correct"] for v in per.values())
    return {"images": n, "correct": k, "accuracy": round(k / n, 4) if n else None, "per_class": per}
