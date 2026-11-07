"""
Derive the severity area thresholds from data (ground-truth boxes of the val split).

    python phase1-vision/scripts/tune_severity.py

For every ground-truth box in data/dataset/labels/val: area ratio = box area / image area
(YOLO labels are normalised, so ratio = w * h). Exact duplicate rows are dropped, as
Ultralytics does. Cutoffs = 40th / 70th / 90th percentiles of ALL boxes, rounded to 3
decimals, so that roughly 40 % of boxes are Low, 30 % Medium, 20 % High and 10 % Critical
(per box, before the 3+ defects escalation).

Writes results/severity_thresholds.json (distribution overall and per class, the proposed
cutoffs, and the share of boxes per level). It does NOT edit severity.py; copy the numbers
into SEVERITY_CONFIG by hand so the change is reviewed.

Caveat: NEU-DET images are 200x200 crops where a defect often fills a large part of the
image. These cutoffs must be re-tuned on real camera frames in Phase 3.

Stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PHASE1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE1_DIR))

from severity import LEVELS  # noqa: E402

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
PCTS = (10, 25, 40, 50, 70, 75, 90, 95, 99)


def load_ratios(labels_dir: Path):
    ratios, classes = [], []
    files = sorted(labels_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no label files in {labels_dir}; run prepare_data.py first")
    for f in files:
        seen = set()
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            c, _, _, w, h = line.split()
            ratios.append(float(w) * float(h))
            classes.append(int(c))
    return np.array(ratios), np.array(classes), len(files)


def describe(r: np.ndarray) -> dict:
    return {"n": int(r.size), "mean": round(float(r.mean()), 4), "min": round(float(r.min()), 4),
            "max": round(float(r.max()), 4),
            "percentiles": {f"p{p}": round(float(np.percentile(r, p)), 4) for p in PCTS}}


def level_shares(r: np.ndarray, t: dict) -> dict:
    lv = np.where(r < t["low_max"], 0, np.where(r < t["medium_max"], 1, np.where(r <= t["high_max"], 2, 3)))
    return {LEVELS[i]: round(float((lv == i).mean()), 4) for i in range(4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=PHASE1_DIR / "data" / "dataset" / "labels" / "val")
    ap.add_argument("--out", type=Path, default=PHASE1_DIR / "results" / "severity_thresholds.json")
    args = ap.parse_args()

    r, c, n_img = load_ratios(args.labels)
    thr = {"low_max": round(float(np.percentile(r, 40)), 3),
           "medium_max": round(float(np.percentile(r, 70)), 3),
           "high_max": round(float(np.percentile(r, 90)), 3)}
    out = {
        "source": f"{args.labels.relative_to(PHASE1_DIR).as_posix()} ({n_img} images, ground-truth boxes)",
        "method": "area_ratio = w*h (normalised YOLO label); cutoffs = p40/p70/p90 of all boxes, rounded to 3 decimals",
        "overall": describe(r),
        "per_class": {name: describe(r[c == i]) for i, name in enumerate(CLASSES) if (c == i).any()},
        "area_thresholds": thr,
        "box_level_share_overall": level_shares(r, thr),
        "box_level_share_per_class": {name: level_shares(r[c == i], thr) for i, name in enumerate(CLASSES)
                                      if (c == i).any()},
        "caveat": "Tuned on NEU-DET 200x200 crops. Re-tune on real camera images in Phase 3.",
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{r.size} boxes in {n_img} images")
    print(f"{'':16s} {'n':>4s} " + " ".join(f"{'p' + str(p):>6s}" for p in PCTS))
    rows = [("ALL", out["overall"])] + list(out["per_class"].items())
    for name, d in rows:
        print(f"{name:16s} {d['n']:4d} " + " ".join(f"{v:6.3f}" for v in d["percentiles"].values()))
    print("cutoffs:", thr)
    print("box share per level:", out["box_level_share_overall"])
    for name, s in out["box_level_share_per_class"].items():
        print(f"  {name:16s} " + "  ".join(f"{k} {v:.2f}" for k, v in s.items()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
