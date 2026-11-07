"""
Post-training checks for the Phase 1 models (runs on any OS with ai-edge-litert + ultralytics,
including Windows; no GPU needed, no export).

    python phase1-vision/scripts/check_models.py --models-dir phase1-vision/models/exported

Expects in --models-dir (download them from Google Drive
MyDrive/qc-defect-detector/phase1/<run-name>/models/):
    best_<run-name>.pt, defect_yolov8n_320_fp32.tflite, defect_yolov8n_320_int8.tflite,
    labels.txt, model_io.json

What it does (all on the 360-image val split made by prepare_data.py, seed 42):
  1. SHA-256 + size of every model file; sizes must match model_io.json from the Colab run.
  2. .pt vs FP32 vs INT8 agreement on the WHOLE val split (conf 0.25, NMS IoU 0.45, same as
     the notebook's 10-image check).
  3. Per-class precision/recall vs ground truth at several confidence thresholds, for each
     model (detections at IoU >= 0.5, same class). Answers "is crazing 0.30 a good threshold?".
  4. Ultralytics mAP for .pt (default rect=True, and rect=False at a fixed 320 input like the
     .tflite), FP32 and INT8, to explain the .pt -> FP32 mAP gap.
  5. Regenerates results/severity_demo.png with the CURRENT severity rule on the same 10 images.
  6. Severity level distribution on all val images (each model, and ground-truth boxes).
  7. Image-level accuracy: % of val images with >= 1 detection of the correct class, overall and per class.

Writes results/phase1_checks.json and results/severity_demo.png. Raises on any missing file
or mismatch; it never substitutes another model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PHASE1_DIR))

from tflite_utils import (  # noqa: E402
    NEU_CLASSES, assert_defect_model, decode_yolov8, describe_io, iou_xyxy, letterbox,
    load_interpreter, match_detections, read_rgb, to_input_tensor,
)
from severity import SEVERITY_CONFIG, assess_severity  # noqa: E402

PRECISIONS = ("fp32", "int8")
SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
CONF, IOU = 0.25, 0.45          # same as the notebook's comparison
SWEEP_BASE_CONF = 0.05          # detections kept for the sweep (filtered further after NMS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_files(models_dir: Path) -> Dict[str, Path]:
    pts = sorted(models_dir.glob("best*.pt"))
    if len(pts) != 1:
        raise FileNotFoundError(f"need exactly one best*.pt in {models_dir}, found {[p.name for p in pts]}")
    files = {"pt": pts[0]}
    for p in PRECISIONS:
        files[p] = models_dir / f"defect_yolov8n_320_{p}.tflite"
    files["labels"] = models_dir / "labels.txt"
    files["model_io"] = models_dir / "model_io.json"
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise FileNotFoundError("missing model files (download them from Google Drive):\n  " + "\n  ".join(missing))
    return files


def load_gt(img: Path, labels_dir: Path, w: int, h: int) -> List[Dict[str, Any]]:
    lbl = labels_dir / f"{img.stem}.txt"
    gt, seen = [], set()
    if lbl.is_file():
        for line in lbl.read_text().splitlines():
            # drop exact duplicate rows, as Ultralytics does (NEU's patches_198 has one)
            if line.strip() and line.strip() not in seen:
                seen.add(line.strip())
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                gt.append({"class_id": int(c), "bbox_xyxy": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
    return gt


def pt_predict(model, img: Path, conf: float, iou: float) -> List[Dict[str, Any]]:
    r = model.predict(str(img), imgsz=320, conf=conf, iou=iou, device="cpu", verbose=False)[0]
    b = r.boxes
    return [{"class_id": int(c), "class_name": NEU_CLASSES[int(c)], "confidence": float(s),
             "bbox_xyxy": [float(v) for v in xyxy]}
            for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]


def tflite_raw(interp, io, img: Path):
    canvas, lb = letterbox(read_rgb(img), io.input_hw)
    interp.set_tensor(interp.get_input_details()[0]["index"], to_input_tensor(canvas, io))
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy(), lb


def pr_counts(dets: List[Dict[str, Any]], gt: List[Dict[str, Any]], thr: float, nc: int) -> np.ndarray:
    """[nc, 3] = TP, FP, FN for detections with conf >= thr (greedy by conf, IoU >= 0.5, same class)."""
    out = np.zeros((nc, 3), dtype=int)
    used = set()
    for d in sorted((d for d in dets if d["confidence"] >= thr), key=lambda d: -d["confidence"]):
        best, best_j = 0.5, None
        for j, g in enumerate(gt):
            if j not in used and g["class_id"] == d["class_id"]:
                v = iou_xyxy(d["bbox_xyxy"], g["bbox_xyxy"])
                if v >= best:
                    best, best_j = v, j
        if best_j is None:
            out[d["class_id"], 1] += 1
        else:
            used.add(best_j)
            out[d["class_id"], 0] += 1
    for j, g in enumerate(gt):
        if j not in used:
            out[g["class_id"], 2] += 1
    return out


def ultra_val(weights: Path, data: Path, project: Path, name: str, **kw) -> Dict[str, Any]:
    from ultralytics import YOLO
    task = {"task": "detect"} if weights.suffix == ".tflite" else {}
    m = YOLO(str(weights), **task).val(data=str(data), imgsz=320, split="val", device="cpu", plots=False,
                                       verbose=False, project=str(project), name=name, exist_ok=True, **kw)
    per = {}
    for i, c in enumerate(m.box.ap_class_index):
        _, _, ap50, ap = m.box.class_result(i)
        per[NEU_CLASSES[int(c)]] = {"mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
            "args": {k: v for k, v in kw.items()}, "per_class": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=PHASE1_DIR / "models" / "exported")
    ap.add_argument("--data", type=Path, default=PHASE1_DIR / "data" / "dataset" / "data.yaml")
    ap.add_argument("--results-dir", type=Path, default=PHASE1_DIR / "results")
    ap.add_argument("--skip-ultralytics-val", action="store_true")
    args = ap.parse_args()

    files = find_files(args.models_dir)
    colab = json.loads((args.results_dir / "phase1_results.json").read_text(encoding="utf-8"))
    model_io = json.loads(files["model_io"].read_text(encoding="utf-8"))

    # ---- 1. hashes + sizes -------------------------------------------------------------
    labels = [l.strip() for l in files["labels"].read_text().splitlines() if l.strip()]
    if labels != NEU_CLASSES:
        raise AssertionError(f"labels.txt {labels} != {NEU_CLASSES}")
    file_info = {}
    for key, f in files.items():
        file_info[f.name] = {"sha256": sha256(f), "size_bytes": f.stat().st_size}
    for p in PRECISIONS:
        want = model_io["models"][p]["size_bytes"]
        got = file_info[files[p].name]["size_bytes"]
        if want != got:
            raise AssertionError(f"{files[p].name}: size {got} != {want} in model_io.json (not the Colab file?)")
        if model_io["models"][p]["size_bytes"] != colab["model_io"]["models"][p]["size_bytes"]:
            raise AssertionError("model_io.json does not match results/phase1_results.json")
    print("[1] files")
    for n, v in file_info.items():
        print(f"    {v['sha256']}  {v['size_bytes']:>10,d}  {n}")

    # ---- dataset ------------------------------------------------------------------------
    ds = args.data.parent
    val_imgs = sorted((ds / "images" / "val").glob("*.jpg"))
    if len(val_imgs) != colab["dataset"]["val_images"]:
        raise AssertionError(f"local val has {len(val_imgs)} images, Colab had {colab['dataset']['val_images']}; "
                             "run prepare_data.py (seed 42)")
    names = {p.name for p in val_imgs}
    demo_names = [r["image"] for r in colab["compare_10_images"]["per_image"]]
    if not set(demo_names) <= names:
        raise AssertionError("local val split differs from the Colab split (demo images missing)")

    # ---- 2+3. inference on the whole val split ---------------------------------------------
    from ultralytics import YOLO
    pt = YOLO(str(files["pt"]))
    if [pt.names[i] for i in sorted(pt.names)] != NEU_CLASSES:
        raise AssertionError(f"{files['pt'].name} is not the 6-class NEU model: {pt.names}")
    interps = {}
    for p in PRECISIONS:
        it = load_interpreter(files[p])
        io = describe_io(it)
        assert_defect_model(io)
        interps[p] = (it, io)

    nc = len(NEU_CLASSES)
    models = ("pt",) + PRECISIONS
    at_conf = {m: {} for m in models}        # conf 0.25 detections, exact (conf filter before NMS)
    base = {m: {} for m in models}           # conf 0.05 detections, for the sweep
    gts = {}
    print(f"[2] inference on {len(val_imgs)} val images ...")
    for k, img in enumerate(val_imgs):
        rgb = read_rgb(img)
        h, w = rgb.shape[:2]
        gts[img.name] = load_gt(img, ds / "labels" / "val", w, h)
        at_conf["pt"][img.name] = pt_predict(pt, img, CONF, IOU)
        base["pt"][img.name] = pt_predict(pt, img, SWEEP_BASE_CONF, IOU)
        for p, (it, io) in interps.items():
            raw, lb = tflite_raw(it, io, img)
            at_conf[p][img.name] = decode_yolov8(raw, io, lb, NEU_CLASSES, CONF, IOU)
            base[p][img.name] = decode_yolov8(raw, io, lb, NEU_CLASSES, SWEEP_BASE_CONF, IOU)
        if (k + 1) % 60 == 0:
            print(f"    {k + 1}/{len(val_imgs)}")

    agreement = {}
    for p in PRECISIONS:
        tot = {"pt_detections": 0, "tflite_detections": 0, "matched": 0, "images_identical": 0}
        ious, dconf = [], []
        for n in names:
            r = match_detections(at_conf["pt"][n], at_conf[p][n])
            tot["pt_detections"] += r["n_ref"]
            tot["tflite_detections"] += r["n_other"]
            tot["matched"] += r["matched"]
            tot["images_identical"] += int(r["matched"] == r["n_ref"] == r["n_other"])
            if r["matched"]:
                ious += [r["mean_iou"]] * r["matched"]
                dconf += [r["mean_abs_conf_diff"]] * r["matched"]
        tot["match_rate"] = round(tot["matched"] / max(tot["pt_detections"], 1), 4)
        tot["mean_iou"] = round(float(np.mean(ious)), 4) if ious else None
        tot["mean_abs_conf_diff"] = round(float(np.mean(dconf)), 4) if dconf else None
        tot["images"] = len(names)
        agreement[p] = tot
        print(f"    {p}: {tot['matched']}/{tot['pt_detections']} .pt detections matched "
              f"({tot['match_rate']:.1%}), {tot['tflite_detections']} TFLite detections, "
              f"{tot['images_identical']}/{len(names)} images identical, mean IoU {tot['mean_iou']}")

    print("[3] precision / recall vs ground truth (IoU >= 0.5)")
    sweep = {}
    for m in models:
        sweep[m] = {c: {} for c in NEU_CLASSES}
        for thr in SWEEP:
            cnt = sum(pr_counts(base[m][n], gts[n], thr, nc) for n in names)
            for i, c in enumerate(NEU_CLASSES):
                tp, fp, fn = (int(x) for x in cnt[i])
                sweep[m][c][f"{thr:.2f}"] = {"tp": tp, "fp": fp, "fn": fn,
                                              "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                                              "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    # crazing at image level: does the image get >= 1 crazing detection (what severity sees)?
    crz_imgs = [n for n in names if n.startswith("crazing_")]
    other_imgs = [n for n in names if not n.startswith("crazing_")]
    crazing_image_level = {}
    for m in models:
        crazing_image_level[m] = {}
        for thr in SWEEP:
            hit = sum(any(d["class_id"] == 0 and d["confidence"] >= thr for d in base[m][n]) for n in crz_imgs)
            fa = sum(any(d["class_id"] == 0 and d["confidence"] >= thr for d in base[m][n]) for n in other_imgs)
            crazing_image_level[m][f"{thr:.2f}"] = {
                "crazing_images_flagged": hit, "crazing_images": len(crz_imgs),
                "other_images_with_crazing_det": fa, "other_images": len(other_imgs)}
    print("    crazing:  thr  | " + " | ".join(f"{m:>4s} P / R  img-hit  false-img" for m in models))
    for thr in SWEEP:
        k = f"{thr:.2f}"
        cells = []
        for m in models:
            s, il = sweep[m]["crazing"][k], crazing_image_level[m][k]
            cells.append(f"{s['precision'] or 0:.2f} / {s['recall'] or 0:.2f}  "
                         f"{il['crazing_images_flagged']:>3d}/{il['crazing_images']}  {il['other_images_with_crazing_det']:>3d}")
        print(f"             {k} | " + " | ".join(cells))

    # ---- 4. Ultralytics mAP ----------------------------------------------------------------
    val_maps = {}
    if not args.skip_ultralytics_val:
        print("[4] Ultralytics val (CPU) ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            val_maps["pt_rect_true_batch32"] = ultra_val(files["pt"], args.data, tmp, "pt_rect", batch=32)
            val_maps["pt_rect_false_batch1"] = ultra_val(files["pt"], args.data, tmp, "pt_320", batch=1, rect=False)
            for p in PRECISIONS:
                val_maps[p] = ultra_val(files[p], args.data, tmp, p, batch=1)
        for k, v in val_maps.items():
            print(f"    {k:22s} mAP50 {v['mAP50']:.4f}  mAP50-95 {v['mAP50_95']:.4f}")

    # ---- 5. severity demo with the current rule --------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    cmc = SEVERITY_CONFIG.get("class_min_confidence", {})
    min_conf = lambda c: float(cmc.get(c, SEVERITY_CONFIG["min_confidence"]))  # noqa: E731
    demo = []
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.6))
    for ax, n in zip(axes.ravel(), demo_names):
        img = ds / "images" / "val" / n
        w, h = Image.open(img).size
        row = {"image": n}
        for p in PRECISIONS:
            row[p] = assess_severity(at_conf[p][n], w, h).to_dict()
        old = next(r for r in colab["severity_demo"] if r["image"] == n)
        row["fp32_old_rule_level"] = old["level"]
        demo.append(row)
        ax.imshow(Image.open(img).convert("RGB"))
        for d in at_conf["fp32"][n]:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            keep = d["confidence"] >= min_conf(d["class_name"])
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.5,
                                           color="red" if keep else "gray"))
            ax.text(x1, y1, f"{d['class_name']} {d['confidence']:.2f}", fontsize=6, color="yellow", va="bottom")
        ax.set_title(f"{n}\nFP32: {row['fp32']['level']} (n={row['fp32']['defect_count']})  |  "
                     f"INT8: {row['int8']['level']} (n={row['int8']['defect_count']})", fontsize=8)
        ax.axis("off")
        print(f"    {n:22s} FP32 {str(row['fp32']['level']):8s} INT8 {str(row['int8']['level']):8s} "
              f"(old rule FP32: {old['level']})")
    t = SEVERITY_CONFIG["area_thresholds"]
    fig.suptitle(f"Severity rule: min conf crazing {min_conf('crazing'):.2f}, others {SEVERITY_CONFIG['min_confidence']:.2f}; "
                 f"area Low < {t['low_max']} <= Medium < {t['medium_max']} <= High <= {t['high_max']} < Critical "
                 "(red = counted, gray = ignored; boxes = FP32 TFLite, conf >= 0.25)", fontsize=10)
    plt.tight_layout()
    demo_png = args.results_dir / "severity_demo.png"
    plt.savefig(demo_png, dpi=120)
    plt.close(fig)

    # ---- 6. severity level distribution on the whole val split ----------------------------
    # Image level (after the 3+ escalation) and detection level (before it), for each model's
    # conf-0.25 detections and for the ground-truth boxes (confidence 1.0) as the reference.
    lv_names = ["Low", "Medium", "High", "Critical", "None"]
    sev_dist = {}
    sources = {m: at_conf[m] for m in models}
    sources["ground_truth"] = {n: [{"class_name": NEU_CLASSES[g["class_id"]], "confidence": 1.0,
                                    "bbox_xyxy": g["bbox_xyxy"]} for g in gts[n]] for n in names}
    for src, dets_by_img in sources.items():
        img_cnt = dict.fromkeys(lv_names, 0)
        det_cnt = dict.fromkeys(lv_names[:4], 0)
        escalated = 0
        for n in names:
            w, h = Image.open(ds / "images" / "val" / n).size
            r = assess_severity(dets_by_img[n], w, h)
            img_cnt[r.level or "None"] += 1
            escalated += int(r.escalated)
            for d in r.detections:
                det_cnt[d.level] += 1
        nd = max(sum(det_cnt.values()), 1)
        sev_dist[src] = {"images": img_cnt,
                         "images_pct": {k: round(100 * v / len(names), 1) for k, v in img_cnt.items()},
                         "images_escalated": escalated,
                         "detections_before_escalation": det_cnt,
                         "detections_pct": {k: round(100 * v / nd, 1) for k, v in det_cnt.items()}}
    print("[6] severity on all val images (image level after escalation | detection level before)")
    for src, d in sev_dist.items():
        print(f"    {src:13s} images " + " ".join(f"{k} {v:3d}" for k, v in d["images"].items())
              + f" (escalated {d['images_escalated']}) | dets " + " ".join(f"{k} {v}%" for k, v in d["detections_pct"].items()))

    # ---- 7. image-level accuracy -------------------------------------------------------------
    # The image's class is its NEU category (filename prefix, e.g. crazing_90.jpg -> crazing). An image
    # is correct if at least one detection that passes the severity rule's per-class min confidence has
    # that class. A few images also carry GT boxes of another class; they are counted and reported.
    def image_class(n: str) -> str:
        c = next((c for c in sorted(NEU_CLASSES, key=len, reverse=True) if n.startswith(c + "_")), None)
        if c is None:
            raise AssertionError(f"cannot get the class from the file name {n}")
        return c
    mixed = sorted(n for n in names if len({g["class_id"] for g in gts[n]}) > 1)
    img_acc = {}
    for m in models:
        per = {c: {"images": 0, "correct": 0, "wrong_class_only": 0, "no_detection": 0} for c in NEU_CLASSES}
        for n in names:
            c = image_class(n)
            kept = [d for d in at_conf[m][n] if d["confidence"] >= min_conf(d["class_name"])]
            per[c]["images"] += 1
            if any(d["class_name"] == c for d in kept):
                per[c]["correct"] += 1
            elif kept:
                per[c]["wrong_class_only"] += 1
            else:
                per[c]["no_detection"] += 1
        for v in per.values():
            v["accuracy"] = round(v["correct"] / v["images"], 4)
        tot = {k: sum(v[k] for v in per.values()) for k in ("images", "correct", "wrong_class_only", "no_detection")}
        tot["accuracy"] = round(tot["correct"] / tot["images"], 4)
        img_acc[m] = {"overall": tot, "per_class": per}
    print(f"[7] image-level accuracy (>= 1 detection of the correct class, per-class min conf); "
          f"images with mixed-class GT: {mixed}")
    for m, d in img_acc.items():
        o = d["overall"]
        print(f"    {m:5s} overall {o['correct']}/{o['images']} ({o['accuracy']:.1%}), wrong class only "
              f"{o['wrong_class_only']}, no detection {o['no_detection']} | "
              + " ".join(f"{c} {v['correct']}/{v['images']}" for c, v in d["per_class"].items()))

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PHASE1_DIR).stdout.strip()
    import ultralytics, ai_edge_litert  # noqa: E401
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": head,
        "colab_run_commit": colab["git_commit"],
        "versions": {"python": sys.version.split()[0], "ultralytics": ultralytics.__version__,
                     "ai-edge-litert": getattr(ai_edge_litert, "__version__", "unknown"), "numpy": np.__version__,
                     "platform": sys.platform},
        "files": file_info,
        "val_images": len(names),
        "agreement_full_val": {"conf": CONF, "nms_iou": IOU, "match_iou": 0.5, **agreement},
        "pr_sweep": {"note": f"detections from conf {SWEEP_BASE_CONF} + NMS IoU {IOU}, then filtered at each "
                             "threshold; TP = same class, IoU >= 0.5 with an unused GT box",
                     "per_class": sweep, "crazing_image_level": crazing_image_level},
        "ultralytics_val": val_maps,
        "severity_config": SEVERITY_CONFIG,
        "severity_demo": demo,
        "severity_distribution_val": sev_dist,
        "image_level_accuracy_val": {"definition": "image correct if >= 1 detection (conf 0.25 decode, then "
                                                   "severity per-class min confidence) has the image's class "
                                                   "(NEU category = filename prefix)",
                                     "images_with_mixed_class_gt": mixed,
                                     **img_acc},
    }
    out_json = args.results_dir / "phase1_checks.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_json} and {demo_png}")


if __name__ == "__main__":
    main()
