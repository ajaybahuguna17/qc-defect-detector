"""Phase 1.2 tests: the not_steel RETAKE gate in fusion.py, the 7-class helpers in fusion_eval.py,
and the --num-classes option of export_cls_tflite.py.

Run from the repo root:
    python -m unittest discover -s phase1-vision/tests -v
"""

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion import (  # noqa: E402
    CLS7_CLASSES, FUSION_CONFIG, NEU_CLASSES, NO_DEFECT, NOT_STEEL, NOT_STEEL_LABEL, RETAKE, REVIEW,
    classifier_classes, classifier_top1, fuse,
)
from fusion_eval import choose_cls_threshold, choose_not_steel_threshold, evaluate, not_steel_rejected  # noqa: E402

W = H = 200


def det(cls, conf, ratio=0.01, y0=0.0):
    return {"class_name": cls, "confidence": conf, "bbox_xyxy": [0.0, y0, float(W), y0 + ratio * H]}


def clf(cls, conf):
    return {"class_name": cls, "confidence": conf}


def cfg(**over):
    c = copy.deepcopy(FUSION_CONFIG)
    c.update(over)
    return c


# fixed test thresholds, independent of the tuned / placeholder values
CFG = cfg(cls_threshold=0.80, not_steel_threshold=0.70, det_min_confidence=0.50,
          det_class_min_confidence={c: 0.50 for c in NEU_CLASSES})


def probs7(top, p, rest_to=None):
    """7 probabilities with `p` on class `top`, the rest spread on `rest_to` (or evenly)."""
    v = [0.0] * 7
    v[CLS7_CLASSES.index(top)] = p
    if rest_to is not None:
        v[CLS7_CLASSES.index(rest_to)] += 1.0 - p
    else:
        others = [i for i in range(7) if i != CLS7_CLASSES.index(top)]
        for i in others:
            v[i] = (1.0 - p) / len(others)
    return v


class NotSteelGateTests(unittest.TestCase):
    def test_confident_not_steel_is_retake(self):
        r = fuse([], clf(NOT_STEEL, 0.9), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert, r.severity_level, r.decision),
                         (NOT_STEEL_LABEL, "classifier", False, None, RETAKE))
        self.assertEqual(r.final_class, "not a steel surface")
        self.assertEqual(r.decision, "RETAKE")
        self.assertIsNone(r.severity)
        self.assertEqual((r.kept_detections, r.final_class_detections), ([], []))

    def test_detector_boxes_are_ignored(self):
        dets = [det("inclusion", 0.99, 0.5), det("scratches", 0.95, 0.2, y0=100)]
        r = fuse(dets, clf(NOT_STEEL, 0.75), W, H, CFG)
        self.assertEqual((r.final_class, r.alert, r.decision), (NOT_STEEL_LABEL, False, RETAKE))
        self.assertEqual(r.kept_detections, [])
        self.assertIn("2 kept detection(s) ignored", r.reason)

    def test_threshold_is_inclusive(self):
        self.assertEqual(fuse([], clf(NOT_STEEL, 0.70), W, H, CFG).decision, RETAKE)
        self.assertIsNone(fuse([], clf(NOT_STEEL, 0.6999), W, H, CFG).decision)

    def test_gate_checked_before_classifier_threshold(self):
        # not_steel above BOTH thresholds is a reject, not a defect class
        r = fuse([det("patches", 0.9)], clf(NOT_STEEL, 0.95), W, H, CFG)
        self.assertEqual(r.final_class, NOT_STEEL_LABEL)

    def test_unsure_not_steel_falls_back_to_detector(self):
        # top-1 not_steel below t_ns but above cls_threshold: it names no defect, the detector decides
        c = cfg(cls_threshold=0.50, not_steel_threshold=0.90, det_class_min_confidence={k: 0.5 for k in NEU_CLASSES})
        r = fuse([det("scratches", 0.8, 0.10)], clf(NOT_STEEL, 0.85), W, H, c)
        self.assertEqual((r.final_class, r.source, r.alert, r.decision), ("scratches", "detector", True, None))
        self.assertEqual(r.severity_level, "Medium")
        self.assertIn("no defect named", r.reason)

    def test_unsure_not_steel_no_boxes_is_no_defect(self):
        r = fuse([det("scratches", 0.3)], clf(NOT_STEEL, 0.6), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert, r.decision), (NO_DEFECT, "none", False, None))

    def test_old_branches_have_no_decision(self):
        self.assertIsNone(fuse([det("scratches", 0.9)], clf("scratches", 0.9), W, H, CFG).decision)
        self.assertIsNone(fuse([], clf("patches", 0.9), W, H, CFG).decision)          # Review
        self.assertEqual(fuse([], clf("patches", 0.9), W, H, CFG).severity_level, REVIEW)
        self.assertIsNone(fuse([], None, W, H, CFG).decision)

    def test_detections_still_validated_on_reject(self):
        with self.assertRaises(ValueError):
            fuse([det("rust", 0.9)], clf(NOT_STEEL, 0.9), W, H, CFG)
        with self.assertRaises(ValueError):
            fuse([det("patches", 1.5)], clf(NOT_STEEL, 0.9), W, H, CFG)

    def test_bad_not_steel_threshold_raises(self):
        with self.assertRaises(ValueError):
            fuse([], clf(NOT_STEEL, 0.9), W, H, cfg(not_steel_threshold=1.5))

    def test_unknown_classifier_class_still_raises(self):
        with self.assertRaises(ValueError):
            fuse([], clf("rust", 0.9), W, H, CFG)

    def test_config_keys(self):
        for k in ("not_steel_class", "not_steel_threshold", "not_steel_label", "retake_decision"):
            self.assertIn(k, FUSION_CONFIG)
        self.assertEqual(FUSION_CONFIG["not_steel_class"], "not_steel")
        self.assertEqual(FUSION_CONFIG["not_steel_label"], "not a steel surface")
        self.assertEqual(FUSION_CONFIG["retake_decision"], "RETAKE")
        self.assertEqual(FUSION_CONFIG["classes"], NEU_CLASSES)     # detector classes unchanged


class ClassifierV2Tests(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual(CLS7_CLASSES, [*NEU_CLASSES, "not_steel"])

    def test_classifier_classes_by_length(self):
        self.assertEqual(classifier_classes(FUSION_CONFIG, 6), NEU_CLASSES)
        self.assertEqual(classifier_classes(FUSION_CONFIG, 7), CLS7_CLASSES)
        with self.assertRaises(ValueError):
            classifier_classes(FUSION_CONFIG, 1000)

    def test_top1_seven_feeds_fuse(self):
        r = fuse([], classifier_top1(probs7(NOT_STEEL, 0.9), CLS7_CLASSES), W, H, CFG)
        self.assertEqual(r.decision, RETAKE)
        r = fuse([det("patches", 0.9)], classifier_top1(probs7("patches", 0.9), CLS7_CLASSES), W, H, CFG)
        self.assertEqual((r.final_class, r.decision), ("patches", None))


def rec(name, true_class, probs, dets=()):
    return {"image": name, "true_class": true_class, "width": W, "height": H,
            "detections": list(dets), "probs": probs}


class FusionEvalV2Tests(unittest.TestCase):
    def test_evaluate_counts_retake_and_not_steel_correct(self):
        recs = [rec("a.jpg", "scratches", probs7("scratches", 0.9), [det("scratches", 0.9)]),
                rec("b.jpg", "scratches", probs7(NOT_STEEL, 0.9), [det("scratches", 0.9)]),   # false reject
                rec("c.jpg", NOT_STEEL, probs7(NOT_STEEL, 0.95)),                             # rejected
                rec("d.jpg", NOT_STEEL, probs7(NOT_STEEL, 0.5, rest_to="patches"))]           # missed
        e = evaluate(recs, CFG)
        self.assertEqual((e["images"], e["correct"], e["retake"]), (4, 2, 2))
        self.assertEqual(e["per_class"]["scratches"], {"images": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(e["per_class"][NOT_STEEL], {"images": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(e["confusion"][NOT_STEEL], {NOT_STEEL_LABEL: 1, NO_DEFECT: 1})

    def test_evaluate_six_class_records_unchanged(self):
        e = evaluate([rec("a.jpg", "patches", [0, 0, 0.9, 0.1, 0, 0], [det("patches", 0.9)])], CFG)
        self.assertEqual((e["correct"], e["retake"]), (1, 0))

    def test_unknown_true_class_raises(self):
        with self.assertRaises(ValueError):
            evaluate([rec("a.jpg", "rust", probs7("patches", 0.9))], CFG)

    def test_not_steel_rejected(self):
        self.assertTrue(not_steel_rejected(probs7(NOT_STEEL, 0.8), 0.8))
        self.assertFalse(not_steel_rejected(probs7(NOT_STEEL, 0.79), 0.8))
        self.assertFalse(not_steel_rejected(probs7("patches", 0.9), 0.3))
        self.assertFalse(not_steel_rejected([0, 0, 0.9, 0.1, 0, 0], 0.3))   # v1 never rejects

    def test_choose_threshold_respects_false_reject_cap(self):
        # 100 steel images: 2 have not_steel top-1 at 0.60 and 0.85 -> t_ns must be > 0.60 for <= 1 %
        steel = [probs7("patches", 0.9)] * 98 + [probs7(NOT_STEEL, 0.60), probs7(NOT_STEEL, 0.85)]
        ns = [probs7(NOT_STEEL, 0.99)] * 50 + [probs7(NOT_STEEL, 0.65)] * 10
        sel = choose_not_steel_threshold([steel], [ns], max_false_reject=0.01)
        t = sel["threshold"]
        row = next(r for r in sel["sweep"] if r["threshold"] == t)
        self.assertTrue(row["feasible"])
        self.assertLessEqual(row["steel_false_reject_rate"], 0.01)
        self.assertEqual(row["not_steel_rejected"], 60)       # all rejected at 0.61..0.65
        self.assertTrue(0.61 <= t <= 0.65, t)

    def test_choose_threshold_infeasible_raises(self):
        steel = [probs7(NOT_STEEL, 0.999)] * 10
        with self.assertRaises(RuntimeError):
            choose_not_steel_threshold([steel], [[probs7(NOT_STEEL, 0.99)]])

    def test_cls_threshold_recheck_with_seven_classes(self):
        base = cfg(not_steel_threshold=0.70)
        val = [rec(f"{c}_{i}.jpg", c, probs7(c, 0.95), [det(c, 0.9)]) for c in NEU_CLASSES for i in range(3)]
        sel = choose_cls_threshold([val], {c: 0.5 for c in NEU_CLASSES}, base=base)
        self.assertIn(sel["threshold"], [r["threshold"] for r in sel["sweep"]])
        self.assertEqual(max(r["correct"] for r in sel["sweep"]), len(val))


try:
    import numpy  # noqa: F401
    from export_cls_tflite import CLASS_LISTS, parse_args
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ClsExportNumClassesTests(unittest.TestCase):
    base = ["--weights", "w.pt", "--data", "cls"]

    def test_default_six(self):
        self.assertEqual(parse_args(self.base).num_classes, 6)

    def test_seven(self):
        a = parse_args(self.base + ["--num-classes", "7", "--precisions", "fp32", "--allow-skip-int8"])
        self.assertEqual((a.num_classes, a.precisions, a.allow_skip_int8), (7, ["fp32"], True))
        self.assertEqual(CLASS_LISTS[7], CLS7_CLASSES)

    def test_bad_num_classes(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(self.base + ["--num-classes", "1000"])


try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False


def noise(size, key):
    import random
    return Image.frombytes("RGB", size, random.Random(key).randbytes(size[0] * size[1] * 3))


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class BuildNotSteelTests(unittest.TestCase):
    """build_not_steel.py on tiny synthetic stand-ins for COCO / DTD / Places365."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.coco, self.dtd, self.places, self.out = t / "coco", t / "dtd", t / "places", t / "cls7"
        self.coco.mkdir()
        self.places.mkdir()
        for i in range(30):
            noise((64 + i, 48), f"coco{i}").save(self.coco / f"{i:012d}.jpg")
        for c in ("cracked", "stratified", "woven"):
            (self.dtd / c).mkdir(parents=True)
            for i in range(8):
                noise((60, 70), f"{c}{i}").save(self.dtd / c / f"{c}_{i:04d}.jpg")
        cats = ["/f/field/wild 0", "/k/kitchen 1", "/s/street 2"]
        (t / "cats.txt").write_text("\n".join(cats) + "\n", encoding="utf-8")
        lines = []
        for i in range(30):
            name = f"Places365_val_{i:08d}.jpg"
            noise((256, 256), f"places{i}").save(self.places / name)
            lines.append(f"{name} {i % 3}")
        (t / "val.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # a steel class folder that must survive
        (self.out / "train" / "crazing").mkdir(parents=True)
        (self.out / "train" / "crazing" / "crazing_1.jpg").write_bytes(b"x")
        self.args = ["--coco-dir", str(self.coco), "--dtd-dir", str(self.dtd), "--places-dir", str(self.places),
                     "--places-val-list", str(t / "val.txt"), "--places-categories", str(t / "cats.txt"),
                     "--out-root", str(self.out), "--n-coco", "20", "--n-dtd", "12", "--n-places", "8"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, extra=()):
        import build_not_steel
        with contextlib.redirect_stdout(io.StringIO()):
            return build_not_steel.main(self.args + list(extra))

    def test_counts_split_and_overlap(self):
        s = self.run_build()
        self.assertEqual(sum(s["totals"].values()), 40)
        self.assertEqual(s["counts"]["train"], {"coco": 14, "dtd": 8, "places365": 6})
        self.assertEqual(s["counts"]["val"], {"coco": 3, "dtd": 2, "places365": 1})
        self.assertEqual(s["counts"]["test"], {"coco": 3, "dtd": 2, "places365": 1})
        self.assertEqual(s["dtd_categories"], 3)                       # round-robin covers every category
        names = {sp: {p.name for p in (self.out / sp / "not_steel").iterdir()} for sp in ("train", "val", "test")}
        self.assertFalse(names["train"] & names["val"] or names["train"] & names["test"] or names["val"] & names["test"])
        self.assertTrue((self.out / "train" / "crazing" / "crazing_1.jpg").is_file())
        self.assertTrue((self.out / "not_steel_manifest.json").is_file())

    def test_places_keyword_filter(self):
        s = self.run_build()
        cats = {r["category"] for r in s["images"] if r["source"] == "places365"}
        self.assertTrue(cats <= {"/f/field/wild", "/s/street"}, cats)       # kitchen filtered out

    def test_transforms(self):
        s = self.run_build(["--gray-frac", "1.0", "--crop-frac", "1.0"])
        for r in s["images"]:
            self.assertTrue(r["grayscale"] and r["cropped_200"])
            self.assertEqual((r["width"], r["height"]), (200, 200))
        f = next((self.out / "train" / "not_steel").iterdir())
        with Image.open(f) as im:
            px = im.convert("RGB").getpixel((5, 5))
        self.assertTrue(abs(px[0] - px[1]) <= 2 and abs(px[1] - px[2]) <= 2, px)   # 3 equal channels (JPEG)
        s0 = self.run_build(["--gray-frac", "0", "--crop-frac", "0"])
        self.assertEqual((s0["grayscale"], s0["cropped_200"]), (0, 0))

    def test_deterministic(self):
        a = [(r["file"], r["split"], r["sha256"]) for r in self.run_build()["images"]]
        b = [(r["file"], r["split"], r["sha256"]) for r in self.run_build()["images"]]
        self.assertEqual(a, b)

    def test_too_few_images_raises(self):
        with self.assertRaises(RuntimeError):
            self.run_build(["--n-coco", "31"])


class ReorderPermutationTests(unittest.TestCase):
    def test_alphabetical_to_contract(self):
        from reorder_cls_classes import permutation
        alpha = sorted(CLS7_CLASSES)
        self.assertEqual(alpha.index("not_steel"), 2)
        perm = permutation(alpha, CLS7_CLASSES)
        self.assertEqual([alpha[i] for i in perm], CLS7_CLASSES)
        self.assertEqual(perm, [0, 1, 3, 4, 5, 6, 2])

    def test_identity_and_errors(self):
        from reorder_cls_classes import names_list, permutation
        self.assertEqual(permutation(CLS7_CLASSES, CLS7_CLASSES), list(range(7)))
        self.assertEqual(names_list({1: "b", 0: "a"}), ["a", "b"])
        with self.assertRaises(ValueError):
            permutation(NEU_CLASSES, CLS7_CLASSES)


P12_JSON = ROOT / "results" / "phase1_2" / "fusion_config.json"


class Phase12TunedConfigTests(unittest.TestCase):
    """FUSION_CONFIG must hold exactly the thresholds the Phase 1.2 run chose on val."""

    def setUp(self):
        import json
        self.tuned = json.loads(P12_JSON.read_text(encoding="utf-8"))

    def test_selected_on_val(self):
        self.assertEqual(self.tuned["selected_on"], "val")
        self.assertEqual(self.tuned["run_name"], "p12_cls7_notsteel_s42")

    def test_thresholds_match_json(self):
        self.assertEqual(FUSION_CONFIG["not_steel_threshold"], self.tuned["not_steel_threshold"])
        self.assertEqual(FUSION_CONFIG["cls_threshold"], self.tuned["cls_threshold"])
        self.assertEqual(FUSION_CONFIG["det_class_min_confidence"], self.tuned["det_class_min_confidence"])
        self.assertEqual(self.tuned["classifier_classes"], CLS7_CLASSES)

    def test_explicit_values(self):
        self.assertEqual(FUSION_CONFIG["not_steel_threshold"], 0.33)
        self.assertEqual(FUSION_CONFIG["cls_threshold"], 0.64)

    def test_tuned_gate(self):
        self.assertEqual(fuse([], clf(NOT_STEEL, 0.33), W, H).decision, RETAKE)
        self.assertIsNone(fuse([], clf(NOT_STEEL, 0.3299), W, H).decision)

if __name__ == "__main__":
    unittest.main()
