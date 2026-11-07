"""Unit tests for phase1-vision/fusion.py: one test per branch of the rule, plus edge cases.

Run from the repo root:
    python -m unittest discover -s phase1-vision/tests -v
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion import FUSION_CONFIG, NEU_CLASSES, NO_DEFECT, REVIEW, classifier_top1, fuse  # noqa: E402
from severity import SEVERITY_CONFIG  # noqa: E402

W = H = 200  # NEU image size, area 40,000 px


def box(ratio, y0=0.0):
    """Full-width box whose area is ratio * W * H."""
    return [0.0, y0, float(W), y0 + ratio * H]


def det(cls, conf, ratio=0.01, y0=0.0):
    return {"class_name": cls, "confidence": conf, "bbox_xyxy": box(ratio, y0)}


def clf(cls, conf):
    return {"class_name": cls, "confidence": conf}


def cfg(**over):
    c = copy.deepcopy(FUSION_CONFIG)
    c.update(over)
    return c


# Fixed test thresholds, independent of the tuned FUSION_CONFIG values (those are pinned to
# results/phase1_1/fusion_config.json by TunedConfigTests below).
CFG = cfg(cls_threshold=0.80, det_min_confidence=0.50,
          det_class_min_confidence={"crazing": 0.25, "inclusion": 0.50, "patches": 0.50,
                                    "pitted_surface": 0.50, "rolled-in_scale": 0.50, "scratches": 0.50})
TUNED_JSON = Path(__file__).resolve().parents[1] / "results" / "phase1_1" / "fusion_config.json"


class ClassifierBranchTests(unittest.TestCase):
    def test_classifier_confident_with_matching_box_gives_severity(self):
        r = fuse([det("scratches", 0.9, ratio=0.10)], clf("scratches", 0.95), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert), ("scratches", "classifier", True))
        self.assertEqual(r.severity_level, "Medium")          # 0.10 is in [0.082, 0.184)
        self.assertEqual(len(r.final_class_detections), 1)
        self.assertEqual(r.severity["defect_count"], 1)

    def test_classifier_overrides_detector_class(self):
        dets = [det("inclusion", 0.95, 0.40), det("patches", 0.6, 0.01, y0=150)]
        r = fuse(dets, clf("patches", 0.9), W, H, CFG)
        self.assertEqual((r.final_class, r.source), ("patches", "classifier"))
        # severity only on the patches box (Low), not the large inclusion box (Critical)
        self.assertEqual(r.severity_level, "Low")
        self.assertEqual([d["class_name"] for d in r.final_class_detections], ["patches"])
        self.assertEqual(len(r.kept_detections), 2)

    def test_classifier_confident_no_box_of_that_class_is_review(self):
        r = fuse([det("inclusion", 0.9)], clf("crazing", 0.85), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert, r.severity_level),
                         ("crazing", "classifier", True, REVIEW))
        self.assertIn("operator must check", r.reason)
        self.assertIn("inclusion", r.reason)
        self.assertIsNone(r.severity)

    def test_classifier_confident_no_boxes_at_all_is_review(self):
        r = fuse([], clf("rolled-in_scale", 0.99), W, H, CFG)
        self.assertEqual((r.final_class, r.alert, r.severity_level), ("rolled-in_scale", True, REVIEW))
        self.assertIn("no box at all", r.reason)

    def test_classifier_box_below_threshold_is_review(self):
        # a pitted_surface box exists but under its 0.50 threshold -> not kept -> Review
        r = fuse([det("pitted_surface", 0.49)], clf("pitted_surface", 0.9), W, H, CFG)
        self.assertEqual(r.severity_level, REVIEW)
        self.assertEqual(r.kept_detections, [])

    def test_classifier_threshold_is_inclusive(self):
        self.assertEqual(fuse([], clf("patches", 0.80), W, H, CFG).source, "classifier")
        self.assertEqual(fuse([], clf("patches", 0.7999), W, H, CFG).source, "none")


class DetectorBranchTests(unittest.TestCase):
    def test_classifier_unsure_uses_top_kept_detection(self):
        dets = [det("inclusion", 0.6, 0.01), det("scratches", 0.9, 0.20, y0=100)]
        r = fuse(dets, clf("inclusion", 0.5), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert), ("scratches", "detector", True))
        self.assertEqual(r.severity_level, "High")            # 0.20 in [0.184, 0.359]

    def test_classifier_not_run_uses_detector(self):
        r = fuse([det("patches", 0.7)], None, W, H, CFG)
        self.assertEqual((r.final_class, r.source), ("patches", "detector"))
        self.assertIsNone(r.classifier_confidence)
        self.assertIn("not run", r.reason)

    def test_per_class_detection_threshold(self):
        # crazing kept at 0.25, scratches needs 0.50
        r = fuse([det("crazing", 0.25), det("scratches", 0.45)], clf("crazing", 0.1), W, H, CFG)
        self.assertEqual(r.final_class, "crazing")
        self.assertEqual([d["class_name"] for d in r.kept_detections], ["crazing"])
        r = fuse([det("crazing", 0.2499)], clf("crazing", 0.1), W, H, CFG)
        self.assertEqual(r.final_class, NO_DEFECT)

    def test_detection_threshold_is_inclusive(self):
        self.assertEqual(fuse([det("scratches", 0.50)], None, W, H, CFG).final_class, "scratches")

    def test_unlisted_class_threshold_uses_default(self):
        c = cfg(cls_threshold=0.8, det_min_confidence=0.7,
                det_class_min_confidence={"crazing": 0.25})
        self.assertEqual(fuse([det("patches", 0.65)], None, W, H, c).final_class, NO_DEFECT)
        self.assertEqual(fuse([det("patches", 0.70)], None, W, H, c).final_class, "patches")

    def test_count_escalation_uses_only_final_class_boxes(self):
        # 2 scratches + 2 inclusion kept: final class scratches has only 2 boxes -> no escalation
        dets = [det("scratches", 0.9, 0.01, 0), det("scratches", 0.85, 0.01, 20),
                det("inclusion", 0.8, 0.01, 40), det("inclusion", 0.8, 0.01, 60)]
        r = fuse(dets, clf("scratches", 0.3), W, H, CFG)
        self.assertEqual(r.final_class, "scratches")
        self.assertEqual(r.severity_level, "Low")
        self.assertFalse(r.severity["escalated"])
        # a 3rd scratches box escalates Low -> Medium
        r = fuse(dets + [det("scratches", 0.8, 0.01, 80)], clf("scratches", 0.3), W, H, CFG)
        self.assertEqual(r.severity_level, "Medium")
        self.assertTrue(r.severity["escalated"])

    def test_tie_on_confidence_picks_lower_class_index(self):
        dets = [det("scratches", 0.7), det("inclusion", 0.7, y0=100)]
        self.assertEqual(fuse(dets, None, W, H, CFG).final_class, "inclusion")
        self.assertEqual(fuse(list(reversed(dets)), None, W, H, CFG).final_class, "inclusion")


class NoDefectBranchTests(unittest.TestCase):
    def test_nothing_at_all(self):
        r = fuse([], clf("crazing", 0.4), W, H, CFG)
        self.assertEqual((r.final_class, r.source, r.alert, r.severity_level), (NO_DEFECT, "none", False, None))
        self.assertEqual(r.final_class, "no defect found")

    def test_only_detections_below_threshold(self):
        r = fuse([det("inclusion", 0.3), det("crazing", 0.1)], clf("inclusion", 0.5), W, H, CFG)
        self.assertEqual((r.final_class, r.alert, r.severity_level), (NO_DEFECT, False, None))
        self.assertEqual(r.kept_detections, [])

    def test_no_classifier_no_detections(self):
        r = fuse([], None, W, H, CFG)
        self.assertEqual((r.final_class, r.alert), (NO_DEFECT, False))


class AlertTests(unittest.TestCase):
    def test_alert_truth_table(self):
        cases = [  # (dets, classifier, expected alert)
            ([], clf("patches", 0.1), False),
            ([], clf("patches", 0.9), True),
            ([det("patches", 0.9)], clf("patches", 0.1), True),
            ([det("patches", 0.9)], clf("patches", 0.9), True),
            ([det("patches", 0.1)], clf("patches", 0.1), False),
        ]
        for dets, c, want in cases:
            with self.subTest(dets=dets, c=c):
                self.assertEqual(fuse(dets, c, W, H, CFG).alert, want)


class DeterminismAndConfigTests(unittest.TestCase):
    def test_input_order_does_not_matter(self):
        dets = [det("scratches", 0.9, 0.01, 0), det("inclusion", 0.6, 0.2, 50), det("scratches", 0.7, 0.05, 120)]
        a = fuse(dets, clf("scratches", 0.5), W, H, CFG).to_dict()
        b = fuse(list(reversed(dets)), clf("scratches", 0.5), W, H, CFG).to_dict()
        self.assertEqual(a, b)

    def test_default_config_used_when_none(self):
        r = fuse([], clf("patches", FUSION_CONFIG["cls_threshold"]), W, H)
        self.assertEqual(r.source, "classifier")

    def test_config_is_not_mutated(self):
        before = copy.deepcopy(FUSION_CONFIG)
        sev_before = copy.deepcopy(SEVERITY_CONFIG)
        fuse([det("scratches", 0.9)], clf("scratches", 0.9), W, H)
        self.assertEqual(FUSION_CONFIG, before)
        self.assertEqual(SEVERITY_CONFIG, sev_before)

    def test_severity_area_thresholds_come_from_config(self):
        c = cfg(cls_threshold=0.8)
        c["severity"]["area_thresholds"] = {"low_max": 0.001, "medium_max": 0.002, "high_max": 0.003}
        r = fuse([det("scratches", 0.9, 0.01)], None, W, H, c)
        self.assertEqual(r.severity_level, "Critical")

    def test_all_thresholds_in_one_dict(self):
        for k in ("cls_threshold", "det_min_confidence", "det_class_min_confidence", "severity"):
            self.assertIn(k, FUSION_CONFIG)
        self.assertEqual(set(FUSION_CONFIG["det_class_min_confidence"]), set(NEU_CLASSES))
        self.assertEqual(FUSION_CONFIG["classes"], NEU_CLASSES)


class TunedConfigTests(unittest.TestCase):
    """FUSION_CONFIG must hold exactly the thresholds the Phase 1.1 run chose on val."""

    def setUp(self):
        import json
        self.tuned = json.loads(TUNED_JSON.read_text(encoding="utf-8"))

    def test_selected_on_val(self):
        self.assertEqual(self.tuned["selected_on"], "val")
        self.assertEqual(self.tuned["run_name"], "p11_split701515_det320_cls224_s42")

    def test_thresholds_match_json(self):
        self.assertEqual(FUSION_CONFIG["cls_threshold"], self.tuned["cls_threshold"])
        self.assertEqual(FUSION_CONFIG["det_min_confidence"], self.tuned["det_min_confidence"])
        self.assertEqual(FUSION_CONFIG["det_class_min_confidence"], self.tuned["det_class_min_confidence"])

    def test_explicit_values(self):
        self.assertEqual(FUSION_CONFIG["cls_threshold"], 0.64)
        self.assertEqual(FUSION_CONFIG["det_class_min_confidence"],
                         {"crazing": 0.10, "inclusion": 0.50, "patches": 0.55, "pitted_surface": 0.15,
                          "rolled-in_scale": 0.15, "scratches": 0.30})

    def test_tuned_config_branches(self):
        # classifier at 0.64 fires, 0.6399 does not
        self.assertEqual(fuse([], clf("patches", 0.64), W, H).source, "classifier")
        self.assertEqual(fuse([], clf("patches", 0.6399), W, H).source, "none")
        # low crazing detector threshold: a 0.10 crazing box is kept and used when the classifier is unsure
        r = fuse([det("crazing", 0.10)], clf("inclusion", 0.5), W, H)
        self.assertEqual((r.final_class, r.source), ("crazing", "detector"))
        self.assertEqual(fuse([det("crazing", 0.0999)], clf("inclusion", 0.5), W, H).final_class, NO_DEFECT)
        # patches needs 0.55
        self.assertEqual(fuse([det("patches", 0.54)], None, W, H).final_class, NO_DEFECT)
        self.assertEqual(fuse([det("patches", 0.55)], None, W, H).final_class, "patches")


class ValidationTests(unittest.TestCase):
    def test_unknown_detection_class_raises(self):
        with self.assertRaises(ValueError):
            fuse([det("rust", 0.9)], None, W, H, CFG)

    def test_unknown_classifier_class_raises(self):
        with self.assertRaises(ValueError):
            fuse([], clf("rust", 0.9), W, H, CFG)

    def test_confidence_out_of_range_raises(self):
        for bad in (-0.1, 1.5, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    fuse([det("patches", bad)], None, W, H, CFG)
                with self.assertRaises(ValueError):
                    fuse([], clf("patches", bad), W, H, CFG)

    def test_bad_image_size_raises(self):
        with self.assertRaises(ValueError):
            fuse([], None, 0, H, CFG)


class ClassifierTop1Tests(unittest.TestCase):
    def test_top1(self):
        r = classifier_top1([0.01, 0.02, 0.9, 0.03, 0.02, 0.02])
        self.assertEqual((r["class_name"], r["class_id"]), ("patches", 2))
        self.assertAlmostEqual(r["confidence"], 0.9)

    def test_tie_picks_lower_index(self):
        self.assertEqual(classifier_top1([0.0, 0.5, 0.0, 0.0, 0.5, 0.0])["class_name"], "inclusion")

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            classifier_top1([0.5, 0.5])

    def test_feeds_fuse(self):
        r = fuse([], classifier_top1([0, 0, 0, 0, 0.95, 0.05]), W, H, CFG)
        self.assertEqual((r.final_class, r.severity_level), ("rolled-in_scale", REVIEW))


if __name__ == "__main__":
    unittest.main()
