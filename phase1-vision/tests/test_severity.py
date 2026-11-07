"""Unit tests for phase1-vision/severity.py (stdlib unittest; also runs under pytest).

Run from the repo root:
    python -m unittest discover -s phase1-vision/tests -v
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from severity import LEVELS, SEVERITY_CONFIG, assess_severity  # noqa: E402

W = H = 200  # NEU-DET image size; image area = 40,000 px


def box_with_ratio(ratio, x0=0.0, y0=0.0):
    """Square-ish box (full width) whose area is exactly ratio * W * H."""
    h = ratio * H
    return [x0, y0, x0 + W, y0 + h]


def det(ratio, conf=0.9, cls="scratches", **kw):
    return {"class_name": cls, "confidence": conf, "bbox_xyxy": box_with_ratio(ratio, **kw)}


class AreaThresholdTests(unittest.TestCase):
    def level(self, ratio):
        return assess_severity([det(ratio)], W, H).level

    def test_low(self):
        self.assertEqual(self.level(0.005), "Low")
        self.assertEqual(self.level(0.0199), "Low")

    def test_medium_boundaries(self):
        self.assertEqual(self.level(0.02), "Medium")   # 0.02 is inclusive lower bound of Medium
        self.assertEqual(self.level(0.0499), "Medium")

    def test_high_boundaries(self):
        self.assertEqual(self.level(0.05), "High")
        self.assertEqual(self.level(0.10), "High")     # 0.10 itself is still High

    def test_critical(self):
        self.assertEqual(self.level(0.1001), "Critical")
        self.assertEqual(self.level(0.8), "Critical")


class ConfidenceFilterTests(unittest.TestCase):
    def test_below_threshold_ignored(self):
        r = assess_severity([det(0.5, conf=0.49)], W, H)
        self.assertIsNone(r.level)
        self.assertEqual(r.defect_count, 0)

    def test_threshold_is_inclusive(self):
        r = assess_severity([det(0.01, conf=0.5)], W, H)
        self.assertEqual(r.level, "Low")
        self.assertEqual(r.defect_count, 1)

    def test_low_conf_detections_do_not_count_toward_escalation(self):
        dets = [det(0.01), det(0.01, conf=0.3), det(0.01, conf=0.2)]
        r = assess_severity(dets, W, H)
        self.assertEqual(r.defect_count, 1)
        self.assertFalse(r.escalated)
        self.assertEqual(r.level, "Low")

    def test_empty(self):
        r = assess_severity([], W, H)
        self.assertIsNone(r.level)
        self.assertEqual(r.max_area_ratio, 0.0)


class ClassMinConfidenceTests(unittest.TestCase):
    def test_config_values(self):
        cmc = SEVERITY_CONFIG["class_min_confidence"]
        self.assertEqual(cmc["crazing"], 0.30)
        for name in ("inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"):
            self.assertEqual(cmc[name], 0.50, name)
        self.assertEqual(SEVERITY_CONFIG["min_confidence"], 0.50)

    def test_crazing_kept_at_0_39(self):
        # Real case from the Colab run: crazing_90.jpg, conf 0.39.
        r = assess_severity([det(0.03, conf=0.39, cls="crazing")], W, H)
        self.assertEqual(r.level, "Medium")
        self.assertEqual(r.defect_count, 1)

    def test_crazing_threshold_inclusive(self):
        self.assertEqual(assess_severity([det(0.01, conf=0.30, cls="crazing")], W, H).defect_count, 1)
        self.assertIsNone(assess_severity([det(0.01, conf=0.29, cls="crazing")], W, H).level)

    def test_other_classes_still_need_0_5(self):
        for name in ("inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"):
            self.assertIsNone(assess_severity([det(0.03, conf=0.39, cls=name)], W, H).level, name)

    def test_unknown_class_uses_global_min(self):
        self.assertIsNone(assess_severity([det(0.03, conf=0.45, cls="unknown")], W, H).level)
        self.assertEqual(assess_severity([det(0.03, conf=0.5, cls="unknown")], W, H).defect_count, 1)

    def test_low_conf_crazing_counts_toward_escalation(self):
        dets = [det(0.01, conf=0.35, cls="crazing")] * 2 + [det(0.01, conf=0.35, cls="scratches")]
        r = assess_severity(dets, W, H)
        self.assertEqual(r.defect_count, 2)
        self.assertFalse(r.escalated)
        r = assess_severity([det(0.01, conf=0.35, cls="crazing")] * 3, W, H)
        self.assertTrue(r.escalated)
        self.assertEqual(r.level, "Medium")

    def test_config_without_class_dict_falls_back(self):
        cfg = copy.deepcopy(SEVERITY_CONFIG)
        del cfg["class_min_confidence"]
        self.assertIsNone(assess_severity([det(0.03, conf=0.39, cls="crazing")], W, H, config=cfg).level)


class CountEscalationTests(unittest.TestCase):
    def test_two_defects_no_escalation(self):
        r = assess_severity([det(0.01), det(0.01)], W, H)
        self.assertEqual(r.level, "Low")
        self.assertFalse(r.escalated)

    def test_three_defects_escalate_one_level(self):
        r = assess_severity([det(0.01), det(0.01), det(0.03)], W, H)
        self.assertEqual(r.level, "High")  # worst is Medium, +1 -> High
        self.assertTrue(r.escalated)
        self.assertEqual(r.defect_count, 3)

    def test_escalation_capped_at_critical(self):
        r = assess_severity([det(0.2), det(0.01), det(0.01)], W, H)
        self.assertEqual(r.level, "Critical")
        self.assertFalse(r.escalated)


class AggregationAndGeometryTests(unittest.TestCase):
    def test_image_level_is_worst_detection(self):
        r = assess_severity([det(0.01), det(0.07)], W, H)
        self.assertEqual(r.level, "High")
        self.assertAlmostEqual(r.max_area_ratio, 0.07, places=6)

    def test_bbox_clipped_to_image(self):
        # box twice the image size -> clipped to full image -> ratio 1.0
        r = assess_severity([{"class_name": "patches", "confidence": 0.9,
                              "bbox_xyxy": [-200, -200, 400, 400]}], W, H)
        self.assertAlmostEqual(r.detections[0].area_ratio, 1.0)
        self.assertEqual(r.level, "Critical")

    def test_inverted_bbox_handled(self):
        r = assess_severity([{"class_name": "patches", "confidence": 0.9,
                              "bbox_xyxy": [100, 100, 0, 0]}], W, H)
        self.assertAlmostEqual(r.detections[0].area_ratio, 0.25)

    def test_non_square_image(self):
        # 640x480 image, 64x48 box -> ratio 0.01 -> Low
        r = assess_severity([{"class_name": "inclusion", "confidence": 0.8,
                              "bbox_xyxy": [0, 0, 64, 48]}], 640, 480)
        self.assertAlmostEqual(r.detections[0].area_ratio, 0.01)
        self.assertEqual(r.level, "Low")

    def test_invalid_image_size(self):
        with self.assertRaises(ValueError):
            assess_severity([det(0.01)], 0, 200)

    def test_deterministic(self):
        dets = [det(0.03), det(0.01, cls="inclusion"), det(0.06, cls="patches")]
        self.assertEqual(assess_severity(dets, W, H).to_dict(), assess_severity(dets, W, H).to_dict())


class ConfigTests(unittest.TestCase):
    def test_class_weight_changes_level(self):
        cfg = copy.deepcopy(SEVERITY_CONFIG)
        cfg["class_area_weight"]["scratches"] = 3.0
        r = assess_severity([det(0.02)], W, H, config=cfg)  # 0.02 * 3 = 0.06 -> High
        self.assertEqual(r.level, "High")
        # default config untouched
        self.assertEqual(assess_severity([det(0.02)], W, H).level, "Medium")

    def test_default_class_weights_are_neutral(self):
        self.assertTrue(all(v == 1.0 for v in SEVERITY_CONFIG["class_area_weight"].values()))

    def test_levels_order(self):
        self.assertEqual(LEVELS, ("Low", "Medium", "High", "Critical"))

    def test_result_serialisable(self):
        import json
        json.dumps(assess_severity([det(0.03)], W, H).to_dict())


if __name__ == "__main__":
    unittest.main()
