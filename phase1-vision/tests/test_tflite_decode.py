"""Unit tests for the reference YOLOv8 decoder in scripts/tflite_utils.py.

Needs numpy only (no TFLite runtime, no model). Skipped automatically if numpy is missing.
    python -m unittest discover -s phase1-vision/tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import numpy as np
    from tflite_utils import (Letterbox, ModelIO, NEU_CLASSES, assert_defect_model, decode_yolov8,
                              letterbox, match_detections)
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


def make_io(layout="NCHW", s=320, n=2100, nc=6):
    shape = [1, 3, s, s] if layout == "NCHW" else [1, s, s, 3]
    return ModelIO(shape, "float32", layout, (s, s), (0.0, 0), [1, 4 + nc, n], "float32", (0.0, 0), nc, n)


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.io = make_io()
        self.lb = Letterbox(scale=1.6, pad_x=0, pad_y=0, orig_w=200, orig_h=200)  # 200x200 -> 320x320

    def raw_with(self, anchors):
        raw = np.zeros((1, 10, 2100), np.float32)
        for i, (cx, cy, w, h, cls, conf) in enumerate(anchors):
            raw[0, :4, i] = [cx, cy, w, h]
            raw[0, 4 + cls, i] = conf
        return raw

    def test_normalized_box_back_to_original_pixels(self):
        # box centred in the 320 input, 0.25 x 0.5 of it -> in 200px original: 50 x 100 centred at 100,100
        raw = self.raw_with([(0.5, 0.5, 0.25, 0.5, 5, 0.9)])
        dets = decode_yolov8(raw, self.io, self.lb, NEU_CLASSES, conf_thr=0.25)
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d["class_name"], "scratches")
        np.testing.assert_allclose(d["bbox_xyxy"], [75, 50, 125, 150], atol=1e-3)
        self.assertAlmostEqual(d["confidence"], 0.9, places=5)

    def test_conf_threshold(self):
        raw = self.raw_with([(0.5, 0.5, 0.2, 0.2, 0, 0.2)])
        self.assertEqual(decode_yolov8(raw, self.io, self.lb, conf_thr=0.25), [])

    def test_nms_same_class_suppressed(self):
        raw = self.raw_with([(0.5, 0.5, 0.3, 0.3, 1, 0.9), (0.51, 0.5, 0.3, 0.3, 1, 0.8)])
        dets = decode_yolov8(raw, self.io, self.lb)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0]["confidence"], 0.9, places=5)

    def test_nms_is_class_aware(self):
        raw = self.raw_with([(0.5, 0.5, 0.3, 0.3, 1, 0.9), (0.5, 0.5, 0.3, 0.3, 2, 0.8)])
        self.assertEqual(len(decode_yolov8(raw, self.io, self.lb)), 2)

    def test_letterbox_padding_undone(self):
        io = make_io()
        rgb = np.zeros((100, 200, 3), np.uint8)   # wide image -> pad top/bottom
        canvas, lb = letterbox(rgb, (320, 320))
        self.assertEqual(canvas.shape, (320, 320, 3))
        self.assertAlmostEqual(lb.scale, 1.6)
        self.assertEqual(lb.pad_y, 80)
        raw = self.raw_with([(0.5, 0.5, 0.5, 0.25, 0, 0.9)])  # 160x80 in input px, centred
        d = decode_yolov8(raw, io, lb)[0]
        np.testing.assert_allclose(d["bbox_xyxy"], [50, 25, 150, 75], atol=1e-3)

    def test_wrong_class_count_rejected(self):
        with self.assertRaises(AssertionError):
            assert_defect_model(make_io(nc=80, n=8400))
        assert_defect_model(make_io())  # 6 classes ok

    def test_match_detections(self):
        a = [{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]
        b = [{"class_id": 0, "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 11]},
             {"class_id": 1, "confidence": 0.7, "bbox_xyxy": [0, 0, 10, 10]}]
        m = match_detections(a, b)
        self.assertEqual(m["matched"], 1)
        self.assertAlmostEqual(m["mean_abs_conf_diff"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
