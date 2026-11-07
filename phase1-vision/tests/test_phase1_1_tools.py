"""Tests for the Phase 1.1 tooling: split sizes / overlap check (prepare_data.py), class from file
name (build_cls_dataset.py), threshold sweep + evaluation (fusion_eval.py), classifier
pre-processing (tflite_utils.py, needs numpy + pillow; skipped without them).

    python -m unittest discover -s phase1-vision/tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from prepare_data import SPLIT_PRESETS, assert_no_overlap, split_sizes  # noqa: E402
from build_cls_dataset import build, class_from_filename  # noqa: E402
from fusion_eval import (choose_cls_threshold, choose_det_thresholds, evaluate, make_config,  # noqa: E402
                         pick_plateau_centre)

try:
    import numpy as np
    from PIL import Image  # noqa: F401
    from tflite_utils import ClsIO, assert_classifier_model, cls_input_tensor, cls_preprocess
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


class SplitTests(unittest.TestCase):
    def test_neu_701515_counts(self):
        self.assertEqual(split_sizes(300, SPLIT_PRESETS["70/15/15"]), (210, 45, 45))

    def test_legacy_8020_counts(self):
        self.assertEqual(split_sizes(300, SPLIT_PRESETS["80/20"]), (240, 60, 0))

    def test_sizes_always_sum(self):
        for n in range(0, 40):
            for ratios in SPLIT_PRESETS.values():
                self.assertEqual(sum(split_sizes(n, ratios)), n)

    def test_overlap_between_any_two_splits_fails(self):
        a, b, c = (Path("crazing_1.jpg"), None), (Path("crazing_2.jpg"), None), (Path("crazing_3.jpg"), None)
        assert_no_overlap({"train": [a], "val": [b], "test": [c]})
        for bad in ({"train": [a], "val": [a], "test": [c]},
                    {"train": [a], "val": [b], "test": [a]},
                    {"train": [a], "val": [b], "test": [b]},
                    {"train": [a, a], "val": [b], "test": [c]}):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    assert_no_overlap(bad)


class ClassFromFilenameTests(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(class_from_filename("crazing_12.jpg"), "crazing")
        self.assertEqual(class_from_filename("rolled-in_scale_7.jpg"), "rolled-in_scale")
        self.assertEqual(class_from_filename("pitted_surface_300.jpg"), "pitted_surface")
        self.assertEqual(class_from_filename("Scratches_1.JPG"), "scratches")

    def test_unknown_raises(self):
        for bad in ("rust_1.jpg", "crazing.jpg", "img_001.jpg"):
            with self.assertRaises(ValueError):
                class_from_filename(bad)

    def test_build_imagefolder(self):
        with tempfile.TemporaryDirectory() as t:
            ds, out = Path(t) / "dataset", Path(t) / "cls"
            files = {"train": ["crazing_1.jpg", "patches_2.jpg"], "val": ["crazing_3.jpg"], "test": ["scratches_4.jpg"]}
            for sp, names in files.items():
                (ds / "images" / sp).mkdir(parents=True)
                for n in names:
                    (ds / "images" / sp / n).write_bytes(b"x")
            s = build(ds, out)
            self.assertTrue((out / "train" / "patches" / "patches_2.jpg").is_file())
            self.assertTrue((out / "test" / "scratches" / "scratches_4.jpg").is_file())
            self.assertEqual(s["totals"], {"train": 2, "val": 1, "test": 1})
            # overlap between splits is refused
            (ds / "images" / "test" / "crazing_1.jpg").write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                build(ds, out)
            # output dir inside the detector dataset is refused (it is deleted before writing)
            with self.assertRaises(ValueError):
                build(ds, ds / "cls")


def rec(name, true, dets=(), probs=None):
    return {"image": name, "true_class": true, "width": 200, "height": 200,
            "detections": [{"class_name": c, "confidence": s, "bbox_xyxy": [0, 0, 20, 20]} for c, s in dets],
            "probs": probs or [1 / 6] * 6}


def onehot(i, p):
    v = [(1 - p) / 5] * 6
    v[i] = p
    return v


class SweepTests(unittest.TestCase):
    def test_plateau_centre(self):
        g = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(pick_plateau_centre(g, [1, 3, 3, 3, 2, 3]), 0.3)    # longest run centre
        self.assertEqual(pick_plateau_centre(g, [1, 3, 3, 1, 3, 3]), 0.2)    # equal runs -> earliest, lower middle
        self.assertEqual(pick_plateau_centre(g, [5, 1, 1, 1, 1, 1]), 0.1)

    def test_det_threshold_separates_classes(self):
        # crazing images score 0.3, a patches image has a spurious crazing box at 0.2
        rs = [rec("crazing_1.jpg", "crazing", [("crazing", 0.3)]),
              rec("crazing_2.jpg", "crazing", [("crazing", 0.35)]),
              rec("patches_1.jpg", "patches", [("crazing", 0.2), ("patches", 0.9)])]
        out = choose_det_thresholds([rs], grid=[0.1, 0.2, 0.25, 0.3, 0.4])
        self.assertEqual(out["thresholds"]["crazing"], 0.25)   # only 0.25 and 0.3 reach 2-0; lower middle
        self.assertGreater(out["thresholds"]["patches"], 0.2)

    def test_evaluate_and_cls_threshold(self):
        rs = [rec("crazing_1.jpg", "crazing", [], onehot(0, 0.9)),       # only the classifier knows
              rec("patches_1.jpg", "patches", [("patches", 0.9)], onehot(5, 0.6)),  # classifier wrong at 0.6
              rec("scratches_1.jpg", "scratches")]                        # nothing -> error
        det = {c: 0.5 for c in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]}
        ev = evaluate(rs, make_config(det, 0.8))
        self.assertEqual((ev["correct"], ev["images"]), (2, 3))
        self.assertEqual(ev["decided_by"], {"classifier": 1, "detector": 1, "none": 1})
        self.assertEqual(ev["review"], 1)                                # crazing: classifier, no box
        self.assertEqual(ev["errors"][0]["final_class"], "no defect found")
        out = choose_cls_threshold([rs], det, grid=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        self.assertEqual(out["threshold"], 0.8)                           # 0.7..0.9 all give 2/3


class ClassifierOnlyAccuracyTests(unittest.TestCase):
    def test_per_class_accuracy_present(self):
        from fusion_eval import classifier_only_accuracy
        rs = [rec("crazing_1.jpg", "crazing", probs=onehot(0, 0.9)), rec("crazing_2.jpg", "crazing", probs=onehot(1, 0.9))]
        out = classifier_only_accuracy(rs)
        self.assertEqual((out["correct"], out["accuracy"]), (1, 0.5))
        self.assertEqual(out["per_class"]["crazing"]["accuracy"], 0.5)
        self.assertIsNone(out["per_class"]["patches"]["accuracy"])


@unittest.skipUnless(HAVE_NP, "numpy / pillow not installed")
class ClsPreprocessTests(unittest.TestCase):
    def test_square_resize_range_and_shape(self):
        rgb = np.full((200, 200, 3), 255, np.uint8)
        x = cls_preprocess(rgb, 224)
        self.assertEqual(x.shape, (224, 224, 3))
        self.assertAlmostEqual(float(x.max()), 1.0)

    def test_centre_crop_keeps_middle(self):
        rgb = np.zeros((100, 300, 3), np.uint8)
        rgb[:, 100:200] = 255                          # white middle third
        x = cls_preprocess(rgb, 224)                    # -> 672x224 resize, crop the middle 224
        self.assertEqual(x.shape, (224, 224, 3))
        self.assertGreater(float(x[:, 20:200].mean()), 0.99)

    def test_layouts(self):
        x = np.zeros((224, 224, 3), np.float32)
        nchw = ClsIO([1, 3, 224, 224], "float32", "NCHW", (224, 224), (0.0, 0), [1, 6], "float32", (0.0, 0), 6)
        nhwc = ClsIO([1, 224, 224, 3], "float32", "NHWC", (224, 224), (0.0, 0), [1, 6], "float32", (0.0, 0), 6)
        self.assertEqual(cls_input_tensor(x, nchw).shape, (1, 3, 224, 224))
        self.assertEqual(cls_input_tensor(x, nhwc).shape, (1, 224, 224, 3))

    def test_output_shape_assert(self):
        good = ClsIO([1, 3, 224, 224], "float32", "NCHW", (224, 224), (0.0, 0), [1, 6], "float32", (0.0, 0), 6)
        assert_classifier_model(good)
        bad = ClsIO([1, 3, 224, 224], "float32", "NCHW", (224, 224), (0.0, 0), [1, 1000], "float32", (0.0, 0), 1000)
        with self.assertRaises(AssertionError):
            assert_classifier_model(bad)


if __name__ == "__main__":
    unittest.main()
