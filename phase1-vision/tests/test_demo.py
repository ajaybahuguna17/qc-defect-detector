"""Tests for the Phase 1 demo (demo/pipeline.py, demo/predict.py, demo/app.py).

Skipped without the demo requirements. The end-to-end tests also need the Phase 1.1 model files.
    python -m unittest discover -s phase1-vision/tests -v
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PHASE1_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = PHASE1_DIR / "demo"
EXAMPLES_DIR = DEMO_DIR / "examples"
sys.path.insert(0, str(DEMO_DIR))

try:
    import pipeline
    import predict
    from fusion import FUSION_CONFIG, NEU_CLASSES, fuse
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False


def models_present():
    return HAVE_DEPS and all(p.is_file() for p in pipeline.required_files(["fp32", "int8"]))


def det(name, conf, box=(0, 0, 100, 100)):
    return {"class_name": name, "confidence": conf, "bbox_xyxy": list(box)}


def result_of(detections, classifier):
    return pipeline.to_result(fuse(detections, classifier, 200, 200), "a.jpg", "fp32", 1.0, 2.0)


CONFIDENT_SCRATCHES = {"class_name": "scratches", "confidence": 0.99}
UNSURE_SCRATCHES = {"class_name": "scratches", "confidence": 0.20}


@unittest.skipUnless(HAVE_DEPS, "demo requirements not installed")
class ModelFileTests(unittest.TestCase):
    def test_missing_models_say_where_to_download(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "MODELS_DIR", Path(tmp)):
            with self.assertRaises(pipeline.ModelFilesError) as ctx:
                pipeline.DefectPipeline()
        message = str(ctx.exception)
        self.assertIn(pipeline.DETECTOR_FILES["fp32"], message)
        self.assertIn(pipeline.CLASSIFIER_FILE, message)
        self.assertIn(pipeline.DRIVE_DIR, message)

    def test_one_missing_file_is_not_replaced_by_the_other_detector(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "MODELS_DIR", Path(tmp)):
            for path in pipeline.required_files(["fp32", "int8"]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(NEU_CLASSES) if path.name == "labels.txt" else "x")
            (Path(tmp) / "detector" / pipeline.DETECTOR_FILES["int8"]).unlink()
            with self.assertRaises(pipeline.ModelFilesError) as ctx:
                pipeline.check_model_files(["int8"])
        self.assertIn(pipeline.DETECTOR_FILES["int8"], str(ctx.exception))

    def test_wrong_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "MODELS_DIR", Path(tmp)):
            for path in pipeline.required_files(["fp32"]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("person\ncar\n" if path.name == "labels.txt" else "x")
            with self.assertRaises(pipeline.ModelFilesError):
                pipeline.check_model_files(["fp32"])

    def test_cli_exits_2_with_the_message_when_models_are_missing(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "MODELS_DIR", Path(tmp)):
            with contextlib.redirect_stderr(stderr):
                code = predict.main([str(DEMO_DIR / "README.md")])
        self.assertEqual(code, 2)
        self.assertIn(pipeline.DRIVE_DIR, stderr.getvalue())

    def test_cli_exits_2_on_a_bad_path(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(predict.main(["no/such/path.jpg"]), 2)
        self.assertIn("no such file", stderr.getvalue())


@unittest.skipUnless(HAVE_DEPS, "demo requirements not installed")
class ResultTests(unittest.TestCase):
    def test_decode_conf_is_below_every_fusion_threshold(self):
        self.assertLess(pipeline.DECODE_CONF, min(FUSION_CONFIG["det_class_min_confidence"].values()))
        self.assertLess(pipeline.DECODE_CONF, FUSION_CONFIG["det_min_confidence"])

    def test_every_class_has_a_colour(self):
        self.assertEqual(set(pipeline.CLASS_COLOURS), set(NEU_CLASSES))

    def test_decided_by_classifier(self):
        r = result_of([det("scratches", 0.9)], CONFIDENT_SCRATCHES)
        self.assertEqual((r.decided_by, r.final_class, r.num_boxes, r.num_final_boxes), ("classifier", "scratches", 1, 1))
        self.assertAlmostEqual(r.largest_area_pct, 25.0)

    def test_decided_by_detector(self):
        r = result_of([det("patches", 0.9)], UNSURE_SCRATCHES)
        self.assertEqual((r.decided_by, r.final_class), ("detector", "patches"))

    def test_decided_by_review_has_boxes_but_no_area(self):
        r = result_of([det("patches", 0.9)], CONFIDENT_SCRATCHES)
        self.assertEqual((r.decided_by, r.severity_level, r.num_boxes, r.num_final_boxes), ("review", "Review", 1, 0))
        self.assertIsNone(r.largest_area_pct)
        self.assertIn("n/a", pipeline.format_card(r))

    def test_decided_by_none(self):
        r = result_of([], UNSURE_SCRATCHES)
        self.assertEqual((r.decided_by, r.alert, r.severity_level), ("none", False, None))
        self.assertEqual(r.final_class, FUSION_CONFIG["no_defect_label"])

    def test_box_below_its_class_threshold_is_not_counted(self):
        r = result_of([det("inclusion", 0.20), det("scratches", 0.9)], CONFIDENT_SCRATCHES)
        self.assertEqual(r.num_boxes, 1)

    def test_card_has_every_field(self):
        card = pipeline.format_card(result_of([det("scratches", 0.9)], CONFIDENT_SCRATCHES))
        for text in ("Alert:", "YES", "Final defect type:", "Decided by:", "classifier", "0.990", "Boxes:",
                     "Largest area:", "25.0 %", "Severity:", "Reason:", "detector 1.0 ms", "classifier 2.0 ms"):
            self.assertIn(text, card)
        card.encode("ascii")  # safe on any Windows console


@unittest.skipUnless(HAVE_DEPS, "demo requirements not installed")
class AnnotateTests(unittest.TestCase):
    def test_small_image_is_enlarged_and_box_scaled_with_its_class_colour(self):
        from PIL import Image, ImageColor
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            Image.new("L", (200, 100), 128).save(path)
            out = pipeline.annotate(path, [det("patches", 0.9, (50, 25, 150, 75))])
        self.assertEqual(out.size, (640, 320))
        self.assertEqual(out.getpixel((320, 240)), ImageColor.getrgb(pipeline.CLASS_COLOURS["patches"]))  # bottom edge


class IndependenceTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_DEPS, "demo requirements not installed")
    def test_no_torch_or_ultralytics_at_runtime(self):
        code = ("import sys; sys.path.insert(0, r'%s'); import pipeline, predict; "
                "bad = {'torch', 'ultralytics'} & set(sys.modules); sys.exit(repr(bad) if bad else 0)" % DEMO_DIR)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_app_never_shares_and_binds_to_localhost(self):
        source = (DEMO_DIR / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("share=True", source)
        self.assertIn("share=False", source)
        self.assertIn('server_name="127.0.0.1"', source)

    def test_app_passes_the_original_file_to_the_model(self):
        source = (DEMO_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn('type="filepath", image_mode=None', source)


class ExampleFileTests(unittest.TestCase):
    def test_two_examples_per_class_from_the_test_split(self):
        names = sorted(p.name for p in EXAMPLES_DIR.iterdir())
        self.assertEqual(len(names), 12)
        for cls in ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"):
            self.assertEqual(sum(n.rsplit("_", 1)[0] == cls for n in names), 2, cls)
        images = PHASE1_DIR / "data" / "dataset" / "images"
        if not (images / "test" / names[0]).is_file():
            self.skipTest("dataset not prepared")
        for name in names:
            self.assertEqual((images / "test" / name).read_bytes(), (EXAMPLES_DIR / name).read_bytes(), name)
            self.assertFalse((images / "train" / name).exists(), name)
            self.assertFalse((images / "val" / name).exists(), name)


@unittest.skipUnless(models_present(), "Phase 1.1 model files not available")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = pipeline.DefectPipeline()

    def test_all_examples_give_the_right_defect_type(self):
        for detector in ("fp32", "int8"):
            for path in sorted(EXAMPLES_DIR.glob("*.jpg")):
                with self.subTest(detector=detector, image=path.name):
                    r = self.pipeline.analyze(path, detector)
                    self.assertEqual(r.final_class, path.stem.rsplit("_", 1)[0])
                    self.assertTrue(r.alert)
                    self.assertGreater(r.detector_ms, 0.0)
                    self.assertGreater(r.classifier_ms, 0.0)

    def test_low_confidence_crazing_boxes_survive_decoding(self):
        """Regression: decoding at the reference default 0.25 would lose boxes fusion keeps at 0.10."""
        seen = [box["confidence"] for path in EXAMPLES_DIR.glob("crazing_*.jpg")
                for box in self.pipeline.analyze(path, "fp32").boxes]
        self.assertTrue(seen)
        self.assertGreaterEqual(min(seen), FUSION_CONFIG["det_class_min_confidence"]["crazing"])


if __name__ == "__main__":
    unittest.main()
