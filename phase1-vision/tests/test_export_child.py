"""Tests for the export timeout / skip logic (scripts/export_child.py) and the classifier export CLI
rules (scripts/export_cls_tflite.py). No Ultralytics, no model: the child is replaced by tiny
Python commands that succeed, fail, or hang.

    python -m unittest discover -s phase1-vision/tests -v
"""

import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_child  # noqa: E402
from export_child import (ExportOutcome, available_precisions, resolve_outcome, run_child,  # noqa: E402
                          run_export, skipped_entry)

PY = sys.executable


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class RunChildTests(unittest.TestCase):
    def test_ok(self):
        with quiet():
            r = run_child([PY, "-c", "print('hello'); print('done')"], timeout_s=30)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["returncode"], 0)
        self.assertEqual(r["tail"], ["hello", "done"])

    def test_failed(self):
        with quiet():
            r = run_child([PY, "-c", "import sys; print('boom'); sys.exit(3)"], timeout_s=30)
        self.assertEqual((r["status"], r["returncode"]), ("failed", 3))
        self.assertIn("boom", r["tail"])

    def test_timeout_kills_hung_child(self):
        t0 = time.monotonic()
        with quiet():
            r = run_child([PY, "-c", "import time; print('calibrating', flush=True); time.sleep(120)"], timeout_s=2)
        self.assertEqual(r["status"], "timeout")
        self.assertLess(time.monotonic() - t0, 30)      # killed, not waited for
        self.assertIn("calibrating", r["tail"])

    def test_tail_is_bounded(self):
        with quiet():
            r = run_child([PY, "-c", "[print(i) for i in range(500)]"], timeout_s=30, tail_lines=5)
        self.assertEqual(r["tail"], ["495", "496", "497", "498", "499"])


class RunExportTests(unittest.TestCase):
    """run_export with the real child replaced by a fake command."""

    def run_fake(self, code, timeout_min=0.05, precision="int8"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path(tmp.name)

        def fake_cmd(weights, imgsz, int8, data, fraction, result):
            return [PY, "-c", code.replace("RESULT", str(result).replace("\\", "/"))
                    .replace("OUT", str(work / "m.tflite").replace("\\", "/"))]

        with mock.patch.object(export_child, "child_cmd", fake_cmd), quiet():
            return run_export(Path("w.pt"), 224, precision, Path("calib"), 0.3, timeout_min, work, "classifier")

    def test_ok_with_result_file(self):
        code = ("import json, pathlib; pathlib.Path('OUT').write_bytes(b'x'); "
                "pathlib.Path('RESULT').write_text(json.dumps({'output': 'OUT'}))")
        out = self.run_fake(code, timeout_min=1, precision="fp32")
        self.assertEqual(out.status, "ok")
        self.assertTrue(out.output.endswith("m.tflite"))

    def test_exit_zero_without_result_is_failure(self):
        out = self.run_fake("print('nothing written')", timeout_min=1)
        self.assertEqual(out.status, "failed")
        self.assertIn("no valid result file", out.reason)

    def test_result_pointing_to_missing_file_is_failure(self):
        out = self.run_fake("import json, pathlib; pathlib.Path('RESULT').write_text(json.dumps({'output': 'OUT'}))",
                            timeout_min=1)
        self.assertEqual(out.status, "failed")
        self.assertIn("does not exist", out.reason)

    def test_crash(self):
        out = self.run_fake("raise SystemExit(7)", timeout_min=1)
        self.assertEqual((out.status, out.returncode), ("failed", 7))

    def test_hang_times_out(self):
        out = self.run_fake("import time; time.sleep(120)", timeout_min=0.05)   # 3 s
        self.assertEqual(out.status, "timeout")
        self.assertIn("timed out after 0.05 min", out.reason)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            run_export(Path("w.pt"), 224, "fp16", None, 0.3, 1, Path("."), "x")
        with self.assertRaises(ValueError):
            run_export(Path("w.pt"), 224, "fp32", None, 0.3, 0, Path("."), "x")


def outcome(precision, status, reason="r"):
    return ExportOutcome(precision=precision, status=status, seconds=900.0, returncode=None,
                         output="/x.tflite" if status == "ok" else None, reason=reason, tail=["last line"])


class PolicyTests(unittest.TestCase):
    def test_ok_returns_none(self):
        for p in ("fp32", "int8"):
            for allow in (False, True):
                self.assertIsNone(resolve_outcome(outcome(p, "ok"), allow, "classifier"))

    def test_fp32_failure_or_timeout_always_raises(self):
        for st in ("failed", "timeout"):
            for allow in (False, True):
                with self.subTest(st=st, allow=allow), self.assertRaises(RuntimeError):
                    resolve_outcome(outcome("fp32", st), allow, "classifier")

    def test_int8_failure_raises_without_flag(self):
        for st in ("failed", "timeout"):
            with self.subTest(st=st), self.assertRaises(RuntimeError) as cm:
                resolve_outcome(outcome("int8", st), False, "detector")
            self.assertIn("--allow-skip-int8", str(cm.exception))

    def test_int8_skip_with_flag_is_recorded_and_loud(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            e = resolve_outcome(outcome("int8", "timeout", "timed out after 15 min"), True, "classifier")
        self.assertEqual(e["status"], "skipped")
        self.assertEqual(e["reason"], "timed out after 15 min")
        self.assertEqual(e["export_status"], "timeout")
        self.assertRegex(e["timestamp"], r"^\d{4}-\d{2}-\d{2}T")
        self.assertIn("WARNING: classifier INT8 EXPORT SKIPPED", buf.getvalue())
        self.assertIn("!" * 50, buf.getvalue())

    def test_skipped_entry(self):
        e = skipped_entry("hung", timestamp="2026-11-07T00:00:00+00:00")
        self.assertEqual(e, {"status": "skipped", "reason": "hung", "timestamp": "2026-11-07T00:00:00+00:00"})


class AvailablePrecisionsTests(unittest.TestCase):
    def test_legacy_entries_without_status_are_ok(self):
        mio = {"models": {"fp32": {"file": "a.tflite"}, "int8": {"file": "b.tflite"}}}
        self.assertEqual(available_precisions(mio), ["fp32", "int8"])

    def test_skipped_int8_is_excluded(self):
        mio = {"models": {"fp32": {"status": "ok", "file": "a"}, "int8": skipped_entry("hung")}}
        self.assertEqual(available_precisions(mio), ["fp32"])

    def test_no_fp32_raises(self):
        with self.assertRaises(ValueError):
            available_precisions({"models": {"int8": {"file": "b"}}})
        with self.assertRaises(ValueError):
            available_precisions({"models": {"fp32": skipped_entry("x")}})

    def test_unknown_status_or_missing_file_raises(self):
        with self.assertRaises(ValueError):
            available_precisions({"models": {"fp32": {"file": "a"}, "int8": {"status": "maybe", "file": "b"}}})
        with self.assertRaises(ValueError):
            available_precisions({"models": {"fp32": {"status": "ok"}}})


try:
    import numpy  # noqa: F401
    from export_cls_tflite import parse_args
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


@unittest.skipUnless(HAVE_NP, "numpy not installed")
class ClsExportCliTests(unittest.TestCase):
    base = ["--weights", "w.pt", "--data", "cls"]

    def bad(self, extra):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(self.base + extra)

    def test_defaults(self):
        a = parse_args(self.base)
        self.assertEqual((a.precisions, a.timeout_min, a.allow_skip_int8), (["fp32", "int8"], 15.0, False))

    def test_timeout_option(self):
        self.assertEqual(parse_args(self.base + ["--timeout-min", "3"]).timeout_min, 3.0)
        self.bad(["--timeout-min", "0"])

    def test_fp32_required(self):
        self.bad(["--precisions", "int8"])

    def test_reason_needs_reuse_and_flag(self):
        self.bad(["--int8-skipped-reason", "hung"])
        self.bad(["--reuse-existing", "--precisions", "fp32", "--int8-skipped-reason", "hung"])
        self.bad(["--allow-skip-int8", "--precisions", "fp32", "--int8-skipped-reason", "hung"])
        self.bad(["--reuse-existing", "--allow-skip-int8", "--int8-skipped-reason", "hung"])  # int8 in precisions
        self.bad(["--reuse-existing", "--allow-skip-int8", "--precisions", "fp32", "--int8-skipped-reason", " "])

    def test_reuse_without_int8_needs_reason(self):
        self.bad(["--reuse-existing", "--precisions", "fp32"])
        self.bad(["--reuse-existing", "--precisions", "fp32", "--allow-skip-int8"])
        a = parse_args(self.base + ["--reuse-existing", "--precisions", "fp32", "--allow-skip-int8",
                                    "--int8-skipped-reason", "hung"])
        self.assertEqual((a.precisions, a.int8_skipped_reason), (["fp32"], "hung"))


if __name__ == "__main__":
    unittest.main()
