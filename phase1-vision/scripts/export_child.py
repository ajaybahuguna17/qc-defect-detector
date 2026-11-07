#!/usr/bin/env python3
"""
Run ONE Ultralytics TFLite export (one precision) in a child process with a hard timeout.

Why: on Colab (2026-11-07) the classifier INT8 export hung for 49 min inside ai-edge-quantizer
calibration with no output and no file activity. Running each precision in its own process
lets the parent kill it after a timeout and decide what that means, instead of hanging forever.

Used by export_tflite.py (detector) and export_cls_tflite.py (classifier):
  * ``run_export(...)``        -> ExportOutcome(status "ok" | "failed" | "timeout", ...)
  * ``resolve_outcome(...)``   -> the policy:
        - any FP32 failure / timeout                       -> hard error (raises)
        - INT8 failure / timeout without allow_skip        -> hard error (raises)
        - INT8 failure / timeout with allow_skip=True      -> {"status": "skipped", reason, timestamp}
          plus a large warning on stdout. Never silent.
    The detector never passes allow_skip; only the classifier has --allow-skip-int8.
  * ``available_precisions(model_io)`` -> precisions whose status is ok (entries without a
    status field, written before this change, count as ok).

As a script it is the child itself (called by run_export, not by hand):
    python export_child.py --weights src.pt --imgsz 320 [--int8 --data calib --fraction 0.3] --result out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_TIMEOUT_MIN = 15.0
WARNING_BAR = "!" * 100


@dataclass
class ExportOutcome:
    precision: str
    status: str                      # "ok" | "failed" | "timeout"
    seconds: float
    returncode: Optional[int]
    output: Optional[str] = None     # path of the produced .tflite when status == "ok"
    reason: str = ""
    tail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # the child runs in its own session/process group
        except ProcessLookupError:
            pass
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        if proc.poll() is None:
            proc.kill()


def run_child(cmd: Sequence[str], timeout_s: float, label: str = "child", tail_lines: int = 40) -> Dict[str, Any]:
    """Run cmd, stream its output live (prefixed), kill the whole process tree after timeout_s.
    Returns {"status": ok|failed|timeout, "returncode", "seconds", "tail"}."""
    kw: Dict[str, Any] = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    else:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    t0 = time.monotonic()
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1, **kw)
    tail: collections.deque = collections.deque(maxlen=tail_lines)

    def pump():
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            print(f"  [{label}] {line}", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout_s)
        status = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.wait()
        status = "timeout"
    reader.join(timeout=10)
    proc.stdout.close()
    return {"status": status, "returncode": proc.returncode, "seconds": round(time.monotonic() - t0, 1),
            "tail": list(tail)}


def child_cmd(weights: Path, imgsz: int, int8: bool, data: Optional[Path], fraction: float, result: Path) -> List[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--weights", str(weights), "--imgsz", str(imgsz),
           "--result", str(result)]
    if int8:
        if data is None:
            raise ValueError("INT8 export needs calibration data")
        cmd += ["--int8", "--data", str(data), "--fraction", str(fraction)]
    return cmd


def run_export(weights: Path, imgsz: int, precision: str, data: Optional[Path], fraction: float,
               timeout_min: float, work: Path, label: str) -> ExportOutcome:
    """Export one precision in a child process. Never raises for a failed/hung export: it reports it."""
    if precision not in ("fp32", "int8"):
        raise ValueError(f"unknown precision {precision!r}")
    if timeout_min <= 0:
        raise ValueError(f"timeout must be > 0 minutes, got {timeout_min}")
    result = Path(work) / f"child_result_{precision}.json"
    if result.exists():
        result.unlink()
    cmd = child_cmd(weights, imgsz, precision == "int8", data, fraction, result)
    print(f"[export] {label} {precision}: child process, timeout {timeout_min:g} min", flush=True)
    r = run_child(cmd, timeout_min * 60.0, label=f"{label} {precision}")
    out = ExportOutcome(precision=precision, status=r["status"], seconds=r["seconds"],
                        returncode=r["returncode"], tail=r["tail"])
    if out.status == "timeout":
        out.reason = f"timed out after {timeout_min:g} min ({out.seconds:.0f} s); child process killed"
    elif out.status == "failed":
        out.reason = f"child exited with code {out.returncode}"
    else:
        try:
            produced = Path(json.loads(result.read_text(encoding="utf-8"))["output"])
        except (OSError, ValueError, KeyError) as e:
            out.status, out.reason = "failed", f"child exited 0 but wrote no valid result file ({e})"
        else:
            if produced.is_file():
                out.output = str(produced)
            else:
                out.status, out.reason = "failed", f"child reported {produced}, which does not exist"
    print(f"[export] {label} {precision}: {out.status} in {out.seconds:.0f} s {out.reason}", flush=True)
    return out


def skipped_entry(reason: str, timestamp: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason, "timestamp": timestamp or utc_now(), **extra}


def print_skip_warning(label: str, precision: str, reason: str) -> None:
    msg = [WARNING_BAR, WARNING_BAR,
           f"!!  WARNING: {label} {precision.upper()} EXPORT SKIPPED  (--allow-skip-int8 was given)",
           f"!!  reason: {reason}",
           f"!!  model_io.json records {precision} as status=skipped. There is NO {precision} {label} model.",
           f"!!  Every result that uses this {label} must say which precision it used.",
           WARNING_BAR, WARNING_BAR]
    print("\n".join(msg), flush=True)
    print("\n".join(msg), file=sys.stderr, flush=True)


def resolve_outcome(outcome: ExportOutcome, allow_skip_int8: bool, label: str) -> Optional[Dict[str, Any]]:
    """Policy. Returns None if the export is ok (caller checks the file), a skipped entry if an
    INT8 failure is explicitly allowed, and raises otherwise."""
    if outcome.status == "ok":
        return None
    detail = outcome.reason + ("\n  last output:\n    " + "\n    ".join(outcome.tail[-15:]) if outcome.tail else "")
    if outcome.precision != "int8":
        raise RuntimeError(f"{label} {outcome.precision} export {outcome.status}: {detail}")
    if not allow_skip_int8:
        raise RuntimeError(f"{label} int8 export {outcome.status}: {detail}\n"
                           f"(only the classifier may skip INT8, and only with --allow-skip-int8)")
    print_skip_warning(label, outcome.precision, outcome.reason)
    return skipped_entry(outcome.reason, export_status=outcome.status, export_seconds=outcome.seconds)


def available_precisions(model_io: Mapping[str, Any]) -> List[str]:
    """Precisions in model_io.json whose status is ok, in fp32, int8 order. A missing status = ok
    (files written before statuses existed). Any status other than ok/skipped raises."""
    out = []
    for p in ("fp32", "int8"):
        m = model_io.get("models", {}).get(p)
        if m is None:
            continue
        st = m.get("status", "ok")
        if st == "ok":
            if "file" not in m:
                raise ValueError(f"model_io.json: {p} has status ok but no file")
            out.append(p)
        elif st != "skipped":
            raise ValueError(f"model_io.json: unknown status {st!r} for {p}")
    if "fp32" not in out:
        raise ValueError("model_io.json has no usable fp32 model")
    return out


def _child_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="child: one Ultralytics LiteRT export")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--result", required=True)
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    kwargs: Dict[str, Any] = dict(format="litert", imgsz=a.imgsz, batch=1)
    if a.int8:
        if not a.data:
            raise ValueError("--int8 needs --data")
        kwargs.update(quantize=8, data=a.data, fraction=a.fraction)
    print(f"YOLO({a.weights}).export({kwargs})", flush=True)
    out = YOLO(a.weights).export(**kwargs)
    out = Path(out) if out else None
    if out is None or not out.is_file():
        raise RuntimeError(f"Ultralytics export returned {out!r}; no .tflite produced")
    # copy to a precision-specific name right away so a later export cannot overwrite it
    final = Path(a.result).with_name(f"exported_{'int8' if a.int8 else 'fp32'}.tflite")
    final.write_bytes(out.read_bytes())
    Path(a.result).write_text(json.dumps({"output": str(final)}), encoding="utf-8")


if __name__ == "__main__":
    _child_main()
