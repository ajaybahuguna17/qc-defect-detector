"""
TFLite helpers for the Phase 1 defect detector: inspect I/O, preprocess, run, decode.

This is the reference decoder for the Android app: whatever this file does, the
Kotlin code must do the same (see phase1-vision/MODEL_IO.md).

Nothing here guesses or falls back silently. If the model does not look like a
6-class YOLOv8 detector, the functions raise.

Expected model (Ultralytics 8.4.198 `format="litert"`, litert-torch backend):
    input : float32, NCHW [1, 3, S, S] or NHWC [1, S, S, 3], RGB, values 0..1, letterboxed
    output: float32 [1, 4 + nc, N]; rows 0..3 = cx, cy, w, h normalised to 0..1 of the
            input size; rows 4.. = per-class scores (already sigmoid). No objectness. No NMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

NEU_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Phase 1.2 classifier v2: the 6 NEU classes in the same order, then the reject class.
NOT_STEEL = "not_steel"
CLS7_CLASSES: List[str] = [*NEU_CLASSES, NOT_STEEL]


# ----------------------------------------------------------------------------
# Loading & inspection
# ----------------------------------------------------------------------------

def load_interpreter(tflite_path: Path, num_threads: int = 4):
    """Create an allocated LiteRT interpreter. Raises if the runtime or file is missing."""
    tflite_path = Path(tflite_path)
    if not tflite_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as e:  # no silent fallback to another runtime
        raise ImportError("ai-edge-litert is required (pip install ai-edge-litert==2.3.0)") from e
    interp = Interpreter(model_path=str(tflite_path), num_threads=num_threads)
    interp.allocate_tensors()
    return interp


@dataclass
class ModelIO:
    input_shape: List[int]
    input_dtype: str
    input_layout: str              # "NCHW" or "NHWC"
    input_hw: Tuple[int, int]
    input_quant: Tuple[float, int]  # (scale, zero_point); (0.0, 0) means not quantised
    output_shape: List[int]
    output_dtype: str
    output_quant: Tuple[float, int]
    num_classes: int
    num_anchors: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def describe_io(interp) -> ModelIO:
    ins, outs = interp.get_input_details(), interp.get_output_details()
    if len(ins) != 1:
        raise ValueError(f"expected 1 input tensor, got {len(ins)}")
    if len(outs) != 1:
        raise ValueError(f"expected 1 output tensor, got {len(outs)}: {[o['shape'] for o in outs]}")
    i, o = ins[0], outs[0]
    shape = [int(v) for v in i["shape"]]
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"unexpected input shape {shape}")
    if shape[1] == 3 and shape[3] != 3:
        layout, hw = "NCHW", (shape[2], shape[3])
    elif shape[3] == 3 and shape[1] != 3:
        layout, hw = "NHWC", (shape[1], shape[2])
    else:
        raise ValueError(f"cannot tell layout from input shape {shape}")
    oshape = [int(v) for v in o["shape"]]
    if len(oshape) != 3 or oshape[0] != 1:
        raise ValueError(f"unexpected output shape {oshape}")
    return ModelIO(
        input_shape=shape,
        input_dtype=np.dtype(i["dtype"]).name,
        input_layout=layout,
        input_hw=hw,
        input_quant=(float(i["quantization"][0]), int(i["quantization"][1])),
        output_shape=oshape,
        output_dtype=np.dtype(o["dtype"]).name,
        output_quant=(float(o["quantization"][0]), int(o["quantization"][1])),
        num_classes=oshape[1] - 4,
        num_anchors=oshape[2],
    )


def assert_defect_model(io: ModelIO, expected_classes: int = 6) -> None:
    """Fail loudly unless output is [1, 4 + expected_classes, N] (e.g. [1, 10, N] for NEU-DET)."""
    want = 4 + expected_classes
    if io.output_shape[1] != want:
        raise AssertionError(
            f"output shape {io.output_shape} is not [1, {want}, N]: model has "
            f"{io.output_shape[1] - 4} classes, expected {expected_classes}. "
            f"(80 classes / [1, 84, N] means a COCO model was exported by mistake.)"
        )
    if io.output_shape[2] <= io.output_shape[1]:
        raise AssertionError(f"output shape {io.output_shape} looks transposed; expected [1, {want}, N] with N >> {want}")


# ----------------------------------------------------------------------------
# Pre-processing (mirrors Ultralytics LetterBox: centred, pad value 114, bilinear)
# ----------------------------------------------------------------------------

@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def read_rgb(image_path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(image_path) as im:
        return np.asarray(im.convert("RGB"))


def letterbox(rgb: np.ndarray, new_hw: Tuple[int, int]) -> Tuple[np.ndarray, Letterbox]:
    h0, w0 = rgb.shape[:2]
    nh, nw = new_hw
    r = min(nh / h0, nw / w0)
    rw, rh = int(round(w0 * r)), int(round(h0 * r))
    if (rw, rh) != (w0, h0):
        try:
            import cv2
            resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.BILINEAR))
    else:
        resized = rgb
    dw, dh = (nw - rw) / 2, (nh - rh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas = np.full((nh, nw, 3), 114, dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = resized
    return canvas, Letterbox(scale=r, pad_x=left, pad_y=top, orig_w=w0, orig_h=h0)


def to_input_tensor(canvas_rgb_uint8: np.ndarray, io: ModelIO) -> np.ndarray:
    x = canvas_rgb_uint8.astype(np.float32) / 255.0          # 0..1
    if io.input_layout == "NCHW":
        x = x.transpose(2, 0, 1)
    x = x[None]
    if io.input_dtype in ("int8", "uint8", "int16"):
        scale, zp = io.input_quant
        info = np.iinfo(np.dtype(io.input_dtype))
        x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(io.input_dtype)
    elif io.input_dtype != "float32":
        raise ValueError(f"unsupported input dtype {io.input_dtype}")
    return np.ascontiguousarray(x)


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        iw = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        ih = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode_yolov8(
    raw: np.ndarray,
    io: ModelIO,
    lb: Letterbox,
    class_names: Sequence[str] = NEU_CLASSES,
    conf_thr: float = 0.25,
    iou_thr: float = 0.45,
    coords_normalized: bool = True,
    max_det: int = 300,
) -> List[Dict[str, Any]]:
    """Decode a [1, 4+nc, N] YOLOv8 output into detections in ORIGINAL image pixels.

    Class-aware NMS (same as Ultralytics default, agnostic=False).
    """
    out = raw.astype(np.float32)
    if io.output_dtype in ("int8", "int16"):
        scale, zp = io.output_quant
        out = (raw.astype(np.float32) - zp) * scale
    p = out[0].T                               # [N, 4+nc]
    nc = p.shape[1] - 4
    if nc != len(class_names):
        raise ValueError(f"model has {nc} classes but {len(class_names)} class names were given")
    xywh, scores = p[:, :4].copy(), p[:, 4:]
    if coords_normalized:
        ih, iw = io.input_hw
        xywh[:, [0, 2]] *= iw
        xywh[:, [1, 3]] *= ih
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf >= conf_thr
    xywh, cls, conf = xywh[m], cls[m], conf[m]
    xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                     xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) if len(xywh) else np.zeros((0, 4))
    # class-aware NMS: offset boxes per class so different classes never suppress each other
    offset = cls[:, None].astype(np.float32) * 4096.0
    keep = nms_xyxy(xyxy + offset, conf, iou_thr)[:max_det]
    dets = []
    for k in keep:
        x1, y1, x2, y2 = xyxy[k]
        # undo letterbox -> original pixels, clip
        x1 = float(np.clip((x1 - lb.pad_x) / lb.scale, 0, lb.orig_w))
        x2 = float(np.clip((x2 - lb.pad_x) / lb.scale, 0, lb.orig_w))
        y1 = float(np.clip((y1 - lb.pad_y) / lb.scale, 0, lb.orig_h))
        y2 = float(np.clip((y2 - lb.pad_y) / lb.scale, 0, lb.orig_h))
        dets.append({
            "class_id": int(cls[k]),
            "class_name": class_names[int(cls[k])],
            "confidence": float(conf[k]),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return dets


def run_tflite_on_image(
    interp,
    io: ModelIO,
    image_path: Path,
    class_names: Sequence[str] = NEU_CLASSES,
    conf_thr: float = 0.25,
    iou_thr: float = 0.45,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Full pipeline for one image. Returns (detections, raw_output)."""
    rgb = read_rgb(Path(image_path))
    canvas, lb = letterbox(rgb, io.input_hw)
    x = to_input_tensor(canvas, io)
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    raw = interp.get_tensor(interp.get_output_details()[0]["index"])
    return decode_yolov8(raw, io, lb, class_names, conf_thr, iou_thr), raw


# ----------------------------------------------------------------------------
# Comparison helpers (used by the notebook's .pt vs .tflite check)
# ----------------------------------------------------------------------------

def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_detections(ref: List[Dict[str, Any]], other: List[Dict[str, Any]], iou_thr: float = 0.5) -> Dict[str, Any]:
    """Greedy same-class IoU matching of `other` against reference detections."""
    used, ious, dconf = set(), [], []
    for r in sorted(ref, key=lambda d: -d["confidence"]):
        best, best_j = 0.0, None
        for j, o in enumerate(other):
            if j in used or o["class_id"] != r["class_id"]:
                continue
            v = iou_xyxy(r["bbox_xyxy"], o["bbox_xyxy"])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= iou_thr:
            used.add(best_j)
            ious.append(best)
            dconf.append(abs(r["confidence"] - other[best_j]["confidence"]))
    return {
        "n_ref": len(ref),
        "n_other": len(other),
        "matched": len(ious),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_abs_conf_diff": float(np.mean(dconf)) if dconf else None,
    }


# ----------------------------------------------------------------------------
# Image classifier (Phase 1.1, YOLOv8n-cls): I/O, pre-processing, run
# ----------------------------------------------------------------------------
# Reference pre-processing = Ultralytics classify_transforms at crop_fraction 1.0:
#   RGB -> resize the SHORTER side to S (bilinear, PIL) -> centre crop S x S -> pixel / 255
#   (mean 0, std 1, i.e. no mean/std normalisation). NEU 200x200 -> plain 224x224 resize.
# The Phase 1.1 notebook checks this against the .pt pipeline on every test image and records
# the max probability difference; the Android code must do the same steps.

@dataclass
class ClsIO:
    input_shape: List[int]
    input_dtype: str
    input_layout: str               # "NCHW" or "NHWC"
    input_hw: Tuple[int, int]
    input_quant: Tuple[float, int]
    output_shape: List[int]
    output_dtype: str
    output_quant: Tuple[float, int]
    num_classes: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def describe_cls_io(interp) -> ClsIO:
    ins, outs = interp.get_input_details(), interp.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError(f"expected 1 input and 1 output tensor, got {len(ins)} / {len(outs)}")
    i, o = ins[0], outs[0]
    shape = [int(v) for v in i["shape"]]
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"unexpected classifier input shape {shape}")
    if shape[1] == 3 and shape[3] != 3:
        layout, hw = "NCHW", (shape[2], shape[3])
    elif shape[3] == 3 and shape[1] != 3:
        layout, hw = "NHWC", (shape[1], shape[2])
    else:
        raise ValueError(f"cannot tell layout from input shape {shape}")
    oshape = [int(v) for v in o["shape"]]
    return ClsIO(
        input_shape=shape, input_dtype=np.dtype(i["dtype"]).name, input_layout=layout, input_hw=hw,
        input_quant=(float(i["quantization"][0]), int(i["quantization"][1])),
        output_shape=oshape, output_dtype=np.dtype(o["dtype"]).name,
        output_quant=(float(o["quantization"][0]), int(o["quantization"][1])),
        num_classes=oshape[-1],
    )


def assert_classifier_model(io: ClsIO, expected_classes: int = 6) -> None:
    """Fail loudly unless the output is exactly [1, expected_classes]."""
    if io.output_shape != [1, expected_classes]:
        raise AssertionError(f"classifier output shape {io.output_shape} != [1, {expected_classes}] "
                             f"(1000 classes means the ImageNet yolov8n-cls was exported by mistake)")


def cls_preprocess(rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 HxWx3 -> HWC float32 S x S x 3 in 0..1 (shorter side -> S, bilinear, centre crop)."""
    from PIL import Image
    h0, w0 = rgb.shape[:2]
    # same arithmetic as torchvision Resize(int): long side = int(S * long / short) (truncated)
    nw, nh = (size, int(size * h0 / w0)) if w0 <= h0 else (int(size * w0 / h0), size)
    im = Image.fromarray(rgb)
    if (nw, nh) != (w0, h0):
        im = im.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    im = im.crop((left, top, left + size, top + size))
    return np.asarray(im, dtype=np.float32) / 255.0


def cls_input_tensor(hwc01: np.ndarray, io: ClsIO) -> np.ndarray:
    x = hwc01.transpose(2, 0, 1) if io.input_layout == "NCHW" else hwc01
    x = x[None].astype(np.float32)
    if io.input_dtype in ("int8", "uint8", "int16"):
        scale, zp = io.input_quant
        info = np.iinfo(np.dtype(io.input_dtype))
        x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(io.input_dtype)
    elif io.input_dtype != "float32":
        raise ValueError(f"unsupported input dtype {io.input_dtype}")
    return np.ascontiguousarray(x)


def run_cls_tflite(interp, io: ClsIO, image_path: Path) -> np.ndarray:
    """Class probabilities [nc] for one image (output dequantised if needed)."""
    if io.input_hw[0] != io.input_hw[1]:
        raise ValueError(f"classifier input must be square, got {io.input_hw}")
    x = cls_input_tensor(cls_preprocess(read_rgb(Path(image_path)), io.input_hw[0]), io)
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    if io.output_dtype in ("int8", "uint8", "int16"):
        scale, zp = io.output_quant
        out = (out.astype(np.float32) - zp) * scale
    return out.reshape(-1).astype(np.float32)
