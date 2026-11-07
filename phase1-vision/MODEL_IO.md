# Phase 1 TFLite model: input / output contract

The contract between the vision model and the Android app. The reference implementation
is `phase1-vision/scripts/tflite_utils.py` (`letterbox`, `to_input_tensor`,
`decode_yolov8`). The Kotlin code must do exactly the same steps.

> **Status: confirmed.** Every value below was confirmed on the real exported files by the
> Colab run of commit `25551ad` (2026-11-07, Tesla T4, Ultralytics 8.4.198, litert-torch 0.10.0,
> ai-edge-litert 2.3.0, torch 2.14.1). Source: `results/phase1_results.json` (`model_io`,
> `val_tflite`, `compare_10_images`). Run name `neu_yolov8n_320_e100_s42`.
> The model files are not in git. They are on Google Drive under
> `MyDrive/qc-defect-detector/phase1/neu_yolov8n_320_e100_s42/models/`.

## Files

| File | What it is |
|---|---|
| `defect_yolov8n_320_fp32.tflite` | FP32 weights and activations. Reference accuracy. |
| `defect_yolov8n_320_int8.tflite` | INT8 weights and activations inside, **float32 input/output** at the graph boundary. Calibrated on ~30 % of the train split. |
| `labels.txt` | Class names, one per line, line index = class id. |
| `model_io.json` | Machine-written I/O report from `export_tflite.py`. |

Sizes (confirmed): FP32 **12.69 MB** (12,693,225 bytes), INT8 **3.44 MB** (3,441,419 bytes).

SHA-256 of the files on Google Drive (`results/phase1_checks.json` -> `files`). Check these after
copying the models into the Android app:

| File | Bytes | SHA-256 |
|---|---|---|
| `best_neu_yolov8n_320_e100_s42.pt` | 6,475,050 | `91d1f243e0e68e39be178a06cd8b7ad33128ed283a4707e743174f3b84a35a9c` |
| `defect_yolov8n_320_fp32.tflite` | 12,693,225 | `cdb2afb1d29b42b71f6a6ac95ad1d7ef55221f5e6ac02e5974b57f2f8547d566` |
| `defect_yolov8n_320_int8.tflite` | 3,441,419 | `54925b0ef977a6b6a0fa89bed75d1365b3c5ecc38d66f215773a699439000714` |
| `labels.txt` | 67 | `950255ac9fb196b614f869f1151f8e132a79d72b7711a4b3f18026ff0ba77298` |
| `model_io.json` | 1,435 | `1ca78eac2e405ab43836c06b2c24894a0016ff323f458386b4f31d985b826df4` |

## Input

| Property | Value | Confirmed |
|---|---|---|
| Shape | `[1, 3, 320, 320]` | ✅ FP32 and INT8 |
| Layout | **NCHW** (channels first). litert-torch keeps PyTorch's layout. This is *not* the usual NHWC of older onnx2tf exports. | ✅ FP32 and INT8 |
| Dtype | `float32` (both FP32 and INT8 files), no input quantisation (scale 0, zero point 0) | ✅ FP32 and INT8 |
| Colour order | RGB | from training pipeline |
| Normalisation | `pixel / 255.0` -> **0.0 to 1.0**. No mean/std. | from export code |
| Resize | Letterbox: scale `r = min(320/h, 320/w)`, bilinear resize, centre on a 320x320 canvas filled with 114 (gray). NEU images are 200x200, so this is a plain 1.6x upscale with no padding. | from export code |

`to_input_tensor()` works out the layout from the input shape, so the Python checks work for either layout.
**The Android code must use NCHW** (confirmed). Write
`buffer[c*H*W + y*W + x]`, not `buffer[(y*W + x)*3 + c]`.

## Output

| Property | Value | Confirmed |
|---|---|---|
| Tensors | 1 | ✅ |
| Shape | `[1, 10, 2100]`: 10 = 4 box values + 6 class scores, 2100 = 40x40 + 20x20 + 10x10 anchors at 320 px | ✅ FP32 and INT8 |
| Dtype | `float32`, no output quantisation (scale 0, zero point 0) | ✅ FP32 and INT8 |
| Box rows 0..3 | `cx, cy, w, h` of the box **in the 320x320 letterboxed input, normalised to 0..1** (Ultralytics' litert export divides by input W/H before quantisation) | ✅ NORMALISED: max abs box value over the 10 check images is 1.014 (FP32) / 1.002 (INT8), not ~320. Values can be slightly above 1 for boxes touching the edge. |
| Rows 4..9 | Per-class confidence, **already sigmoid-ed** (0..1). There is **no objectness** score. | from export code |
| NMS included? | **No.** The app must run NMS. | from export code |

`export_tflite.py` refuses any file whose output is not `[1, 10, N]`. For example, `[1, 84, 8400]` is a COCO model.

## Decoding (what the app must do)

```
out = output[0]                       # [10, N]
for each anchor i in 0..N-1:
    scores = out[4:10, i]
    cls    = argmax(scores); conf = scores[cls]
    if conf < CONF_THRESHOLD: skip    # 0.25 for drawing boxes; severity uses its own per-class min (crazing 0.25, others 0.50)
    cx, cy, w, h = out[0:4, i] * 320  # normalised -> input pixels (confirmed normalised)
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
class-aware NMS, IoU 0.45 (boxes of different classes never suppress each other)
undo letterbox:  x = (x - pad_x) / r,  y = (y - pad_y) / r,  clip to the original image
```

Then pass `{class_name, confidence, bbox_xyxy}` (original pixels) plus the original image
width/height to the severity rule (`phase1-vision/severity.py`, port it 1:1 to Kotlin).

## Class order

Fixed by `data.yaml` / `labels.txt`:

| id | class |
|---|---|
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |

## Accuracy (val split, 360 images)

| Model | Input the val ran at | mAP50 | mAP50-95 | Source |
|---|---|---|---|---|
| `best.pt` (Ultralytics default val) | 352x352 (rect, pad 0.5) | 0.769 | 0.446 | Colab and local, identical |
| `best.pt` at the deployed size | **320x320** | **0.744** | **0.420** | local (`phase1_checks.json`) |
| FP32 `.tflite` | 320x320 | **0.744** | **0.420** | Colab and local, identical |
| INT8 `.tflite` | 320x320 | **0.715** | **0.372** | local x86 CPU; Colab gave 0.707 / 0.367 |

**The `.pt` -> FP32 gap (0.769 -> 0.744) is not a conversion loss.** `Model.val()` runs a `.pt`
with rectangular batches and pad 0.5, so a 320 px image is evaluated on a 352x352 input.
Exported models are always evaluated at their fixed 320x320 input. Evaluated at 320, the
`.pt` gives exactly the FP32 numbers (0.7442 / 0.4204). **0.744 is the number to expect on the
phone**, not 0.769. (Running at 352 on the phone would need a re-export at that size.)

INT8 differs slightly between Colab and the local CPU (0.715 vs 0.707 mAP50) because
quantised CPU kernels round differently on different hardware. Expect ~0.71 on the phone and
measure it there.

Agreement with `best.pt` on **all 360 val images** (conf 0.25, NMS IoU 0.45, match = same class,
IoU >= 0.5), from `results/phase1_checks.json` -> `agreement_full_val`:

| Model | `.pt` detections | TFLite detections | Matched | Match rate | Images identical | Mean IoU |
|---|---|---|---|---|---|---|
| FP32 | 949 | 949 | 949 | **100 %** | 360 / 360 | 1.000 |
| INT8 | 949 | 911 | 868 | **91.5 %** | 268 / 360 | 0.907 |

(The 10-image check in `phase1_results.json` gave 100 % / 92 %; the full set confirms it.)

INT8 is 3.7x smaller than FP32 and loses 0.029 mAP50 / 0.048 mAP50-95 at the same 320 input.
Pick FP32 if the phone is fast enough, otherwise INT8 (to be decided with the Phase 3
on-device benchmarks).
