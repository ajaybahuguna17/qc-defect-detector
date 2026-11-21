# Phase 1 TFLite model: input / output contract

> **Phase 1.1 (run `p11_split701515_det320_cls224_s42`, 2026-11-07):** the detector was retrained
> on the new 70/15/15 split and an image classifier was added (section *Image classifier* at the
> end). The detector I/O contract below is unchanged (same export script, same 320 input, output
> `[1, 10, 2100]`). Phase 1.1 detector files: FP32 12,693,227 bytes, INT8 3,441,424 bytes; TEST
> (270 images, 320 px) mAP50 / mAP50-95: `.pt` 0.741 / 0.413 = FP32 0.741 / 0.413, INT8 0.693 / 0.340.
> Source: `results/phase1_1/phase1_1_results.json`. Models on Drive under
> `MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/`. The SHA-256 table
> and the val-split tables below are still those of the Phase 1 run `neu_yolov8n_320_e100_s42`.

## Phase 1.1 model files (USE THESE IN PHASE 3)

**The Android app (Phase 3) must use the Phase 1.1 models below**, not the Phase 1 files of run
`neu_yolov8n_320_e100_s42` listed further down. Locations (not in git):
- Google Drive: `MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/{detector,classifier}/`
- local: `phase1-vision/models/exported/phase1_1/{detector,classifier}/` (gitignored)

SHA-256 and size of every file in `models/exported/phase1_1/` (computed locally on 2026-11-08; sizes
and the classifier FP32 hash match `results/phase1_1/phase1_1_results.json`, and both `model_io.json`
files are identical to the copies in that results file). Check these after copying the models into the app:

| File | Bytes | SHA-256 |
|---|---|---|
| `detector/best_p11_split701515_det320_cls224_s42_detector.pt` | 6,475,050 | `4bdd364d2cc1e301fc4e5db918a21f2100840e901d1260f21dd7a0c1fb877c56` |
| `detector/defect_yolov8n_320_fp32.tflite` | 12,693,227 | `b453d1b99fe83a340c159ca44e99a9511cb3ee5b2dc2f813575aa2732f9262e0` |
| `detector/defect_yolov8n_320_int8.tflite` | 3,441,424 | `ceed1ecd20c6d8cb715b39cf7f2bc1cff238f4aac3f4fca6fb6278c547fb9f2f` |
| `detector/labels.txt` | 67 | `950255ac9fb196b614f869f1151f8e132a79d72b7711a4b3f18026ff0ba77298` |
| `detector/model_io.json` | 1,450 | `cbb97612a160b1c3dde7f1d3a1d40f32048805daeb09e296aac1e15136562710` |
| `classifier/best_p11_split701515_det320_cls224_s42_classifier.pt` | 2,976,584 | `b895725b140feafa0f111aa80f4ef04d67e01274bd1d1e0113c6b50b7ec66e9e` |
| `classifier/defect_cls_yolov8n_224_fp32.tflite` | 5,815,905 | `34889a816f4319bc693e38fd9faad0560ce2ff07102b543c446704128b51831d` |
| `classifier/labels.txt` | 67 | `950255ac9fb196b614f869f1151f8e132a79d72b7711a4b3f18026ff0ba77298` |
| `classifier/model_io.json` | 1,984 | `d6fea14410ad4e572e0d57a6414b2415b4246e3b7f783bc426b7b346ec4a8559` |

There is no `classifier/defect_cls_yolov8n_224_int8.tflite`: the classifier INT8 export was skipped
(see *Image classifier* below). On the phone, the "INT8 pipeline" = detector INT8 + classifier FP32.

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

### Image-level accuracy (what the operator sees)

% of the 360 val images where at least one detection has the image's class, after the conf-0.25
decode and the severity rule's per-class min confidence (crazing 0.25, others 0.50). The image's
class is its NEU category (filename prefix). From `results/phase1_checks.json` ->
`image_level_accuracy_val`:

| Class | `.pt` / FP32 | INT8 |
|---|---|---|
| crazing | 56 / 60 (93 %) | 51 / 60 (85 %) |
| inclusion | 60 / 60 (100 %) | 59 / 60 (98 %) |
| patches | 60 / 60 (100 %) | 60 / 60 (100 %) |
| pitted_surface | 55 / 60 (92 %) | 53 / 60 (88 %) |
| rolled-in_scale | 44 / 60 (73 %) | 45 / 60 (75 %) |
| scratches | 59 / 60 (98 %) | 59 / 60 (98 %) |
| **overall** | **334 / 360 (92.8 %)** | **327 / 360 (90.8 %)** |
| misses: wrong class only / nothing kept | 1 / 25 | 2 / 31 |

Almost every miss is "no detection kept", not a wrong class. 28 val images also have a GT box of a
second class; they are scored by their category only.

INT8 is 3.7x smaller than FP32 and loses 0.029 mAP50 / 0.048 mAP50-95 at the same 320 input.
Pick FP32 if the phone is fast enough, otherwise INT8 (to be decided with the Phase 3
on-device benchmarks).

---

# Image classifier (Phase 1.1): input / output contract

> **Status: confirmed (FP32).** Values below come from the Phase 1.1 Colab run
> `p11_split701515_det320_cls224_s42` (Tesla T4, Ultralytics 8.4.198, resumed from the Drive backup
> `p11_backup`), file `results/phase1_1/phase1_1_results.json` -> `model_io.classifier` and `test`.
> Models on Google Drive under
> `MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/classifier/` and locally in
> `phase1-vision/models/exported/phase1_1/classifier/` (hashes: *Phase 1.1 model files* at the top).
>
> **Classifier INT8: SKIPPED. There is no INT8 classifier.** The INT8 export hung in the
> ai-edge-quantizer calibration step (49 min with no output and no file activity) and was
> interrupted. `model_io.json` records
> `"int8": {"status": "skipped", "reason": "hung >45 min in ai-edge-quantizer calibration on Colab, 2026-11-07", "timestamp": "2026-11-07T18:39:43+00:00"}`.
> The **"INT8 pipeline" is detector INT8 + classifier FP32** (5.8 MB FP32 classifier on top of the
> 3.4 MB INT8 detector). Exports now run in a child process with a timeout
> (`scripts/export_child.py`, default 15 min); a classifier INT8 failure is only tolerated with
> `--allow-skip-int8`, never silently.

YOLOv8n-cls, ImageNet-pretrained, trained on the same 70/15/15 split as the detector
(`data/cls/{train,val,test}/<class>/`, built by `scripts/build_cls_dataset.py`), imgsz 224,
patience 20: it early-stopped after **34 epochs** (of 100). One label per image (the NEU category).
Reference code: `scripts/tflite_utils.py` (`cls_preprocess`, `cls_input_tensor`, `run_cls_tflite`).
Export: `scripts/export_cls_tflite.py`.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `defect_cls_yolov8n_224_fp32.tflite` | 5,815,905 (5.82 MB) | `34889a816f4319bc693e38fd9faad0560ce2ff07102b543c446704128b51831d` |
| `defect_cls_yolov8n_224_int8.tflite` | **does not exist: skipped** (see above) | - |
| `labels.txt` | same 6 classes, same order as the detector | |
| `model_io.json` | machine-written I/O report, including the pre-processing read from the checkpoint | |

## Input

| Property | Value | Confirmed |
|---|---|---|
| Shape | `[1, 3, 224, 224]` | ✅ FP32 |
| Layout | **NCHW** (channels first, like the detector). Write `buffer[c*H*W + y*W + x]`. | ✅ FP32 |
| Dtype | `float32`, no input quantisation (scale 0, zero point 0) | ✅ FP32 |
| Colour order | RGB | from the training pipeline |
| Resize | Shorter side -> 224 (bilinear), centre crop 224x224. NEU 200x200 -> plain 224x224 resize. **Not** the detector's letterbox. | ✅ checkpoint transforms: `Resize(224, bilinear, antialias=True)`, `CenterCrop(224)` |
| Normalisation | `pixel / 255.0` -> 0..1, **mean 0, std 1** (no ImageNet mean/std) | ✅ checkpoint transforms: `ToTensor()`, `Normalize(mean=[0,0,0], std=[1,1,1])` |

Reference pre-processing parity: `cls_preprocess` vs the `.pt` pipeline on all 540 val + test
images, max probability difference **5.8e-6**.

## Output

| Property | Value | Confirmed |
|---|---|---|
| Shape | `[1, 6]` | ✅ FP32 |
| Dtype | `float32`, no output quantisation | ✅ FP32 |
| Meaning | Class probabilities, **softmax already applied** (each 0..1, sum 1; checked on a real image by the export script) | ✅ FP32 |

Top-1 = argmax, confidence = that probability. Feed it to `fusion.py` (`classifier_top1`).

## Accuracy (TEST split, 270 images, 45 per class)

| Model | Top-1 | Same top-1 as `.pt` |
|---|---|---|
| `.pt` | **269 / 270 (99.6 %)** (Ultralytics val: top-1 0.996, top-5 1.000) | - |
| FP32 `.tflite` | **269 / 270 (99.6 %)** | 270 / 270, max probability difference 0.0 |
| INT8 `.tflite` | skipped | skipped |

The one error: a rolled-in_scale image classified as crazing (`.pt` and FP32 alike). Confusion
matrix: `results/phase1_1/cls_confusion_test.png`.

## Fusion (what the app shows)

`phase1-vision/fusion.py` combines both models for one image. Thresholds in `FUSION_CONFIG`,
chosen on VAL by the notebook (`results/phase1_1/fusion_config.json`, FP32 and INT8 pipelines
summed): **classifier 0.64**; detector per class crazing 0.10, inclusion 0.50, patches 0.55,
pitted_surface 0.15, rolled-in_scale 0.15, scratches 0.30.

1. kept detections = detections with confidence >= their per-class threshold;
2. `final_class` = classifier class if its confidence >= `cls_threshold`, else the class of the
   highest-confidence kept detection, else `"no defect found"`;
3. `alert` = any kept detection OR classifier confidence >= `cls_threshold`;
4. severity = `severity.py` on the kept boxes of `final_class`; alert without such a box ->
   level `"Review"` (the operator must check the part).

TEST result: fused class correct on **270 / 270** images for the FP32 pipeline and for the INT8
pipeline (detector INT8 + classifier FP32). The classifier decided 267 images, the detector 3.
"Review" level: 0 (FP32 pipeline), 5 (INT8 pipeline).

**Caveats.** NEU-DET has no defect-free images, so false alarms on good parts are **not
measured**. The detector thresholds are low (crazing 0.10) for that reason and must be re-tuned
with clean-part camera images in Phase 3. All numbers are on NEU lab crops, not on factory camera
images.

# Image classifier v2 (Phase 1.2): 7 classes with `not_steel`

> **Status: PLANNED, values TODO until the Phase 1.2 Colab run** `p12_cls7_notsteel_s42`
> (`notebooks/train_colab_phase1_2.ipynb`). Nothing below marked TODO has been measured yet. Until then
> the app keeps using the Phase 1.1 6-class classifier above.

Same architecture and pre-processing as the Phase 1.1 classifier (YOLOv8n-cls from `yolov8n-cls.pt`,
imgsz 224, patience 20, seed 42), plus a 7th class `not_steel` trained on ~2,000 COCO / DTD / Places365
images (`DATA_SOURCES.md`). The **detector is unchanged** (Phase 1.1 files above).

## Class order (labels.txt)

| Index | Class |
|---|---|
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |
| 6 | **not_steel** |

Indices 0-5 are the detector order, unchanged. Ultralytics' ImageFolder would put `not_steel` at
index 2 (alphabetical); `scripts/reorder_cls_classes.py` permutes the trained head to this order and
the notebook checks it on real images. `fusion.CLS7_CLASSES` holds this list.

## Files (TODO after the run)

| File | Bytes | SHA-256 |
|---|---|---|
| `defect_cls_yolov8n_224_fp32.tflite` (v2, 7 classes) | TODO | TODO |
| `defect_cls_yolov8n_224_int8.tflite` | **not exported** (recorded as skipped: FP32 only, the INT8 calibration hung in Phase 1.1) | - |
| `labels.txt` | the 7 classes above | TODO |
| `model_io.json` | machine-written I/O report | TODO |

Drive: `MyDrive/qc-defect-detector/phase1/p12_cls7_notsteel_s42/models/classifier/`.

## Input / output

| Property | Value | Confirmed |
|---|---|---|
| Input | same as Phase 1.1: `[1, 3, 224, 224]` NCHW float32, RGB, shorter side -> 224 + centre crop, pixel / 255, mean 0 std 1 | TODO |
| Output | `[1, 7]` float32, softmax inside (asserted by `export_cls_tflite.py --num-classes 7`) | TODO |

## Fusion: the RETAKE rule (`fusion.py`, checked first)

If the classifier top-1 is `not_steel` with probability `>= not_steel_threshold` (t_ns):
`final_class = "not a steel surface"`, `alert = false`, no severity, `decision = "RETAKE"`, and the
detector boxes are ignored. A `not_steel` top-1 below t_ns names no defect: the classifier does not
fire and the detector decides alone (rest of the Phase 1.1 rule). All other branches have
`decision = null`.

| Threshold | Value | Source |
|---|---|---|
| `not_steel_threshold` (t_ns) | **TODO** (placeholder 0.50 in `FUSION_CONFIG`) | VAL: most not_steel rejected with steel false rejection <= 1 % |
| `cls_threshold` | 0.64 (Phase 1.1), **TODO re-check** | VAL re-check with classifier v2; kept if 0.64 still reaches the best val score |
| detector thresholds | Phase 1.1, unchanged | |

## Accuracy (TEST, TODO)

| Metric | Value |
|---|---|
| 7x7 confusion (`.pt`, FP32) | TODO (`results/phase1_2/cls7_confusion_test.png`) |
| Steel fused image-level accuracy, Phase 1.1 detector (target ~270/270) | TODO |
| not_steel rejection rate (300 test images) | TODO |
| Steel false-rejection rate (270 test images) | TODO |
