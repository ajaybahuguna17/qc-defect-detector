# Phase 1 demo

Local, offline, CPU-only demo of the Phase 1.1 pipeline: TFLite detector (FP32 or INT8) + TFLite
classifier (FP32) -> `fusion.py` + `severity.py`. No `.pt` files or ultralytics at runtime, no share link.

## Run

From the repo root (Windows; on Linux/macOS use `.venv/bin/python`):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r phase1-vision/requirements-demo.txt

.venv\Scripts\python phase1-vision/demo/app.py                       # UI at http://127.0.0.1:7860
.venv\Scripts\python phase1-vision/demo/predict.py phase1-vision/demo/examples
.venv\Scripts\python phase1-vision/demo/predict.py <image or folder> --detector int8
```

The CLI saves annotated images to `phase1-vision/demo/out/` (gitignored).

## Models

Not in git. Copy them from Google Drive
`MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/{detector,classifier}/` to
`phase1-vision/models/exported/phase1_1/{detector,classifier}/` (hashes in `../MODEL_IO.md`). If a file is
missing, the demo stops and says so; it never uses another model.

## Reading the result

- **Boxes** are the kept detections, any class (per-class thresholds in `FUSION_CONFIG`).
- **Largest area** and **Severity** use only the boxes of the final defect type.
- **Decided by** `review`: the classifier fired but the detector kept no box of that type, so the
  operator must check the part.
- **Inference times** cover the interpreter call only, not pre-processing or decoding.

## Limits

NEU-DET has no defect-free images, so false alarms on good parts are not measured, and the results do not
carry over to other kinds of image. The classifier always picks one of the 6 classes.

## Examples

`examples/` holds the first 2 filenames per class (sorted) of the 70/15/15 TEST split, unmodified.
Dataset: NEU-DET, K. Song and Y. Yan, *Applied Surface Science*, 2013; for research use.
