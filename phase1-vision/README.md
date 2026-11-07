# Phase 1: Vision defect detector (NEU-DET -> YOLOv8n -> TFLite)

**Target:** Arm Android phone, TFLite (LiteRT) runtime.
**Status:** COMPLETE. Trained, evaluated and exported in Colab on 2026-11-07 (commit `25551ad`). Val mAP50 0.769 (`.pt`), 0.744 (FP32 TFLite), 0.707 (INT8 TFLite). Results in `results/`, models on Google Drive (not in git). See the Phase 1 tracker in `../PROJECT_CONTEXT.md`.

Pipeline: train in Google Colab -> evaluate -> severity rule -> export TFLite FP32 + INT8 -> sanity-check -> document.

## Layout

```
phase1-vision/
├── notebooks/train_colab.ipynb   # THE entry point: end-to-end Colab run (GPU)
├── scripts/
│   ├── prepare_data.py           # NEU-DET download, VOC->YOLO, seeded 80/20 split, overlap check, data.yaml
│   ├── export_tflite.py          # best.pt -> FP32/INT8 .tflite, asserts [1, 10, N], writes model_io.json
│   └── tflite_utils.py           # reference preprocess + decode + NMS (the Android code must match it)
├── severity.py                   # deterministic Low/Medium/High/Critical rule (one config dict)
├── tests/                        # unit tests: severity rule, decoder
├── MODEL_IO.md                   # TFLite input/output contract for the Android app
├── requirements.txt              # pinned Colab/export environment
├── data/                         # raw/ and dataset/ are gitignored; data.yaml has no absolute path
└── models/exported/              # gitignored export output
```

## Run it (Colab)

1. Open `phase1-vision/notebooks/train_colab.ipynb` in Colab (badge at the top of the notebook,
   or *File -> Open notebook -> GitHub*), choose a GPU runtime, then *Run all*.
2. Outputs go to Google Drive `MyDrive/qc-defect-detector/phase1/<run-name>/`:
   `models/` (best.pt, `defect_yolov8n_320_{fp32,int8}.tflite`, labels.txt, model_io.json) and
   `results/` (phase1_results.json, confusion matrices, PR curves, training curves, severity demo).

Defaults: YOLOv8n from COCO-pretrained weights, `imgsz=320` (NEU images are 200x200), 100 epochs
(patience 30), batch 32, seed 42.

## Local scripts

```bash
# dataset (any OS, stdlib only; downloads NEU-DET if data/raw is empty)
python phase1-vision/scripts/prepare_data.py            # --seed 42 --split-ratio 0.8

# export (Linux x86_64 / macOS only; needs requirements.txt)
python phase1-vision/scripts/export_tflite.py --weights best.pt \
    --data phase1-vision/data/dataset/data.yaml --imgsz 320 --precisions fp32,int8

# unit tests (Python >= 3.10; decoder tests need numpy, they are skipped without it)
python -m unittest discover -s phase1-vision/tests -v
```

`prepare_data.py` clears `images/{train,val}` and `labels/{train,val}` before writing, uses a
private seeded RNG, and fails if any image is in both splits. With the real NEU-DET it produces
**1,440 train / 360 val** images (3,322 / 867 boxes). Synthetic data is generated only with `--synthetic`.

`export_tflite.py` has no fallbacks. If the weights are missing, the checkpoint is not the 6 NEU
classes, the export fails, or the output is not `[1, 10, N]`, it raises.

## Severity rule (`severity.py`)

Ignore detections below their class's min confidence (crazing 0.30, all others 0.50; the model is under-confident on crazing). Per detection, area ratio = bbox area / image area:
`< 0.02` Low, `0.02-0.05` Medium, `0.05-0.10` High, `> 0.10` Critical. Image level = worst
detection, raised one level if there are 3+ valid defects. All thresholds are in `SEVERITY_CONFIG`.
Per-class area weights exist but are all 1.0 (class-neutral) until we have domain input.

## Dataset

NEU Surface Defect Database (NEU-DET), Northeastern University (China): 1,800 grayscale
200x200 images, 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale,
scratches), 300 images per class. Cite: K. Song and Y. Yan, "A noise robust method based on
completed local binary patterns for hot-rolled steel strip surface defects", *Applied Surface
Science*, 2013; Y. He et al., "An End-to-end Steel Surface Defect Detection Approach via Fusing
Multiple Hierarchical Features", *IEEE TIM*, 2020.
