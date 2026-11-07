# Phase 1: Vision defect detector (NEU-DET -> YOLOv8n -> TFLite)

**Target:** Arm Android phone, TFLite (LiteRT) runtime.
**Status:** COMPLETE. Trained, evaluated and exported in Colab on 2026-11-07 (commit `25551ad`). Val mAP50 at the deployed 320 input: 0.744 (`.pt` and FP32 TFLite, identical), 0.715 (INT8 TFLite). The headline 0.769 for `.pt` was measured at 352 px, see `MODEL_IO.md`. Results in `results/`, models on Google Drive (not in git). See the Phase 1 tracker in `../PROJECT_CONTEXT.md`.

Pipeline: train in Google Colab -> evaluate -> severity rule -> export TFLite FP32 + INT8 -> sanity-check -> document.

**Phase 1.1 (COMPLETE, 2026-11-08):** goal near-zero image-level error.
New 70/15/15 train/val/test split (210/45/45 per class, seed 42), detector retrained on it, an
image classifier (YOLOv8n-cls) added, and a deterministic fusion rule (`fusion.py`) that
combines both. Thresholds are chosen on **val**; every reported number is on **test** (270 images):
fused class correct **270/270** (FP32 pipeline and INT8 pipeline = detector INT8 + classifier FP32;
classifier INT8 export was skipped, it hung), classifier alone 269/270, detector alone 267/270 (FP32) /
256/270 (INT8), detector mAP50 0.741 (FP32) / 0.693 (INT8). Results in `results/phase1_1/`, models on
Drive under `MyDrive/qc-defect-detector/phase1/p11_split701515_det320_cls224_s42/models/` and locally in
`models/exported/phase1_1/` (gitignored). **Phase 3 must use these Phase 1.1 models** (hashes in `MODEL_IO.md`).
Notebook: `notebooks/train_colab_phase1_1.ipynb`. **Caveats:** NEU-DET has no defect-free images, so
false alarms on good parts are not measured; the low detector thresholds must be re-tuned with
clean-part images in Phase 3; lab images only.

## Layout

```
phase1-vision/
├── notebooks/
│   ├── train_colab.ipynb         # Phase 1 run (80/20 split, detector only) - kept for reproducibility
│   └── train_colab_phase1_1.ipynb  # Phase 1.1 run: 70/15/15, detector + classifier, val sweep, test report
├── scripts/
│   ├── prepare_data.py           # NEU-DET download, VOC->YOLO, seeded 70/15/15 split (or legacy 80/20), overlap check, data.yaml
│   ├── build_cls_dataset.py      # ImageFolder copy of the SAME split for the classifier (data/cls/)
│   ├── export_cls_tflite.py      # classifier best.pt -> FP32/INT8 .tflite, asserts [1, 6] + softmax, writes model_io.json
│   ├── export_child.py           # runs each export precision in a child process with a timeout (default 15 min)
│   ├── fusion_eval.py            # val threshold sweep + image-level evaluation of the fusion rule
│   ├── export_tflite.py          # best.pt -> FP32/INT8 .tflite, asserts [1, 10, N], writes model_io.json
│   ├── check_models.py           # post-run checks (any OS): hashes, full-val .pt vs TFLite, P/R sweep, severity, image accuracy
│   ├── tune_severity.py          # severity area cutoffs = p40/p70/p90 of val GT box area ratios
│   └── tflite_utils.py           # reference preprocess + decode + NMS, classifier preprocess (Android must match)
├── severity.py                   # deterministic Low/Medium/High/Critical rule (one config dict)
├── fusion.py                     # deterministic detector + classifier fusion (final class, alert, severity / Review)
├── tests/                        # unit tests: severity, fusion, split/sweep tools, decoder
├── MODEL_IO.md                   # TFLite input/output contract for the Android app
├── requirements.txt              # pinned Colab/export environment
├── data/                         # raw/, dataset/ images+labels and cls/ are gitignored; data.yaml has no absolute path
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
python phase1-vision/scripts/prepare_data.py            # default --split 70/15/15 --seed 42
python phase1-vision/scripts/prepare_data.py --split 80/20   # legacy Phase 1 split (identical to the old one)
python phase1-vision/scripts/build_cls_dataset.py       # data/dataset -> data/cls/{train,val,test}/<class>/

# export (Linux x86_64 / macOS only; needs requirements.txt)
python phase1-vision/scripts/export_tflite.py --weights best.pt \
    --data phase1-vision/data/dataset/data.yaml --imgsz 320 --precisions fp32,int8

# post-training checks of the PHASE 1 models (needs the legacy split: prepare_data.py --split 80/20)
python phase1-vision/scripts/check_models.py

# unit tests (Python >= 3.10; decoder tests need numpy, they are skipped without it)
python -m unittest discover -s phase1-vision/tests -v
```

`prepare_data.py` clears `images/{train,val,test}` and `labels/{train,val,test}` before writing,
uses a private seeded RNG, and fails if any image is in two splits (checked in memory and on disk).
With the real NEU-DET the default 70/15/15 split gives **1,260 train / 270 val / 270 test** images
(210 / 45 / 45 per class; 2,899 / 661 / 629 boxes) and `data.yaml` gets `test: images/test`.
`--split 80/20` reproduces the Phase 1 split exactly (**1,440 / 360**, same val images).
Synthetic data is generated only with `--synthetic`.

## Fusion rule (`fusion.py`, Phase 1.1)

Per image: kept detections = conf >= per-class threshold. `final_class` = classifier class if its
confidence >= `cls_threshold`, else the top kept detection's class, else `"no defect found"`.
`alert` = any kept detection OR classifier confident. Severity = `severity.py` on the kept boxes of
`final_class`; alert without such a box -> `"Review"` (operator must check). All thresholds in
`FUSION_CONFIG`, chosen on val (`results/phase1_1/fusion_config.json`): classifier 0.64; detector
crazing 0.10, inclusion 0.50, patches 0.55, pitted_surface 0.15, rolled-in_scale 0.15, scratches 0.30.

`export_tflite.py` has no fallbacks. If the weights are missing, the checkpoint is not the 6 NEU
classes, the export fails, or the output is not `[1, 10, N]`, it raises.

## Severity rule (`severity.py`)

Ignore detections below their class's min confidence (crazing 0.25, all others 0.50; the model is under-confident on crazing, 0.25 chosen from the val sweep in `results/phase1_checks.json`). Per detection, area ratio = bbox area / image area:
`< 0.082` Low, `0.082-0.184` Medium, `0.184-0.359` High, `> 0.359` Critical. Image level = worst
detection, raised one level if there are 3+ valid defects. All thresholds are in `SEVERITY_CONFIG`.
Per-class area weights exist but are all 1.0 (class-neutral) until we have domain input.

The area cutoffs are the 40th / 70th / 90th percentiles of the 866 ground-truth box area ratios in
the val split (`scripts/tune_severity.py` -> `results/severity_thresholds.json`), so ~40 % of boxes
are Low, 30 % Medium, 20 % High, 10 % Critical before escalation. **They were tuned on NEU 200x200
crops and must be re-tuned on real camera images in Phase 3.**

## Dataset

NEU Surface Defect Database (NEU-DET), Northeastern University (China): 1,800 grayscale
200x200 images, 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale,
scratches), 300 images per class. Cite: K. Song and Y. Yan, "A noise robust method based on
completed local binary patterns for hot-rolled steel strip surface defects", *Applied Surface
Science*, 2013; Y. He et al., "An End-to-end Steel Surface Defect Detection Approach via Fusing
Multiple Hierarchical Features", *IEEE TIM*, 2020.
