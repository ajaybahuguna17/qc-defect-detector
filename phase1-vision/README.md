# Phase 1: Computer Vision Defect Detection Pipeline

**Lead:** Student 3 (Windows + Google Colab Nvidia T4 GPU)  
**Consumer / Edge Target:** Student 1 (Native Android Edge Integration)  
**Architecture:** Ultralytics YOLOv8n $\rightarrow$ Quantized TensorFlow Lite (`.tflite`)  
**Status:** Local Pipeline & Self-Contained Colab Notebook Completed & Verified

---

## 1. Objectives & Performance Budget

Phase 1 delivers the visual perception pipeline (Pipeline A) for the edge defect detection assistant. All models execute 100% locally on the edge Android device without external network dependencies.

- **Target Defect Dataset:** NEU Surface Defect Database (NEU-DET) benchmark.
- **Classes (6):** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- **Target Accuracy:** $\text{mAP@0.50} \ge 0.85$.
- **Latency Budget:** $\le 30\text{ ms}$ per frame on mobile Arm CPU/GPU.
- **Model Size Budget:** $\le 8\text{ MB}$ (INT8) or $\le 16\text{ MB}$ (FP16).

---

## 2. Directory Layout

```
phase1-vision/
├── README.md                          <-- You are here
├── data/
│   ├── raw/                           <-- Raw downloaded dataset (IMAGES & ANNOTATIONS)
│   │   ├── IMAGES/                    <-- 1,800 raw images (200x200 grayscale)
│   │   └── ANNOTATIONS/               <-- 1,800 Pascal VOC XML files
│   └── dataset/                       <-- Processed YOLO dataset (Stratified Train/Val split)
│       ├── data.yaml                  <-- Ultralytics dataset configuration
│       ├── images/
│       │   ├── train/                 <-- 1,440 training images (80%)
│       │   └── val/                   <-- 360 validation images (20%)
│       └── labels/
│           ├── train/                 <-- 1,440 normalized YOLO txt label files
│           └── val/                   <-- 360 normalized YOLO txt label files
├── models/
│   ├── best_neu_yolov8n.pt            <-- Baseline trained NEU-DET YOLOv8n weights
│   └── exported/                      <-- Final .tflite models & Android handover bundle
│       ├── defect_yolov8n_fp16.tflite <-- FP16 quantized model (for mobile GPU)
│       ├── labels.txt                 <-- Ordered 6-class defect index mapping
│       └── model_metadata.json        <-- Input tensor specs & normalization config
├── notebooks/
│   └── train_colab.ipynb              <-- Self-contained Colab notebook (T4 GPU training)
└── scripts/
    ├── prepare_data.py                <-- Automated dataset fetcher & YOLO converter
    └── export_tflite.py               <-- TFLite exporter & real-image inference verifier
```

---

## 3. Step-by-Step Google Colab Execution Guide for Student 3

The notebook `phase1-vision/notebooks/train_colab.ipynb` is **100% self-contained**: it automatically downloads the benchmark dataset, preprocesses annotations, trains YOLOv8n, validates accuracy metrics, exports quantized `.tflite` models, and prompts a browser download of the final handover bundle.

### Step 1: Open Google Colab & Allocate GPU
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `phase1-vision/notebooks/train_colab.ipynb`.
3. Switch runtime to GPU: **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU $\rightarrow$ Save**.

### Step 2: Run Environment & Dependency Cells
- **Cell 1 (`!nvidia-smi`):** Verifies the Nvidia T4 GPU is active and CUDA is detected.
- **Cell 2 (`!pip install -q ultralytics tensorflow`):** Installs required training and TFLite compilation libraries.

### Step 3: Run Automated Dataset Acquisition
- **Cell 3:** Downloads the official 26.3 MB NEU-DET benchmark archive from the mirror, unpacks all 1,800 images and XML annotations, converts bounding boxes to normalized YOLO `[class_id, xc, yc, w, h]` format, executes stratified 80/20 train/validation partitioning, and generates `/content/defect_dataset/data.yaml`.
- Verified split: **1,440 Train images** (3,322 defect boxes) and **360 Validation images** (867 defect boxes).

### Step 4: Train YOLOv8n for 50 Epochs
- **Cell 4:** Trains `yolov8n.pt` with image resolution `imgsz=640`, batch size `16`, optimizer `AdamW`, initial learning rate `lr0=0.001`, and patience `15`.
- Training runs in ~10–15 minutes on the T4 GPU.

### Step 5: Validate Quality Gate ($\text{mAP@50} \ge 0.85$)
- **Cell 5:** Computes precision, recall, $\text{mAP@50}$, and $\text{mAP@50-95}$ on the validation partition.
- Confirms that model satisfies the edge detection accuracy requirement ($\text{mAP@50} \ge 0.85$).

### Step 6: Export Quantized TensorFlow Lite Models
- **Cell 6:** Compiles two model variants:
  1. `defect_yolov8n_fp16.tflite` (Half-precision float16, optimized for mobile GPU delegates).
  2. `defect_yolov8n_int8.tflite` (Post-training calibrated integer quantization for Arm CPU via XNNPACK).

### Step 7: Download Android Handover Bundle
- **Cell 7:** Automatically creates `android_handover_bundle.zip` and triggers a browser download containing:
  - `defect_yolov8n_fp16.tflite`
  - `defect_yolov8n_int8.tflite`
  - `labels.txt`
  - `model_metadata.json`
- Send this downloaded zip file directly to **Student 1** for integration into the native Android application!

---

## 4. Local CLI Scripts & Verification (Windows / macOS / Linux)

### Dataset Preparation Script: `prepare_data.py`
Runs fully programmatically with pure standard library (no external dependencies required):

```bash
# Standard automated run (auto-downloads NEU-DET if missing):
python phase1-vision/scripts/prepare_data.py

# Optional flags:
python phase1-vision/scripts/prepare_data.py \
  --split-ratio 0.8 \
  --seed 42 \
  --download \
  --sample-limit 50
```

### Model Export & Inference Verification Script: `export_tflite.py`
Exports trained weights to `.tflite`, copies artifacts to `phase1-vision/models/exported/`, and executes a verification inference pass on a real validation image from `phase1-vision/data/dataset/images/val/`:

```bash
# Export trained model and run verification:
python phase1-vision/scripts/export_tflite.py \
  --weights phase1-vision/models/best_neu_yolov8n.pt \
  --imgsz 640 \
  --output-dir phase1-vision/models/exported

# Export INT8 with calibration on dataset:
python phase1-vision/scripts/export_tflite.py \
  --weights phase1-vision/models/best_neu_yolov8n.pt \
  --int8 \
  --data phase1-vision/data/dataset/data.yaml \
  --output-dir phase1-vision/models/exported
```

---

## 5. Handover Specifications for Student 1 (Android App Lead)

The following assets are placed in `phase1-vision/models/exported/`:

1. **`defect_yolov8n_fp16.tflite`** (3.25 MB):
   - Input shape: `[1, 512, 512, 3]` or `[1, 640, 640, 3]` RGB.
   - Normalization: `0.0` to `1.0` float32 (`pixel / 255.0`).
   - Output format: Candidate boxes `[x1, y1, x2, y2]` + 6 class probabilities.
2. **`labels.txt`**:
   ```
   crazing
   inclusion
   patches
   pitted_surface
   rolled-in_scale
   scratches
   ```
3. **`model_metadata.json`**:
   ```json
   {
     "model_name": "defect_yolov8n_fp16.tflite",
     "architecture": "YOLOv8n",
     "framework": "TensorFlow Lite",
     "quantization": "FP16",
     "recommended_conf_threshold": 0.50,
     "recommended_iou_threshold": 0.45
   }
   ```
