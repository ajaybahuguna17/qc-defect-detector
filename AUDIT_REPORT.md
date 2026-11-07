# PS5 Submission Audit: qc-defect-detector

**Challenge:** Bharat AI-SoC Challenge 2026–27, Problem Statement 5: Real-time industrial defect detection & QC assistant
**Audit date:** 2026-11-07. Phase 1 deadline is 2026-11-23, **16 days away**.
**Commit audited:** `8b19e4e` (branch `master`, remote `github.com/ajaybahuguna17/qc-defect-detector`)
**Method:** I read every tracked file and the untracked model files. I ran the tests, the scripts and the models on a **copy of the repo in a temp directory**, so nothing in the repo was changed except this report. Python 3.14.7 on Windows / AMD Ryzen 3 7320U (x86, **not Arm**).

---

## 0. Bottom line

The repository has **a working trained defect detector (`.pt`) and a well-written prompt/parser module. Nothing else in the PS5 pipeline actually runs.**

- **The mandatory Llama + ExecuTorch requirement is not met.** No Llama weights are ever loaded. The committed `.pte` is a 1 KB placeholder filled with the repeating bytes `AA BB CC DD`. No code anywhere runs LLM inference.
- **The "verified" TFLite models are the wrong model.** Both `.tflite` files output `[1, 84, 8400]`, which means **80 COCO classes**. They were exported from stock `yolov8n.pt`, not from the defect checkpoint. That is why their metadata reports mAP50 = 0.0001.
- **There is no application.** There is no camera loop, alerting, logging, UI or Android code. Phase 3 has not started.
- **No edge benchmarks exist** for either pipeline on any Arm device.
- **Several PROJECT_CONTEXT.md claims are false**: "verified TFLite", "Completed" `.pte` export, "<750 MB RAM", and the training-run status. See §5.

Getting from here to a submittable state in 16 days is achievable. It needs focused work on real ExecuTorch export and runtime, plus a minimal end-to-end app. See §8.

---

## 1. What the project currently does

| Area | Reality today |
|---|---|
| **Planned architecture** | Android app: CameraX → YOLOv8n TFLite (Pipeline A) → prompt → Llama-3.2-1B ExecuTorch/XNNPACK (Pipeline B) → Compose UI + Room DB. |
| **Hardware target** | Arm Android phone, which PS5 allows ("Arm Android with camera"). No Raspberry Pi or Hailo path. No Android code exists. |
| **Dataset** | Real **NEU-DET**: 1,800 images and 1,800 VOC XMLs in `phase1-vision/data/raw/` (gitignored). Split into 1,442 train / 363 val. That is 1,805 files: **5 images are in both train and val** (see §4). |
| **Vision model** | `phase1-vision/models/best_neu_yolov8n.pt` is a real 6-class YOLOv8n. I validated it on the project val split: **mAP50 0.861, mAP50-95 0.559, P 0.806, R 0.799** (per-class: crazing is weakest at 0.661). It is **not produced by this repo**: its checkpoint says it was trained 2025-10-24 in Colab on `/content/neudet.yaml` for 100 epochs, resumed from `last.pt`. Its own recorded val mAP50 is 0.741, so 0.861 on this split may be inflated by overlap with its unknown training split. `best_neu_yolov8n.onnx` is correctly 6-class (`[1,10,8400]`). |
| **TFLite models** | `models/exported/defect_yolov8n_{fp16,int8}.tflite` are **COCO yolov8n**, not the defect model (output `[1,84,8400]`). Input is NCHW float32 `[1,3,640,640]`, not the documented NHWC `[1,640,640,3]`. The "int8" file also takes float32 input. |
| **LLM part** | `phase2-reasoning/prompts/qc_system_prompt.py` is a solid prompt builder: Llama 3 chat template, defect knowledge base, JSON and regex parsing into root cause / severity / advisory / safety plus a stop-line flag. `export_llama_pte.py` **does not export Llama** (see §3). No inference code. |
| **UI** | None. |
| **Logging** | None. No SQLite, CSV or JSON logging of inspections. |
| **Tests** | `phase2-reasoning/eval/test_reasoning_pipeline.py`: 13 tests, **all pass**. They cover string formatting, parsing and the placeholder file. **No test touches a real model.** |

---

## 2. Requirement checklist

### Functional requirements (PS5)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Detect surface defects, cracks, anomalies | **PARTIAL** | A real detector exists (`phase1-vision/models/best_neu_yolov8n.pt`, untracked, mAP50 0.861) but nothing in the repo runs it on images or video at runtime. `export_tflite.py` only does a single verification inference. The deployable TFLite files are the wrong model. There is no anomaly detection for unseen defect types: NEU-DET only covers 6 known classes. |
| 2 | Classify defect type **and severity** | **PARTIAL** | Type comes from the YOLO classes. Severity exists only as LLM prompt rules plus parser heuristics (`qc_system_prompt.py:181-185`, `:281-288`, `:323-325`), which are inconsistent with each other and never run on a real LLM. There is no deterministic severity from vision. |
| 3 | Alert operators with actionable recommendations | **MISSING** | `QCAdvisoryResult.to_markdown()` formats a card, but no alert mechanism or UI exists, and no LLM produces the recommendations. |
| 4 | Log inspection results | **MISSING** | No code. Room/SQLite is only planned (PROJECT_CONTEXT.md §1.4, Phase 3). |
| 5 | Real-time at the edge, no cloud | **MISSING / UNPROVEN** | No real-time loop and no Arm measurements. The only network use is the one-time dataset download (`prepare_data.py:75`, notebook cell 6), which is acceptable. **No cloud API calls anywhere**, which is good. |

### Mandatory technical requirement

| Requirement | Status | Evidence |
|---|---|---|
| Llama-family model used for advisory | **MISSING** | No Llama weights, tokenizer or checkpoint anywhere. `--model-id` is accepted by `export_llama_pte.py` but **never used**. |
| Exported to `.pte` with ExecuTorch (XNNPACK) | **MISSING** (placeholder) | `phase2-reasoning/models/llama3_2_1b_int4_xnnpack.pte` is 1,056 bytes: `"ET01"`, 28 zero bytes, then `AA BB CC DD` × 256. It was written by `_lower_dry_run_simulation()` (`export_llama_pte.py:263-271`). |
| Inference run through the ExecuTorch runtime on Arm CPU | **MISSING** | No runtime code (Python, C++ or Android). ExecuTorch is not installed in any Python on this machine. |
| Vision and LLM pipelines separate and documented | **PARTIAL** | Separated in the design (PROJECT_CONTEXT.md §3 diagram) and in the folder layout. No implementation to document. |

### Deliverables

| Deliverable | Status | Evidence |
|---|---|---|
| Source code on GitHub | **PARTIAL** | Repo exists and is pushed. Scripts only, no application. |
| Application | **MISSING** | No Android, Pi or desktop app. |
| Demonstration video | **MISSING** | None. |
| Technical report | **MISSING** | No report. No root `README.md`. Phase READMEs exist (`phase1-vision/README.md`, `phase2-reasoning/README.md`). |

---

## 3. Is the Llama + ExecuTorch part real? **No.**

`phase2-reasoning/scripts/export_llama_pte.py` has three paths, and **none of them produces a Llama model**:

1. **Default when ExecuTorch is not installed (always, on this machine):** falls into `_lower_dry_run_simulation()`. It sleeps 3 × 50 ms, prints "Simulating…", and writes the fake 1 KB file. **The run reports `EXPORT PIPELINE COMPLETED SUCCESSFULLY` even without `--dry-run`**, and the metadata then records `"dry_run": false`. I reproduced this: without `--dry-run`, the output had `{'dry_run': False}`. A teammate or judge could reasonably believe a real export happened.
2. **"Native" path when ExecuTorch is installed (`:215-257`):** builds a toy `EdgeAttentionBlock` (two `Linear` layers and a SiLU on random weights) and lowers *that* to a `.pte`. It is not Llama, has no quantization, and the `--quantization` flag is ignored.
3. **On any exception in the native path:** silently falls back to the fake file again and still returns `SUCCESS`.

Other details:
- The "mobile memory plan" (`calculate_mobile_memory_plan`, `:68-117`) is arithmetic from hard-coded hyperparameters. The "<750 MB RAM" figure is an **estimate, not a measurement**. The activation term `seq_len × hidden × 4 × 8` is a made-up factor.
- `test_exporter_dry_run_execution` asserts the first 4 bytes are `ET01`. In a real ExecuTorch `.pte` (a FlatBuffer) the first 4 bytes are a root offset, and the file identifier (e.g. `ET12`) sits at bytes 4–7. **This test will fail against a genuine export.**
- No tokenizer handling. The README handover lists `tokenizer.bin`, but nothing produces it.
- `setup_executorch.py` reports `STATUS: ENVIRONMENT READY` even when ExecuTorch is "NOT INSTALLED" and Python 3.14 is out of range.

**Cloud / non-ExecuTorch runtimes:** I grepped for `openai`, `anthropic`, `ollama`, `llama_cpp`, `transformers` usage, `from_pretrained`, `InferenceClient`, `requests` and API keys. **No cloud or alternative LLM runtime is used in code.** `transformers` and `huggingface_hub` appear only as packages that `setup_executorch.py` checks for, and in install instructions. The only network calls are dataset downloads (Google Drive URL, optional `kagglehub`). (Ollama is installed on this PC per PATH, but the project does not use it. Keep it that way.)

**What "real" looks like:** export with the official ExecuTorch Llama flow, e.g. `python -m executorch.examples.models.llama.export_llama` or the newer `executorch.extension.llm.export` / `optimum-executorch`, with XNNPACK delegation (`-X`), 8da4w quantization, KV cache and SDPA. The result is a **~1 GB** `.pte` for Llama-3.2-1B, plus `tokenizer.model`. Then run it with the ExecuTorch `llama_main` runner (C++), the Android `LlmModule` (Java/Kotlin), or the Python runner, and record real TTFT and tokens/s. Check the current ExecuTorch docs for exact flags, because they change between releases. The export needs Linux or macOS (or WSL2) with Python 3.10–3.12.

---

## 4. Bugs and issues found

### Correctness / integrity (high impact)
1. **Fake `.pte` committed as the deliverable**: `phase2-reasoning/models/llama3_2_1b_int4_xnnpack.pte`. It is gitignored (`*.pte`), but `llama_pte_metadata.json` *is* committed and describes it as an "ExecuTorch PTE Binary" with `mobile_budget_pass: true`.
2. **Wrong model in the TFLite handover**: `export_tflite.py:335-349`. If the weights path is missing it **silently switches to COCO `yolov8n.pt`**. If the TFLite export fails (it does on Windows), it copies any `test_yolov8n.tflite` lying around (`:405-410`). Both shipped `.tflite` files are COCO, 80 classes. `labels.txt` still lists 6 defect classes, so Android would map COCO class 0 ("person") to "crazing".
3. **`model_metadata.json` on disk does not come from `export_tflite.py`.** It carries notebook-style fields (`mAP50: 0.00013`), so it came from a run of the notebook's validation cell against the wrong model. The notebook makes the same mistake: cell 10 falls back to `yolov8n.pt` if `best.pt` is missing.
4. **Train/val leakage: 5 images are in both splits** (`crazing_1.jpg`, `crazing_10.jpg`, `crazing_101.jpg`, `crazing_108.jpg`, `crazing_109.jpg`, …). These are stale files from an earlier run, and `prepare_data.py` does not clear the output dirs before copying. Counts are 1,442/363, not the claimed 1,440/360.
5. **Documented tensor layout is wrong**: the metadata, notebook and PROJECT_CONTEXT all say `[1, 640, 640, 3]`. The actual TFLite input is `[1, 3, 640, 640]`. Re-check after re-export, because Ultralytics TFLite is usually NHWC. The NCHW layout is a further sign these came from an unusual export path.
6. **Severity logic is inconsistent and fails "Critical"** (`qc_system_prompt.py`):
   - Prompt rule: Critical if area > 0.10. JSON-path override: only if area > 0.15, or > 0.08 with conf > 0.90. Regex fallback: area > 0.10 **or conf ≥ 0.85**. These are three different rules.
   - `parse_qc_response("model crashed", "scratches")` returns **Critical** with stop-line, because the helper hard-codes conf 0.85 (`:405`).
   - A tiny scratch (area 0.005, conf 0.86) with unparseable output also becomes **Critical**, and so stops the line.
   - `_clean_severity("not high risk")` returns **Critical** (substring match on "high").
   - In a factory, false line-stops are expensive. Severity should be deterministic from vision features, with the LLM explaining it rather than deciding it.
7. **`setup_executorch.py` says "READY"** with ExecuTorch missing.

### Portability (Linux / Raspberry Pi / Colab)
8. **Hard-coded Windows absolute path committed**: `phase1-vision/data/dataset/data.yaml:5`, `path: C:/Users/Ajay/Desktop/qc-defect-detector/...`. It breaks training and val on any other machine. Use a relative path, or regenerate it.
9. All script defaults are **CWD-relative** (`"phase1-vision/models/..."`, `"phase2-reasoning/models"`). Run from any other directory, they create nested folders or miss files. Resolve paths from `Path(__file__)`.
10. Hyphenated package dirs (`phase1-vision`, `phase2-reasoning`) cannot be imported as Python packages. The README example `from phase2_reasoning.prompts...` fails with `ModuleNotFoundError` (verified). The tests work only because they edit `sys.path`.
11. No other `\`-style or `C:` paths were found in code. `pathlib` is used consistently, which is good.

### Dependencies / environment
12. **No `requirements.txt`, `pyproject.toml` or `environment.yml` anywhere.** Needed at minimum: `ultralytics`, `numpy`, `pillow`, `ai-edge-litert` (or `tflite-runtime` on Pi), `torch`, `executorch`, `torchao`, `pytest` (optional). Python **3.10–3.12** is required for ExecuTorch. The dev machine runs **3.14** (ExecuTorch absent); 3.12 is installed but has no packages.
13. Notebook cell 4 installs `ultralytics tensorflow` with no version pins, and none of the notebook cells has been executed (no outputs saved), so there is no evidence the notebook runs end to end.

### Missing model files (fresh clone)
14. A fresh `git clone` has **no models at all**: `.pt`, `.onnx`, `.tflite` and `.pte` are all gitignored (correctly, for size), but there is no download script or GitHub Release. Judges cannot reproduce anything.
15. No Llama tokenizer, no real `.pte`, and no correct TFLite.

### Secrets / repo hygiene
16. **No secrets found**: no API keys, HF tokens or passwords. ✅
17. Untracked root files `yolov8n.pt` and `yolov8n.onnx` (stock COCO) are the source of the wrong-model fallbacks. Delete them or move them out of the repo.
18. No `LICENSE`. Using Llama requires following the Llama 3.2 Community License, including "Built with Llama" attribution. NEU-DET also needs a citation.
19. Committed `__pycache__` is not an issue: it is ignored and untracked.

### Test run results
```
python -m unittest discover -s phase2-reasoning/eval -t phase2-reasoning -v  → Ran 13 tests, OK
python -m unittest phase2-reasoning/eval/test_reasoning_pipeline.py         → Ran 13 tests, OK
python phase2-reasoning/prompts/qc_system_prompt.py                         → self-test OK (parses a hard-coded "simulated" response)
```
The tests pass, but they only prove the string handling works. They give no evidence for any PS5 requirement.

---

## 5. PROJECT_CONTEXT.md claims vs reality

| Claim (PROJECT_CONTEXT.md) | Reality |
|---|---|
| "local verified TFLite model (`defect_yolov8n_fp16.tflite`)" (§4, Phase 1) | It is COCO yolov8n, not the defect model. "Verification" ran it on one image with the wrong class mapping. |
| "Verified `export_tflite.py` with real-image inference" (Changelog) | The export failed on Windows and fell back to a cached COCO TFLite. See §4.2. |
| "1,440 train / 360 val" | 1,442 / 363, with 5 duplicate images across splits. |
| "Baseline trained NEU-DET checkpoint (`best_neu_yolov8n.pt`)" (Completed) | True that it is a NEU model. But it was trained in 2025 elsewhere (100 epochs, its own val mAP50 0.741), not by this repo's notebook (50 epochs). Provenance must be stated or the model retrained. |
| "`export_llama_pte.py`: INT4/INT8/FP16 quantization, XNNPACK Arm CPU delegation, memory planning (<750MB RAM for INT4)" (Completed) | No quantization is performed and no Llama is loaded. XNNPACK is applied only to a toy 2-layer block, and only if ExecuTorch is installed. <750 MB is a formula output, not a measurement. Note: a real 1B 8da4w export is ~1 GB on disk. |
| "13 passing unit and integration tests" | True, but none is an integration test of a real model. |
| "Vision models must be exported with input `[1, 640, 640, 3]`" | Current files are `[1,3,640,640]`. |
| Vision target "15–30 FPS", LLM "TTFT < 1.5 s" | No measurements exist. |
| "Room / SQLite" logging | Not implemented. |
| Status header "Phase 1 & Phase 2 Active", Phase 2 items "Completed" | Phase 2's core deliverable (a real `.pte`) is not done. |
| Diagram: "YOLOv8n INT8/FP16 TFLite / GPU or NNAPI delegate" | Fine as a design. Note that NNAPI is deprecated on recent Android, so use the GPU delegate or XNNPACK. |

---

## 6. Benchmarking

**Status: MISSING.** There is no latency, FPS, memory, power or thermal measurement code, and no results, for either pipeline on Arm.

For reference only, measured during this audit on the **x86 dev laptop (Ryzen 3 7320U, 4 threads), so not representative of Arm**:

| Model | Runtime | Latency / image |
|---|---|---|
| `best_neu_yolov8n.pt` | PyTorch CPU (Ultralytics val, 640) | 94.9 ms inference + 3.7 ms pre/post |
| `defect_yolov8n_fp16.tflite` (COCO!) | LiteRT + XNNPACK | 106 ms |
| `defect_yolov8n_int8.tflite` (COCO!) | LiteRT + XNNPACK | 68 ms |
| Llama `.pte` | — | **cannot run (placeholder)** |

Takeaway: at 640×640, YOLOv8n will **not** reach 15–30 FPS on a phone CPU or Pi 5 CPU without INT8 quantization or a GPU delegate, and probably not without a smaller input. **NEU images are only 200×200, so training and inference at `imgsz=320` (or 256) is justified** and roughly 4× cheaper.

What judges will want, measured on the actual Arm device:
- **Vision:** mean, p50 and p95 latency, FPS of the full loop (capture → pre → infer → NMS → draw), model size, peak RSS, mAP for FP32 vs FP16 vs INT8.
- **LLM:** model load time, TTFT, decode tokens/s, peak RSS, `.pte` size, and a quality check (JSON-valid rate across the 6 classes × severities).
- **Both together:** vision FPS while the LLM is generating. This is important on a 4-core CPU.
- Device, OS, thread counts, quantization scheme and thermal state for every number.

---

## 7. README / technical report

| Item | Status |
|---|---|
| Root `README.md` | **MISSING.** The GitHub landing page shows nothing useful. |
| Setup steps | Partial, split across phase READMEs. Missing a requirements file and Python version pinning. Phase 2 instructions are macOS-only. |
| Architecture diagram | Present (Mermaid, PROJECT_CONTEXT.md §3), but it describes the planned Android system, not what exists. |
| Dataset | Named (NEU-DET) with class counts. Missing citation and licence. Download URL is a third-party Google Drive mirror. |
| Results | **MISSING.** No mAP table, confusion matrix, latency or LLM samples. |
| How to reproduce | Not possible from a fresh clone (no models, no app, Windows `data.yaml`). |
| Technical report | **MISSING.** |
| Demo video | **MISSING.** |
| Two-pipeline separation write-up | Design only. |

---

## 8. Prioritized TODO to be submission-ready by 23 Nov 2026

Effort is in person-days for one competent developer. There are 3 team members and 16 calendar days. Items marked ⛓ block other work.

### 🔴 Critical: without these the submission fails PS5's mandatory requirements

| # | Task | Effort | Owner suggestion |
|---|---|---|---|
| C0 ⛓ | **Decide the demo target today.** Native Android (current plan, 0% built), or **Arm Linux / Raspberry Pi 5 with a Python app**. The Pi route is much faster to build: ExecuTorch has a Python runtime and C++ runner, OpenCV handles the camera, and the requirement text is written with the Pi in mind. Only keep Android if someone can commit ~6–8 days to it. | 0.5 h | Team |
| C1 ⛓ | **Real Llama-3.2-1B-Instruct → `.pte` export** with the official ExecuTorch Llama flow (XNNPACK, 8da4w, KV cache), on Linux/macOS/WSL2 with Python 3.10–3.12. Commit the exact command and versions. Ship `tokenizer.model`. Replace the simulation code, or make it fail loudly instead of reporting SUCCESS. | 1–2 d | Student 2 |
| C2 ⛓ | **On-device ExecuTorch inference wrapper** (`advisor.generate(DefectContext) -> QCAdvisoryResult`) on the chosen target, reusing `qc_system_prompt.py`. Add a test that loads the real `.pte` and checks JSON-valid output. Record TTFT and tokens/s. | 1–2 d | Student 2 |
| C3 | **Re-export the correct TFLite** from `best_neu_yolov8n.pt` on Colab/Linux. Remove the COCO fallbacks in `export_tflite.py` and notebook cell 10 (fail if weights are missing). Verify output `[1,10,N]` and the real input layout. Strongly consider `imgsz=320`. If on Pi, exporting YOLO to ExecuTorch/XNNPACK or ONNX Runtime is also fine. | 0.5–1 d | Student 3 |
| C4 | **Fix the dataset split and the model provenance.** Clear the output dirs in `prepare_data.py`, regenerate the split (removes the 5 leaked images), make `data.yaml` relative, and **retrain with the repo's own notebook** so the reported mAP is reproducible and honest. Otherwise, clearly document where `best_neu_yolov8n.pt` came from. | 0.5–1 d (Colab) | Student 3 |
| C5 | **End-to-end application**: camera or video file → detect → deterministic severity → async LLM advisory (don't block the vision loop) → on-screen alert with recommendation → **SQLite log** (timestamp, class, conf, bbox, severity, advisory, latency, snapshot path). Include a log viewer or CSV export. | 4–6 d (Pi/Python) / 7–9 d (Android) | Student 1 (+ all) |
| C6 | **Deterministic severity** from class, bbox area, confidence and defect count, with one rule used everywhere. The LLM explains the severity and gives actions. The parser defaults to the rule result, never to "Critical". Fix `_clean_severity` and `parse_qc_response`. | 0.5 d | Student 2 |
| C7 | **Benchmarks on the real Arm device** (§6 list), as a script that writes `benchmarks/results.json` and a markdown table. | 1 d | Student 1/3 |
| C8 | **Deliverables**: root README (overview, diagram showing the separate vision and LLM pipelines, setup, model downloads, how to run, results), **technical report** (PDF), **demo video** (3–5 min: live detection → alert → advice → log, plus an airplane-mode proof of no cloud use). | 2–3 d | All |
| C9 | **Correct PROJECT_CONTEXT.md and `llama_pte_metadata.json`** so no claim is false. Delete the fake `.pte` and metadata. | 1 h | Anyone |

### 🟠 Important: noticeably affects judging or reproducibility

| # | Task | Effort |
|---|---|---|
| I1 | `requirements.txt` (pinned) + `requirements-export.txt` (torch/executorch versions); state Python 3.11 or 3.12 | 2 h |
| I2 | Model distribution: GitHub Release (or a `scripts/download_models.py`) for `.pt`, `.tflite` and `.pte` (~1 GB, too big for git; Releases allow 2 GB per file) | 2 h |
| I3 | Make scripts CWD-independent (`Path(__file__).resolve()`); turn the phase dirs into an importable package, or document `PYTHONPATH` | 2–3 h |
| I4 | `setup_executorch.py`: report NOT READY when ExecuTorch is missing or Python is out of range | 30 min |
| I5 | Fix the `.pte` header test to check the real FlatBuffer identifier (bytes 4–7), or drop it | 30 min |
| I6 | Proper INT8 TFLite with calibration data; report the accuracy drop vs FP32 | 0.5 d |
| I7 | Advisory quality eval: 30–50 scripted defect contexts → JSON-valid %, severity agreement with rules, a few human-rated samples; include in the report | 0.5–1 d |
| I8 | LICENSE, Llama 3.2 licence notice and "Built with Llama", NEU-DET citation | 1 h |
| I9 | Remove the stray root `yolov8n.pt` / `yolov8n.onnx` | 5 min |
| I10 | Consider whether 1B is fast enough. If TTFT is too slow, cap `max_new_tokens` around 96, shorten the system prompt (it is long, which hurts prefill), cache advisories per (class, severity), or use a SpinQuant/QAT Llama-3.2-1B checkpoint | 0.5 d |

### 🟢 Nice-to-have

| # | Task | Effort |
|---|---|---|
| N1 | Unsupervised anomaly detection (e.g. PatchCore/PaDiM on MVTec AD) to cover "anomalies" beyond the 6 trained classes | 2–3 d |
| N2 | Hailo AI HAT+ path for vision (HailoRT), keeping Llama on the CPU via ExecuTorch, as PS5 describes | 2–3 d (needs hardware) |
| N3 | Operator feedback (confirm / false-positive button) saved to the DB | 0.5 d |
| N4 | GitHub Actions CI running the unit tests | 1 h |
| N5 | Thermal and long-run soak test (30 min continuous) with an FPS-over-time plot | 0.5 d |
| N6 | Second dataset (KolektorSDD / MVTec) to show generalization | 1–2 d |

### Suggested 16-day schedule
- **Nov 7–8:** C0 decision. C9 cleanup. C1 starts (Student 2). C3 + C4 retrain on Colab (Student 3). C5 skeleton (Student 1).
- **Nov 9–12:** C1/C2 working on device. C5 vision loop + SQLite. C6.
- **Nov 13–16:** Integrate LLM into the app (async). I1–I5. First benchmarks (C7).
- **Nov 17–19:** Benchmarks final, I7 eval, README + report drafting (C8).
- **Nov 20–21:** Demo video. Fresh-clone reproducibility test on a clean machine or Pi.
- **Nov 22:** Buffer. Tag the release and submit by Nov 23.

---

*Audit performed read-only. No source files were modified. This file is the only change.*
