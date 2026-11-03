# Phase 2: On-Device LLM & ExecuTorch Optimization (Pipeline B)

**Project:** Real-Time Industrial Defect Detection & Quality-Control Assistant  
**Theme:** Manufacturing & Industry 4.0  
**Responsible Team Member:** Student 2 (macOS Apple Silicon / Intel)  
**Handover Consumer:** Student 1 (Android Edge Integration)  

---

## 1. Overview & Dual-Pipeline Architecture

In this system, high-throughput visual perception (Pipeline A: YOLOv8n TFLite) is paired with a deterministic cognitive advisory engine (Pipeline B: ExecuTorch Llama 3.2 1B).

```
+--------------------------------------------------------------------------------+
| PIPELINE A: Visual Perception (YOLOv8n TFLite)                                 |
| Frame (640x640) -> BBox + Defect Class + Confidence >= 0.50 + Area Ratio       |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| PIPELINE B: Cognitive Advisory (ExecuTorch Llama 3.2 1B Instruct)              |
| Defect Telemetry -> SOP-Grounded Prompt -> ExecuTorch XNNPACK -> Advisory JSON|
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| OPERATOR HUD & LOCAL PERSISTENCE (Android Compose + SQLite / Room)             |
| UI Alert Card + Stop-Line Mandate + Local Audit Trail Record                   |
+--------------------------------------------------------------------------------+
```

When Pipeline A detects a surface defect above threshold, the defect metadata is mapped into a structured prompt. Pipeline B computes:
1. **Defect Root Cause** (metallurgical and mechanical diagnosis)
2. **Severity Level** (`Low`, `Medium`, or `Critical`)
3. **Actionable Operator Advisory** (immediate machine/stand adjustments)
4. **Immediate Safety Protocol** (operator PPE, safety lockout, or conveyor stop mandate)

All inference runs **100% locally and offline** on device hardware using PyTorch ExecuTorch with XNNPACK targeting Arm CPUs.

---

## 2. Directory Structure

```
phase2-reasoning/
├── README.md                           # Operational playbook for Student 2
├── prompts/
│   ├── __init__.py                     # Module exports
│   └── qc_system_prompt.py             # Prompt builder, domain SOPs, & output parser
├── scripts/
│   ├── __init__.py
│   ├── setup_executorch.py             # Environment & backend diagnostic validator
│   └── export_llama_pte.py             # Lowering, INT4/INT8 quantization, & .pte exporter
├── models/
│   ├── .gitkeep
│   ├── llama3_2_1b_int4_xnnpack.pte    # Compiled ExecuTorch binary for Arm CPU
│   └── llama_pte_metadata.json         # Android runtime specs & memory budget
└── eval/
    ├── __init__.py
    └── test_reasoning_pipeline.py      # Automated 13-point test & validation suite
```

---

## 3. Domain Prompt Engineering (`prompts/qc_system_prompt.py`)

### Input Telemetry Contract (`DefectContext`)
| Field | Type | Description | Bounds |
| :--- | :--- | :--- | :--- |
| `defect_class` | `str` | Detected defect (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) | Non-empty |
| `confidence` | `float` | YOLO detector confidence score | `[0.0, 1.0]` |
| `bbox_area_ratio` | `float` | Ratio of defect bounding box area to total image area | `[0.0, 1.0]` |
| `surface_type` | `str` | Substrate type (`hot_rolled_steel`, `cold_rolled_steel`, `galvanized_sheet`, etc.) | Non-empty |

### Output Reasoning Contract (`QCAdvisoryResult`)
Outputs are strictly parsed into:
- **`Defect Root Cause`**: Metallurgical or mechanical origin.
- **`Severity Level`**: `Low` | `Medium` | `Critical`.
- **`Actionable Operator Advisory`**: 2–3 concise corrective actions for machine operators.
- **`Immediate Safety Protocol`**: Safety containment, PPE, or line-stop mandate.
- **`stop_line_recommended`**: Boolean flag (`True` if `Severity Level == Critical`).

### Usage Example:
```python
from phase2_reasoning.prompts.qc_system_prompt import (
    DefectContext,
    QCPromptBuilder,
    QCOutputParser,
)

# 1. Create defect context from vision pipeline
ctx = DefectContext(
    defect_class="rolled-in_scale",
    confidence=0.91,
    bbox_area_ratio=0.11,
    surface_type="hot_rolled_steel",
)

# 2. Generate Llama 3.2 Instruct chat formatted prompt
prompt = QCPromptBuilder.format_llama3_chat(ctx)

# 3. Parse LLM inference output
llm_output = '{"defect_root_cause": "Descaler header clog", "severity_level": "Critical", ...}'
advisory = QCOutputParser.parse(llm_output, ctx)
print(advisory.to_markdown())
```

---

## 4. Environment Setup for Student 2 (macOS Apple Silicon)

Student 2 runs on macOS to compile and lower Llama 3.2 via ExecuTorch.

### Step 1: Create Python 3.11 Environment
```bash
conda create -n executorch-qc python=3.11 -y
conda activate executorch-qc
```

### Step 2: Install PyTorch Nightly & ExecuTorch
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install --pre torchao --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install --pre executorch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### Step 3: Install HuggingFace & Tokenizers
```bash
pip install transformers huggingface_hub accelerate tiktoken sentencepiece
huggingface-cli login
```

### Step 4: Validate Environment
```bash
python phase2-reasoning/scripts/setup_executorch.py
# Or machine-readable check:
python phase2-reasoning/scripts/setup_executorch.py --json
```

---

## 5. Lowering, Quantization & `.pte` Export (`scripts/export_llama_pte.py`)

### Mobile Memory Planning & Precision Trade-offs
ExecuTorch runtime calculates static arena memory to guarantee execution without Android OOM crashes:

| Quantization | Weight Size | KV Cache (512 seq) | Peak Activations | Runtime Arena | Total RAM | Mobile Budget (<1.2 GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INT4 (Recommended)** | **~636 MB** | **16 MB** | **32 MB** | **64 MB** | **~748 MB** | **PASSED (Optimal)** |
| **INT8** | ~1,180 MB | 16 MB | 32 MB | 64 MB | ~1,292 MB | WARNING (Tight on 4GB RAM) |
| **FP16** | ~2,360 MB | 16 MB | 32 MB | 64 MB | ~2,472 MB | FAILED (Exceeds Mobile RAM) |

### Export Commands

#### Dry-run verification (any workstation):
```bash
python phase2-reasoning/scripts/export_llama_pte.py --dry-run
```

#### Production compilation on macOS (Student 2):
```bash
# 4-bit weight-only quantization via XNNPACK backend (Target: Arm64 Mobile CPU)
python phase2-reasoning/scripts/export_llama_pte.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct \
    --quantization int4 \
    --max-seq-len 512 \
    --backend xnnpack \
    --output-dir phase2-reasoning/models
```

---

## 6. Self-Test & Verification Suite (`eval/test_reasoning_pipeline.py`)

Run the 13 automated unit and integration tests:
```bash
python -m unittest phase2-reasoning/eval/test_reasoning_pipeline.py -v
```

Tests verify:
- Bounding box telemetry and input bounds validation.
- Prompt formatting for all 6 NEU defect classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).
- Output parsing across valid JSON, markdown code blocks, and raw regex fallbacks.
- Severity escalation logic (e.g., `bbox_area_ratio > 0.10` triggering critical stop-line mandates).
- Environment diagnostics and memory budget calculations.
- Dry-run export container validation (`ET01` binary magic header).

---

## 7. Android Handover Bundle (For Student 1)

Upon export completion, Student 2 hands over the following bundle to Student 1:

1. **Compiled ExecuTorch Model:** `phase2-reasoning/models/llama3_2_1b_int4_xnnpack.pte`
2. **Runtime Configuration:** `phase2-reasoning/models/llama_pte_metadata.json`
3. **Tokenizer Assets:** `tokenizer.bin` (BPE vocabulary compatible with ExecuTorch C++ runtime)
4. **Prompt Format Contract:** Documented in Section 3 above and implemented in `qc_system_prompt.py`
