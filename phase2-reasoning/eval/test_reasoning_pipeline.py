"""
Comprehensive Unit & Integration Test Suite for Phase 2 Reasoning Pipeline.

Tests:
  1. DefectContext validation and edge case rejection.
  2. Llama 3.2 chat prompt generation across all 6 NEU defect classes.
  3. QCOutputParser robustness across JSON, markdown code fence, and unstructured fallback.
  4. Severity escalation and safety stop-line mandate triggers.
  5. EnvironmentChecker diagnostics and reporting.
  6. Memory plan calculations and dry-run export pipeline execution.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

# Add project root and phase2 to path
CURRENT_DIR = Path(__file__).resolve().parent
PHASE2_DIR = CURRENT_DIR.parent
PROJECT_ROOT = PHASE2_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE2_DIR))

from prompts.qc_system_prompt import (
    DefectContext,
    QCAdvisoryResult,
    QCOutputParser,
    QCPromptBuilder,
    SeverityLevel,
    format_llama3_defect_prompt,
    parse_qc_response,
)
from scripts.setup_executorch import EnvironmentChecker
from scripts.export_llama_pte import (
    LlamaPTEExporter,
    calculate_mobile_memory_plan,
)


class TestQCPromptEngineering(unittest.TestCase):
    """Test suite for domain prompt engineering and output parsing."""

    def test_defect_context_valid(self):
        """Test valid DefectContext instantiation and normalization."""
        ctx = DefectContext(
            defect_class="rolled-in_scale",
            confidence=0.88,
            bbox_area_ratio=0.045,
            surface_type="hot_rolled_steel",
        )
        ctx.validate()
        self.assertEqual(ctx.normalized_defect_name(), "rolled_in_scale")
        self.assertIn("defect_class", ctx.to_dict())

    def test_defect_context_invalid_confidence(self):
        """Test rejection of out-of-bound confidence."""
        ctx = DefectContext(
            defect_class="crazing",
            confidence=1.5,
            bbox_area_ratio=0.02,
        )
        with self.assertRaises(ValueError):
            ctx.validate()

    def test_defect_context_invalid_area(self):
        """Test rejection of out-of-bound area ratio."""
        ctx = DefectContext(
            defect_class="crazing",
            confidence=0.9,
            bbox_area_ratio=-0.1,
        )
        with self.assertRaises(ValueError):
            ctx.validate()

    def test_prompt_generation_all_six_classes(self):
        """Verify prompt templates render correctly for all 6 NEU defect classes."""
        classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
        for c in classes:
            prompt = format_llama3_defect_prompt(
                defect_class=c,
                confidence=0.85,
                bbox_area_ratio=0.05,
                surface_type="cold_rolled_steel",
            )
            self.assertIn("<|begin_of_text|>", prompt)
            self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
            self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
            self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)
            self.assertIn(c, prompt)

    def test_parser_valid_json(self):
        """Test parser with clean JSON."""
        sample_json = json.dumps({
            "defect_root_cause": "Hydraulic descaling pump pressure drop below 160 bar.",
            "severity_level": "Medium",
            "actionable_operator_advisory": "Inspect descaler header spray manifold and adjust pressure.",
            "immediate_safety_protocol": "Wear protective thermal gloves and eye shield.",
        })
        ctx = DefectContext(defect_class="rolled-in_scale", confidence=0.75, bbox_area_ratio=0.04)
        result = QCOutputParser.parse(sample_json, ctx)

        self.assertEqual(result.severity_level, SeverityLevel.MEDIUM.value)
        self.assertFalse(result.stop_line_recommended)
        self.assertIn("Hydraulic", result.root_cause)
        self.assertIn("thermal gloves", result.immediate_safety_protocol)

    def test_parser_markdown_wrapped_json(self):
        """Test parser with markdown code fence wrapper."""
        raw_text = (
            "Here is the quality control advisory for the detected defect:\n\n"
            "```json\n"
            "{\n"
            '  "defect_root_cause": "Thermal fatigue microcracking due to cooling spray gradient.",\n'
            '  "severity_level": "Critical",\n'
            '  "actionable_operator_advisory": "Halt line immediately and inspect mold primary cooling.",\n'
            '  "immediate_safety_protocol": "Conveyor emergency stop. Clear casting zone."\n'
            "}\n"
            "```\n"
        )
        ctx = DefectContext(defect_class="crazing", confidence=0.92, bbox_area_ratio=0.14)
        result = QCOutputParser.parse(raw_text, ctx)

        self.assertEqual(result.severity_level, SeverityLevel.CRITICAL.value)
        self.assertTrue(result.stop_line_recommended)
        self.assertIn("Thermal fatigue", result.root_cause)

    def test_parser_unstructured_fallback(self):
        """Test regex fallback when LLM emits plain text without JSON."""
        unstructured = (
            "Defect Root Cause: Entrapped non-metallic slag during continuous casting.\n"
            "Severity Level: Critical\n"
            "Actionable Operator Advisory: Flag coil for metallurgical rejection.\n"
            "Immediate Safety Protocol: Lockout tagout finishing stand."
        )
        ctx = DefectContext(defect_class="inclusion", confidence=0.88, bbox_area_ratio=0.09)
        result = QCOutputParser.parse(unstructured, ctx)

        self.assertEqual(result.severity_level, SeverityLevel.CRITICAL.value)
        self.assertTrue(result.stop_line_recommended)
        self.assertIn("Entrapped non-metallic", result.root_cause)

    def test_markdown_and_json_serialization(self):
        """Verify export methods on QCAdvisoryResult."""
        res = QCAdvisoryResult(
            defect_class="scratches",
            root_cause="Pinch roll guide burr.",
            severity_level="Low",
            actionable_advisory="Deburr entry guide.",
            immediate_safety_protocol="Standard PPE.",
            stop_line_recommended=False,
        )
        md = res.to_markdown()
        self.assertIn("[LOW]", md)
        self.assertIn("Pinch roll guide burr", md)

        js = res.to_json()
        data = json.loads(js)
        self.assertEqual(data["defect_class"], "scratches")
        self.assertEqual(data["severity_level"], "Low")


class TestEnvironmentChecker(unittest.TestCase):
    """Test environment checking tool."""

    def test_environment_report_structure(self):
        """Check all required keys are present in diagnostic report."""
        checker = EnvironmentChecker()
        report = checker.get_full_report()

        self.assertIn("platform", report)
        self.assertIn("python", report)
        self.assertIn("pytorch", report)
        self.assertIn("executorch", report)
        self.assertIn("quantization", report)
        self.assertIn("huggingface", report)
        self.assertIn("ready_for_export", report)

    def test_python_version_fields(self):
        """Verify python version fields."""
        checker = EnvironmentChecker()
        py_info = checker.check_python()
        self.assertIn("version", py_info)
        self.assertIn("executable", py_info)
        self.assertIn("supported", py_info)


class TestExportPipeline(unittest.TestCase):
    """Test memory planning and export dry-run execution."""

    def test_memory_plan_int4(self):
        """Verify INT4 memory budget fits within mobile constraints."""
        plan = calculate_mobile_memory_plan(quantization="int4", max_seq_len=512)
        self.assertTrue(plan.mobile_budget_pass)
        self.assertLess(plan.total_estimated_ram_mb, 1000.0)
        self.assertGreater(plan.weights_bytes, 0)
        self.assertEqual(plan.quantization, "int4")

    def test_memory_plan_int8(self):
        """Verify INT8 memory budget calculation."""
        plan = calculate_mobile_memory_plan(quantization="int8", max_seq_len=512)
        self.assertGreater(plan.total_estimated_ram_mb, 1200.0)
        self.assertEqual(plan.quantization, "int8")

    def test_exporter_dry_run_execution(self):
        """Test dry-run export produces valid files and metadata."""
        test_out_dir = PHASE2_DIR / "eval" / "test_output"
        exporter = LlamaPTEExporter(
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            output_dir=str(test_out_dir),
            quantization="int4",
            max_seq_len=256,
            backend="xnnpack",
            dry_run=True,
        )
        res = exporter.run()

        self.assertEqual(res["status"], "SUCCESS")
        pte_file = Path(res["model_path"])
        metadata_file = Path(res["metadata_path"])

        self.assertTrue(pte_file.exists())
        self.assertTrue(metadata_file.exists())

        # Verify PTE file header
        with open(pte_file, "rb") as f:
            header = f.read(4)
            self.assertEqual(header, b"ET01")

        # Verify JSON metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["model_metadata"]["quantization"], "INT4")
            self.assertEqual(meta["model_metadata"]["backend"], "XNNPACK")
            self.assertIn("crazing", meta["defect_classes"])

        # Clean up test output
        if pte_file.exists():
            pte_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        if test_out_dir.exists():
            try:
                test_out_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
