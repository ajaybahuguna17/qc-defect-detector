"""
Industrial Defect Reasoning & Quality Control Prompt Engineering.

Pipeline B: Cognitive Advisory (Llama 3.2 1B Instruct)
This module defines domain-specific prompt templates, context data structures,
and parsing utilities to transform computer vision defect telemetry into actionable,
SOP-grounded operator advisories on edge devices.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class SeverityLevel(str, Enum):
    """Manufacturing defect severity classifications."""
    LOW = "Low"
    MEDIUM = "Medium"
    CRITICAL = "Critical"


# Canonical defect classes aligned with NEU Surface Defect Benchmark (Phase 1)
VALID_DEFECT_CLASSES = {
    "crazing": {
        "typical_cause": "Thermal fatigue, cooling water spray imbalance, or rapid temperature gradients during casting/hot-rolling.",
        "risk_profile": "Surface tensile microcracks leading to fatigue failure under cyclic stress.",
    },
    "inclusion": {
        "typical_cause": "Entrapped non-metallic slag, deoxidation products, or mold flux contamination during solidification.",
        "risk_profile": "Internal voids, stress concentration points, and structural weakness during bending or welding.",
    },
    "patches": {
        "typical_cause": "Localized oxidation, slab scabbing, or metal flakes re-welded onto strip surface during roughing passes.",
        "risk_profile": "Severe surface discontinuity, coating adhesion failure, and laminar tearing.",
    },
    "pitted_surface": {
        "typical_cause": "Acid over-pickling, hydraulic scale residue, or chemical corrosion cavities.",
        "risk_profile": "Localized wall-thickness loss, accelerated corrosion pitting, and galvanic degradation.",
    },
    "rolled_in_scale": {
        "typical_cause": "Hydraulic descaler header pressure drop or clogged descaling nozzles prior to hot finishing stands.",
        "risk_profile": "Abrasive iron oxide embedded into matrix, tooling wear, and catastrophic paint delamination.",
    },
    "rolled-in_scale": {
        "typical_cause": "Hydraulic descaler header pressure drop or clogged descaling nozzles prior to hot finishing stands.",
        "risk_profile": "Abrasive iron oxide embedded into matrix, tooling wear, and catastrophic paint delamination.",
    },
    "scratches": {
        "typical_cause": "Work roll mechanical friction, guide plate burrs, foreign metal debris on conveyor tables, or pinch roll misalignment.",
        "risk_profile": "Linear stress risers, aesthetic rejection, and premature notch fatigue rupture.",
    },
}

SUPPORTED_SURFACE_TYPES = [
    "hot_rolled_steel",
    "cold_rolled_steel",
    "galvanized_sheet",
    "stainless_steel",
    "aluminum_billet",
    "raw_billet",
    "general_metal",
]


@dataclass
class DefectContext:
    """
    Input telemetry from Pipeline A (Computer Vision) passed into Pipeline B.
    
    Attributes:
        defect_class: Label of detected defect (e.g., 'rolled-in_scale', 'crazing').
        confidence: Object detector confidence score [0.0, 1.0].
        bbox_area_ratio: Bounding box area divided by total frame area [0.0, 1.0].
        surface_type: Substrate material type being inspected.
        line_id: Manufacturing line or conveyor identifier.
        line_speed_mpm: Optional strip speed in meters per minute.
        metadata: Optional dictionary of additional sensor telemetry.
    """
    defect_class: str
    confidence: float
    bbox_area_ratio: float
    surface_type: str = "hot_rolled_steel"
    line_id: str = "Line-1"
    line_speed_mpm: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate telemetry sanity bounds."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence}")
        if not (0.0 <= self.bbox_area_ratio <= 1.0):
            raise ValueError(f"bbox_area_ratio must be between 0.0 and 1.0, got {self.bbox_area_ratio}")
        if not self.defect_class or not isinstance(self.defect_class, str):
            raise ValueError(f"defect_class must be a non-empty string, got '{self.defect_class}'")
        if not self.surface_type or not isinstance(self.surface_type, str):
            raise ValueError(f"surface_type must be a non-empty string, got '{self.surface_type}'")

    def normalized_defect_name(self) -> str:
        """Return standardized defect identifier string."""
        return self.defect_class.strip().lower().replace(" ", "_").replace("-", "_")

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return asdict(self)


@dataclass
class QCAdvisoryResult:
    """
    Structured reasoning output delivered to Operator UI (Phase 3) and Room DB.
    
    Attributes:
        defect_class: Detected defect class.
        root_cause: Diagnostic engineering root cause.
        severity_level: Evaluated severity ('Low', 'Medium', 'Critical').
        actionable_advisory: Concrete, actionable SOP steps for machine operator.
        immediate_safety_protocol: Safety containment instructions or line-stop mandate.
        stop_line_recommended: Boolean flag indicating if immediate conveyor halt is advised.
        raw_response: Unmodified LLM output for verification.
    """
    defect_class: str
    root_cause: str
    severity_level: str
    actionable_advisory: str
    immediate_safety_protocol: str
    stop_line_recommended: bool = False
    raw_response: str = ""

    @property
    def safety_protocol(self) -> str:
        """Alias for immediate_safety_protocol."""
        return self.immediate_safety_protocol

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Format as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate human-readable summary card for Operator UI."""
        badge = "[CRITICAL]" if self.severity_level.upper() == "CRITICAL" else (
            "[MEDIUM]" if self.severity_level.upper() == "MEDIUM" else "[LOW]"
        )
        stop_alert = "**[STOP LINE MANDATE: YES]** Immediate line halt required!\n" if self.stop_line_recommended else ""
        return (
            f"### Defect Inspection Advisory: {self.defect_class.replace('_', ' ').title()}\n\n"
            f"- **Severity Level:** {badge} ({self.severity_level})\n"
            f"{stop_alert}"
            f"- **Defect Root Cause:**\n  {self.root_cause}\n"
            f"- **Actionable Operator Advisory:**\n  {self.actionable_advisory}\n"
            f"- **Immediate Safety Protocol:**\n  {self.immediate_safety_protocol}\n"
        )


class QCPromptBuilder:
    """
    Constructs optimized prompt templates for Llama 3.2 1B Instruct.
    
    Adheres strictly to Llama 3.2 Instruct chat template formatting:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    SYSTEM_PROMPT = (
        "You are an expert On-Device Industrial Quality Control (QC) & Safety Edge Assistant "
        "deployed on a factory floor. You analyze real-time surface defect telemetry from computer "
        "vision sensors and provide immediate, deterministic root cause diagnostics and operator SOP protocols.\n\n"
        "Rules:\n"
        "1. Strictly adhere to standard steel/metal manufacturing standard operating procedures (SOP).\n"
        "2. Keep diagnostics concise, factual, and actionable for busy line operators under edge latency constraints.\n"
        "3. Provide output strictly in the following JSON schema:\n"
        "{\n"
        '  "defect_root_cause": "<1-2 sentences on metallurgical or mechanical root cause>",\n'
        '  "severity_level": "Low" | "Medium" | "Critical",\n'
        '  "actionable_operator_advisory": "<2-3 clear, numbered corrective actions for machine operator>",\n'
        '  "immediate_safety_protocol": "<immediate operator PPE, safety lockdown, or line-stop condition>"\n'
        "}\n"
        "4. Severity criteria:\n"
        "   - 'Critical': Large defect area (bbox_area_ratio > 0.10) or high-risk defect (crazing, inclusion, patches) with conf >= 0.70. Mandates conveyor halt.\n"
        "   - 'Medium': Moderate defect area (0.02 - 0.10) or defects requiring roll/nozzle adjustment without immediate shutdown.\n"
        "   - 'Low': Small blemish (bbox_area_ratio < 0.02) or minor scratch/pitting, require continuous monitoring.\n"
        "5. Output valid JSON only, without markdown fences or extraneous preamble."
    )

    @classmethod
    def build_user_prompt(cls, context: DefectContext) -> str:
        """
        Build domain-engineered user prompt from defect telemetry.
        """
        context.validate()
        norm_defect = context.normalized_defect_name()
        domain_info = VALID_DEFECT_CLASSES.get(norm_defect, {})
        known_cause = domain_info.get("typical_cause", "Mechanical guide or thermal rolling anomaly.")
        
        speed_info = f"{context.line_speed_mpm:.1f} m/min" if context.line_speed_mpm else "Standard speed"
        area_pct = context.bbox_area_ratio * 100.0

        user_content = (
            f"INSPECTION EVENT TELEMETRY:\n"
            f"- Defect Class: {context.defect_class}\n"
            f"- Detection Confidence: {context.confidence:.2f} ({context.confidence*100.0:.1f}%)\n"
            f"- Bounding Box Area Ratio: {context.bbox_area_ratio:.4f} ({area_pct:.2f}% of total inspection field)\n"
            f"- Surface Substrate Type: {context.surface_type}\n"
            f"- Production Zone: {context.line_id} (Line Speed: {speed_info})\n"
            f"- Baseline Metallurgical Correlation: {known_cause}\n\n"
            f"Generate the QC advisory JSON according to your SOP instructions."
        )
        return user_content

    @classmethod
    def format_llama3_chat(cls, context: DefectContext) -> str:
        """
        Render prompt in official Llama 3.2 Instruct token format.
        """
        user_prompt = cls.build_user_prompt(context)
        formatted = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{cls.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return formatted

    @classmethod
    def format_plain_prompt(cls, context: DefectContext) -> str:
        """
        Render flat prompt without special tokens for standard completion engines.
        """
        user_prompt = cls.build_user_prompt(context)
        return f"### SYSTEM INSTRUCTIONS:\n{cls.SYSTEM_PROMPT}\n\n### TELEMETRY:\n{user_prompt}\n\n### ADVISORY RESPONSE (JSON):\n"


class QCOutputParser:
    """
    Robust output parser for edge LLM responses.
    
    Handles strict JSON, markdown-wrapped JSON (```json ... ```),
    and unstructured key-value fallbacks when token budgets truncate output.
    """

    @classmethod
    def parse(cls, response_text: str, context: Optional[DefectContext] = None) -> QCAdvisoryResult:
        """
        Parse raw LLM output into a structured QCAdvisoryResult.
        """
        clean_text = response_text.strip()
        defect_class = context.defect_class if context else "unknown_defect"

        # 1. Strip markdown code fence if present
        json_pattern = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        candidate_json = json_pattern.group(1) if json_pattern else clean_text

        # 2. Try direct JSON parsing
        try:
            # Find outermost curly braces
            start_idx = candidate_json.find("{")
            end_idx = candidate_json.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = candidate_json[start_idx : end_idx + 1]
                data = json.loads(json_str)
                return cls._build_result_from_dict(data, defect_class, clean_text, context)
        except Exception:
            pass

        # 3. Fallback: Heuristic regex extraction for unstructured responses
        root_cause = cls._extract_field(clean_text, [
            r"defect[_\s]*root[_\s]*cause[:\s]+(.*?)(?=(?:severity|actionable|immediate|$))",
            r"root[_\s]*cause[:\s]+(.*?)(?=(?:severity|actionable|immediate|$))",
        ], default="Surface integrity anomaly detected during rolling or casting process.")

        severity = cls._extract_field(clean_text, [
            r"severity[_\s]*level[:\s]+[\"']?(Low|Medium|Critical)[\"']?",
            r"severity[:\s]+[\"']?(Low|Medium|Critical)[\"']?",
        ], default="")

        # Compute severity heuristic if not extracted
        if not severity:
            if context and (context.bbox_area_ratio > 0.10 or context.confidence >= 0.85):
                severity = SeverityLevel.CRITICAL.value
            elif context and context.bbox_area_ratio > 0.03:
                severity = SeverityLevel.MEDIUM.value
            else:
                severity = SeverityLevel.LOW.value

        advisory = cls._extract_field(clean_text, [
            r"actionable[_\s]*(?:operator)?[_\s]*advisory[:\s]+(.*?)(?=(?:immediate|safety|$))",
            r"advisory[:\s]+(.*?)(?=(?:immediate|safety|$))",
        ], default="Inspect work rolls, verify hydraulic descaler pressure, and flag coil for quality review.")

        safety = cls._extract_field(clean_text, [
            r"immediate[_\s]*safety[_\s]*protocol[:\s]+(.*)$",
            r"safety[_\s]*protocol[:\s]+(.*)$",
        ], default="Standard PPE required. Avoid direct contact with hot moving substrate.")

        severity_clean = cls._clean_severity(severity)
        is_critical = (severity_clean == SeverityLevel.CRITICAL.value)

        return QCAdvisoryResult(
            defect_class=defect_class,
            root_cause=root_cause.strip(" \n\"'"),
            severity_level=severity_clean,
            actionable_advisory=advisory.strip(" \n\"'"),
            immediate_safety_protocol=safety.strip(" \n\"'"),
            stop_line_recommended=is_critical,
            raw_response=clean_text,
        )

    @classmethod
    def _build_result_from_dict(
        cls,
        data: Dict[str, Any],
        defect_class: str,
        raw_text: str,
        context: Optional[DefectContext] = None,
    ) -> QCAdvisoryResult:
        """Map JSON dict keys into QCAdvisoryResult."""
        root_cause = str(data.get("defect_root_cause") or data.get("root_cause") or "Root cause under analysis.")
        severity_raw = str(data.get("severity_level") or data.get("severity") or "Medium")
        severity_clean = cls._clean_severity(severity_raw)

        # Safety rule: enforce Critical if context indicates high risk
        if context and (context.bbox_area_ratio > 0.15 or (context.bbox_area_ratio > 0.08 and context.confidence > 0.90)):
            severity_clean = SeverityLevel.CRITICAL.value

        advisory_val = data.get("actionable_operator_advisory") or data.get("operator_advisory") or data.get("advisory")
        if isinstance(advisory_val, list):
            advisory = " ".join([f"{i+1}. {item}" for i, item in enumerate(advisory_val)])
        else:
            advisory = str(advisory_val or "Verify roller alignment and check water nozzle pressure.")

        safety_val = data.get("immediate_safety_protocol") or data.get("safety_protocol") or "Standard safety lockout SOP applies."
        if isinstance(safety_val, list):
            safety = " ".join(safety_val)
        else:
            safety = str(safety_val)

        is_critical = (severity_clean == SeverityLevel.CRITICAL.value)

        return QCAdvisoryResult(
            defect_class=defect_class,
            root_cause=root_cause.strip(),
            severity_level=severity_clean,
            actionable_advisory=advisory.strip(),
            immediate_safety_protocol=safety.strip(),
            stop_line_recommended=is_critical,
            raw_response=raw_text,
        )

    @staticmethod
    def _clean_severity(val: str) -> str:
        """Normalize severity text to Low/Medium/Critical."""
        val_lower = val.lower()
        if "crit" in val_lower or "high" in val_lower or "halt" in val_lower or "stop" in val_lower:
            return SeverityLevel.CRITICAL.value
        elif "med" in val_lower:
            return SeverityLevel.MEDIUM.value
        elif "low" in val_lower or "minor" in val_lower:
            return SeverityLevel.LOW.value
        return SeverityLevel.MEDIUM.value

    @staticmethod
    def _extract_field(text: str, patterns: List[str], default: str) -> str:
        """Attempt regex extractions in order."""
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if match:
                res = match.group(1).strip()
                if res:
                    return res
        return default


def format_llama3_defect_prompt(
    defect_class: str,
    confidence: float,
    bbox_area_ratio: float,
    surface_type: str = "hot_rolled_steel",
    line_id: str = "Line-1",
    line_speed_mpm: Optional[float] = None,
) -> str:
    """
    Convenience function to generate a fully-formatted Llama 3.2 Instruct prompt.
    """
    ctx = DefectContext(
        defect_class=defect_class,
        confidence=confidence,
        bbox_area_ratio=bbox_area_ratio,
        surface_type=surface_type,
        line_id=line_id,
        line_speed_mpm=line_speed_mpm,
    )
    return QCPromptBuilder.format_llama3_chat(ctx)


def parse_qc_response(response_text: str, defect_class: str = "unknown") -> QCAdvisoryResult:
    """
    Convenience function to parse an LLM response string into a QCAdvisoryResult.
    """
    ctx = DefectContext(defect_class=defect_class, confidence=0.85, bbox_area_ratio=0.05)
    return QCOutputParser.parse(response_text, context=ctx)


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2 Domain Prompt Engineering Verification")
    print("=" * 70)

    sample_ctx = DefectContext(
        defect_class="rolled-in_scale",
        confidence=0.92,
        bbox_area_ratio=0.12,
        surface_type="hot_rolled_steel",
        line_id="Finishing-Stand-4",
        line_speed_mpm=18.5,
    )

    print("\n[1] Generated Llama 3.2 Chat Formatted Prompt:")
    chat_prompt = QCPromptBuilder.format_llama3_chat(sample_ctx)
    print(chat_prompt)

    # Simulated LLM response
    simulated_llm_json = (
        "{\n"
        '  "defect_root_cause": "Hydraulic descaler header pressure drop (<150 bar) or nozzle clog leading to un-cleared secondary iron oxide scale rolled into the steel matrix.",\n'
        '  "severity_level": "Critical",\n'
        '  "actionable_operator_advisory": "1. Check descaling pump pressure at Stand 4. 2. Purge and clear spray nozzles on bottom header. 3. Tag current coil section for scrap or re-grinding.",\n'
        '  "immediate_safety_protocol": "Initiate controlled line pause. Wear heat-resistant safety gloves and safety face shield prior to inspecting hydraulic descaling manifolds."\n'
        "}"
    )

    print("\n[2] Parsing Simulated LLM JSON Response:")
    advisory = QCOutputParser.parse(simulated_llm_json, sample_ctx)
    print(advisory.to_markdown())

    print("[3] Self-Test Verification Succeeded.")
