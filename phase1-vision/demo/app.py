"""
Phase 1 demo UI (Gradio), local and offline: python phase1-vision/demo/app.py

Same pipeline as demo/predict.py (demo/pipeline.py). Serves on 127.0.0.1 only; no share link.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # no telemetry: must be set before importing gradio

import gradio as gr  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from pipeline import DemoResult, DefectPipeline, ModelFilesError, annotate, line_decision  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
# (background, text) of the severity badge
SEVERITY_COLOURS = {
    "Low": ("#2E7D32", "#FFFFFF"),
    "Medium": ("#F9A825", "#000000"),
    "High": ("#EF6C00", "#000000"),
    "Critical": ("#C62828", "#FFFFFF"),
    "Review": ("#6A1B9A", "#FFFFFF"),
}
NO_SEVERITY_COLOURS = ("#616161", "#FFFFFF")
CAVEAT = (
    "Trained on NEU-DET steel crops only, which include no defect-free images: false alarms on good parts "
    "are not measured, and any other image gets a meaningless answer."
)


def _badge(text: str, colours: Tuple[str, str]) -> str:
    style = f"background:{colours[0]};color:{colours[1]};padding:2px 10px;border-radius:4px;font-weight:600"
    return f'<span style="{style}">{html.escape(text)}</span>'


def card_html(r: DemoResult) -> str:
    level = r.severity_level or "none"
    if r.classifier_confidence is None:
        classifier = "not run"
    else:
        classifier = f"{r.classifier_confidence:.3f} ({r.classifier_class})"
    area = "n/a (no box of the final type)" if r.largest_area_pct is None else f"{r.largest_area_pct:.1f} %"
    rows = [
        ("Decision", f"<b>{html.escape(line_decision(r))}</b>"),
        ("Alert", _badge("ALERT", SEVERITY_COLOURS["Critical"]) if r.alert else _badge("No alert", SEVERITY_COLOURS["Low"])),
        ("Final defect type", f"<b>{html.escape(r.final_class)}</b>"),
        ("Decided by", html.escape(r.decided_by)),
        ("Classifier confidence", html.escape(classifier)),
        ("Boxes", f"{r.num_boxes} kept ({r.num_final_boxes} of the final type)"),
        ("Largest area", area),
        ("Severity", _badge(level, SEVERITY_COLOURS.get(level, NO_SEVERITY_COLOURS))),
        ("Reason", html.escape(r.reason)),
        ("Detector time", f"{r.detector_ms:.1f} ms ({r.detector.upper()})"),
        ("Classifier time", f"{r.classifier_ms:.1f} ms ({r.classifier_version} FP32)"),
    ]
    body = "".join(
        f'<tr><th style="text-align:left;vertical-align:top;padding:4px 12px 4px 0;white-space:nowrap">{name}</th>'
        f'<td style="padding:4px 0">{value}</td></tr>'
        for name, value in rows
    )
    return f"<table>{body}</table>"


def build_app(pipeline: DefectPipeline) -> gr.Blocks:
    def run(image_path: Optional[str], detector: str):
        if not image_path:
            return None, ""
        try:
            result = pipeline.analyze(Path(image_path), detector.lower())
            return annotate(Path(image_path), result.boxes), card_html(result)
        except (OSError, UnidentifiedImageError) as e:
            raise gr.Error(f"Cannot read this image: {e}")

    with gr.Blocks(title="QC Defect Demo", analytics_enabled=False) as demo:
        gr.Markdown(f"# Steel surface defect demo (Phase 1)\nTFLite on this CPU, fully offline. {CAVEAT}")
        with gr.Row():
            with gr.Column():
                # image_mode=None + filepath: the model reads the original file. Gradio's default
                # image_mode="RGB" re-encodes grayscale images first, which changes the pixels.
                image = gr.Image(type="filepath", image_mode=None, sources=["upload"], label="Image")
                detector = gr.Radio(["FP32", "INT8"], value="FP32", label="Detector model (classifier is always FP32)")
                button = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    examples=sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")),
                    inputs=image,
                    examples_per_page=12,
                    cache_examples=False,
                )
            with gr.Column():
                annotated = gr.Image(
                    type="pil", format="png", label="Detections", show_label=False, interactive=False,
                    buttons=["download", "fullscreen"],
                )
                card = gr.HTML(label="Result")
        gr.on(
            triggers=[image.change, detector.change, button.click],
            fn=run,
            inputs=[image, detector],
            outputs=[annotated, card],
            api_name="analyze",
            concurrency_limit=1,  # one interpreter per model: never invoke the same one twice at once
        )
    return demo


def main() -> int:
    try:
        pipeline = DefectPipeline()
    except ModelFilesError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    build_app(pipeline).launch(server_name="127.0.0.1", share=False, show_error=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
