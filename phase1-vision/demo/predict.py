"""
Phase 1 demo CLI: prints the result card and saves annotated images to demo/out/.

    python phase1-vision/demo/predict.py <image or folder> [--detector fp32|int8]

A folder is read non-recursively. Exit code 0 = every image processed, 1 = at least one image could
not be processed, 2 = bad input or missing model files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import UnidentifiedImageError

from pipeline import DefectPipeline, ModelFilesError, annotate, format_card

OUT_DIR = Path(__file__).resolve().parent / "out"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"no such file or folder: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 defect demo: TFLite detector + classifier, fusion, severity.")
    parser.add_argument("path", type=Path, help="an image, or a folder of images")
    parser.add_argument("--detector", choices=["fp32", "int8"], default="fp32", help="detector model (default: fp32)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    try:
        images = find_images(args.path)
        if not images:
            raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in folder: {args.path}")
        pipeline = DefectPipeline([args.detector])
    except (FileNotFoundError, ModelFilesError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for image in images:
        try:
            result = pipeline.analyze(image, args.detector)
            saved = OUT_DIR / f"{image.stem}_{args.detector}.png"
            annotate(image, result.boxes).save(saved)
        except (OSError, UnidentifiedImageError) as e:
            print(f"ERROR: cannot process {image}: {e}\n", file=sys.stderr)
            failed += 1
            continue
        print(format_card(result))
        print(f"{'Saved:':<19}{saved}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
