#!/usr/bin/env python3
"""
Phase 1.2: put the classifier v2 outputs in the contract order (6 NEU classes, then not_steel).

Ultralytics builds a classification dataset with torchvision ImageFolder, which sorts the class
folders ALPHABETICALLY: crazing, inclusion, not_steel, patches, pitted_surface, rolled-in_scale,
scratches -> not_steel would be output 2. The app contract (MODEL_IO.md, fusion.CLS7_CLASSES)
needs it LAST. This script permutes the rows of the final Linear layer (weight and bias) of the
trained checkpoint and rewrites ``names``, which reorders the softmax outputs exactly (softmax is
permutation-equivariant). It does this for every model stored in the checkpoint ("model" and
"ema"). The notebook then checks on real images that new_probs == old_probs[perm].

Note: after reordering, Ultralytics ``model.val(data=<ImageFolder>)`` would compare the new
output indices with the alphabetical folder indices and report a wrong top-1. Evaluate with
``fusion_eval`` / the notebook's own confusion matrix instead.

Usage:
    python phase1-vision/scripts/reorder_cls_classes.py --src best.pt --dst best_cls7_ordered.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tflite_utils import CLS7_CLASSES  # noqa: E402


def permutation(current: Sequence[str], target: Sequence[str]) -> List[int]:
    """perm with target[i] == current[perm[i]]; new_output[i] = old_output[perm[i]]."""
    if sorted(current) != sorted(target) or len(set(current)) != len(current):
        raise ValueError(f"class sets differ or repeat: {list(current)} vs {list(target)}")
    return [list(current).index(c) for c in target]


def names_list(names) -> List[str]:
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _final_linear(model):
    head = model.model[-1]
    lin = getattr(head, "linear", None)
    if lin is None or not hasattr(lin, "weight"):
        raise AssertionError(f"classifier head {type(head).__name__} has no .linear layer")
    return lin


def reorder_checkpoint(src: Path, dst: Path, target: Sequence[str] = CLS7_CLASSES) -> Dict:
    import torch

    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    done, perm, before = [], None, None
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is None:
            continue
        cur = names_list(m.names)
        p = permutation(cur, target)
        if perm is not None and p != perm:
            raise AssertionError(f"'{key}' class order {cur} differs from the other model in the checkpoint")
        perm, before = p, cur
        lin = _final_linear(m)
        if lin.weight.shape[0] != len(target):
            raise AssertionError(f"final layer has {lin.weight.shape[0]} outputs, expected {len(target)}")
        idx = torch.tensor(p, dtype=torch.long)
        with torch.no_grad():
            lin.weight.copy_(lin.weight[idx].clone())
            if lin.bias is not None:
                lin.bias.copy_(lin.bias[idx].clone())
        m.names = {i: c for i, c in enumerate(target)}
        done.append(key)
    if not done:
        raise AssertionError(f"{src}: no 'model' or 'ema' in the checkpoint")
    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"]["note_class_order"] = f"outputs reordered by reorder_cls_classes.py, perm {perm}"
    torch.save(ckpt, str(dst))
    info = {"src": str(src), "dst": str(dst), "before": before, "after": list(target), "perm": perm, "models": done}
    print(f"[reorder] {before} -> {list(target)} (perm {perm}) in {done} -> {dst}")
    return info


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.src.resolve() == a.dst.resolve():
        ap.error("--dst must differ from --src (keep the trained checkpoint)")
    return reorder_checkpoint(a.src, a.dst)


if __name__ == "__main__":
    main()
