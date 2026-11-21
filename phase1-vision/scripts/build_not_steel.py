#!/usr/bin/env python3
"""
Phase 1.2: build the ``not_steel`` class of classifier v2 from public research datasets.

Sources (downloaded and extracted by notebooks/train_colab_phase1_2.ipynb; licences in
phase1-vision/DATA_SOURCES.md):
  * COCO val2017      (``val2017/*.jpg``)                         people, objects, indoor/outdoor scenes
  * DTD r1.0.1        (``dtd/images/<category>/*.jpg``, 47 cats)  textures: hard look-alikes
  * Places365 val_256 (``val_256/*.jpg`` + ``places365_val.txt`` + ``categories_places365.txt``)
                      fields, ground, roads, buildings (categories filtered by PLACES_KEYWORDS)

Steps (all deterministic from --seed):
  1. sample: COCO uniform; DTD round-robin over all 47 categories (equal per category); Places
     uniform over the images whose category matches a keyword;
  2. transform each image (so the model cannot learn "colour = not steel" or "not 200x200 =
     not steel"): with probability --crop-frac a random square crop (50-100 % of the short side)
     resized to 200x200; independently with probability --gray-frac converted to grayscale
     (stored as 3 identical channels, like NEU). Saved as JPEG q95;
  3. split 70/15/15 per source (so every split has every source), then write
     ``<out-root>/{train,val,test}/not_steel/<source>__<stem>.jpg``;
  4. verify: zero overlap between splits by source id AND by content hash (exact duplicates
     across splits raise), counts match, and write ``not_steel_manifest.json`` (per image: source,
     source file, category, split, grayscale, cropped, sha256).

The out-root's ``*/not_steel`` folders are cleared first; other class folders are not touched.
Needs Pillow.

Usage (from the repo root, in Colab):
    python phase1-vision/scripts/build_not_steel.py --coco-dir /content/ns/coco/val2017 \\
        --dtd-dir /content/ns/dtd/dtd/images --places-dir /content/ns/places/val_256 \\
        --places-val-list /content/ns/places/places365_val.txt \\
        --places-categories /content/ns/places/categories_places365.txt \\
        --out-root phase1-vision/data/cls7
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

NOT_STEEL = "not_steel"
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_SIZE = 200          # NEU crop size
JPEG_QUALITY = 95

# Places365 categories kept (substring match on the category path, e.g. "/f/field/cultivated").
PLACES_KEYWORDS: Tuple[str, ...] = (
    "field", "pasture", "farm", "wheat", "corn", "rice_paddy", "vineyard", "lawn", "yard", "park",
    "desert", "sand", "beach", "badlands", "canyon", "rock", "cliff", "mountain", "valley", "forest",
    "road", "street", "highway", "crosswalk", "parking", "driveway", "alley", "plaza", "courtyard",
    "runway", "railroad", "bridge", "building", "facade", "house", "skyscraper", "tower", "church",
    "construction_site", "village", "downtown", "residential", "patio", "porch", "playground",
)

Item = Dict[str, str]   # {"source", "path", "category", "id"}


def _images(d: Path) -> List[Path]:
    if not Path(d).is_dir():
        raise FileNotFoundError(f"folder not found: {d}")
    return sorted(p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def sample_coco(coco_dir: Path, n: int, rng: random.Random) -> List[Item]:
    imgs = _images(coco_dir)
    if len(imgs) < n:
        raise RuntimeError(f"COCO: {len(imgs)} images in {coco_dir}, need {n}")
    return [{"source": "coco", "path": str(p), "category": "", "id": f"coco__{p.stem}"}
            for p in sorted(rng.sample(imgs, n))]


def sample_dtd(dtd_images_dir: Path, n: int, rng: random.Random) -> List[Item]:
    """Round-robin over all categories (shuffled within each), so the n images cover every category."""
    cats = sorted(d for d in Path(dtd_images_dir).iterdir() if d.is_dir())
    if not cats:
        raise FileNotFoundError(f"DTD: no category folders in {dtd_images_dir}")
    pools = {c.name: rng.sample(_images(c), len(_images(c))) for c in cats}
    out: List[Item] = []
    k = 0
    while len(out) < n:
        progressed = False
        for c in sorted(pools):
            if k < len(pools[c]) and len(out) < n:
                p = pools[c][k]
                out.append({"source": "dtd", "path": str(p), "category": c, "id": f"dtd__{p.stem}"})
                progressed = True
        if not progressed:
            raise RuntimeError(f"DTD: only {len(out)} images available, need {n}")
        k += 1
    return out


def read_places_lists(val_list: Path, categories: Path) -> Dict[str, str]:
    """file name -> category path, from places365_val.txt ("name idx") and categories_places365.txt ("/a/x idx")."""
    idx2cat = {}
    for line in Path(categories).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, i = line.rsplit(maxsplit=1)
            idx2cat[int(i)] = name
    out = {}
    for line in Path(val_list).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, i = line.rsplit(maxsplit=1)
            out[name] = idx2cat[int(i)]
    return out


def places_category_ok(cat: str, keywords: Sequence[str] = PLACES_KEYWORDS) -> bool:
    return any(k in cat for k in keywords)


def sample_places(places_dir: Path, file2cat: Mapping[str, str], n: int, rng: random.Random,
                  keywords: Sequence[str] = PLACES_KEYWORDS) -> List[Item]:
    imgs = [p for p in _images(places_dir) if places_category_ok(file2cat.get(p.name, ""), keywords)]
    if len(imgs) < n:
        raise RuntimeError(f"Places365: {len(imgs)} images in matching categories, need {n}")
    return [{"source": "places365", "path": str(p), "category": file2cat[p.name], "id": f"places365__{p.stem}"}
            for p in sorted(rng.sample(imgs, n))]


def split_items(items: Sequence[Item], rng: random.Random, ratios=RATIOS) -> Dict[str, List[Item]]:
    """70/15/15 per source (train gets the rounding remainder)."""
    out: Dict[str, List[Item]] = {sp: [] for sp in SPLITS}
    for src in sorted({it["source"] for it in items}):
        group = sorted((it for it in items if it["source"] == src), key=lambda it: it["id"])
        rng.shuffle(group)
        n = len(group)
        n_val, n_test = round(n * ratios[1]), round(n * ratios[2])
        n_train = n - n_val - n_test
        out["train"] += group[:n_train]
        out["val"] += group[n_train:n_train + n_val]
        out["test"] += group[n_train + n_val:]
    return out


def transform(img, rng: random.Random, gray_frac: float, crop_frac: float, size: int = OUT_SIZE):
    """-> (PIL RGB image, grayscale?, cropped?)."""
    from PIL import Image
    img = img.convert("RGB")
    cropped = rng.random() < crop_frac
    if cropped:
        w, h = img.size
        side = max(1, int(min(w, h) * rng.uniform(0.5, 1.0)))
        x0, y0 = rng.randint(0, w - side), rng.randint(0, h - side)
        img = img.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.BILINEAR)
    gray = rng.random() < gray_frac
    if gray:
        img = img.convert("L").convert("RGB")
    return img, gray, cropped


def build(items_by_split: Mapping[str, Sequence[Item]], out_root: Path, seed: int,
          gray_frac: float = 0.5, crop_frac: float = 0.5) -> Dict:
    from PIL import Image
    out_root = Path(out_root)
    for sp in SPLITS:
        d = out_root / sp / NOT_STEEL
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    ids = {sp: {it["id"] for it in items_by_split[sp]} for sp in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if ids[a] & ids[b]:
                raise RuntimeError(f"{len(ids[a] & ids[b])} source images in both {a} and {b}")

    records, hashes = [], {}
    for sp in SPLITS:
        for it in sorted(items_by_split[sp], key=lambda x: x["id"]):
            # per-image RNG: the transform of an image does not depend on the others
            rng = random.Random(f"{seed}:{it['id']}")
            with Image.open(it["path"]) as im:
                img, gray, cropped = transform(im, rng, gray_frac, crop_frac)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            data = buf.getvalue()
            h = hashlib.sha256(data).hexdigest()
            if h in hashes and hashes[h] != sp:
                raise RuntimeError(f"identical image content in {hashes[h]} and {sp}: {it['id']}")
            hashes[h] = sp
            name = f"{it['id']}.jpg"
            (out_root / sp / NOT_STEEL / name).write_bytes(data)
            records.append({"file": name, "split": sp, "source": it["source"], "source_file": Path(it["path"]).name,
                            "category": it["category"], "grayscale": gray, "cropped_200": cropped,
                            "width": img.size[0], "height": img.size[1], "sha256": h})

    for sp in SPLITS:
        on_disk = {p.name for p in _images(out_root / sp / NOT_STEEL)}
        want = {f"{it['id']}.jpg" for it in items_by_split[sp]}
        if on_disk != want:
            raise RuntimeError(f"{sp}: written files differ from the split")

    def count(key=None, val=None, sp=None):
        return sum(1 for r in records if (sp is None or r["split"] == sp) and (key is None or r[key] == val))

    summary = {
        "seed": seed, "gray_frac": gray_frac, "crop_frac": crop_frac, "out_size": OUT_SIZE,
        "split_ratios": list(RATIOS),
        "counts": {sp: {src: sum(1 for r in records if r["split"] == sp and r["source"] == src)
                        for src in sorted({r["source"] for r in records})} for sp in SPLITS},
        "totals": {sp: count(sp=sp) for sp in SPLITS},
        "grayscale": count("grayscale", True), "cropped_200": count("cropped_200", True),
        "pairwise_overlap": 0, "duplicate_content_across_splits": 0,
        "dtd_categories": len({r["category"] for r in records if r["source"] == "dtd"}),
        "places_categories": len({r["category"] for r in records if r["source"] == "places365"}),
        "images": records,
    }
    (out_root / "not_steel_manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{'source':10s} " + " ".join(f"{sp:>6s}" for sp in SPLITS))
    for src in sorted(summary["counts"]["train"]):
        print(f"{src:10s} " + " ".join(f"{summary['counts'][sp].get(src, 0):6d}" for sp in SPLITS))
    print(f"{'TOTAL':10s} " + " ".join(f"{summary['totals'][sp]:6d}" for sp in SPLITS))
    print(f"grayscale {summary['grayscale']}/{len(records)}, cropped to {OUT_SIZE}x{OUT_SIZE} "
          f"{summary['cropped_200']}/{len(records)}; DTD categories {summary['dtd_categories']}, "
          f"Places categories {summary['places_categories']}")
    print("[CHECK] pairwise overlap = 0 (ids and content)")
    return summary


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", type=Path, required=True)
    ap.add_argument("--dtd-dir", type=Path, required=True, help="dtd/images (one folder per category)")
    ap.add_argument("--places-dir", type=Path, required=True, help="val_256")
    ap.add_argument("--places-val-list", type=Path, required=True, help="places365_val.txt")
    ap.add_argument("--places-categories", type=Path, required=True, help="categories_places365.txt")
    ap.add_argument("--out-root", type=Path, required=True, help="ImageFolder root (train/val/test)")
    ap.add_argument("--n-coco", type=int, default=1000)
    ap.add_argument("--n-dtd", type=int, default=600)
    ap.add_argument("--n-places", type=int, default=400)
    ap.add_argument("--gray-frac", type=float, default=0.5)
    ap.add_argument("--crop-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    items = (sample_coco(a.coco_dir, a.n_coco, rng) + sample_dtd(a.dtd_dir, a.n_dtd, rng)
             + sample_places(a.places_dir, read_places_lists(a.places_val_list, a.places_categories), a.n_places, rng))
    if len({it["id"] for it in items}) != len(items):
        raise RuntimeError("duplicate source ids")
    return build(split_items(items, random.Random(a.seed + 1)), a.out_root, a.seed, a.gray_frac, a.crop_frac)


if __name__ == "__main__":
    main()
