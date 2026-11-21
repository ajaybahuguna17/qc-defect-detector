# Data sources

Every dataset used to train or evaluate the Phase 1 vision models. Images are **not** in git.
They are downloaded in Colab by the notebooks.

## Steel (defect classes): NEU-DET

| | |
|---|---|
| Used by | Phase 1, 1.1, 1.2 (detector and classifier, the 6 defect classes) |
| Download | `scripts/prepare_data.py` (see that script for the mirror it uses) |
| Split | Phase 1.1 / 1.2: 70/15/15, seed 42 (210 / 45 / 45 images per class). The exact file list is in `results/phase1_2/neu_split_manifest_s42.json`; the Phase 1.2 notebook asserts its split is identical |
| Licence | NEU surface defect database (Northeastern University, China), released for academic research. Cite: K. Song, Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects", Applied Surface Science, 2013 |

## not_steel (Phase 1.2, classifier v2 class 7): about 2,000 images

Downloaded by `notebooks/train_colab_phase1_2.ipynb` from the official URLs below (all returned
HTTP 200 on 2026-11-21; sizes from the servers). The notebook writes the size and SHA-256 of
each archive it got into `phase1_2_results.json` -> `data_sources`. Images are sampled, transformed
and split by `scripts/build_not_steel.py` (seed 42); the per-image list (source file, category,
split, grayscale, crop, SHA-256) is `not_steel_manifest.json` in the run's results.

| Source | Official URL | Size | Images used | What it covers | Licence / terms |
|---|---|---|---|---|---|
| **COCO 2017 val** | http://images.cocodataset.org/zips/val2017.zip | 815,585,330 B | 1,000 of 5,000, uniform random | people, everyday objects, indoor and outdoor scenes | COCO terms of use (cocodataset.org/#termsofuse). Images come from Flickr and each keeps its own Flickr licence (Creative Commons variants, some non-commercial). Annotations (not used) CC BY 4.0. Cite: T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014 |
| **DTD r1.0.1** (Describable Textures Dataset) | https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz (redirects to https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz) | 625,239,812 B | 600 of 5,640, round-robin over **all 47 categories** (12 or 13 each) | textures: concrete-like, stone, fabric, wood, cracked, pitted, scaly, grooved, stained... the **hard look-alikes** | Made available "for research purposes" by the Oxford VGG group. Cite: M. Cimpoi et al., "Describing Textures in the Wild", CVPR 2014 |
| **Places365-Standard val, 256x256** | https://data.csail.mit.edu/places/places365/val_256.tar + https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar (labels) | 525,158,400 B + 67,498,496 B | 400 of 36,500, uniform random over the categories that match `PLACES_KEYWORDS` in `build_not_steel.py` (fields, pasture, desert, roads, streets, parking lots, buildings, facades, construction sites, ...) | fields, ground, roads, buildings | "Non-commercial research and educational purposes only" (places2.csail.mit.edu). Cite: B. Zhou et al., "Places: A 10 Million Image Database for Scene Recognition", TPAMI 2017 |

### Transformations (so the model cannot learn "colour = not steel")

NEU images are 200x200 grayscale. For each not_steel image, independently and deterministically
(per-image RNG seeded from the seed and the image id):
- with probability 0.5: a random square crop (50-100 % of the short side) resized to **200x200**;
- with probability 0.5: converted to **grayscale** (stored as 3 equal channels, like NEU).

Then a 70/15/15 split **per source** (seed 42): 1,400 train / 300 val / 300 test, with zero overlap
by image id and by content hash (`build_not_steel.py` raises otherwise).

### Caveats

- **Licences: research only.** DTD and Places365 are research / non-commercial; COCO images keep
  their Flickr licences, some of them non-commercial. A model trained on them is fine for this
  course project. **Check the licences before any commercial use** of the classifier v2 weights.
- DTD categories such as *pitted*, *scaly*, *grooved*, *cracked* or *stained* may contain a few photos
  of real metal surfaces. They are labelled not_steel anyway; watch the steel false-rejection rate.
- None of these are factory images. A phone photo of a machine, a conveyor or a painted steel part
  is not covered; re-check the rejection rate on Phase 3 camera images.
