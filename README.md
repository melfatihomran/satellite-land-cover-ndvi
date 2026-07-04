# Satellite Land Cover Classification & NDVI Vegetation Mapping

A compact end-to-end remote sensing pipeline, built as internship prep for
**NARSS (National Authority for Remote Sensing and Space Sciences)**, Egypt.

It combines the two techniques most visible in NARSS-affiliated research:

1. **Land cover classification** — a ResNet-50 fine-tuned on the EuroSAT
   Sentinel-2 benchmark (10 classes: crops, forest, urban, water, etc.),
   the same transfer-learning approach used in published work classifying
   EgyptSat-1 imagery.
2. **NDVI vegetation health mapping** — computed directly from Red/NIR band
   math, the standard technique behind NARSS's own Nile Delta crop-monitoring
   work.

## Why this project

NARSS's Remote Sensing Applications sector works on land use, agriculture,
and environmental monitoring using exactly this pipeline: multispectral
imagery → engineered indices (NDVI, NDWI) → CNN/ML classification → maps
for decision-making. This repo demonstrates the full chain — from raw
satellite bands to an interactive tool — using a stack I already know
(PyTorch, ResNet-50 transfer learning, Gradio deployment).

## Project structure

```
narss-prep-project/
├── README.md
├── requirements.txt
├── app.py                      # Gradio demo — classification + NDVI in one UI
├── models/                     # trained model checkpoints land here
├── data/                       # EuroSAT downloads land here
├── assets/                     # confusion_matrix.png, ndvi_map.png
└── src/
    ├── dataset_utils.py        # EuroSAT loading, transforms, dataloaders
    ├── train_classifier.py     # ResNet-50 training loop + evaluation
    └── ndvi_analysis.py        # NDVI computation, stats, heatmap rendering
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1 — Land Cover Classification (EuroSAT + ResNet-50)

```bash
cd src
python train_classifier.py --epochs 10 --batch-size 32
```

This downloads EuroSAT automatically on first run (~90MB), fine-tunes a
ResNet-50 (ImageNet-pretrained), and saves the best checkpoint to
`../models/resnet50_eurosat.pt`. Prints per-class precision/recall/F1
at the end.

Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake.

**Note:** run this on a machine with normal (non-sandboxed) internet
access — the dataset download needs it.

## Part 2 — NDVI Vegetation Mapping (real Sentinel-2 imagery)

NDVI = (NIR − Red) / (NIR + Red), computed per-pixel. Values range from
-1 to 1; healthy vegetation typically scores 0.3–0.8.

```python
from src.ndvi_analysis import load_bands, compute_ndvi, ndvi_summary_stats, plot_ndvi_heatmap

red, nir, profile = load_bands("B04.tif", "B08.tif")
ndvi = compute_ndvi(red, nir)
print(ndvi_summary_stats(ndvi))
plot_ndvi_heatmap(ndvi, save_path="ndvi_map.png")
```

**Getting a free real Sentinel-2 tile over Egypt (e.g. the Nile Delta):**
the Copernicus Browser (browser.dataspace.copernicus.eu) lets you search
Sentinel-2 L2A scenes by date/location and download individual bands
(B04 = Red, B08 = NIR) with a free account, no approval process.

The `ndvi_analysis.py` module was validated with a synthetic
bare-soil/dense-vegetation test scene (run `python src/ndvi_analysis.py`)
before pointing it at real imagery — confirms the band math and edge-case
handling (division by zero, clipping) are correct.

## Results

### Land Cover Classification (EuroSAT, ResNet-50, 8 epochs)

**Test accuracy: 98.35%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| AnnualCrop | 0.97 | 0.98 | 0.98 | 472 |
| Forest | 1.00 | 0.99 | 0.99 | 442 |
| HerbaceousVegetation | 0.97 | 0.99 | 0.98 | 458 |
| Highway | 0.98 | 1.00 | 0.99 | 391 |
| Industrial | 0.99 | 0.98 | 0.99 | 378 |
| Pasture | 0.97 | 0.97 | 0.97 | 299 |
| PermanentCrop | 0.98 | 0.96 | 0.97 | 379 |
| Residential | 0.99 | 0.99 | 0.99 | 450 |
| River | 0.99 | 0.98 | 0.98 | 375 |
| SeaLake | 0.99 | 1.00 | 1.00 | 406 |

See `assets/confusion_matrix.png` for the full confusion matrix.

**Notes on the weaker classes:** PermanentCrop and Pasture have the lowest
F1-scores (0.97), which lines up with known EuroSAT literature — different
crop/vegetation types have overlapping spectral signatures in RGB-only
imagery. This is a real limitation of RGB-based classification, and part
of why NDVI/multispectral analysis (Part 2) adds value beyond what a
plain RGB classifier can distinguish.

### NDVI Vegetation Mapping (Nile Delta, near Rosetta/Idku, Sentinel-2, July 2026)

| Metric | Value |
|---|---|
| Mean NDVI | 0.276 |
| Median NDVI | 0.097 |
| Bare soil / water | 55.1% |
| Sparse vegetation | 11.9% |
| Dense vegetation | 33.0% |

See `assets/ndvi_map.png` for the rendered heatmap.

The map correctly resolves the Rosetta branch of the Nile, dense
agricultural fields on both banks, and a visible NDVI dip over
Rosetta's urban area. Open sea reads close to neutral (~-0.015) rather
than strongly negative — verified via a Red/NIR band sanity check to be
a real coastal turbidity/sediment effect near the river mouth, not a
band mix-up (confirmed NIR < Red over water, the physically correct
direction).

## Part 3 — Combined Demo

```bash
python app.py
```

Launches a local Gradio app with two tabs: upload a patch for land-cover
classification, or a pair of Sentinel-2 bands for an NDVI heatmap.

## Honest notes on scope

This was built over a few days as internship prep, not a production
system. The classifier is trained and evaluated on EuroSAT's benchmark
labels, not NARSS-verified ground truth — strong accuracy here
demonstrates the methodology works, not that it's validated for
operational use. The NDVI map is a single-date snapshot, not a
time-series.

Next steps I'd want to explore at NARSS: multi-temporal change
detection (tracking NDVI across a growing season), additional indices
(NDWI for water content, SWIR-based soil moisture), and validating
against real ground-truth data rather than benchmark labels.

## References

- Helber et al., 2019 — *EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification*
- NARSS-affiliated research on EgyptSat-1 CNN classification and
  Nile Delta NDVI-based crop monitoring (see project write-up)
