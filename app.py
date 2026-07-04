"""
app.py

Gradio demo combining both pipelines into one interface:
  Tab 1: Land cover classification (upload a patch -> ResNet-50 prediction)
  Tab 2: NDVI vegetation mapping (upload Red + NIR bands -> heatmap + stats)

Run with:
    python app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import torch
import numpy as np
import rasterio
from PIL import Image

from dataset_utils import get_transforms, EUROSAT_CLASSES
from train_classifier import build_model
from ndvi_analysis import compute_ndvi, ndvi_summary_stats, plot_ndvi_heatmap

# Use an absolute path so this works regardless of the working directory
# the script is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet50_eurosat.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def get_model():
    """Lazily loads the trained classifier (only once)."""
    global _model
    if _model is None:
        _model = build_model(num_classes=len(EUROSAT_CLASSES))
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        else:
            print(f"WARNING: no trained weights found at {MODEL_PATH}. "
                  f"Run train_classifier.py first for real predictions.")
        _model.to(device)
        _model.eval()
    return _model


def classify_image(image: Image.Image):
    if image is None:
        return "Please upload an image."

    model = get_model()
    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k=3)
    results = {EUROSAT_CLASSES[i]: float(p) for p, i in zip(top_probs, top_idx)}
    return results


def analyze_ndvi(red_file, nir_file):
    if red_file is None or nir_file is None:
        return None, "Please upload both a Red band (.tif) and a NIR band (.tif) file."

    # Real Sentinel-2 analytical bands are (typically 32-bit float) GeoTIFFs —
    # read them with rasterio, not PIL, to preserve actual reflectance values.
    try:
        with rasterio.open(red_file.name) as src:
            red = src.read(1).astype("float32")
        with rasterio.open(nir_file.name) as src:
            nir = src.read(1).astype("float32")
    except Exception as e:
        return None, f"Could not read one of the files as a GeoTIFF: {e}"

    if red.shape != nir.shape:
        return None, f"Band size mismatch: Red={red.shape}, NIR={nir.shape}. Use matching crops/extents."

    ndvi = compute_ndvi(red, nir)
    stats = ndvi_summary_stats(ndvi)

    fig = plot_ndvi_heatmap(ndvi, title="NDVI Vegetation Health Map")

    stats_text = (
        f"Mean NDVI: {stats['mean_ndvi']:.3f}\n"
        f"Median NDVI: {stats['median_ndvi']:.3f}\n"
        f"Range: [{stats['min_ndvi']:.3f}, {stats['max_ndvi']:.3f}]\n\n"
        f"Bare soil / water: {stats['pct_water_or_bare']:.1f}%\n"
        f"Sparse vegetation: {stats['pct_sparse_vegetation']:.1f}%\n"
        f"Dense vegetation: {stats['pct_dense_vegetation']:.1f}%"
    )
    return fig, stats_text


with gr.Blocks(title="NARSS Prep: Land Cover + NDVI") as demo:
    gr.Markdown(
        "# Satellite Land Cover Classification & NDVI Vegetation Mapping\n"
        "Built as internship prep for NARSS (National Authority for Remote "
        "Sensing and Space Sciences), Egypt."
    )

    with gr.Tab("Land Cover Classification"):
        gr.Markdown(
            "Upload a satellite image patch (RGB). The model, a ResNet-50 "
            "fine-tuned on EuroSAT, predicts the land cover class."
        )
        with gr.Row():
            img_input = gr.Image(type="pil", label="Satellite Image Patch")
            label_output = gr.Label(num_top_classes=3, label="Predicted Land Cover")
        classify_btn = gr.Button("Classify")
        classify_btn.click(fn=classify_image, inputs=img_input, outputs=label_output)

    with gr.Tab("NDVI Vegetation Mapping"):
        gr.Markdown(
            "Upload matching Red (B04) and NIR (B08) band **GeoTIFF files** "
            "(same scene, same extent) from Sentinel-2 or Landsat to compute NDVI."
        )
        with gr.Row():
            red_input = gr.File(label="Red Band (.tif)", file_types=[".tif", ".tiff"])
            nir_input = gr.File(label="NIR Band (.tif)", file_types=[".tif", ".tiff"])
        ndvi_btn = gr.Button("Compute NDVI")
        with gr.Row():
            ndvi_plot = gr.Plot(label="NDVI Heatmap")
            ndvi_stats = gr.Textbox(label="Summary Stats", lines=8)
        ndvi_btn.click(fn=analyze_ndvi, inputs=[red_input, nir_input], outputs=[ndvi_plot, ndvi_stats])


if __name__ == "__main__":
    demo.launch()