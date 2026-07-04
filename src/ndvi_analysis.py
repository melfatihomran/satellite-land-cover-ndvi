"""
ndvi_analysis.py

Computes NDVI (Normalized Difference Vegetation Index) from multispectral
satellite imagery (e.g. Sentinel-2 or Landsat GeoTIFFs) and renders a
vegetation-health heatmap.

NDVI = (NIR - Red) / (NIR + Red)

Values range from -1 to 1:
  < 0        : water, clouds, snow
  0 - 0.2    : bare soil, rock, urban areas
  0.2 - 0.5  : sparse to moderate vegetation (shrubland, grassland)
  0.5 - 1.0  : dense, healthy vegetation (crops, forest)

This is the exact index used in NARSS-linked agricultural monitoring
work in the Nile Delta and Al-Sharkia region, where time-series NDVI
is tracked to monitor crop phenology and health.

Band notes:
  - Sentinel-2: Band 4 = Red (665nm), Band 8 = NIR (842nm)
  - Landsat 8/9: Band 4 = Red, Band 5 = NIR
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_bands(red_path: str, nir_path: str):
    with rasterio.open(red_path) as red_src:
        red = red_src.read(1).astype("float32")
        profile = red_src.profile

    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1).astype("float32")

    if red.shape != nir.shape:
        raise ValueError(
            f"Band shape mismatch: red={red.shape}, nir={nir.shape}. "
            "Make sure both bands are from the same scene/resolution."
        )

    return red, nir, profile


def load_bands_from_stack(multiband_path: str, red_band_idx: int, nir_band_idx: int):
    with rasterio.open(multiband_path) as src:
        red = src.read(red_band_idx).astype("float32")
        nir = src.read(nir_band_idx).astype("float32")
        profile = src.profile
    return red, nir, profile


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide="ignore", invalid="ignore")
    denominator = nir + red
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi


def ndvi_summary_stats(ndvi: np.ndarray) -> dict:
    valid = ndvi[~np.isnan(ndvi)]
    return {
        "mean_ndvi": float(np.mean(valid)),
        "median_ndvi": float(np.median(valid)),
        "min_ndvi": float(np.min(valid)),
        "max_ndvi": float(np.max(valid)),
        "pct_water_or_bare": float(np.mean(valid < 0.2) * 100),
        "pct_sparse_vegetation": float(np.mean((valid >= 0.2) & (valid < 0.5)) * 100),
        "pct_dense_vegetation": float(np.mean(valid >= 0.5) * 100),
    }


def plot_ndvi_heatmap(ndvi: np.ndarray, title: str = "NDVI Vegetation Health Map", save_path: str = None):
    ndvi_cmap = LinearSegmentedColormap.from_list(
        "ndvi_cmap",
        ["#8B4513", "#D2B48C", "#FFFF99", "#90EE90", "#228B22", "#006400"],
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(ndvi, cmap=ndvi_cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved NDVI heatmap to {save_path}")

    return fig


def save_ndvi_geotiff(ndvi: np.ndarray, profile: dict, output_path: str):
    profile = profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndvi.astype("float32"), 1)
    print(f"Saved NDVI GeoTIFF to {output_path}")


if __name__ == "__main__":
    np.random.seed(0)
    red = np.concatenate(
        [np.random.uniform(0.15, 0.25, (100, 50)), np.random.uniform(0.02, 0.08, (100, 50))],
        axis=1,
    )
    nir = np.concatenate(
        [np.random.uniform(0.15, 0.25, (100, 50)), np.random.uniform(0.35, 0.55, (100, 50))],
        axis=1,
    )

    ndvi = compute_ndvi(red, nir)
    stats = ndvi_summary_stats(ndvi)
    print("NDVI summary stats (synthetic test scene):")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}")

    fig = plot_ndvi_heatmap(ndvi, title="Synthetic Test Scene NDVI", save_path="/tmp/ndvi_test.png")
    print("Smoke test passed.")
