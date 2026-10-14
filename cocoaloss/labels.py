"""Weak training labels from Hansen Global Forest Change, aligned to a grid.

Hansen is used ONLY to train. It is never the test set: the models are
judged against independently interpreted reference samples (see
docs/validation.md), because scoring a model against its own training
labels would only measure how well it copies Hansen.
"""

from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject

from cocoaloss import config

HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"
HANSEN_VERSION = "GFC-2024-v1.12"


def _tile_tag(lon: float, lat: float) -> str:
    import math

    top = math.ceil(lat / 10) * 10
    left = math.floor(lon / 10) * 10
    return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'W' if left < 0 else 'E'}"


def _read_layer_on_grid(layer: str, tag: str, like: xr.DataArray) -> np.ndarray:
    """Reproject one Hansen layer onto the grid of `like` (nearest neighbour)."""
    url = f"/vsicurl/{HANSEN_BASE}/Hansen_{HANSEN_VERSION}_{layer}_{tag}.tif"
    dst = np.zeros(like.shape, dtype="uint8")
    transform = like.odc.transform
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(url) as src:
            reproject(
                rasterio.band(src, 1), dst,
                dst_transform=transform, dst_crs=like.odc.crs.to_wkt(),
                resampling=Resampling.nearest,
            )
    return dst


def forest_loss_label(grid: xr.DataArray, tag: str = "10N_010W") -> np.ndarray:
    """uint8 mask on `grid`: 1 = forest (>=10% canopy in 2000) lost in
    2021-2024, 0 = otherwise. The EUDR-relevant event."""
    tree = _read_layer_on_grid("treecover2000", tag, grid)
    loss = _read_layer_on_grid("lossyear", tag, grid)
    y0, y1 = config.LOSS_YEARS[0] - 2000, config.LOSS_YEARS[-1] - 2000
    return ((tree >= config.FOREST_CANOPY_MIN_PCT) & (loss >= y0) & (loss <= y1)).astype("uint8")
