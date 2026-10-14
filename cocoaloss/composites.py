"""Cloud-masked Sentinel-2 median composites from the public Earth Search STAC.

No credentials are needed: the COGs are on public S3. Pixels flagged by the
scene classification layer (SCL) as cloud, shadow, cirrus or no-data are
dropped before taking the per-pixel median over the period.
"""

from __future__ import annotations

import numpy as np
import odc.stac
import pystac_client
import xarray as xr

from cocoaloss import config

Bounds = tuple[float, float, float, float]


def search_items(bounds: Bounds, period: tuple[str, str], max_cloud: float = 60.0):
    client = pystac_client.Client.open(config.STAC_URL)
    search = client.search(
        collections=[config.S2_COLLECTION],
        bbox=bounds,
        datetime=f"{period[0]}/{period[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    # Some items reference a legacy requester-pays JP2 bucket; keep only
    # the public COG copies.
    return [i for i in search.items()
            if all(i.assets[a].href.startswith("https") for a in ("red", "scl"))]


def mask_invalid(scl: xr.DataArray) -> xr.DataArray:
    """True where the SCL class is usable."""
    return ~scl.isin(list(config.SCL_MASK))


def median_composite(bounds: Bounds, period: tuple[str, str]) -> xr.Dataset:
    """Per-pixel median of valid observations, one variable per band, in
    reflectance units (0-1). Also returns `n_valid`, the observation count."""
    items = search_items(bounds, period)
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes for {bounds} in {period}")
    cube = odc.stac.load(
        items,
        bands=[*config.S2_BANDS, "scl"],
        bbox=bounds,
        resolution=config.S2_RESOLUTION_M,
        crs="EPSG:32630",
        chunks={"x": 1024, "y": 1024},
        groupby="solar_day",
    )
    valid = mask_invalid(cube["scl"])
    bands = []
    for name in config.S2_BANDS:
        # Level-2A processing baseline >= 04.00 adds a +1000 DN offset.
        refl = (cube[name].astype("float32") - 1000.0) / 10000.0
        bands.append(refl.where(valid & (refl > -0.1)).median("time").rename(name))
    out = xr.merge(bands)
    out["n_valid"] = valid.sum("time").astype("int16")
    return out.compute()


def spectral_indices(ds: xr.Dataset) -> xr.Dataset:
    """NDVI, NBR and NDMI, the standard forest-disturbance indices."""
    eps = 1e-6
    ndvi = (ds["nir"] - ds["red"]) / (ds["nir"] + ds["red"] + eps)
    nbr = (ds["nir"] - ds["swir22"]) / (ds["nir"] + ds["swir22"] + eps)
    ndmi = (ds["nir"] - ds["swir16"]) / (ds["nir"] + ds["swir16"] + eps)
    return xr.Dataset({"ndvi": ndvi, "nbr": nbr, "ndmi": ndmi})


def stack_array(pre: xr.Dataset, post: xr.Dataset) -> np.ndarray:
    """Model input (C, H, W): pre bands, post bands, then NBR/NDVI deltas.
    NaNs (never-valid pixels) are filled with 0 after differencing."""
    ip, iq = spectral_indices(pre), spectral_indices(post)
    layers = [pre[b].values for b in config.S2_BANDS]
    layers += [post[b].values for b in config.S2_BANDS]
    layers += [(iq["nbr"] - ip["nbr"]).values, (iq["ndvi"] - ip["ndvi"]).values]
    return np.nan_to_num(np.stack(layers).astype("float32"))
