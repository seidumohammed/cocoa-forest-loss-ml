"""Study design constants: regions, periods, class definitions."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_CHIPS = REPO_ROOT / "data" / "chips"
DATA_REFERENCE = REPO_ROOT / "data" / "reference"

STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
S2_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir16", "swir22"]
S2_RESOLUTION_M = 20  # SWIR native is 20 m; resample 10 m bands down
# Sentinel-2 SCL classes treated as unusable: no data, saturated, shadow,
# cloud (medium/high prob), thin cirrus, snow.
SCL_MASK = (0, 1, 3, 8, 9, 10, 11)

# Study regions (west, south, east, north), EPSG:4326, Ghana cocoa belt.
# Spatial split: models are trained on TRAIN blocks and judged on TEST
# blocks that share no pixels and no neighbouring chips with training data.
REGIONS = {
    "train_sefwi_wiawso": (-2.60, 6.10, -2.40, 6.30),
    "train_bibiani": (-2.40, 6.30, -2.20, 6.50),
    "test_juaboso": (-2.90, 6.30, -2.70, 6.50),
}

# Epochs. Dry-season (Dec-Mar) composites minimise cloud in the forest zone.
PRE_PERIOD = ("2019-12-01", "2020-03-31")   # EUDR baseline (cutoff 31 Dec 2020)
POST_PERIOD = ("2023-12-01", "2024-03-31")
LOSS_YEARS = (2021, 2022, 2023, 2024)       # Hansen lossyear codes 21..24

FOREST_CANOPY_MIN_PCT = 10   # EUDR / FAO forest definition
CHIP_SIZE = 64               # pixels (64 x 20 m = 1.28 km)
RANDOM_SEED = 42
