"""Load tiles, cut chips, and keep train/test spatially disjoint."""

from __future__ import annotations

import numpy as np

from cocoaloss import config


def load_chips(region_prefix: str):
    """All 64x64 chips from regions whose name starts with `region_prefix`
    ('train' or 'test'). Returns x (N,16,64,64), y (N,64,64), meta list."""
    c = config.CHIP_SIZE
    xs, ys, meta = [], [], []
    for path in sorted(config.DATA_PROCESSED.glob(f"{region_prefix}*/*.npz")):
        d = np.load(path)
        x, y = d["x"], d["y"]
        for r in range(0, y.shape[0], c):
            for k in range(0, y.shape[1], c):
                xs.append(x[:, r:r + c, k:k + c]); ys.append(y[r:r + c, k:k + c])
                meta.append((path.parent.name, path.stem, r, k))
    return np.stack(xs), np.stack(ys), meta


def usable(x: np.ndarray) -> np.ndarray:
    """Mask of chips with real data (not all-zero from missing observations)."""
    return (np.abs(x[:14]).sum(axis=(0, 1, 2)) if False else np.abs(x[:, :14]).mean(axis=(1, 2, 3))) > 1e-4
