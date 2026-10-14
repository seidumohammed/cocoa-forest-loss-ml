"""Download composites + Hansen labels for every region as 0.05-degree tiles.

Resumable: tiles already on disk are skipped. Each tile is cropped to a
multiple of CHIP_SIZE and saved as data/processed/<region>/<i>_<j>.npz with
x (16,H,W) float32, y (H,W) uint8, and the affine transform + CRS.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from cocoaloss import config
from cocoaloss.composites import median_composite, stack_array
from cocoaloss.labels import forest_loss_label

STEP = 0.05
WORKERS = 3


def tiles(region_bounds):
    w, s, e, n = region_bounds
    nx, ny = round((e - w) / STEP), round((n - s) / STEP)
    for i in range(nx):
        for j in range(ny):
            yield i, j, (w + i * STEP, s + j * STEP, w + (i + 1) * STEP, s + (j + 1) * STEP)


def build(region, i, j, bounds):
    out = config.DATA_PROCESSED / region / f"{i}_{j}.npz"
    if out.exists():
        return region, i, j, "cached"
    pre = median_composite(bounds, config.PRE_PERIOD)
    post = median_composite(bounds, config.POST_PERIOD)
    x = stack_array(pre, post)
    y = forest_loss_label(pre["red"])
    c = config.CHIP_SIZE
    h, w = (x.shape[1] // c) * c, (x.shape[2] // c) * c
    t = pre["red"].odc.transform
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x[:, :h, :w], y=y[:h, :w], transform=np.array(t)[:6],
                        crs=str(pre["red"].odc.crs), bounds=np.array(bounds))
    return region, i, j, f"ok loss={y.mean():.3f}"


if __name__ == "__main__":
    only = sys.argv[1:] or list(config.REGIONS)
    jobs = [(r, i, j, b) for r in only for i, j, b in tiles(config.REGIONS[r])]
    print(f"{len(jobs)} tiles", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(build, *job): job for job in jobs}
        for f in as_completed(futs):
            try:
                print(*f.result(), flush=True)
            except Exception as err:  # keep going; rerun picks up failures
                print("FAIL", futs[f][:3], repr(err)[:150], flush=True)
