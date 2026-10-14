"""Fetch a small real composite pair + Hansen label and print a sanity report."""

import time

import numpy as np

from cocoaloss import config
from cocoaloss.composites import median_composite, spectral_indices, stack_array
from cocoaloss.labels import forest_loss_label

# ~4 km square inside the Sefwi-Wiawso training block
BOUNDS = (-2.52, 6.18, -2.48, 6.22)

t = time.time()
pre = median_composite(BOUNDS, config.PRE_PERIOD)
print(f"pre   {dict(pre.sizes)}  median valid obs/pixel = {float(pre['n_valid'].median())}  ({time.time()-t:.0f}s)")
post = median_composite(BOUNDS, config.POST_PERIOD)
print(f"post  {dict(post.sizes)}  median valid obs/pixel = {float(post['n_valid'].median())}  ({time.time()-t:.0f}s)")

x = stack_array(pre, post)
y = forest_loss_label(pre["red"])
print("input", x.shape, "label", y.shape, "loss fraction", round(float(y.mean()), 4))

# Signal check: NBR should fall where Hansen says forest was lost.
dnbr = x[-2]
print("mean dNBR  loss px:", round(float(dnbr[y == 1].mean()), 3) if y.any() else "n/a",
      "| non-loss px:", round(float(dnbr[y == 0].mean()), 3))
np.savez_compressed(config.DATA_PROCESSED / "smoke_chip.npz", x=x, y=y)
