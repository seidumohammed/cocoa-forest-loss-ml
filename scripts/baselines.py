"""B1 (dNBR threshold) and B2 (random forest) vs Hansen weak labels,
trained on train regions, scored on the held-out test region.
NOTE: scoring against Hansen here is a development check only; the
headline numbers come from the independent reference sample."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cocoaloss import config
from cocoaloss.dataset import load_chips
from cocoaloss.metrics import prf

rng = np.random.default_rng(config.RANDOM_SEED)
xtr, ytr, _ = load_chips("train"); xte, yte, _ = load_chips("test")
print("train chips", len(xtr), "test chips", len(xte),
      "| loss frac train", ytr.mean().round(4), "test", yte.mean().round(4))

# B1: pick the dNBR cut-off that maximises F1 on TRAIN, apply to TEST.
dtr, dte = xtr[:, 14], xte[:, 14]
best = max(((prf(dtr < t, ytr)["f1"], t) for t in np.arange(-0.05, -0.6, -0.025)))
print("B1 threshold", round(best[1], 3), "train F1", best[0])
print("B1 test ", prf(dte < best[1], yte))

# B2: random forest on balanced pixel samples (per-pixel, spectral only).
def flat(x): return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])
Xf, yf = flat(xtr), ytr.ravel()
pos, neg = np.where(yf == 1)[0], np.where(yf == 0)[0]
idx = np.concatenate([rng.choice(pos, min(len(pos), 100_000), replace=False),
                      rng.choice(neg, min(len(neg), 200_000), replace=False)])
rf = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=config.RANDOM_SEED,
                            class_weight="balanced_subsample").fit(Xf[idx], yf[idx])
print("B2 test ", prf(rf.predict(flat(xte)).reshape(yte.shape), yte))
