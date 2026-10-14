"""M1: U-Net (ResNet18 encoder, trained from scratch on 16-channel input)
on train regions with Hansen weak labels. CPU-sized: 64 px chips.

Validation chips are held out by tile (not by random chip) so neighbouring
chips never leak across the train/validation boundary.
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, TensorDataset

from cocoaloss import config
from cocoaloss.dataset import load_chips
from cocoaloss.metrics import prf

torch.manual_seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED)
EPOCHS, BATCH = 25, 32

x, y, meta = load_chips("train")
ok = np.abs(x[:, :14]).mean(axis=(1, 2, 3)) > 1e-4
x, y, meta = x[ok], y[ok], [m for m, k in zip(meta, ok) if k]
mu = x.mean(axis=(0, 2, 3), keepdims=True); sd = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
np.savez(config.DATA_PROCESSED / "norm.npz", mu=mu, sd=sd)

tiles = sorted({(m[0], m[1]) for m in meta})
val_tiles = set(tiles[:: 5])  # every 5th tile held out for validation
isval = np.array([(m[0], m[1]) in val_tiles for m in meta])
print(f"{len(x)} chips | train {int((~isval).sum())} val {int(isval.sum())} | loss frac {y.mean():.4f}")

def loader(mask, shuffle):
    xt = torch.from_numpy((x[mask] - mu) / sd); yt = torch.from_numpy(y[mask]).float().unsqueeze(1)
    return DataLoader(TensorDataset(xt, yt), BATCH, shuffle=shuffle)

tr, va = loader(~isval, True), loader(isval, False)
model = smp.Unet("resnet18", encoder_weights=None, in_channels=16, classes=1)
loss_fn = smp.losses.DiceLoss("binary") if False else None
dice = smp.losses.DiceLoss("binary"); bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * len(tr))

best = 0
for ep in range(EPOCHS):
    model.train()
    for xb, yb in tr:
        if np.random.rand() < 0.5: xb, yb = xb.flip(-1), yb.flip(-1)   # augment
        if np.random.rand() < 0.5: xb, yb = xb.flip(-2), yb.flip(-2)
        opt.zero_grad(); out = model(xb); l = bce(out, yb) + dice(out, yb); l.backward(); opt.step(); sched.step()
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for xb, yb in va: P.append((model(xb).sigmoid() > 0.5).numpy()); Y.append(yb.numpy())
    m = prf(np.concatenate(P), np.concatenate(Y))
    print(ep, f"loss {l.item():.3f}", m, flush=True)
    if m["f1"] > best: best = m["f1"]; torch.save(model.state_dict(), config.DATA_PROCESSED / "unet_best.pt")
print("best val F1", best)
