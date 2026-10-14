import numpy as np, segmentation_models_pytorch as smp, torch
from cocoaloss import config
from cocoaloss.dataset import load_chips
from cocoaloss.metrics import prf

x, y, _ = load_chips("test")
n = np.load(config.DATA_PROCESSED / "norm.npz")
model = smp.Unet("resnet18", encoder_weights=None, in_channels=16, classes=1)
model.load_state_dict(torch.load(config.DATA_PROCESSED / "unet_best.pt")); model.eval()
with torch.no_grad():
    p = torch.cat([model(torch.from_numpy((x[i:i+64]-n["mu"])/n["sd"])).sigmoid() for i in range(0, len(x), 64)]).numpy()[:, 0]
for t in (0.5, 0.6, 0.7, 0.8):
    print("U-Net test thr", t, prf(p > t, y))
