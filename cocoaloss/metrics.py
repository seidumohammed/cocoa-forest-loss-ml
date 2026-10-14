import numpy as np


def prf(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Precision / recall / F1 / IoU for the positive (loss) class."""
    p, r = pred.astype(bool).ravel(), ref.astype(bool).ravel()
    tp, fp, fn = (p & r).sum(), (p & ~r).sum(), (~p & r).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=round(float(prec), 4), recall=round(float(rec), 4),
                f1=round(float(f1), 4), iou=round(float(tp / max(tp + fp + fn, 1)), 4))
