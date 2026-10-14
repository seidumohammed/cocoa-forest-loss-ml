# Cocoa-belt forest-loss detection (Sentinel-2 + deep learning)

Detects forest loss after the EUDR cutoff (31 Dec 2020) in Ghana's cocoa belt
and, unlike a threshold screen, reports how accurate that detection is, with
area estimates and confidence intervals from an independent reference sample.
It is the modelling layer behind [eudr-plot-risk-checker](../eudr-plot-risk-checker).

## Research question
Does a learned model detect post-2020 forest loss in cocoa landscapes more
accurately than (a) Hansen GFC alone and (b) a spectral-index rule, and does
the gain hold in a region the model never saw?

## Design
| Item | Choice |
|---|---|
| Input | Sentinel-2 L2A dry-season median composites, pre (Dec 2019-Mar 2020) and post (Dec 2023-Mar 2024), SCL cloud mask, 20 m |
| Features | 7 bands x 2 dates + dNBR + dNDVI = 16 channels |
| Training labels | Hansen GFC v1.12: canopy >=10% in 2000 and loss 2021-2024 (weak labels) |
| Models | B0 Hansen only; B1 dNBR threshold; B2 random forest on pixels; M1 U-Net (ResNet18) |
| Split | Spatial: train on Sefwi-Wiawso + Bibiani, test on Juaboso (no shared area) |
| Independent test | Stratified random reference sample, interpreted by hand from imagery (Olofsson et al. 2014 design) |
| Metrics | Precision, recall, F1 for the loss class; area-adjusted loss estimate with 95% CI |
| Cross-check | RADD (Sentinel-1) alerts, an independent radar-based source |

Hansen is a training signal only. Scoring against Hansen would reward copying it.

## Status
- [x] Cloud-masked composites and Hansen label alignment (`cocoaloss/`), verified on real data (`scripts/smoke_test.py`)
- [x] Chip dataset: 48 tiles, 512 train chips, 256 test chips (`scripts/build_dataset.py`)
- [x] Baselines B1-B2 and U-Net M1 trained (`scripts/baselines.py`, `train_unet.py`, `eval_test.py`)
- [ ] Reference sample + labelling tool
- [ ] Accuracy assessment and area estimates
- [ ] Hook into the screener API; deploy; preprint

## Development results (scored against Hansen weak labels, unseen test region)
These are a development check, not the headline. Hansen is a noisy 30 m annual
product, so scores are capped by label error.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| B1 dNBR threshold | 0.29 | 0.50 | 0.37 |
| B2 random forest (pixel) | 0.16 | 0.56 | 0.25 |
| M1 U-Net (threshold 0.6) | 0.36 | 0.50 | 0.42 |

The U-Net gain over the threshold is modest. The independent reference
sample decides whether it is real.
