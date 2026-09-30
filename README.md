# Smartphone Magnetic Indoor Localization

## Scope
Corridor-level classification from smartphone magnetic and inertial trajectories using UJIIndoorLoc-Mag.

## Dataset
UJIIndoorLoc-Mag contains 270 public training trajectories and 11 public test trajectories with magnetometer, accelerometer, orientation, timestamps, and corridor metadata. Source: UCI dataset 343; DOI 10.24432/C5D311; CC BY 4.0.

## Evaluation Protocol
Trajectory IDs are isolation groups. Development uses saved two-fold StratifiedGroupKFold, segment-pure 2 s windows with 1 s stride, macro-F1 as primary metric, and fold-local transformations. Final public test access occurred once after candidate freeze.

## Development Results
| Model | Macro-F1 |
|---|---:|
| KNN body XYZ | 0.7028 |
| ExtraTrees body XYZ | 0.6872 |
| MLP body XYZ | 0.3591 |

## Final Holdout
Closed-set macro-F1: KNN 0.0499; ExtraTrees 0.0339; MLP 0.1000. The final set contains 364 windows, including 9 windows from two unseen labels.

## Structure
`src/` contains pipeline code. `scripts/` contains reproducible entry points. `configs/` contains feature and frozen configs. `data/manifests/` contains metadata only. `experiments/` contains curated tables and figures. `docs/` contains methodology and evaluation documentation.

## Reproduction
`python -m pytest -q`

Raw data is intentionally excluded. Acquire UJI data with `python -m scripts.download_uji_mag`, then parse with `python -m scripts.parse_uji_mag`.
