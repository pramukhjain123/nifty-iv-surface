# Arbitrage-free neural IV surface: zero calendar and zero butterfly arbitrage, guaranteed

October 2026. This is exploratory work, kept outside `paper/`; the manuscript was not changed.

## What the model is

The model has two layers.

### 1. Neural prior (the learning part)
Two MLPs are trained on the history of *visible* NIFTY smiles. One learns how smiles behave between listed strikes; the other learns how they extend just beyond the outermost strikes.

For a new day they produce pseudo-quotes at generic locations: midpoints between visible strikes, and 1–2 spacings beyond each edge. Each fold's prior sees only dates before that fold's test window. It never sees a held-out strike, and never sees a 30–60-day slice.

### 2. Arbitrage-free layer (the guarantee)
Each day's visible quotes and pseudo-quotes are fit jointly by one quadratic program.

- **Per maturity.** The price curve is the expectation of a mean-one *lognormal-kernel mixture* with non-negative weights. That is a genuine positive density, so call prices are convex in strike and within bounds for *any* weights. This gives no butterfly arbitrage, by construction.
- **Across maturities.** The surface at maturity m is the running maximum of the slice prices up to m, equivalently `w_m(k) = max_{j<=m} w_j(k)`. This is still convex and bounded, and is non-decreasing in maturity. This gives no calendar arbitrage, by construction. The QP also imposes the ordering on a grid, so the maximum rarely binds.
- **Between listed maturities.** Call prices are blended as `P_lo + a(tau)(P_hi - P_lo)`, with `a` non-decreasing in [0, 1] (variance-time weighting). Both guarantees hold at every tau up to the last listed expiry. This is what answers Task C.

Scope of the guarantee: exact arithmetic, the paper's normalized-forward convention (k = ln K/F_tau), and maturities up to the last listed expiry (no extrapolation beyond it).

## Arbitrage, using the paper's own dense-grid rule (all 18 folds, 1,065 test-window days)

| Check | Paper's rule | Result |
|---|---|---|
| Butterfly | Durrleman g(k) by central differences (h = 1e-3), 61 points per maturity, visible range ±20% padding; 7,339 slices, 447,679 points | **0 violations** (observed range and padding) |
| Calendar | consecutive expiries, shared per-day k grid; 6,274 pairs | **0 violations** (minimum Δw = 0) |
| Between listed maturities | 25 τ × 121 k per day | **0** calendar violations of 3,092,760; **0** convexity violations of 3,168,375 |

For comparison (paper Table 5):

| Method | Butterfly, slices with any violation | Calendar, pairs with any violation |
|---|---|---|
| ThinPlate | 90.8% | 35.6% |
| CubicSpline | 84.3% | 91.2% |
| SVI | 4.0% | 60.0% |
| Fengler | 44.7% (w-space) | 1.7% |
| SSVI, v3 | 0 / 0 | 0 / 0 |

## Accuracy, using the paper's Table 6/7 metric

The metric is test-window held-out RMSE averaged over the 17 blind folds, fold 16 excluded, in vol points. The baseline rows reproduce the paper exactly.

| Method | All held-out RMSE | MAE | Median abs err | 10%-trimmed mean | Zero arbitrage? |
|---|---|---|---|---|---|
| **Arbitrage-free neural (new)** | **0.518** | **0.171** | 0.058 | 0.081 | **guaranteed, both** |
| ThinPlate | 0.633 | 0.181 | 0.051 | 0.073 | no |
| Fengler | 0.646 | 0.203 | 0.055 | 0.080 | butterfly in price space |
| SSVI | 2.179 | 1.469 | 1.006 | 1.189 | guaranteed, both |
| SVI | 2.573 | 1.212 | 0.485 | 0.723 | no |
| Paper NN v4 | 3.35 | — | — | — | calendar only |
| Paper NN v3 | 5.10 | — | — | — | calendar only |

### By range

| Method | In-range RMSE | Out-of-range RMSE |
|---|---|---|
| New | 0.379 | **1.385** |
| ThinPlate | **0.374** | 1.963 |
| Fengler | 0.440 | 2.006 |
| SSVI | 1.997 | 3.925 |

### Head-to-head

- **Folds.** The new model beats ThinPlate in 16 of 17 blind folds and Fengler in 16 of 17.
- **Paired daily loss (paper §VI-F).** It beats ThinPlate on 635 of 1,004 days (sign test p = 4e-17, Wilcoxon p = 4e-23) and Fengler on 784 of 1,004 (p = 9e-75). Against SSVI it wins 998 of 1,004 days.

## Task C: withheld 30–60-day band, paper's exact quote set and Table 11 metric

| Method | c1 RMSE / MAE | c2 RMSE / MAE |
|---|---|---|
| **New** | 1.625 / 1.120 | **1.101** / 0.763 |
| Fengler + PCHIP | **1.473 / 0.906** | 1.210 / **0.708** |
| SSVI + PCHIP | 2.107 / 1.493 | 1.728 / 1.239 |
| ThinPlate | 2.507 / 1.581 | 1.648 / 1.018 |
| Paper NN v3 (band withheld) | 4.650 / 3.589 | 4.416 / 4.103 |

## Read honestly

- **Both zeros are guaranteed by construction.** No measured violation appeared anywhere on the paper's grids. This holds alongside the best Table 6 accuracy of any method, constrained or not: 0.518, against ThinPlate's 0.633, which has violations on 91% of slices.
- **The accuracy gain comes from outside the visible strike range** (1.39 vs 1.96–2.01). Inside the range the new model ties ThinPlate (0.379 vs 0.374) and beats Fengler (0.440). On typical quotes the median is slightly worse than ThinPlate's (0.058 vs 0.051).
- **The guarantee has a cost on stressed days.** In the June 2024 election week, steep 7-day wings were smoothed (fold 10, in-range 0.63 vs ThinPlate 0.50).
- **The unconstrained learned interpolator is still more accurate in-range** (0.327). The guarantee costs about 0.05 vol points there.
- **Task C is mixed.** The new model is best on c2 RMSE and second to Fengler + PCHIP on c1 and on MAE. It is 3–4× better than the paper's v3.
- **Disclosure.** The atom grid was changed after a numerical failure on test fold 9 (see `CONFIG_FROZEN.md`); the change was re-validated on development days only. 41% of held-out strikes lie within 1e-3 in k of a generic midpoint pseudo-quote. That follows from the midpoint rule, not from the held-out locations; it is disclosed rather than hidden.
- **Audit.** An independent audit re-derived the guarantees and found zero violations on 150 random days with its own code. It found no leakage and reproduced every number.

## Files

- `code/`: model and pipeline.
  - `qpmix.py`, `surface.py`: arbitrage-free layer.
  - `prior.py`, `mlp.py`: neural prior.
  - `gen_pseudo.py`, `fit_windows.py`: pipeline.
  - `evaluate_v2.py`, `taskc_agg.py`, `arb_final.py`: scoring.
- `pseudo/`, `ab/`, `c/`: per-fold pseudo-quotes, surfaces + predictions, and Task C predictions.
- `final/`: `table6_metric.csv`, `table6_inrange.csv`, `table6_oor.csv`, `taskc_table11.csv`, `arb_summary.txt`, `final_predictions.csv`.
