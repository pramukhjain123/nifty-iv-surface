# Pre-paper checks for the arbitrage-free neural surface (October 2026)

All checks use the paper's own metrics. Table 6/7 figures are averaged over the 17 blind folds, with fold 16 excluded; units are vol points.

## 1. Ablation: is the neural part doing the work? Yes.
Same QP, same guarantees, no neural pseudo-quotes, all 18 test folds.

| Table 6 metric | With neural prior | No prior | ThinPlate | Fengler |
|---|---|---|---|---|
| All held-out RMSE | **0.518** | 0.687 | 0.633 | 0.646 |
| In-range RMSE | **0.379** | 0.419 | 0.374 | 0.440 |
| Out-of-range RMSE | **1.385** | 2.098 | 1.963 | 2.006 |
| Task C c1 RMSE | **1.625** | 1.771 | 2.507 | 1.473 |
| Task C c2 RMSE | **1.101** | 1.289 | 1.648 | 1.210 |

- The prior helps in **17 of 17** folds, and on **807 of 1,004** days (sign test p = 4e-88, Wilcoxon p = 9e-87).
- **Without the prior, the guaranteed surface loses to both ThinPlate and Fengler.** The accuracy advantage comes from the learned component; the zero-arbitrage guarantee comes from the construction.

## 2. Seed variance of the prior (all 18 folds, seeds (2,3) vs (0,1))

| Table 6 metric | Seeds (0,1) | Seeds (2,3) |
|---|---|---|
| All held-out | 0.518 | 0.517 |
| In-range | 0.379 | 0.380 |
| Out-of-range | 1.385 | 1.379 |

- The mean per-fold |difference| is 0.005 (maximum 0.026).
- The across-seed sd of the 17-fold mean is 0.0005.
- For comparison, the paper's v3 seed noise floor is ±0.45 on Task B.

## 3. Table 6, training-window column ("Train hold-out")

| Method | Train hold-out |
|---|---|
| **New model** | **0.622** |
| Fengler | 0.833 |
| ThinPlate | 0.947 |
| SSVI | 2.791 |
| SVI | 3.946 |
| v4 | 4.14 |
| v3 | 6.68 |

The baseline values are recomputed here and match the paper exactly.

How the training-window days were predicted:
- Days from 2021-10 onward use the out-of-time prediction from the fold whose test window contains them. That prior never saw the day.
- The 431 days before 2021-10 use fold 0's prior, which was trained on visible strikes of dates before 2021-10 (never on held-out ones). Pooled over those days: in-range 0.394, out-of-range 1.895.

## 4. Fold 16 (excluded from the paper's aggregates), reported separately

| Fold 16 | New | ThinPlate | Fengler | SSVI |
|---|---|---|---|---|
| All held-out | 0.601 | 0.616 | **0.400** | 2.112 |
| In-range | **0.202** | 0.278 | 0.224 | 1.937 |
| Out-of-range | 2.215 | 2.160 | **1.437** | 3.808 |
| Task C c1 / c2 | **1.219 / 0.760** | — | 1.291 / 0.780 (Fengler + PCHIP) | — |

Fengler wins fold 16 overall because of out-of-range quotes. On the paper's main metric this is one of the 16-of-17 per-fold comparisons the new model does not win.

## 5. Cost (Table 10)

| Step | Cost |
|---|---|
| QP surface fit + prediction | mean 23.6 s per day, single process (median 20.5 s, range 11.5–43.6 s; 7.2 expiries per day) |
| Neural prior training | about 37 s per seed per fold window on 2 cores; 1–3 minutes per fold for 2 seeds |
| Prior inference | about 4 s per day |

- A full-panel refit takes about 10 CPU-hours.
- That is far slower than SSVI (37.7 ms per day) or ThinPlate (16 ms per day). It is a real cost, and the paper should report it.

## 6. Arbitrage
The construction is identical in every run, so the guarantee holds for every fit. The main run was checked on the paper's dense grids: 0 violations over 1,065 days, 7,339 slices, 447,679 butterfly points, 6,274 calendar pairs and about 6.3 million between-maturity points.
