# Can a neural model beat the classical baselines? — Findings (Oct 2026)

Exploratory work, kept separate from the paper. Nothing in `paper/` was changed.

## Why the paper's network loses

The paper's neural model is one pooled surface `w(k, tau)` fit across 18+ months, with no date input. Every classical baseline (SVI, SSVI, thin-plate, cubic, Fengler) is refit on each day's own visible strikes. That mismatch explains the gap on held-out strikes (Task A, about 6.7 vol-pts versus 0.42 for thin-plate).

## What works: a learned local interpolator

- **Method.** For each held-out strike, take the 4 nearest visible strikes on each side of the same (date, expiry) slice. Start from linear interpolation. An MLP (256-256-128, GELU, Huber loss) predicts the correction from the neighbours' offsets and residuals, plus tau, moneyness and ATM vol.
- **Training data.** The network learns from leave-one-out on visible strikes only, using all dates before the fold's test start. The paper's held-out strikes are never seen.
- **Evaluation.** The paper's 18 walk-forward folds, scored on the paper's fixed Task-A held-out strikes inside the visible range. These are exactly the quotes the baselines are scored on. The final model is a 5-seed ensemble.

## Results (vol-pts; common quote set n = 42,227, test windows 2021-10 to 2026-01)

| Method | RMSE | MAE | Median abs err | Folds where NN wins (of 18) |
|---|---|---|---|---|
| **Neural interpolator** | **0.320** | **0.101** | 0.042 | — |
| Akima (per slice)* | 0.345 | 0.108 | 0.042 | 17 |
| Linear (per slice)* | 0.349 | 0.113 | 0.043 | 17 |
| PCHIP (per slice)* | 0.350 | 0.111 | 0.043 | 18 |
| Thin-plate (paper) | 0.368 | 0.115 | 0.043 | 18 |
| Fengler (paper) | 0.432 | 0.135 | 0.048 | 18 |
| Cubic spline (paper) | 1.434 | 0.179 | 0.048 | 18 |
| SSVI (paper) | 1.990 | 1.383 | 0.968 | 18 |
| SVI (paper) | 2.439 | 1.114 | 0.372 | 18 |

\* These per-slice interpolants are not in the paper. Plain linear interpolation already beats every baseline the paper uses, so the neural model has to beat those too.

### Robustness

- **Clean folds.** Folds 0–3 and 9–17 never touched any design choice. On them: NN 0.337, Akima 0.361, thin-plate 0.386, Fengler 0.439.
- **Day-clustered bootstrap, 95% CI of the RMSE difference.**
  - NN − Akima: [−0.039, −0.014]
  - NN − thin-plate: [−0.062, −0.036]
  - NN − Fengler: [−0.150, −0.080]
- **Diebold–Mariano tests** (squared error, day-clustered Newey–West): t = −4.0 vs Akima, −6.8 vs thin-plate, −5.4 vs Fengler.
- **Single seeds.** Each seed alone scores 0.322–0.325, so the result does not depend on ensembling.
- **Buckets.** The NN wins in every maturity bucket and every moneyness bucket. The gain is largest for deep puts and for maturities over 3 months.

### Where the gain comes from

The gain is in the error tails: 99th-percentile error is 1.02 vs 1.29 for Akima. The median is a tie at about 0.042, which is likely the quote-noise floor. The headline improvement is about 7% RMSE over the best classical interpolant and 13% over the paper's best baseline (thin-plate).

## Arbitrage (the paper's theme)

The NN is a pointwise interpolator, so it has no no-arbitrage guarantee. The real market quotes themselves fail the discrete butterfly test in 14.4% of strike triples. The NN's completed smiles fail in 10.3%, versus 8.0% for Akima and 8.8% for thin-plate.

To compare on equal footing, the same convex call-price projection (vega-weighted QP: decreasing, convex, within bounds) was applied to every method's completed smile. After that, every smile is butterfly-free on the strike grid, and the ranking holds:

| Method | RMSE after repair | MAE after repair |
|---|---|---|
| **NN** | **0.393** | **0.116** |
| Akima | 0.408 | 0.121 |
| Thin-plate | 0.415 | 0.124 |
| Linear | 0.416 | 0.127 |
| Fengler | 0.446 | 0.138 |

(Repaired-comparison set: n = 44,156.)

## Caveats

- **Contamination.** Fold 4's test window (2022Q4) was the validation set used for the configuration choice. The feature-set choice was made while looking at 2023, which is folds 5–8. The clean-fold numbers above exclude both.
- **Scope.** This covers in-range interpolation only. Out-of-range extrapolation (about 7% of held-out quotes) is not addressed.
- **What the paper's experiment answers.** The paper's model is a global, cross-time surface; this one is a per-day interpolator with a learned prior. They answer different questions. The new result does not overturn the paper's null result for pooled surfaces. It shows that a neural model that uses each day's visible quotes beats the classical interpolators.
- **Audit.** An independent audit found no leakage. Training uses visible strikes only and dates before the test start. Test features use only the visible strikes of the same slice.

## Files

- `code/`
  - `load.py`, `feats2.py`, `mlp.py`: data loading, features and the MLP.
  - `wf.py`: walk-forward driver.
  - `compare.py`: matched comparison.
  - `arb.py`: butterfly check.
  - `repair.py`: arbitrage repair.
  - `sweep.py`, `proto.py`, `proto2.py`: design exploration.
- `results/`
  - `wf_preds.csv`, `matched.csv`: per-quote predictions.
  - `summary_common.csv`, `per_fold_rmse.csv`, `arb_summary.csv`, `repair_summary.csv`, `sweep.csv`.
- To rerun, use the cloud-workspace paths at the top of `load.py`. Point `R` at `E:\Research\iv_surface`.

## Against the paper's own tables (Full_Paper.pdf, Tables 6–7)

The metric is the paper's Table 6/7 rule: test-window held-out RMSE averaged over the 17 blind folds, with fold 16 excluded. The baseline rows reproduce the paper exactly: thin-plate 0.633 / 0.181 / 0.051 / 0.073; Fengler 0.646; SSVI 2.179; SVI 2.573.

The neural interpolator only covers in-range strikes. For the paper's all-strikes number, the 6.6% of out-of-range strikes are filled with thin-plate's prediction (full coverage).

| Method | RMSE | MAE | Median | 10% trimmed mean |
|---|---|---|---|---|
| **New neural model + thin-plate for out-of-range** | **0.607** | **0.167** | **0.050** | **0.069** |
| Thin-plate (paper's best) | 0.633 | 0.181 | 0.051 | 0.073 |
| Fengler | 0.646 | 0.203 | 0.055 | 0.080 |
| SSVI | 2.179 | 1.469 | 1.006 | 1.189 |
| SVI | 2.573 | 1.212 | 0.485 | 0.723 |
| Paper's NN v4 (state) | 3.35 | — | — | — |
| Paper's NN v3 (static) | 5.10 | — | — | — |

### In-range strikes only (pure interpolation, same 17-fold metric)

| Method | RMSE |
|---|---|
| **New neural model** | **0.327** |
| Thin-plate | 0.374 |
| Fengler | 0.440 |
| Cubic spline | 1.164 |
| SSVI | 1.997 |

- **In-range wins.** The new model beats thin-plate and Fengler in 17 of 17 blind folds.
- **All-strikes wins.** It beats thin-plate in 17 of 17 folds. It beats Fengler in only 10 of 17, because Fengler's out-of-range behaviour sometimes wins.
- **Clean folds only.** Excluding folds 4–8 (used in my design choices) as well as fold 16 leaves 12 folds:
  - All strikes: 0.620 vs thin-plate 0.648 vs Fengler 0.657.
  - In-range: 0.351 vs 0.399 vs 0.452.

Script: `code/paper_metric.py`.
