# Frozen configuration (final, "v2")
Chosen on development days only (all dates < 2021-10-01, never in any test window).

- **QP layer.** Per-day joint QP across all listed maturities.
  - Each maturity's density is a lognormal-kernel mixture on a two-resolution atom dictionary:
    - fine: 32 atoms within ±4 ATM std, kernels (1, 4) × spacing;
    - coarse: 50 atoms over the padded range, kernels (1, 4, 16) × spacing.
  - λ = 1e-7 (second-difference smoothness on the weights); vega floor 5e-3.
  - Calendar ordering is imposed on a 100-point k grid.
  - Running max over maturity.
  - Variance-time price interpolation between listed maturities.
- **Neural prior.**
  - Interior MLP 256-256-128 and edge MLP 128-128, Huber loss, weight decay 1e-3, about 37 epochs, seeds (0, 1).
  - Trained per fold on VISIBLE strikes of dates < test_start, with every 30–60-day slice excluded.
- **Pseudo-quotes.**
  - Placed at midpoints between visible strikes, and 2 strike spacings beyond each edge (within k ∈ [−0.35, 0.25]).
  - Weight α = 1.0 relative to a visible quote.

## History, disclosed
v1 used a single uniform atom grid per slice (50 atoms × (1, 4, 16)), frozen on the same development days. On test fold 9 it failed numerically: on a 7-day weekly with very wide strike coverage, the atom spacing was about the ATM std, so the slice could not be resolved.

The two-resolution dictionary (v2) was then selected on development days only. On those days it matched v1's combined score (0.544 vs 0.537), with better in-range and typical-quote error.

v1 results for folds 0–10 are kept for comparison:

| Folds 0–10 | Table 6 RMSE |
|---|---|
| v1 | 0.595 |
| v2 | 0.557 |
| ThinPlate | 0.714 |
| Fengler | 0.696 |
