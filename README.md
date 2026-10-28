# NIFTY IV surface: replication folder

Copy of the data and code behind "Arbitrage-Constrained Neural Interpolation of the Implied Volatility Surface:
Evidence from NSE Index Options" (Jain and Sinha). The raw NSE bhavcopy zips are not in this repository (1.1 GB); they are archived on Zenodo (DOI to be added).

```
data/
  raw_bhavcopy/    (not in git; Zenodo) NSE F&O bhavcopy zips, 2020-01-01 to 2026-01-30 (MM-YYYY folders), plus holiday and missing-date lists
  intraday_1min/   Upstox 1-minute NIFTY index and weekly-option candles, 2024-10 to 2026-10 (Section VII)
  pipeline/        pipeline outputs: 00_inventory, 02_parsed, 03_filtered (filter_waterfall.csv = Table 1),
                   04_iv/panel.parquet (frozen panel: 307,458 quotes, 9,438 slices), 05_splits (folds, held-out strikes)
code/
  pipeline/        src/ (numbered scripts 00-15), config.yaml, requirements.txt, PIPELINE_NOTES.md,
                   arbfree_neural/, neural_interp/, ablation/, seed_full/
  intraday_backtest/  Section VII scripts and results_long/ (124 sampled sessions)
  pnmix/           PNmix library (neural prior + lognormal-mixture QP)
  data_download/   NSE bhavcopy downloader
results/
  pipeline_results/  every phase's notes, tables and figures
  06_nn_runs/        neural-network training runs
```

Notes
- `code/pipeline/config.yaml` still points to the original paths (`E:\Research\data`, `E:\Research\iv_surface`).
  Edit `data_root`, `raw_bhavcopy` and `iv_surface_root` to run from this folder.
- `.env` files (API keys) and the private PNmix Pro engine are deliberately not copied.
