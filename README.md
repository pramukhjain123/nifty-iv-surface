# NIFTY IV surface: replication folder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23728872.svg)](https://doi.org/10.5281/zenodo.23728872)

Code and derived data behind "Arbitrage-Constrained Neural Interpolation of the Implied Volatility Surface:
Evidence from NSE Index Options" (Jain and Sinha). Third-party market data (NSE bhavcopy, Upstox candles) are not redistributed; see "Data not included" below.

```
data/
  raw_bhavcopy/    (not included) NSE F&O bhavcopy zips, 2020-01-01 to 2026-01-30 (MM-YYYY folders), plus holiday and missing-date lists
  intraday_1min/   (candles not included) Upstox 1-minute NIFTY index and weekly-option candles, 2024-10 to 2026-10 (Section VII)
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

## Data not included

NSE's and Upstox's terms do not allow these files to be redistributed, so they are left out. The scripts that recreate them are included.

- `data/raw_bhavcopy/`: NSE F&O bhavcopy zips, 2020-01 to 2026-01. Download with `code/data_download/nse_udiff_downloader_enhanced.py`.
- `data/pipeline/02_parsed/*.parquet` and `data/pipeline/03_filtered/nifty_filtered.parquet`: quote-level bhavcopy rows. Rebuild from the zips with stages 01-04 in `code/pipeline/src/`.
- `data/intraday_1min/*.csv.gz`: Upstox 1-minute NIFTY index and option candles for Section VII. Fetch with `code/intraday_backtest/fetch_long.py` using your own Upstox API token (`UPSTOX_ANALYTICS_TOKEN` in `code/.env`).

Without these files you can still reproduce every table and figure in Sections I-VI from the derived panel in `data/pipeline/04_iv/`. The Section VII trade logs and summary are included in `code/intraday_backtest/results_long/`.

## How to cite

See `CITATION.cff` (GitHub shows it under "Cite this repository"). Archived on Zenodo: v1.0.0 is [10.5281/zenodo.23728872](https://doi.org/10.5281/zenodo.23728872); all versions are [10.5281/zenodo.23728871](https://doi.org/10.5281/zenodo.23728871).
