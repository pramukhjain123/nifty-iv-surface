# Backtests

Research scripts behind the backtest reports (run from this folder, with `pnmix_live` installed: `pip install -e ..`).

| Script | What it does |
|---|---|
| `fetch_week.py`, `fetch_long.py` | Download Upstox 1-minute option candles (one week / about 2 years) into `BT_DATA` |
| `prep_days.py`, `run_long.py` | Build per-day files and fit PNmix every minute on the sampled days |
| `backtest.py` | Minute-by-minute strategy simulation (`python backtest.py fit`, then `python backtest.py trade`) |
| `ten_lots.py`, `no_charges.py`, `diag.py` | Variants: 10 lots / no wait, P&L before charges, diagnostics |
| `make_report.py`, `month_report.py`, `long_report.py` | HTML reports (1 week, 1 month, 2 years) |

Environment: `BT_DATA` (input folder, default `data`), `BT_OUT` (results folder, default `results`), `BT_FRESH`, `BT_TAG`.
The 2-year results used in the paper are in `results_long/` and `pnmix_backtest_2years.html`.

These are one-off research scripts: they are kept as they were run (only their imports were updated) and are not
covered by the package's lint and tests.
