# Phase 0 Data Inventory — NIFTY IV Surface Paper

Scan date: 2026-08-27. Produced by `src/00_inventory.py`. Raw listing: `file_inventory.csv`
(1,506 files, ~15.1 GB, scoped to the folders actually relevant to this project).

## Sample period
Raw NSE F&O daily bhavcopy: `E:\Research\data\code downloaded data\MM-YYYY\FO_BHAV_YYYYMMDD.zip`,
77 month folders, **2020-01-01 through 2026-01-30** — continuous, ~6.1 years (double the
plan's 3-year minimum). Pre-combined panels at the same daily-contract grain already exist:
`combined_bhavcopy_2026_clean.parquet`/`.csv` (70,467,231 rows, all F&O symbols/instruments)
and slimmer variants (`clean_options_fast.parquet`, `bigfile.parquet` — same row count, fewer
columns). Only 3 missing trading dates across the whole span, per an existing
`code downloaded data/missing_dates.csv`: 2020-04-14, 2021-03-30, 2026-01-15 — confirm these
are NSE holidays before Phase 1 (minutes of work, not a project risk).

## Schema / fields available
Raw daily zip columns: `INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW,
CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP`. **SETTLE_PR is present
in the raw zips but was dropped when they were combined** into the parquet/CSV panels (those
keep OHLC + Close only). Close and settlement clearly diverge on stale/illiquid rows — verified
directly (NIFTYIT, 27-Feb-2020 expiry: Close 15341.00 vs Settle 15864.00 on a zero-volume row).
Per the plan's own preference for settlement over close, **Phase 1 should parse `SETTLE_PR`
from the raw zips**, not from the pre-combined tables. Volume (`CONTRACTS`) and open interest
(`OPEN_INT`) are present throughout in both raw and combined form, so filter F2 is fully
supported.

## Futures / forward
NIFTY futures sit in the same daily file as the options (`Instrument = FUTIDX`, later `IDF`),
matched by `(Date, ExpiryDate)`. No separate futures file and no put-call-parity fallback are
needed for any part of the 6-year span.

## Known data-format break (verified, and already fixed once in this repo)
NSE switched the F&O bhavcopy to the "UDiFF" format on **2024-07-08**, renaming instrument
codes: `FUTIDX→IDF`, `OPTIDX→IDO` (and `FUTSTK→STF`, `OPTSTK→STO` for single names); `CE/PE`
stays unchanged on the index side. Confirmed directly on the NIFTY series: `FUTIDX`/`OPTIDX`
rows run 2020-01-01–2024-07-05, `IDF`/`IDO` rows run 2024-07-08–2026-01-30, no overlap or gap
at the boundary. **This has already been solved once**: `E:\Research\changes\
08_fix_bhavcopy_post2024.py` (from an unrelated sibling project in this same E:\Research tree)
diagnoses and normalizes exactly this break, and its output already exists at
`E:\Research\pramukh\data\combined_bhavcopy_normalized.parquet` (567 MB, built 2025-07-28,
FUTIDX/OPTIDX convention restored across the full 2020–2026 span). Phase 1 can reuse this file
directly instead of re-deriving the mapping — but it still needs `SETTLE_PR` merged back in
from the raw zips, since it was itself built from the settle-less combined parquet.

## Existing reusable code (sibling project, same repo)
`E:\Research\changes\00_build_iv_skew.py` already implements Black-76 pricing + Brent's-method
IV inversion (`scipy.optimize.brentq`, vol bracket, forward = near-month futures close) for
single-name stock options from this same bhavcopy source. The architecture — forward-matching
by nearest expiry, liquidity filter, vol inversion via brentq — transfers directly to the NIFTY
index case in Phase 1 §1.5 of the plan; it needs adapting for settlement price (it currently
uses Close), Black-76 on the index future specifically, and the wider log-moneyness/tau grid
this paper's plan specifies.

## Gaps / limitations to carry into the paper
- **Risk-free rate**: no 91-day T-bill series found under `E:\Research`; pull from RBI/FBIL per
  plan §1.3. The existing sibling skew script uses a flat 6.5% placeholder, which is not
  sufficient for this paper's discounting.
- **pyarrow** could not be installed in a reasonable time in the Claude device-bridge shell
  (very slow pip download on that path); `duckdb` installed in ~20s and reads/queries the
  parquet files directly. Recommend standardizing the pipeline scripts on duckdb (or polars)
  rather than pandas+pyarrow, or pre-installing pyarrow once, off-hours, before Phase 1 starts.
- Background/detached shell jobs (`nohup ... &`) do not survive between separate device-bridge
  calls — each call's process tree is torn down when the call returns. Not a data issue, just
  a constraint on how Phase 1's longer jobs will need to be kicked off from this environment
  (run to completion within one call, or split into checkpointed steps) if run this way again;
  running directly on the Windows machine has no such limit.

## Decision gate (per plan §Phase 0)
All required fields — trade date, expiry, CE/PE, strike, a price, matched futures, open
interest, volume — are present for the full 2020–2026 span, and a working instrument-code
harmonization and a working Black-76/Brent inversion pattern both already exist in this repo.
**Gate passed**: proceed straight to Phase 1 when the calendar allows.

## Per the plan's own timeline
27 Aug 2026 falls inside the "Phase 0 only" window (through 5 Sep); Phase 1 is scheduled to
start 1 Oct, after the two other papers due 15 Oct. This inventory is the complete Phase 0
deliverable — no pipeline, inversion, or model code has been written or run beyond this scan,
per the plan's own stated discipline ("nothing but Phase 0 before 15 October").
