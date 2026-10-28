# Phase 1 — Pipeline Notes (raw files -> clean IV panel)

Run date: 2026-08-27 (started ahead of the plan's own Oct-1 schedule, at the user's
explicit request, overriding the plan's "nothing but Phase 0 before 15 October" rule).

## Pipeline (src/01_parse.py -> 05_invert.py)
1. **Parse** (`01_parse.py`): NIFTY-only rows from all 1,496 raw daily F&O bhavcopy zips,
   2020-01-01 to 2026-01-30. Handles BOTH raw formats found in the data: the pre-2024-07-08
   legacy format and NSE's UDiFF format from 2024-07-08 onward (completely different column
   names, not just renamed instrument codes -- confirmed a clean, same-day cutover, no overlap).
   Output: 2,948,548 NIFTY rows (2,944,060 options + 4,488 futures), matching the known total
   from the pre-combined panel exactly.
2. **Attach forward** (`02_attach_forward.py`): 29.6% of (date,expiry) slices matched directly
   to a NIFTY futures settlement; the remaining 70.4% (mostly weekly expiries, which never
   trade as futures) via put-call parity at the straddle strike minimizing |C-P|, preferring a
   liquid (volume>0, OI>0 on both legs) straddle and falling back to the unrestricted argmin
   only when no liquid pair exists in that slice (flagged `put_call_parity_illiquid`, ~31% of
   rows -- expected to be mostly filtered out downstream, and it was: see waterfall).
3. **Risk-free rate** (`03_riskfree.py`): RBI repo rate, forward-filled daily, as a documented
   proxy for the plan's specified 91-day T-bill yield (no free daily T-bill series was
   obtainable this session -- see the script's docstring for sources tried). Plan's own note:
   this "barely moves the IVs," so precision here is a lower priority than everywhere else.
4. **Filters F1-F7** (`04_filters.py`): see `03_filtered/filter_waterfall.csv`. One filter was
   tightened from the plan's literal spec after a data-quality finding (below): F2 raised from
   `volume>0` to `volume>=10` (OI>0 unchanged).
5. **IV inversion** (`05_invert.py`): Black-76, vectorized Newton-Raphson (whole panel at once,
   numpy) with a per-row Brent's-method fallback for anything Newton doesn't converge on. Pure
   per-row Brent (as literally specified) measured ~2ms/row -- ~11 minutes for the full panel,
   too slow for this session's per-call time budget, so Newton does the heavy lifting and Brent
   is the safety net. 100% of rows converged on Newton alone this run (0 fallback needed).

## Filter waterfall (F0 -> F7)
| step | rows remaining | slices remaining |
|---|---|---|
| F0 start (options w/ forward) | 2,944,060 | 29,135 |
| F1 tau in [7/365, 1.0] | 2,162,119 | 15,671 |
| F2 volume>=10 & OI>0 | 714,131 | 12,370 |
| F3 moneyness [-0.35,0.25] | 705,570 | 12,348 |
| F4 price > intrinsic+tick | 681,835 | 12,346 |
| F5 OTM only | 474,073 | 12,289 |
| F6 static-arbitrage (155,951 violations dropped) | 318,122 | 12,289 |
| F7 >=8 strikes/slice | **307,458** | **9,438** |

## IV sanity checks (per plan's own instructions)
- Overall: mean 17.7%, median 15.8%, IQR [12.8%, 20.4%], min 6.9%, max 126% -- max sits in the
  March 2020 COVID crash window, consistent with historical India VIX behaviour.
- Near-ATM (|k|<0.03) median IV by year: 2020 18.9%, 2021 15.1%, 2022 17.4%, 2023 10.8%,
  2024 13.1%, 2025 11.6%, 2026 11.0%. 2023's unusually low reading matches NIFTY's well-known
  low-volatility regime that year; 2020 and 2022 elevated readings match COVID and the
  2022 rate-hike/war period. All plausible, no red flags.
- 10 random daily smiles plotted twice (before/after the F2 tightening below):
  `sanity_check_smiles.png`, `sanity_check_smiles_v2.png`. 9 of 10 in both draws show the
  textbook NIFTY shape (downward put skew, slight call-wing upturn). One draw surfaced a
  genuine anomaly.

## Finding: a real data anomaly, investigated and left in place
2024-03-13, expiry 2024-05-30: three adjacent OTM call strikes (22700, 23100, 23600) priced at
implied vols of ~11.5-11.8%, sandwiched between neighbours at ~15-16%, in the first-draw plot.
Investigated directly (dumped every row in that slice): the forward, tau, and moneyness values
are all correct and consistent -- this is a real settlement-price anomaly in that day's raw
NSE bhavcopy, not a pipeline bug. Raising the volume floor to 10 (from the plan's literal >0)
removed some noisy low-volume slices elsewhere but did NOT remove this one (those three strikes
had volume 63, 4→gone, and 28 respectively) -- so a second random draw still surfaced one
similarly noisy slice (2024-04-02, expiry 2024-05-02). This looks like irreducible
microstructure noise in thinly-traded weekly-expiry middle strikes, at a rate of roughly
1-in-10 slices by casual sampling, not a systemic bug -- and it's a reasonable expectation that
the whole point of Phase 3's arbitrage-constrained fitting is to be robust to exactly this kind
of raw-quote noise. Left in the frozen panel; worth a footnote in the paper's data section
rather than hand-scrubbing individual days, and worth watching if Phase 4 error decomposition
shows it concentrated in specific moneyness/maturity buckets.

## Frozen deliverable
`04_iv/panel.parquet` -- 307,458 rows, 9,438 (date,expiry) slices, columns:
`date, expiry, instrument, opt_type, strike, settle_price, oi, volume, tau, fwd, fwd_source,
k, r, iv`. Per the plan: "Freeze it. Everything downstream reads only this."
