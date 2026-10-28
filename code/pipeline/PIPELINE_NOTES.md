# Arbitrage-Constrained Neural IV Surface — NIFTY Index Options

Working paper for IEEE Access. Full plan: see `paper/PLAN.md` in this repo (the design doc
this project was set up from). Target submission: mid-December 2026.

## Status (updated 2026-08-28)
Phases 0-4 are complete; Phase 5 (writing) has a full draft manuscript. Current state:

- **Phase 0 (data inventory)** — done. `00_inventory/phase0_inventory_note.md`.
- **Phase 1 (pipeline)** — done, ahead of the plan's original 1 Oct schedule (started early
  at the user's request). Frozen panel: `04_iv/panel.parquet` (307,458 rows, 9,438 slices).
  Notes: `04_iv/phase1_notes.md`.
- **Phase 2 (splits & baselines)** — done. Four classical baselines (SVI, SSVI, ThinPlate,
  CubicSpline) fit and scored. Notes: `results/phase2_notes.md`.
- **Phase 3 (model)** — done. v3 hard-calendar-constraint architecture selected
  (λ=0.01). Notes: `results/phase3_notes.md` (includes a disclosed pre-registration
  violation during model selection — see Phase 4 notes for how it's handled).
- **Phase 4 (evaluation)** — done. Full 18-fold walk-forward evaluation, baseline
  arbitrage-violation measurement, Diebold-Mariano tests, robustness cuts, seed-variance
  check. Notes: `results/phase4_notes.md`.
- **Phase 5 (writing)** — draft manuscript in `paper/`, revised against a full review
  (mapping in `paper/REVIEW_RESPONSE.md`).
- **Phase 6 (reviewer-risk revision)** — done. Three additions, all in the manuscript:
  - **Fengler (2009) implemented** as a fifth classical baseline (`src/11a_fengler.py`,
    scored by `11b`, summarised by `11d`). Constrained QP on normalised call prices;
    fits all 9,438 slices; zero price-space convexity/slope/density violations; Task A
    in-range RMSE 0.445 vol pts vs thin-plate 0.422, and best of all methods
    out-of-range at 2.31. This materially reframes the paper — see Results §VI-C.
  - **Task C, maturity holdout** (`src/12a`–`12e`). The [30,60]-day band is removed from
    training on all 18 folds and reconstructed. Withholding it changes the network's
    accuracy on that band by +0.7% / -0.1%, so the architecture does interpolate in
    maturity by construction; but classical baselines given a monotone PCHIP rule are
    two to four times more accurate on the same band.
  - **Multi-strike forward sensitivity** (`src/11c`). Median difference from the Phase-1
    forward is 3.8 bp; full panel rebuild moves F6 rejection 34.651% -> 34.506% and IVs
    by a mean absolute 0.121 vol points. Closes REVIEW_RESPONSE item 10.
  - `src/14_verify_paper_numbers.py` re-derives every number added in this revision from
    the results files and checks it against the .tex sources (33/33 pass).
  - `src/13_build_repro_archive.py` builds `repro/iv_surface_replication.zip`.
  - Manuscript compiles to 13 pages on the IEEE Access class.

  Still needs before submission: author ORCID, affiliation, biography, acknowledgments,
  and a Zenodo DOI for the Data Availability Statement (upload `repro/` zip).

## Layout
- `00_inventory/` — Phase 0 output
- `01_raw/` .. `05_splits/` — Phase 1 pipeline stages and Phase 2 splits
- `06_nn/` — neural network training runs (v1/v2/v3, all folds, seed-variance runs)
- `src/` — numbered, re-runnable pipeline/model/eval scripts
- `results/` — every phase's notes.md and output tables/figures/CSVs
- `paper/` — Phase 5 manuscript: `main.tex`, `sections/*.tex`, `refs.bib`, `figures/`,
  compiled `main.pdf`, and `PLAN.md` (the original design doc)
- `config.yaml`, `requirements.txt` — pinned paths/params/deps, per the plan's reproducibility
  section

## Key facts from Phase 0 (details in the inventory note)
- 2020-01-01 to 2026-01-30 of NSE F&O bhavcopy, continuous, ~6.1 years.
- Settlement price (`SETTLE_PR`) lives only in the raw daily zips, not the combined
  parquet/CSV panels — Phase 1 parsing reads it from there.
- NSE renamed index instrument codes on 2024-07-08 (`FUTIDX/OPTIDX` -> `IDF/IDO`); Phase 1's
  parser handles both formats with a clean, non-overlapping cutover.
- A sibling project's working Black-76 + Brent IV-inversion architecture
  (`changes/00_build_iv_skew.py`) informed Phase 1's inversion step.

## Headline results (Phases 4 and 6)
- The hard-constrained neural network (v3, λ=0.01) achieves **0% butterfly and calendar
  arbitrage violation across all 18 walk-forward folds**.
- Measuring the same violation rate for the four classical baselines (new in Phase 4 §9):
  SVI, ThinPlate, and CubicSpline all show material violation rates (4-91%); **SSVI is
  also empirically arbitrage-free**, so the network's guarantee differentiates it from
  three of four baselines, not all four.
- The network does **not** close the fit-accuracy gap to any classical baseline
  (Diebold-Mariano rejects equal accuracy vs. SVI/SSVI/ThinPlate at p<0.0001) — reported as
  an honest null result, not reframed.
- **Phase 6:** Fengler's constrained smoother is arbitrage-free by construction *and* about
  as accurate as the best unconstrained smoother, so "constrained and accurate" is not a
  property that distinguishes the neural surface here. What survives is the maturity
  claim, which Task C confirms structurally (withholding a maturity band costs the
  network essentially nothing) and refutes practically (classical methods with a cheap
  interpolation rule beat it on the same band).
