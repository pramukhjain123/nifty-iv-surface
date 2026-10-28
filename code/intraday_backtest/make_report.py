"""Write results/report.html from the backtest results (self-contained, opens in any browser)."""
import html
import math
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'summary.json')))
R1 = json.load(open(os.path.join(HERE, 'results_fresh1', 'summary.json'))) if os.path.exists(os.path.join(HERE, 'results_fresh1', 'summary.json')) else None
panel = pd.concat([pd.read_pickle(os.path.join(HERE, 'data', 'Nifty50_panel', f)) for f in sorted(os.listdir(os.path.join(HERE, 'data', 'Nifty50_panel')))])
gap_rs = (panel.mid - panel.fair).abs(); gap_v = ((panel.iv - panel.iv_fit) * 100).abs()
ctrl_rs = (panel.mid - panel.fair_ctrl).abs().dropna()

main = R['runs']['PNmix | spread x1.0']
A = main['all']
rs = lambda x, d=2: '–' if x is None else ('−' if x < 0 else '') + '₹' + f'{abs(x):,.{d}f}'
esc = html.escape

variants = [(k, v['all']) for k, v in R['runs'].items()]
if R1:
    variants += [(k + ' | price must trade in the same minute', v['all']) for k, v in R1['runs'].items()
                 if k in ('PNmix | spread x1.0', 'Neighbours (no model) | spread x1.0')]


def bar_svg(items):
    """Horizontal bars of average net per lot (single series)."""
    W, rowh, left, right = 760, 30, 300, 70
    vals = [v.get('avg_net') or 0 for _, v in items]
    lo, hi = min(0, min(vals)) * 1.15, max(0, max(vals)) * 1.15 or 1
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    H = rowh * len(items) + 40
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Average net profit per lot by variant">']
    step = 20 if hi - lo < 160 else 40
    for t in np.arange(math.ceil(lo / step) * step, hi + 1e-9, step):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="10" y2="{H - 26}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{rs(t, 0)}</text>')
    out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="6" y2="{H - 26}" class="zero"/>')
    for i, (name, v) in enumerate(items):
        y = 14 + i * rowh; val = v.get('avg_net') or 0; a, b = sorted([x(0), x(val)])
        out.append(f'<text x="{left - 8}" y="{y + 15}" class="lab" text-anchor="end">{esc(name)}</text>')
        out.append(f'<rect x="{a:.1f}" y="{y + 4}" width="{max(b - a, 1):.1f}" height="16" rx="3" class="bar{" alt" if "impossible" in name else ""}">'
                   f'<title>{esc(name)}: {rs(val)} per lot over {v.get("trades", 0)} trades</title></rect>')
        tx = b + 6 if val >= 0 else x(0) + 6                    # labels never collide with the category names
        out.append(f'<text x="{tx:.1f}" y="{y + 16}" class="val" text-anchor="start">{rs(val)}</text>')
    out.append('</svg>')
    return ''.join(out)


def short(k):
    k = (k.replace('PNmix | filled at the signal price (not achievable) | spread x', 'Signal-price fill (impossible), x')
          .replace('PNmix | spread x', 'PNmix, spread x').replace('Neighbours (no model) | spread x', 'No model, spread x')
          .replace(' | price must trade in the same minute', ', same-minute prices'))
    return k
chart_items = [(short(k), v) for k, v in variants if 'hold' not in k and 'x0.0' not in k]


def row(name, s):
    return (f'<tr><td>{esc(name)}</td><td>{s.get("trades", 0):,}</td><td>{s.get("gap_closed_pct", "–")}%</td>'
            f'<td>{s.get("median_min_to_close") if s.get("median_min_to_close") is not None else "–"}</td>'
            f'<td>{s.get("win_pct", "–")}%</td><td>{rs(s.get("avg_expected"))}</td><td>{rs(s.get("avg_gross"))}</td>'
            f'<td>{rs(s.get("avg_costs"))}</td><td class="b">{rs(s.get("avg_net"))}</td><td>± {rs(s.get("se_net"))}</td>'
            f'<td>{rs(s.get("avg_ex_index"))}</td><td>{rs(s.get("total_after_brokerage"), 0)}</td><td>{s.get("days_positive", "–")}</td></tr>')


rows_all = ''.join(row(k, v) for k, v in variants)
rows_day = ''.join(row(d, s) for d, s in main['by_day'].items())
spread_rows = ''.join(f'<tr><td>{esc(b)}</td><td>₹{v:.3f}</td><td>{n}</td></tr>' for b, v, n in R['spread_model'])
reasons = ', '.join(f'{k} {v}' for k, v in main['by_reason'].items())
sm = R['runs']['PNmix | filled at the signal price (not achievable) | spread x1.0']['all']
fresh_line = ''
if R1:
    f1 = R1['runs']['PNmix | spread x1.0']['all']
    fresh_line = (f'<li>With the stricter rule that every price must come from a trade in that same minute, PNmix made '
                  f'{f1["trades"]} trades averaging {rs(f1["avg_net"])} per lot ({f1["days_positive"]} days positive).</li>')

page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PNmix Backtest</title><style>
:root{{--bg:#f7f6f3;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e4e3df;--line:#d9d7d1;--bar:#2a78d6;--alt:#9b9a95;--neg-bg:#fdecec;--neg:#9b1c1c}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#121212;--panel:#1b1b1a;--ink:#f2f1ee;--ink2:#bdbbb5;--muted:#8f8d87;--grid:#2c2b29;--line:#3a3936;--bar:#3987e5;--alt:#6b6964;--neg-bg:#3a1414;--neg:#ffb4b4}}}}
:root[data-theme="dark"]{{--bg:#121212;--panel:#1b1b1a;--ink:#f2f1ee;--ink2:#bdbbb5;--muted:#8f8d87;--grid:#2c2b29;--line:#3a3936;--bar:#3987e5;--alt:#6b6964;--neg-bg:#3a1414;--neg:#ffb4b4}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:20px 16px 40px}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:0 0 10px}}
.sub{{color:var(--ink2);margin:0 0 16px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}}
.verdict{{background:var(--neg-bg);color:var(--neg);border-radius:10px;padding:12px 14px;margin-bottom:14px;font-weight:600}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}} .tile{{border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.tile b{{display:block;font-size:20px}} .tile span{{color:var(--ink2);font-size:12px}}
.tw{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th{{color:var(--ink2);font-weight:600}}
td:first-child,th:first-child{{text-align:left}} td.b{{font-weight:600}} ul{{margin:6px 0 0;padding-left:20px}} li{{margin:4px 0}}
svg .grid{{stroke:var(--grid)}} svg .zero{{stroke:var(--ink2);stroke-width:1.5}} svg .bar{{fill:var(--bar)}} svg .bar.alt{{fill:var(--alt)}}
svg text{{fill:var(--ink2);font-size:12px}} svg .val{{fill:var(--ink);font-weight:600}} svg rect:hover{{opacity:.8}}
.note{{color:var(--ink2);font-size:13px}} footer{{color:var(--muted);font-size:12px;margin-top:10px}}
</style></head><body><main>
<h1>PNmix backtest: NIFTY options, 25 Sep – 1 Oct 2026</h1>
<p class="sub">5 sessions of Upstox 1-minute candles · {R["minutes_fitted"]:,} minutes fitted · {R["quotes"]:,} option prices · expiries 29 Sep, 6 Oct, 13 Oct · lot 65</p>
<div class="verdict">Trading PNmix gaps lost money on every day after costs: {rs(A["avg_net"])} per lot on average over {A["trades"]} trades
({rs(A["total_after_brokerage"], 0)} for the week at 1 lot each, including brokerage).</div>

<section class="card"><h2>Main run (realistic: fill one minute after the signal, modelled spread, all charges)</h2><div class="tiles">
<div class="tile"><b>{A["trades"]}</b><span>paper trades</span></div>
<div class="tile"><b>{A["gap_closed_pct"]}%</b><span>gaps that closed (mid reached fair), median {A["median_min_to_close"]:g} min</span></div>
<div class="tile"><b>{rs(A["avg_expected"])}</b><span>expected per lot at the signal</span></div>
<div class="tile"><b>{rs(A["avg_net"])}</b><span>actual net per lot after STT and charges (± {rs(A["se_net"])})</span></div>
<div class="tile"><b>{A["win_pct"]}%</b><span>trades that made money</span></div>
<div class="tile"><b>{A["days_positive"]}</b><span>days with a profit</span></div></div>
<p class="note">Closed because: {esc(reasons)}.</p></section>

<section class="card"><h2>What went wrong: the gaps are real, but they close before a one-minute trader can act</h2><ul>
<li>PNmix fair values sit very close to where options trade: median gap {rs(float(gap_rs.median()), 3)} ({gap_v.median():.3f} vol points);
 with no model (each strike from its neighbours) the median gap is {rs(float(ctrl_rs.median()), 3)}.</li>
<li>Gaps that beat costs do close: {A["gap_closed_pct"]}% of them, median {A["median_min_to_close"]:g} minutes.</li>
<li>But most of the move happens inside the first minute. Filled at the signal price itself (impossible for anyone not watching the
 book tick by tick), the same rules make {rs(sm["avg_net"])} per lot ({sm["days_positive"]} days positive). Waiting one minute turns that into {rs(A["avg_net"])}.</li>
<li>Even the impossible version loses after the {rs(47.2)} brokerage per round trip at 1 lot ({rs(sm["total_after_brokerage"], 0)} for the week): the edge per lot is smaller than the fixed cost.</li>
<li>Against a no-model control (each strike priced from its two neighbours), PNmix lost less in the main run at every spread level, but with the stricter same-minute price rule it lost more at 1× and 1.5×. One week is too short to separate them; neither is profitable at one-minute speed.</li>
{fresh_line}</ul></section>

<section class="card"><h2>Average net per lot after charges, every variant</h2>{bar_svg(chart_items)}
<p class="note">Grey bar: filled at the signal price, which no one trading on one-minute data can get. Spread x0.5 / x1 / x1.5 scales the modelled bid-ask spread.</p></section>

<section class="card"><h2>All variants</h2><div class="tw"><table><thead><tr><th>Variant</th><th>Trades</th><th>Gaps closed</th><th>Median min</th><th>Win rate</th>
<th>Expected / lot</th><th>Gross / lot</th><th>Charges / lot</th><th>Net / lot</th><th>Std error</th><th>Ex-index / lot</th><th>Week, 1 lot each, incl. brokerage</th><th>Days positive</th></tr></thead>
<tbody>{rows_all}</tbody></table></div></section>

<section class="card"><h2>Main run by day</h2><div class="tw"><table><thead><tr><th>Day</th><th>Trades</th><th>Gaps closed</th><th>Median min</th><th>Win rate</th>
<th>Expected / lot</th><th>Gross / lot</th><th>Charges / lot</th><th>Net / lot</th><th>Std error</th><th>Ex-index / lot</th><th>Day, incl. brokerage</th><th>Positive</th></tr></thead>
<tbody>{rows_day}</tbody></table></div></section>

<section class="card"><h2>How it was run</h2><ul class="note">
<li><b>Data:</b> Upstox 1-minute candles for the NIFTY index and every option within ±8% of spot (550 contracts; the expired 29 Sep series from Upstox's expired-instruments API).</li>
<li><b>Fair value:</b> every minute, the same pipeline as PNmix Live: last traded price of each option (only candles with volume, at most 1 minute old), forward from put-call parity, OTM strikes, nearest two expiries with 2+ hours left, PNmix fit.</li>
<li><b>Signal:</b> the live track-record rule. The gap must beat STT, exchange, SEBI, stamp duty and GST at the modelled bid/ask, and the option must trade at least one lot in the next minute.</li>
<li><b>Fills:</b> the next minute's opening trade, minus half the spread to sell or plus half to buy. Exits the same way when the mid crosses the current fair value, after 60 minutes (15 and 30 also tested), at 15:25, or after 10 minutes without trades. No new trades in an expiry with under 3 hours left.</li>
<li><b>Spread model:</b> candles have no bid/ask, so the half-spread comes from a real Upstox option chain (OTM strikes, median by price level, table below) and is tested at 0.5×, 1× and 1.5×.</li>
<li><b>Ex-index:</b> net minus option delta × index move, to separate the volatility gap from market direction.</li></ul>
<div class="tw"><table style="max-width:420px"><thead><tr><th>Option price</th><th>Median half-spread</th><th>Quotes</th></tr></thead><tbody>{spread_rows}</tbody></table></div></section>

<section class="card"><h2>Limits</h2><ul class="note">
<li>Only one week. It was a strongly falling week (NIFTY roughly 23,100 → 22,400), and the expiry day (29 Sep) had the most signals.</li>
<li>Last-traded prices bounce between bid and ask, which creates false gaps. Live bid/ask quotes (what PNmix Live uses) are cleaner, so the live track record is the better test.</li>
<li>The spread model comes from one after-hours snapshot; intraday spreads near the open can be wider.</li>
<li>Trades on nearby strikes at the same moment are related, so there are fewer independent bets than trades.</li></ul></section>
<footer>Research only, not investment advice. Hypothetical results from historical data; they do not predict future results.</footer>
</main></body></html>'''
open(os.path.join(HERE, 'results', 'report.html'), 'w').write(page)
print('wrote results/report.html')
