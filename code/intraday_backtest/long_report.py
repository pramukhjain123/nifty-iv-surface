"""Two-year PNmix backtest (every 4th trading day): every variant (wait / no wait, 1 / 10 lots, with / without charges, spreads, slippage,
no-model control), statistics by day, and a self-contained HTML report.

Run after run_long.py.   BT_DATA=data_long BT_OUT=results_long python long_report.py
"""
import html
import json
import math
import os

import numpy as np
import pandas as pd

import backtest as bt
from pnmix_live import charges

FIX = charges.fixed_round_trip()
P = pd.concat([pd.read_pickle(os.path.join(bt.PANEL, f)) for f in sorted(os.listdir(bt.PANEL)) if f.endswith('.pkl')])
h, spread_table = bt.spread_model()
NOW = P.copy(); NOW['next_open'] = NOW.mid                     # "no wait": filled at the signal price
DAYS = sorted(P.day.unique())
os.makedirs(bt.OUT, exist_ok=True)


def run(panel, lots, fair='fair', mult=1.0, min_net=0.0, vol_lots=1, slip=0):
    bt.SLIP_TICKS = slip
    T = bt.simulate(panel, fair, mult, h=h, min_net=min_net, min_vol_lots=vol_lots)
    bt.SLIP_TICKS = 0
    return T


def stats(T, lots, with_charges=True):
    if T is None or len(T) == 0:
        return {'trades': 0, 'month': 0.0, 'days_positive': f'0/{len(DAYS)}'}
    pnl = lots * (T.net_per_lot if with_charges else T.gross_per_lot) - (FIX if with_charges else 0.0)
    day = pnl.groupby(T.day).sum().reindex(DAYS, fill_value=0.0)
    cum = day.cumsum(); dd = float((cum - cum.cummax()).min())
    rng = np.random.default_rng(0)
    boot = [rng.choice(day.values, len(day)).sum() for _ in range(5000)]       # resample whole days
    srt = pnl.sort_values(ascending=False)
    t = day.mean() / (day.std(ddof=1) / math.sqrt(len(day))) if day.std(ddof=1) > 0 else None
    return {'trades': int(len(T)), 'win_pct': round(100 * float((pnl > 0).mean()), 1), 'month': round(float(pnl.sum()), 0),
            'avg_trade': round(float(pnl.mean()), 0), 'days_positive': f'{int((day > 0).sum())}/{len(day)}',
            'best_day': round(float(day.max()), 0), 'worst_day': round(float(day.min()), 0), 'max_drawdown': round(dd, 0),
            'boot_5': round(float(np.percentile(boot, 5)), 0), 'boot_95': round(float(np.percentile(boot, 95)), 0),
            'p_loss': round(float(np.mean(np.array(boot) < 0)), 3), 't_daily': None if t is None else round(float(t), 2),
            'top5_share': round(float(srt.head(5).sum() / pnl.sum()), 2) if pnl.sum() > 0 else None,
            'without_top10': round(float(srt.iloc[10:].sum()), 0),
            'charges': round(float((lots * T.costs_per_lot).sum() + len(T) * FIX), 0),
            'gap_closed_pct': round(100 * float((T.reason == 'gap closed').mean()), 1),
            'median_min': float(T[T.reason == 'gap closed'].minutes.median()) if (T.reason == 'gap closed').any() else None,
            'daily': {d: round(float(v), 0) for d, v in day.items()}}


V = [  # name, panel, lots, fair, spread mult, min net per lot, lots traded next minute, slippage ticks
    ('Wait 1 min, 1 lot', P, 1, 'fair', 1.0, 0.0, 1, 0),
    ('Wait 1 min, 10 lots', P, 10, 'fair', 1.0, FIX / 10, 1, 0),
    ('No wait, 1 lot', NOW, 1, 'fair', 1.0, 0.0, 1, 0),
    ('No wait, 10 lots', NOW, 10, 'fair', 1.0, FIX / 10, 10, 0),
    ('No wait, 10 lots, +1 tick per fill', NOW, 10, 'fair', 1.0, FIX / 10, 10, 1),
    ('No wait, 10 lots, +2 ticks per fill', NOW, 10, 'fair', 1.0, FIX / 10, 10, 2),
    ('No wait, 10 lots, spread x0.5', NOW, 10, 'fair', 0.5, FIX / 10, 10, 0),
    ('No wait, 10 lots, spread x1.5', NOW, 10, 'fair', 1.5, FIX / 10, 10, 0),
    ('Wait 1 min, 1 lot, no model (neighbours)', P, 1, 'fair_ctrl', 1.0, 0.0, 1, 0),
    ('No wait, 10 lots, no model (neighbours)', NOW, 10, 'fair_ctrl', 1.0, FIX / 10, 10, 0),
]
res = {'days': DAYS, 'minutes_fitted': int(P.groupby(['day', 'j']).ngroups), 'quotes': int(len(P)),
       'median_gap_rs': round(float((P.mid - P.fair).abs().median()), 3),
       'median_gap_vol': round(float(((P.iv - P.iv_fit) * 100).abs().median()), 3),
       'median_gap_ctrl_rs': round(float((P.mid - P.fair_ctrl).abs().dropna().median()), 3),
       'spread_model': spread_table, 'variants': []}
for name, panel, lots, fair, mult, mn, vl, slip in V:
    T = run(panel, lots, fair, mult, mn, vl, slip)
    T.to_csv(os.path.join(bt.OUT, 'trades_' + ''.join(c if c.isalnum() else '_' for c in name).strip('_') + '.csv'), index=False)
    a, b = stats(T, lots, True), stats(T, lots, False)
    res['variants'].append({'name': name, 'lots': lots, 'with_charges': a, 'before_charges': b})
    print(f"{name:45s} trades {a['trades']:5d}  with charges {a['month']:>10,.0f} ({a['days_positive']})  "
          f"before charges {b['month']:>10,.0f} ({b['days_positive']})  P(loss) {a.get('p_loss')}", flush=True)
json.dump(res, open(os.path.join(bt.OUT, 'long_summary.json'), 'w'), indent=1)

# ------------------------------------------------------------------------------------------------ HTML report
rs = lambda x, d=0: '–' if x is None else ('−' if x < 0 else '') + '₹' + f'{abs(x):,.{d}f}'
esc = html.escape
by = {v['name']: v for v in res['variants']}
main, fast = by['Wait 1 min, 1 lot'], by['No wait, 10 lots']


def bars(key):
    items = [(v['name'], v[key]['month']) for v in res['variants']]
    W, rowh, left, right = 820, 28, 300, 90
    vals = [x for _, x in items]; lo, hi = min(0, min(vals)) * 1.12, max(0, max(vals)) * 1.12 or 1
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    H = rowh * len(items) + 36; o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Profit or loss over the sampled days, by variant">']
    step = 10 ** math.floor(math.log10(max(hi - lo, 1) / 4)); step *= 5 if (hi - lo) / step > 20 else 2 if (hi - lo) / step > 8 else 1
    for t in np.arange(math.ceil(lo / step) * step, hi + 1e-9, step):
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="{H - 24}" class="grid"/><text x="{x(t):.1f}" y="{H - 6}" class="tick" text-anchor="middle">{('−' if t < 0 else '') + '₹' + format(abs(t) / 1e5, '.0f')}L</text>')
    o.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="4" y2="{H - 24}" class="zero"/>')
    for i, (n, v) in enumerate(items):
        y = 10 + i * rowh; a, b = sorted([x(0), x(v)])
        o.append(f'<text x="{left - 8}" y="{y + 14}" class="lab" text-anchor="end">{esc(n)}</text>'
                 f'<rect x="{a:.1f}" y="{y + 3}" width="{max(b - a, 1):.1f}" height="15" rx="3" class="{"pos" if v >= 0 else "neg"}"><title>{esc(n)}: {rs(v)}</title></rect>'
                 f'<text x="{(b + 6) if v >= 0 else (x(0) + 6):.1f}" y="{y + 15}" class="val">{rs(v)}</text>')
    return ''.join(o) + '</svg>'


def vrow(v, key):
    s = v[key]
    return (f'<tr><td>{esc(v["name"])}</td><td>{s["trades"]:,}</td><td>{s.get("win_pct", "–")}%</td><td class="b">{rs(s["month"])}</td>'
            f'<td>{s["days_positive"]}</td><td>{rs(s.get("worst_day"))}</td><td>{rs(s.get("max_drawdown"))}</td>'
            f'<td>{rs(s.get("boot_5"))} to {rs(s.get("boot_95"))}</td><td>{s.get("p_loss", "–")}</td><td>{rs(s.get("without_top10"))}</td></tr>')


head = ('<tr><th>Variant</th><th>Trades</th><th>Win rate</th><th>Total P&amp;L</th><th>Days positive</th><th>Worst day</th>'
        '<th>Max drawdown</th><th>90% range (resampling days)</th><th>Chance of a losing total</th><th>Without best 10 trades</th></tr>')
daily_rows = ''.join(f'<tr><td>{d}</td><td>{rs(main["with_charges"]["daily"][d])}</td><td>{rs(fast["with_charges"]["daily"][d])}</td>'
                     f'<td>{rs(fast["before_charges"]["daily"][d])}</td></tr>' for d in DAYS)
F, M = fast['with_charges'], main['with_charges']
css = '''<style>:root{--bg:#f7f6f3;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e4e3df;--line:#d9d7d1;--pos:#2a78d6;--neg:#c2410c;--note:#fff4e5;--note-ink:#7a4100}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121212;--panel:#1b1b1a;--ink:#f2f1ee;--ink2:#bdbbb5;--muted:#8f8d87;--grid:#2c2b29;--line:#3a3936;--pos:#3987e5;--neg:#e0703f;--note:#3a2a12;--note-ink:#ffd7a1}}
:root[data-theme="dark"]{--bg:#121212;--panel:#1b1b1a;--ink:#f2f1ee;--ink2:#bdbbb5;--muted:#8f8d87;--grid:#2c2b29;--line:#3a3936;--pos:#3987e5;--neg:#e0703f;--note:#3a2a12;--note-ink:#ffd7a1}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0 0 10px}.sub{color:var(--ink2);margin:0 0 14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}.verdict{background:var(--note);color:var(--note-ink);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.tile{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.tile b{display:block;font-size:20px}.tile span{color:var(--ink2);font-size:12px}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}th{color:var(--ink2);font-weight:600}
td:first-child,th:first-child{text-align:left}td.b{font-weight:600}ul{margin:6px 0 0;padding-left:20px}li{margin:4px 0}.note{color:var(--ink2);font-size:13px}
svg .grid{stroke:var(--grid)}svg .zero{stroke:var(--ink2);stroke-width:1.5}svg .pos{fill:var(--pos)}svg .neg{fill:var(--neg)}svg text{fill:var(--ink2);font-size:12px}svg .val{fill:var(--ink);font-weight:600}footer{color:var(--muted);font-size:12px}</style>'''
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PNmix Two-Year Backtest</title>{css}</head><body><main>
<h1>PNmix backtest: NIFTY options, two years ({DAYS[0]} to {DAYS[-1]})</h1>
<p class="sub">{len(DAYS)} sampled sessions (every 4th trading day) of Upstox 1-minute candles · {res["minutes_fitted"]:,} minutes fitted · {res["quotes"]:,} option prices · NIFTY weekly expiries</p>
<div class="verdict"><b>Realistic (act one minute after the gap, 1 lot):</b> {rs(M["month"])} over the {len(DAYS)} sampled days, {M["days_positive"]} days positive.<br>
<b>No wait, 10 lots:</b> {rs(F["month"])} over the {len(DAYS)} sampled days after all charges ({F["days_positive"]} days positive; 90% range {rs(F["boot_5"])} to {rs(F["boot_95"])}),
{rs(fast["before_charges"]["month"])} before charges. Charges for that variant: {rs(F["charges"])}.</div>
<section class="card"><h2>P&amp;L over the sampled days after all charges, every variant</h2>{bars("with_charges")}</section>
<section class="card"><h2>After all charges</h2><div class="tw"><table><thead>{head}</thead><tbody>{"".join(vrow(v, "with_charges") for v in res["variants"])}</tbody></table></div></section>
<section class="card"><h2>Before charges (spread still paid)</h2><div class="tw"><table><thead>{head}</thead><tbody>{"".join(vrow(v, "before_charges") for v in res["variants"])}</tbody></table></div></section>
<section class="card"><h2>Day by day</h2><div class="tw"><table style="max-width:640px"><thead><tr><th>Day</th><th>Wait 1 min, 1 lot</th><th>No wait, 10 lots</th><th>No wait, 10 lots, before charges</th></tr></thead><tbody>{daily_rows}</tbody></table></div></section>
<section class="card"><h2>How close is PNmix to the market?</h2><p class="note">Median distance between the traded price and PNmix fair value: {rs(res["median_gap_rs"], 3)}
({res["median_gap_vol"]:.3f} vol points). Pricing each strike from its two neighbours instead: {rs(res["median_gap_ctrl_rs"], 3)}.</p></section>
<section class="card"><h2>How it was run</h2><ul class="note">
<li>Every minute: last traded price of every NIFTY option within about ±6% of spot (candles with volume, at most 2 minutes old), forward from put-call parity, nearest two expiries with 2+ hours left, PNmix fit. Same pipeline as PNmix Live.</li>
<li>Signal: the gap beats STT, exchange, SEBI, stamp duty and GST at the modelled bid/ask; for 10 lots it must also cover the ₹{FIX:.2f} brokerage (₹{FIX / 10:.2f} per lot).</li>
<li>"Wait 1 min": filled at the next minute's opening trade ± half-spread. "No wait": filled at the price that showed the gap ± half-spread, which needs an automated order within seconds; 10-lot runs also need 10+ lots traded in the next minute.</li>
<li>Exit when the mid crosses the current fair value, after 60 minutes, at 15:25, or after 10 minutes without trades; exit pays the spread again.</li>
<li>Spread model (no historical bid/ask in candles): median half-spread by price level from a real Upstox option-chain snapshot; tested at 0.5×, 1×, 1.5× and with 1–2 extra ticks per fill.</li>
<li>The 90% range and chance of a losing total come from resampling whole days 5,000 times, because trades on the same day are related.</li></ul></section>
<section class="card"><h2>Limits</h2><ul class="note"><li>Every 4th trading day only (about 125 of about 495 sessions); results scale roughly 4× for all days. Lot size changed 25→75→65 over the period and per-lot figures use the lot in force; STT and other charges use the October 2026 rates throughout. The spread model comes from one October 2026 snapshot.</li><li>Last-traded prices bounce between bid and ask, which flatters "no wait" results.
The live track record, which uses real bid/ask quotes, is the better test.</li><li>Order-book depth for 10 lots is not known; the slippage rows show how fast the result shrinks.</li></ul></section>
<footer>Research only, not investment advice. Hypothetical results from historical data.</footer></main></body></html>'''
open(os.path.join(bt.OUT, 'long_report.html'), 'w').write(page)
print('wrote', os.path.join(bt.OUT, 'long_report.html'))
