"""PNmix minute-by-minute backtest on Upstox 1-minute candles (data from fetch_week.py).

Step 1 (fit):    for every minute, take each option's price from a candle that actually traded in that minute or the
                 one before, build the same clean OTM quotes as PNmix Live (forward from put-call parity, nearest two
                 expiries with 2+ hours left), fit PNmix and store the fair value of every strike. A control "fair"
                 is stored too: each strike's IV interpolated from its two neighbours (no model at all).
Step 2 (trade):  the live track-record rules, with no look-ahead:
                 * signal at the end of minute t when the gap beats STT and all variable charges at the modelled
                   bid/ask, and the option traded at least one lot in minute t+1;
                 * fill at the OPEN of minute t+1, minus/plus the modelled half-spread (SELL at the bid, BUY at the ask);
                 * close when the mid crosses the current fair, after 60 minutes, at 15:25, or if the strike stops
                   trading for 10 minutes; exit filled at the next minute's open on the other side of the spread;
                 * no new trades in an expiry with under 3 hours left; 5-minute cooldown per strike.
Candles have no bid/ask, so the spread is modelled from a real Upstox option-chain snapshot (median half-spread by
price level) and the results are shown at 0.5x, 1x and 1.5x that spread.

Usage:  python backtest.py fit      (slow: ~1 fit per minute per day, uses 2 processes)
        python backtest.py trade    (fast: runs every variant and writes results/)
"""
import gzip
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import pnmix
from pnmix.bhavcopy import _black, _implied_vol
from pnmix.rates import repo_rate
from scipy.stats import norm

from pnmix_live import charges, engine  # the live package (pip install -e .. from the pnmix_live folder)

TAG = os.environ.get('BT_TAG', 'Nifty50')
SUFFIX = '' if os.environ.get('BT_FRESH', '2') == '2' else f"_fresh{os.environ['BT_FRESH']}"
DATA = os.path.join(HERE, os.environ.get('BT_DATA', 'data')); OUT = os.path.join(HERE, os.environ.get('BT_OUT', 'results') + SUFFIX); PANEL = os.path.join(DATA, f'{TAG}_panel' + SUFFIX)
FRESH = int(os.environ.get('BT_FRESH', 2))   # a price counts if its candle traded in this minute (1) or also the one before (2)
TICK = 0.05
FIRST_SIGNAL, LAST_SIGNAL = 5, 369         # 09:20 ... 15:24 (minute index from 09:15)
HOLD, COOLDOWN, STALE_EXIT, MIN_ENTRY_HOURS = 60, 5, 10, 3.0


def load(day=None):
    meta = json.load(open(os.path.join(DATA, f'{TAG}_meta.json')))
    idx = pd.read_csv(os.path.join(DATA, f'{TAG}_index.csv.gz'))
    if day is not None and os.path.isdir(os.path.join(DATA, 'days')):        # long history: one options file per day
        opt = pd.read_csv(os.path.join(DATA, 'days', f'{day}.csv.gz'))
        idx = idx[idx.ts.str[:10] == day].copy()
    else:
        opt = pd.read_csv(os.path.join(DATA, f'{TAG}_options.csv.gz'))
    for d in (idx, opt):
        d['day'] = d.ts.str[:10]; d['j'] = (d.ts.str[11:13].astype(int) - 9) * 60 + d.ts.str[14:16].astype(int) - 15
    opt = opt[(opt.j >= 0) & (opt.j < 375)]
    return meta, idx, opt


# ------------------------------------------------------------------------------------------- step 1: fit every minute
def fit_day(day):
    meta, idx, opt = load(day)
    ix = idx[idx.day == day].set_index('j').sort_index()
    o = opt[opt.day == day]
    r = repo_rate(pd.Timestamp(day).date())
    keys = sorted(set(zip(o.expiry, o.strike, o.type)))
    col = {k: i for i, k in enumerate(keys)}
    n = 375; C = np.full((n, len(keys)), np.nan); O = C.copy(); V = np.zeros_like(C)
    ci = np.array([col[k] for k in zip(o.expiry, o.strike, o.type)])
    C[o.j.values, ci] = o.close.values; O[o.j.values, ci] = o.open.values; V[o.j.values, ci] = o.volume.values
    traded = V > 0
    fresh = np.full_like(C, np.nan)                      # last traded close within FRESH minutes
    for lag in range(FRESH - 1, -1, -1):
        sh = np.full_like(C, np.nan); sh[lag:] = np.where(traded, C, np.nan)[:n - lag]
        fresh = np.where(np.isnan(sh), fresh, sh)
    nxt_open = np.full_like(C, np.nan); nxt_open[:-1] = np.where(traded, O, np.nan)[1:]
    nxt_vol = np.zeros_like(C); nxt_vol[:-1] = V[1:]
    S_close = ix.close.reindex(range(n)).ffill().values; S_next = ix.open.reindex(range(1, n + 1)).ffill().values
    exps = sorted({k[0] for k in keys})
    rows = []; t0 = time.time(); fails = 0
    for j in range(FIRST_SIGNAL, 375):
        now = pd.Timestamp(f'{day} 09:15', tz=engine.IST).to_pydatetime() + pd.Timedelta(minutes=j + 1)
        el = [e for e in exps if engine.time_to_expiry(e, now) * 365 * 24 >= engine.MIN_HOURS][:2]
        S = S_close[j]; frames = []
        for e in el:
            tau = engine.time_to_expiry(e, now)
            ks = sorted({k[1] for k in keys if k[0] == e})
            c_ = np.array([fresh[j, col[(e, K, 'CE')]] if (e, K, 'CE') in col else np.nan for K in ks])
            p_ = np.array([fresh[j, col[(e, K, 'PE')]] if (e, K, 'PE') in col else np.nan for K in ks])
            K_ = np.array(ks, float); both = ~np.isnan(c_) & ~np.isnan(p_)
            if both.sum() < 3:
                continue
            near = np.argsort(np.where(both, np.abs(K_ - S), np.inf))[:3]
            F = float(np.median(K_[near] + (c_[near] - p_[near]) * math.exp(r * tau)))
            k = np.log(K_ / F); call = k >= 0; mid = np.where(call, c_, p_)
            intr = np.where(call, np.maximum(F - K_, 0), np.maximum(K_ - F, 0)) * math.exp(-r * tau)
            ok = (k >= engine.K_RANGE[0]) & (k <= engine.K_RANGE[1]) & ~np.isnan(mid) & (mid > intr + 0.05)
            if ok.sum() < engine.MIN_STRIKES:
                continue
            q = pd.DataFrame({'strike': K_[ok], 'k': k[ok], 'opt_type': np.where(call[ok], 'CE', 'PE'), 'mid': mid[ok]})
            m = len(q)
            q['iv'] = _implied_vol(q.mid.values, np.full(m, F), q.strike.values, np.full(m, tau), np.full(m, r), call[ok])
            q = q.dropna(subset=['iv'])
            if len(q) < engine.MIN_STRIKES:
                continue
            q['tau'] = tau; q['fwd'] = F; q['r'] = r; q['expiry'] = pd.Timestamp(e); q['exp'] = e
            frames.append(q)
        if not frames:
            continue
        Q = pd.concat(frames, ignore_index=True); Q['date'] = pd.Timestamp(day)
        try:
            surf = pnmix.fit_quotes(Q)
        except Exception:
            fails += 1; continue
        for e, g in Q.groupby('exp'):
            tau = float(g.tau.iloc[0]); F = float(g.fwd.iloc[0]); kk = g.k.values; K = g.strike.values
            call = (g.opt_type == 'CE').values; m = len(g)
            ivf = surf.iv_k(kk, tau)
            fair = _black(np.full(m, F), K, np.full(m, tau), ivf, np.full(m, r), call)
            order = np.argsort(kk); ivc = np.full(m, np.nan)            # control: neighbours only
            ko, io = kk[order], g.iv.values[order]
            for a in range(1, m - 1):
                w = (ko[a] - ko[a - 1]) / (ko[a + 1] - ko[a - 1]); ivc[order[a]] = io[a - 1] * (1 - w) + io[a + 1] * w
            fair_c = _black(np.full(m, F), K, np.full(m, tau), np.where(np.isnan(ivc), 0.2, ivc), np.full(m, r), call)
            fair_c = np.where(np.isnan(ivc), np.nan, fair_c)
            d1 = (np.log(F / K) + 0.5 * ivf ** 2 * tau) / (ivf * math.sqrt(tau))
            delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1) * math.exp(-r * tau) * F / S
            cidx = [col[(e, Kx, t)] for Kx, t in zip(K, g.opt_type)]
            rows.append(pd.DataFrame({'day': day, 'j': j, 'expiry': e, 'strike': K, 'type': g.opt_type.values,
                                      'hours_left': tau * 365 * 24, 'k': kk, 'F': F, 'S': S, 'S_next': S_next[j],
                                      'mid': g.mid.values, 'iv': g.iv.values, 'iv_fit': ivf, 'fair': fair,
                                      'fair_ctrl': fair_c, 'delta': delta, 'next_open': nxt_open[j, cidx],
                                      'next_vol': nxt_vol[j, cidx], 'lot': meta['lot_size'][e]}))
        if j % 60 == 0:
            print(day, f'minute {j}/375', f'{time.time() - t0:.0f}s', flush=True)
    P = pd.concat(rows, ignore_index=True)
    P.to_pickle(os.path.join(PANEL, f'{day}.pkl'))
    return day, len(P), fails, round(time.time() - t0)


# ------------------------------------------------------------------------------------------- step 2: trade
def spread_model():
    s = pd.DataFrame(json.load(open(os.path.join(DATA, f'{TAG}_chain_snapshot.json'))))
    s = s[(s.bid > 0) & (s.ask > 0) & (s.ask >= s.bid)].copy()
    s['otm'] = np.where(s.type == 'CE', s.strike >= s.spot, s.strike <= s.spot)
    s = s[s.otm]; s['mid'] = (s.bid + s.ask) / 2; s['half'] = (s.ask - s.bid) / 2
    edges = [0, 2, 5, 10, 20, 50, 100, 200, 400, 1e9]
    g = s.groupby(pd.cut(s.mid, edges), observed=True).half.median()
    centers = np.array([math.sqrt(max(iv.left, 0.5) * min(iv.right, 800)) for iv in g.index]); vals = g.values
    table = [(str(iv), round(float(v), 3), int(c)) for iv, v, c in zip(g.index, vals, s.groupby(pd.cut(s.mid, edges), observed=True).size())]
    return (lambda p: np.maximum(np.interp(np.log(np.maximum(p, 0.05)), np.log(centers), vals), TICK / 2)), table


def _bidask(mid, half):
    return max(math.floor((mid - half) / TICK + 1e-9) * TICK, TICK), math.ceil((mid + half) / TICK - 1e-9) * TICK


SLIP_TICKS = 0          # extra ticks paid on every fill (e.g. for walking the book with a large order)


def simulate(P, fair_col='fair', mult=1.0, hold=HOLD, h=None, min_net=0.0, min_vol_lots=1):
    """min_net: net per lot the signal must promise; min_vol_lots: lots the option must trade in the fill minute."""
    P = P.dropna(subset=[fair_col]).sort_values(['day', 'j'])
    trades = []; open_ = {}; cool = {}
    for (day, j), g in P.groupby(['day', 'j'], sort=True):
        quotes = {(r.expiry, r.strike, r.type): r for r in g.itertuples(index=False)}
        for key, tr in list(open_.items()):
            q = quotes.get(key)
            if q is None:
                if j - tr['last_j'] >= STALE_EXIT or j >= LAST_SIGNAL:
                    _exit(tr, trades, open_, cool, key, tr['last_mid'], tr['last_S'], j, 'stopped trading', h, mult)
                continue
            tr['last_j'] = j; tr['last_mid'] = q.mid; tr['last_S'] = q.S_next
            fair = getattr(q, fair_col)
            why = ('gap closed' if (q.mid <= fair if tr['side'] == 'SELL' else q.mid >= fair) else
                   'time limit' if j + 1 - tr['j_in'] >= hold else 'end of day' if j >= LAST_SIGNAL else None)
            if why:
                px = q.next_open if not np.isnan(q.next_open) else q.mid
                _exit(tr, trades, open_, cool, key, px, q.S_next, j + 1, why, h, mult)
        if j > LAST_SIGNAL - 1:
            continue
        for key, q in quotes.items():
            if key in open_ or j - cool.get(key, -99) < COOLDOWN or q.hours_left < MIN_ENTRY_HOURS:
                continue
            half = h(q.mid) * mult; bid, ask = _bidask(q.mid, half)
            fair = getattr(q, fair_col)
            x = charges.trade_to_fair(bid, ask, fair, q.lot)
            if x['net_per_lot'] <= min_net or np.isnan(q.next_open) or q.next_vol < q.lot * min_vol_lots:
                continue
            hb, ha = _bidask(q.next_open, h(q.next_open) * mult)
            entry = hb - SLIP_TICKS * TICK if x['side'] == 'SELL' else ha + SLIP_TICKS * TICK
            open_[key] = dict(day=day, expiry=key[0], strike=key[1], type=key[2], side=x['side'], lot=q.lot,
                              j_sig=j, j_in=j + 1, entry=entry, mid_sig=q.mid, fair_sig=fair, gap_vol=(q.iv - q.iv_fit) * 100,
                              expected=x['net_per_lot'], S_in=q.S_next, delta_per_lot=q.delta * q.lot,
                              last_j=j, last_mid=q.mid, last_S=q.S_next, hours_left=q.hours_left)
        if j == LAST_SIGNAL:                                  # anything left at 15:24 closes at 15:25
            for key, tr in list(open_.items()):
                _exit(tr, trades, open_, cool, key, tr['last_mid'], tr['last_S'], j + 1, 'end of day', h, mult)
    for key, tr in list(open_.items()):
        _exit(tr, trades, open_, cool, key, tr['last_mid'], tr['last_S'], tr['last_j'], 'end of data', h, mult)
    return pd.DataFrame(trades)


def _exit(tr, trades, open_, cool, key, px, S, j_out, why, h, mult):
    hb, ha = _bidask(px, h(px) * mult)
    ex = ha + SLIP_TICKS * TICK if tr['side'] == 'SELL' else max(hb - SLIP_TICKS * TICK, 0.0)   # buy back at the ask / sell at the bid
    lot = tr['lot']
    if tr['side'] == 'SELL':
        gross = (tr['entry'] - ex) * lot; c = charges.variable_round_trip(ex, tr['entry'], lot)
    else:
        gross = (ex - tr['entry']) * lot; c = charges.variable_round_trip(tr['entry'], ex, lot)
    net = gross - c['total']; pos = 1 if tr['side'] == 'BUY' else -1
    trades.append({**{k: v for k, v in tr.items() if not k.startswith('last_')}, 'j_out': j_out, 'minutes': j_out - tr['j_in'],
                   'reason': why, 'exit': ex, 'S_out': S, 'gross_per_lot': gross, 'costs_per_lot': c['total'],
                   'stt_per_lot': c['stt'], 'net_per_lot': net,
                   'ex_index_per_lot': net - pos * tr['delta_per_lot'] * (S - tr['S_in'])})
    open_.pop(key, None); cool[key] = j_out


def stats(T):
    if T is None or len(T) == 0:
        return {'trades': 0}
    n = len(T); net = T.net_per_lot; byday = T.groupby('day').net_per_lot.sum()
    return {'trades': n, 'gap_closed_pct': round(100 * (T.reason == 'gap closed').mean(), 1),
            'median_min_to_close': float(T[T.reason == 'gap closed'].minutes.median()) if (T.reason == 'gap closed').any() else None,
            'win_pct': round(100 * (net > 0).mean(), 1), 'avg_expected': round(T.expected.mean(), 2),
            'avg_gross': round(T.gross_per_lot.mean(), 2), 'avg_costs': round(T.costs_per_lot.mean(), 2),
            'avg_net': round(net.mean(), 2), 'se_net': round(net.std(ddof=1) / math.sqrt(n), 2) if n > 1 else None,
            'avg_ex_index': round(T.ex_index_per_lot.mean(), 2), 'total_net_1lot_each': round(net.sum(), 0),
            'total_after_brokerage': round(net.sum() - n * charges.fixed_round_trip(), 0),
            'days_positive': f'{int((byday > 0).sum())}/{len(byday)}'}


def trade_all():
    os.makedirs(OUT, exist_ok=True)
    P = pd.concat([pd.read_pickle(os.path.join(PANEL, f)) for f in sorted(os.listdir(PANEL)) if f.endswith('.pkl')])
    h, table = spread_model()
    res = {'spread_model': table, 'minutes_fitted': int(P.groupby(['day', 'j']).ngroups), 'quotes': len(P), 'runs': {}}
    for fair_col, label in (('fair', 'PNmix'), ('fair_ctrl', 'Neighbours (no model)')):
        for mult in (0.5, 1.0, 1.5):
            T = simulate(P, fair_col, mult, h=h); name = f'{label} | spread x{mult}'
            res['runs'][name] = {'all': stats(T), 'by_day': {d: stats(g) for d, g in T.groupby('day')} if len(T) else {},
                                 'by_side': {s: stats(g) for s, g in T.groupby('side')} if len(T) else {},
                                 'by_reason': T.reason.value_counts().to_dict() if len(T) else {}}
            T.to_csv(os.path.join(OUT, f"trades_{'pnmix' if fair_col == 'fair' else 'neighbours'}_x{mult}.csv"), index=False)
            print(name, res['runs'][name]['all'], flush=True)
    S = P.copy(); S['next_open'] = S.mid; S['next_vol'] = 1e9      # benchmark: filled at the signal price itself
    for mult in (0.0, 1.0):
        T = simulate(S, 'fair', mult, h=h); name = f'PNmix | filled at the signal price (not achievable) | spread x{mult}'
        res['runs'][name] = {'all': stats(T), 'by_day': {d: stats(g) for d, g in T.groupby('day')}}
        print(name, stats(T), flush=True)
    for hold in (15, 30):
        T = simulate(P, 'fair', 1.0, hold=hold, h=h); res['runs'][f'PNmix | spread x1.0 | hold {hold} min'] = {'all': stats(T)}
        print('hold', hold, stats(T), flush=True)
    json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    return res


if __name__ == '__main__':
    if sys.argv[1:2] == ['fit']:
        os.makedirs(PANEL, exist_ok=True)
        if os.path.isdir(os.path.join(DATA, 'days')):
            alld = sorted(f[:-7] for f in os.listdir(os.path.join(DATA, 'days')) if f.endswith('.csv.gz'))
            step = int(os.environ.get('BT_STEP', 1)); off = int(os.environ.get('BT_OFFSET', 0))
            days = sys.argv[2:] or alld[off::step]
        else:
            days = sys.argv[2:] or sorted(load()[1].day.unique())
        days = [d for d in days if not os.path.exists(os.path.join(PANEL, f'{d}.pkl'))]
        with Pool(int(os.environ.get('BT_PROCS', 2))) as pool:
            for out in pool.imap_unordered(fit_day, days):
                print('done', out, flush=True)
    else:
        trade_all()
