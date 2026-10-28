"""What if the trade is taken at the signal price itself (no 1-minute wait) and with 10 lots?"""
import os, json, math
import numpy as np, pandas as pd
import backtest as bt
from pnmix_live import charges
P = pd.concat([pd.read_pickle(os.path.join(bt.PANEL, f)) for f in sorted(os.listdir(bt.PANEL))])
h, _ = bt.spread_model()
FIX = charges.fixed_round_trip()
S = P.copy(); S['next_open'] = S.mid          # fill at the signal minute's own price (no wait)
# liquidity proxy for the no-wait case: lots traded in the following minute
def run(name, panel, lots, mult=1.0, min_net=0.0, vol_lots=1, slip=0):
    bt.SLIP_TICKS = slip
    T = bt.simulate(panel, 'fair', mult, h=h, min_net=min_net, min_vol_lots=vol_lots)
    bt.SLIP_TICKS = 0
    if len(T) == 0:
        return {'variant': name, 'trades': 0}
    T['pnl'] = lots * T.net_per_lot - FIX                      # brokerage is per order, not per lot
    T['t_in'] = T.day + T.j_in.astype(str).str.zfill(3); T['t_out'] = T.day + T.j_out.astype(str).str.zfill(3)
    # most positions open at the same time
    ev = pd.concat([pd.DataFrame({'t': T.t_in, 'd': 1}), pd.DataFrame({'t': T.t_out, 'd': -1})]).sort_values(['t', 'd'])
    conc = int(ev.d.cumsum().max())
    byday = T.groupby('day').pnl.sum(); cum = T.sort_values('t_out').pnl.cumsum()
    dd = float((cum - cum.cummax()).min())
    return {'variant': name, 'trades': len(T), 'lots': lots, 'win_pct': round(100 * (T.pnl > 0).mean(), 1),
            'avg_pnl_per_trade': round(T.pnl.mean(), 0), 'week_pnl': round(T.pnl.sum(), 0),
            'best_day': round(byday.max(), 0), 'worst_day': round(byday.min(), 0), 'days_positive': f'{(byday > 0).sum()}/{len(byday)}',
            'max_drawdown': round(dd, 0), 'max_open_at_once': conc,
            'stt_week': round((lots * T.stt_per_lot).sum(), 0), 'charges_week': round((lots * T.costs_per_lot).sum() + len(T) * FIX, 0),
            'by_day': byday.round(0).to_dict()}
out = []
out.append(run('Wait 1 min, 1 lot (earlier main run)', P, 1))
out.append(run('Wait 1 min, 10 lots', P, 10, min_net=FIX / 10))
out.append(run('No wait, 1 lot', S, 1))
out.append(run('No wait, 10 lots, same signals', S, 10, vol_lots=10))
out.append(run('No wait, 10 lots, only gaps worth > Rs 4.72/lot', S, 10, min_net=FIX / 10, vol_lots=10))
out.append(run('  same, +1 tick slippage each fill', S, 10, min_net=FIX / 10, vol_lots=10, slip=1))
out.append(run('  same, +2 ticks slippage each fill', S, 10, min_net=FIX / 10, vol_lots=10, slip=2))
out.append(run('  same, spread x0.5', S, 10, mult=0.5, min_net=FIX / 10, vol_lots=10))
out.append(run('  same, spread x1.5', S, 10, mult=1.5, min_net=FIX / 10, vol_lots=10))
for o in out:
    print({k: v for k, v in o.items() if k != 'by_day'})
for o in out[3:5]:
    print(o['variant'], o.get('by_day'))
json.dump(out, open(os.path.join(bt.OUT, 'ten_lots.json'), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
