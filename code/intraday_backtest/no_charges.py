"""Same backtest variants, (a) same trades with P&L before charges, (b) re-run as if there were no charges at all."""
import os, json
import numpy as np, pandas as pd
import backtest as bt
from pnmix_live import charges
P = pd.concat([pd.read_pickle(os.path.join(bt.PANEL, f)) for f in sorted(os.listdir(bt.PANEL))]); h, _ = bt.spread_model()
S = P.copy(); S['next_open'] = S.mid
FIX = charges.fixed_round_trip()
variants = [('Wait 1 min, 1 lot', P, 1, 0.0, 1, 0), ('Wait 1 min, 10 lots', P, 10, FIX / 10, 1, 0),
            ('No wait, 1 lot', S, 1, 0.0, 1, 0), ('No wait, 10 lots', S, 10, FIX / 10, 10, 0),
            ('No wait, 10 lots, +1 tick per fill', S, 10, FIX / 10, 10, 1), ('No wait, 10 lots, +2 ticks per fill', S, 10, FIX / 10, 10, 2)]
def summ(T, lots, with_charges):
    pnl = lots * (T.net_per_lot if with_charges else T.gross_per_lot) - (FIX if with_charges else 0)
    byday = pnl.groupby(T.day).sum()
    return dict(trades=len(T), win=round(100 * (pnl > 0).mean(), 1), week=round(pnl.sum()), days=f'{(byday > 0).sum()}/{len(byday)}',
                charges=round((lots * T.costs_per_lot).sum() + len(T) * FIX))
out = []
for name, panel, lots, mn, vl, slip in variants:
    bt.SLIP_TICKS = slip
    T = bt.simulate(panel, 'fair', 1.0, h=h, min_net=mn, min_vol_lots=vl)
    a = summ(T, lots, True); b = summ(T, lots, False)
    # (b) a world with zero charges: every gap that beats the spread is taken
    saved = (charges.BROKERAGE_PER_ORDER, charges.STT_SELL, charges.NSE_TXN, charges.SEBI, charges.STAMP_BUY)
    charges.BROKERAGE_PER_ORDER = charges.STT_SELL = charges.NSE_TXN = charges.SEBI = charges.STAMP_BUY = 0.0
    T0 = bt.simulate(panel, 'fair', 1.0, h=h, min_net=0.0, min_vol_lots=vl)
    charges.BROKERAGE_PER_ORDER, charges.STT_SELL, charges.NSE_TXN, charges.SEBI, charges.STAMP_BUY = saved
    c = summ(T0, lots, False); bt.SLIP_TICKS = 0
    out.append(dict(variant=name, with_charges=a, same_trades_no_charges=b, zero_charge_world=c))
    print(name, '| with', a, '| same trades, no charges', b, '| zero-charge world', c, flush=True)
json.dump(out, open(os.path.join(bt.OUT, 'no_charges.json'), 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
