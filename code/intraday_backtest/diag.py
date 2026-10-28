import os, pandas as pd, numpy as np
import backtest as bt
P = pd.concat([pd.read_pickle(os.path.join(bt.PANEL, f)) for f in sorted(os.listdir(bt.PANEL))])
h, _ = bt.spread_model()
T = pd.read_csv('results/trades_pnmix_x1.0.csv')
sgn = np.where(T.side == 'SELL', 1, -1)
T['gap_sig_rs'] = sgn * (T.mid_sig - T.fair_sig)
# what was left of the gap at fill: entry fill vs signal fair
T['mid_in'] = np.where(T.side == 'SELL', T.entry, T.entry)  # fill includes half spread
print(T[['gap_sig_rs', 'expected', 'gross_per_lot', 'net_per_lot', 'minutes']].describe().round(2))
print(T.groupby('reason').agg(n=('net_per_lot', 'size'), net=('net_per_lot', 'mean'), gross=('gross_per_lot', 'mean')).round(2))
print(T.sample(8, random_state=1)[['day', 'expiry', 'strike', 'type', 'side', 'mid_sig', 'fair_sig', 'entry', 'exit', 'minutes', 'reason', 'gross_per_lot', 'net_per_lot']].to_string())
# variants: fill at the signal minute's own price (impossible in practice: shows the cost of waiting one minute)
Q = P.copy(); Q['next_open'] = Q.mid; Q['next_vol'] = 1e9
for m in (0.0, 1.0):
    print('same-minute fill, spread x', m, bt.stats(bt.simulate(Q, 'fair', m, h=h)))
print('next-minute fill, spread x 0', bt.stats(bt.simulate(P, 'fair', 0.0, h=h)))
