"""Pipeline driver: as the download completes expiries, build the per-day files and fit the sampled days (every STEP-th
trading day). Loops until every sampled day is fitted. Safe to restart."""
import os, sys, time, json, datetime as dt
import pandas as pd
os.environ.setdefault('BT_DATA', 'data_long')
import prep_days, backtest as bt
from multiprocessing import Pool
STEP = int(os.environ.get('BT_STEP', 4)); OFF = int(os.environ.get('BT_OFFSET', 0))
D = bt.DATA; meta = json.load(open(os.path.join(D, 'Nifty50_meta.json')))
idx = pd.read_csv(os.path.join(D, 'Nifty50_index.csv.gz'), usecols=['ts']); days = sorted(idx.ts.str[:10].unique())[OFF::STEP]
os.makedirs(bt.PANEL, exist_ok=True)
fetch_done = lambda: 'DONE' in open(os.path.join(bt.HERE, 'fetch_long.out')).read()
def started():
    return sorted({f[:10] for f in os.listdir(os.path.join(D, 'raw')) if f.endswith('.csv')})
while True:
    st = started(); fin = fetch_done(); exps = sorted(meta['expiries'])
    ready = []
    for d in days:
        if os.path.exists(os.path.join(bt.PANEL, d + '.pkl')): continue
        nxt = [e for e in exps if e >= d][:3]                       # two nearest expiries plus the one after (guard)
        if len(nxt) < 2: continue
        if fin or (len(nxt) == 3 and nxt[2] in st and nxt[1] in st):    # later expiries already started => these are complete
            ready.append(d)
    if ready:
        batch = ready[:8]
        prep_days.build(batch)
        print(time.strftime('%H:%M:%S'), 'fitting', batch, flush=True)
        with Pool(2) as pool:
            for out in pool.imap_unordered(bt.fit_day, batch): print('done', out, flush=True)
    elif fin and all(os.path.exists(os.path.join(bt.PANEL, d + '.pkl')) for d in days if len([e for e in exps if e >= d]) >= 2):
        print('ALL FITTED'); break
    else:
        time.sleep(120)
