"""Long-history downloader (about 2 years of NIFTY options, 1-minute candles) for the PNmix backtest.

For every weekly expiry E it fetches, for the 14 days before E (the days E is one of the two nearest expiries), every
option within +-6% of the index range of that window. Resumable: rerun until it prints DONE. Token read from ../.env.
Env: BT_DATA (output folder, default data_long), START (default 2024-10-03), END (default 2026-10-02), RATE (req/s)."""
import csv, gzip, json, os, sys, threading, time, urllib.parse, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import httpx

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, os.environ.get('BT_DATA', 'data_long'))
os.makedirs(OUT, exist_ok=True)
INDEX = 'NSE_INDEX|Nifty 50'; TAG = 'Nifty50'
START, END = os.environ.get('START', '2024-10-03'), os.environ.get('END', '2026-10-02')
RATE = float(os.environ.get('RATE', 5)); BAND = 0.06; LEAD = 14
tok = [l.split('=', 1)[1].strip().strip('"\'') for l in open(os.path.join(HERE, '..', '.env')) if l.startswith('UPSTOX_ANALYTICS_TOKEN=')][0]
c = httpx.Client(headers={'Authorization': f'Bearer {tok}', 'Accept': 'application/json'}, timeout=40,
                 limits=httpx.Limits(max_connections=8))
q = lambda s: urllib.parse.quote(s, safe='')
lock = threading.Lock(); last = [0.0]
log = open(os.path.join(OUT, 'fetch.log'), 'a')
def say(*a):
    print(time.strftime('%H:%M:%S'), *a, file=log, flush=True)
def get(url, **params):
    r = None
    for attempt in range(40):
        with lock:
            wait = 1.0 / RATE - (time.time() - last[0])
            if wait > 0: time.sleep(wait)
            last[0] = time.time()
        try:
            r = c.get(url, params=params or None)
        except Exception as e:
            say('EXC', repr(e)[:80]); time.sleep(3); continue
        if r.status_code == 429: say('429'); time.sleep(30); continue
        if r.status_code >= 500: time.sleep(2); continue
        return r
    return r
D = lambda s: dt.date.fromisoformat(s)

# ---- index candles, month by month
ipath = os.path.join(OUT, f'{TAG}_index.csv.gz')
if not os.path.exists(ipath):
    rows = []; a = D(START)
    while a <= D(END):
        b = min(a + dt.timedelta(days=27), D(END))
        r = get(f'https://api.upstox.com/v3/historical-candle/{q(INDEX)}/minutes/1/{b}/{a}')
        rows += r.json()['data']['candles']; say('index', a, b, len(rows)); a = b + dt.timedelta(days=1)
    with gzip.open(ipath, 'wt', newline='') as fh:
        w = csv.writer(fh); w.writerow(['ts', 'open', 'high', 'low', 'close'])
        for x in sorted({tuple(x[:5]) for x in rows}): w.writerow(x)
import pandas as pd
idx = pd.read_csv(ipath); idx['day'] = idx.ts.str[:10]
dl = idx.groupby('day').agg(lo=('low', 'min'), hi=('high', 'max'))

# ---- expiries and contracts
cpath = os.path.join(OUT, 'contracts.json')
if os.path.exists(cpath):
    cons = json.load(open(cpath))
else:
    exps = [e for e in get('https://api.upstox.com/v2/expired-instruments/expiries', instrument_key=INDEX).json()['data'] if e >= START]
    cons = []
    for e in exps:
        d = get('https://api.upstox.com/v2/expired-instruments/option/contract', instrument_key=INDEX, expiry_date=e).json().get('data') or []
        for x in d: x['_expired'] = True
        cons += d; say('expired expiry', e, len(d))
    live = get('https://api.upstox.com/v2/option/contract', instrument_key=INDEX).json()['data']
    cons += live; say('live contracts', len(live))
    cons = [x for x in cons if x['instrument_type'] in ('CE', 'PE')]
    json.dump(cons, open(cpath, 'w'))
all_exp = sorted({x['expiry'] for x in cons})
RAW = os.path.join(OUT, 'raw'); os.makedirs(RAW, exist_ok=True)
todo = []; meta_exp = {}
for E in all_exp:
    a = max(D(E) - dt.timedelta(days=LEAD), D(START)); b = min(D(E), D(END))
    w = dl[(dl.index >= str(a)) & (dl.index <= str(b))]
    if w.empty: continue
    lo, hi = w.lo.min() * (1 - BAND), w.hi.max() * (1 + BAND)
    meta_exp[E] = {'from': str(a), 'to': str(b)}
    for x in cons:
        k_ = x['strike_price']
        mid_lo, mid_hi = w.lo.min() * 0.97, w.hi.max() * 1.03          # all strikes near the money, every 100 further out
        if x['expiry'] == E and lo <= k_ <= hi and (mid_lo <= k_ <= mid_hi or k_ % 100 == 0):
            f = os.path.join(RAW, f"{E}_{x['strike_price']:g}_{x['instrument_type']}.csv")
            if not os.path.exists(f): todo.append((x, f, str(a), str(b)))
meta = {'index': INDEX, 'from': START, 'to': END, 'expiries': meta_exp,
        'lot_size': {E: next(x['lot_size'] for x in cons if x['expiry'] == E) for E in meta_exp}}
json.dump(meta, open(os.path.join(OUT, f'{TAG}_meta.json'), 'w'))
say('expiries', len(meta_exp), 'to fetch', len(todo))
done = [0]; T0 = time.time()
def one(job):
    x, f, a, b = job
    if x.get('_expired'):
        url = f'https://api.upstox.com/v2/expired-instruments/historical-candle/{q(x["instrument_key"])}/1minute/{b}/{a}'
    else:
        url = f'https://api.upstox.com/v3/historical-candle/{q(x["instrument_key"])}/minutes/1/{b}/{a}'
    r = get(url)
    if r is None or r.status_code != 200:
        say('FAIL', x['trading_symbol'], getattr(r, 'status_code', None), (r.text[:100] if r is not None else '')); return
    cs = (r.json().get('data') or {}).get('candles') or []
    with open(f + '.tmp', 'w', newline='') as fh:
        w = csv.writer(fh)
        for cnd in sorted(cs): w.writerow([x['expiry'], x['strike_price'], x['instrument_type'], *cnd[:7]])
    os.replace(f + '.tmp', f)
    done[0] += 1
    if done[0] % 200 == 0: say('fetched', done[0], 'of', len(todo), f'{time.time() - T0:.0f}s')
with ThreadPoolExecutor(4) as ex:
    list(ex.map(one, todo))
left = sum(1 for x, f, a, b in todo if not os.path.exists(f))
say('DONE' if left == 0 else f'PARTIAL {left} missing'); print('DONE' if left == 0 else f'PARTIAL {left}')
