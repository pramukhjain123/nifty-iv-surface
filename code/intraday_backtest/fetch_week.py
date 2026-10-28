"""Download 1-minute candles for a PNmix backtest: the index plus every option within +-8% of spot for the expiries that
were nearest during the window (live ones via /v3/historical-candle, expired ones via the expired-instruments API).
Also saves the current option chain (bid/ask) to calibrate the spread model. Token is read from ../.env."""
import csv, gzip, json, os, sys, time, urllib.parse
import httpx
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, os.environ.get('BT_DATA', 'data'))
INDEX = sys.argv[1] if len(sys.argv) > 1 else 'NSE_INDEX|Nifty 50'
FROM, TO = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('2026-09-25', '2026-10-01')
N_EXP = int(os.environ.get('N_EXP', 3))
BUDGET = float(os.environ.get('FETCH_BUDGET', 150))   # seconds per run; rerun until it prints DONE (resumes)
T0 = time.time()
tok = [l.split('=', 1)[1].strip().strip('"\'') for l in open(os.path.join(HERE, '..', '.env')) if l.startswith('UPSTOX_ANALYTICS_TOKEN=')][0]
c = httpx.Client(headers={'Authorization': f'Bearer {tok}', 'Accept': 'application/json'}, timeout=30)
q = lambda s: urllib.parse.quote(s, safe='')
tag = INDEX.split('|')[1].replace(' ', '')
last = [0.0]
def get(url, **params):
    for attempt in range(6):
        wait = 0.25 - (time.time() - last[0])
        if wait > 0: time.sleep(wait)
        last[0] = time.time(); r = c.get(url, params=params or None)
        if r.status_code == 429: time.sleep(5 * (attempt + 1)); continue
        if r.status_code >= 500: time.sleep(2); continue
        return r
    return r
log = open(os.path.join(OUT, f'{tag}_fetch.log'), 'a')
def say(*a):
    print(*a, file=log, flush=True)
# index candles
r = get(f'https://api.upstox.com/v3/historical-candle/{q(INDEX)}/minutes/1/{TO}/{FROM}')
idx = r.json()['data']['candles']; say('index candles', len(idx))
with gzip.open(os.path.join(OUT, f'{tag}_index.csv.gz'), 'wt', newline='') as fh:
    w = csv.writer(fh); w.writerow(['ts', 'open', 'high', 'low', 'close'])
    for x in sorted(idx): w.writerow(x[:5])
lo, hi = min(x[3] for x in idx) * 0.92, max(x[2] for x in idx) * 1.08
# contracts: expired ones that expire on/after FROM, then live ones
live = get('https://api.upstox.com/v2/option/contract', instrument_key=INDEX).json()['data']
expired_exps = [e for e in get('https://api.upstox.com/v2/expired-instruments/expiries', instrument_key=INDEX).json()['data'] if e >= FROM]
cons = []
for e in expired_exps:
    d = get('https://api.upstox.com/v2/expired-instruments/option/contract', instrument_key=INDEX, expiry_date=e).json().get('data') or []
    for x in d: x['_expired'] = True
    cons += d; say('expired expiry', e, len(d))
live_exps = sorted({x['expiry'] for x in live})
need = sorted(set(expired_exps) | set(live_exps))[:N_EXP]
cons += [x for x in live if x['expiry'] in need]
cons = [x for x in cons if x['expiry'] in need and lo <= x['strike_price'] <= hi and x['instrument_type'] in ('CE', 'PE')]
say('expiries', need, 'contracts', len(cons))
meta = {'index': INDEX, 'from': FROM, 'to': TO, 'expiries': need, 'lot_size': {x['expiry']: x['lot_size'] for x in cons}}
json.dump(meta, open(os.path.join(OUT, f'{tag}_meta.json'), 'w'))
RAW = os.path.join(OUT, 'raw_' + tag); os.makedirs(RAW, exist_ok=True)
todo = []
for x in cons:
    f = os.path.join(RAW, f"{x['expiry']}_{x['strike_price']:g}_{x['instrument_type']}.csv")
    if not os.path.exists(f): todo.append((x, f))
say('to fetch', len(todo), 'of', len(cons))
for x, f in todo:
    if time.time() - T0 > BUDGET:
        print('PARTIAL', len(todo), 'left at start; run again'); sys.exit(0)
    if x.get('_expired'):
        url = f'https://api.upstox.com/v2/expired-instruments/historical-candle/{q(x["instrument_key"])}/1minute/{TO}/{FROM}'
    else:
        url = f'https://api.upstox.com/v3/historical-candle/{q(x["instrument_key"])}/minutes/1/{TO}/{FROM}'
    r = get(url)
    if r.status_code != 200:
        say('FAIL', x['trading_symbol'], r.status_code, r.text[:150]); continue
    cs = (r.json().get('data') or {}).get('candles') or []
    with open(f + '.tmp', 'w', newline='') as fh:
        w = csv.writer(fh)
        for cnd in sorted(cs): w.writerow([x['expiry'], x['strike_price'], x['instrument_type'], *cnd[:7]])
    os.replace(f + '.tmp', f)
with gzip.open(os.path.join(OUT, f'{tag}_options.csv.gz'), 'wt', newline='') as fh:
    fh.write('expiry,strike,type,ts,open,high,low,close,volume,oi\n')
    for x in cons:
        f = os.path.join(RAW, f"{x['expiry']}_{x['strike_price']:g}_{x['instrument_type']}.csv")
        if os.path.exists(f): fh.write(open(f).read())
# current chain snapshot (bid/ask) for the spread model
snap = []
for e in [x for x in live_exps[:2]]:
    for row in get('https://api.upstox.com/v2/option/chain', instrument_key=INDEX, expiry_date=e).json().get('data') or []:
        for t, side in (('CE', 'call_options'), ('PE', 'put_options')):
            md = (row.get(side) or {}).get('market_data') or {}
            snap.append({'expiry': e, 'strike': row['strike_price'], 'type': t, 'spot': row.get('underlying_spot_price'),
                         'ltp': md.get('ltp'), 'bid': md.get('bid_price'), 'ask': md.get('ask_price'),
                         'bid_qty': md.get('bid_qty'), 'ask_qty': md.get('ask_qty'), 'volume': md.get('volume')})
json.dump(snap, open(os.path.join(OUT, f'{tag}_chain_snapshot.json'), 'w'))
say('DONE', len(snap), 'snapshot rows'); print('DONE')
