"""Build one options file per trading day (days/YYYY-MM-DD.csv.gz) from the raw per-contract files of fetch_long.py.
Usage: python prep_days.py [day ...]   (no days = all)"""
import gzip, os, sys, collections, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, os.environ.get('BT_DATA', 'data_long'))
RAW = os.path.join(D, 'raw'); OUT = os.path.join(D, 'days'); os.makedirs(OUT, exist_ok=True)

def build(days):
    want = set(days) if days else None
    lo = min(want) if want else '0000'; hi = (dt.date.fromisoformat(max(want)) + dt.timedelta(days=21)).isoformat() if want else '9999'
    by = collections.defaultdict(list)
    for f in sorted(os.listdir(RAW)):
        if not f.endswith('.csv') or not (lo <= f[:10] <= hi): continue
        for line in open(os.path.join(RAW, f)):
            d = line.split(',')[3][:10]
            if want is None or d in want: by[d].append(line)
    for day, ls in sorted(by.items()):
        tmp = os.path.join(OUT, day + '.tmp')
        with gzip.open(tmp, 'wt') as fh:
            fh.write('expiry,strike,type,ts,open,high,low,close,volume,oi\n'); fh.writelines(ls)
        os.replace(tmp, os.path.join(OUT, day + '.csv.gz'))
    return len(by)

if __name__ == '__main__':
    print(build(sys.argv[1:]), 'days')
