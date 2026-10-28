"""Fit the 431 pre-test days (< 2021-10-01) with fold-0's prior pseudo-quotes, same frozen config, for the
Table 6 training-window column. Chunked + resumable."""
import sys,os,pickle,time; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from multiprocessing import Pool
import run_ab_container as RA
if __name__=='__main__':
    p=RA.p; ps=pd.read_csv('pseudo_pre/pre.csv',parse_dates=['date','expiry'])
    days=sorted(p[p.date<'2021-10-01'].date.unique()); os.makedirs('pre',exist_ok=True)
    for c0 in range(0,len(days),60):
        out=f'pre/chunk{c0:03d}.pkl'
        if os.path.exists(out): continue
        t0=time.time(); jobs=[(d,p[p.date==d],ps[ps.date==d]) for d in days[c0:c0+60]]
        with Pool(2) as pool: res=pool.map(RA.job,jobs,chunksize=2)
        pickle.dump(dict(preds=pd.concat([r[1] for r in res if r[1] is not None]),surf={r[0]:r[2] for r in res}),open(out,'wb'))
        print(f'pre chunk {c0}: {time.time()-t0:.0f}s',flush=True)
    print('PRE_DONE')
