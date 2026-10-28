"""Per fold: train the neural prior on VISIBLE strikes of dates < test_start, excluding every [30,60]-day slice
(so Task C bands are never seen), then emit pseudo-quotes for (a) every slice of the fold's test-window days
(Task A/B) and (b) every non-band slice of the fold's Task C quote days (c1 and c2)."""
import sys,time,os,glob; sys.path.insert(0,'.')
from prep import load
import prior, numpy as np, pandas as pd
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
p=load(); p['dte']=(p.tau*365).round().astype(int); p['in_band']=(p.dte>=30)&(p.dte<=60)
folds=pd.read_csv(f'{R}/05_splits/folds.csv',parse_dates=['test_start','test_end'])
Q=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/results/phase6_taskC_fold*_raw_d0_1000000000_m30.csv'))])
Q=Q[Q.method=='Fengler+PCHIP'][['fold','date']].drop_duplicates(); Q['date']=pd.to_datetime(Q.date)
os.makedirs('pseudo',exist_ok=True)
for f in folds.itertuples():
    out=f'pseudo/fold{f.fold}.csv'
    if os.path.exists(out): continue
    t0=time.time(); models=prior.train(p[(p.date<f.test_start)&(~p.in_band)])
    days_ab=set(p[(p.date>=f.test_start)&(p.date<f.test_end)].date.unique()); days_c=set(Q[Q.fold==f.fold].date)
    rows=[]
    for day in sorted(days_ab|days_c):
        gd=p[p.date==day]
        for e,g in gd.groupby('expiry'):
            if day not in days_ab and g.in_band.iloc[0]: continue
            v=g[~g.is_heldout]
            if len(v)<3: continue
            k,iv,kind=prior.pseudo_for_slice(models,g.tau.iloc[0],v.k.values,v.iv.values)
            if len(k): rows.append(pd.DataFrame({'fold':f.fold,'date':day,'expiry':e,'k':k,'iv':iv,'kind':kind}))
    pd.concat(rows).to_csv(out,index=False)
    print(f"fold {f.fold:2d}: {time.time()-t0:.0f}s, days A/B={len(days_ab)} C={len(days_c)}",flush=True)
print('PSEUDO_DONE')
