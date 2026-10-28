"""(1) seed-variance: identical prior pipeline with seeds (2,3) -> pseudo_s23/fold*.csv
   (2) training-window column: pseudo-quotes for days < 2021-10-01 (before any test window) from fold 0's prior
       (seeds 0,1; trained on visible strikes of dates < 2021-10-01, band slices excluded) -> pseudo_pre/pre.csv"""
import sys,time,os,glob; sys.path.insert(0,'.')
from prep import load
import prior, numpy as np, pandas as pd
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
p=load(); p['dte']=(p.tau*365).round().astype(int); p['in_band']=(p.dte>=30)&(p.dte<=60)
folds=pd.read_csv(f'{R}/05_splits/folds.csv',parse_dates=['test_start','test_end'])
Q=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/results/phase6_taskC_fold*_raw_d0_1000000000_m30.csv'))])
Q=Q[Q.method=='Fengler+PCHIP'][['fold','date']].drop_duplicates(); Q['date']=pd.to_datetime(Q.date)
def emit(models,days,skip_band_days=set()):
    rows=[]
    for day in sorted(days):
        gd=p[p.date==day]
        for e,g in gd.groupby('expiry'):
            if day in skip_band_days and g.in_band.iloc[0]: continue
            v=g[~g.is_heldout]
            if len(v)<3: continue
            k,iv,kind=prior.pseudo_for_slice(models,g.tau.iloc[0],v.k.values,v.iv.values)
            if len(k): rows.append(pd.DataFrame({'date':day,'expiry':e,'k':k,'iv':iv,'kind':kind}))
    return pd.concat(rows)
os.makedirs('pseudo_pre',exist_ok=True); os.makedirs('pseudo_s23',exist_ok=True)
if not os.path.exists('pseudo_pre/pre.csv'):
    t0=time.time(); f0=folds.iloc[0]; m=prior.train(p[(p.date<f0.test_start)&(~p.in_band)],seeds=(0,1))
    emit(m,set(p[p.date<f0.test_start].date.unique())).to_csv('pseudo_pre/pre.csv',index=False); print(f'pre-period pseudo {time.time()-t0:.0f}s',flush=True)
for f in folds.itertuples():
    out=f'pseudo_s23/fold{f.fold}.csv'
    if os.path.exists(out): continue
    t0=time.time(); m=prior.train(p[(p.date<f.test_start)&(~p.in_band)],seeds=(2,3))
    ab=set(p[(p.date>=f.test_start)&(p.date<f.test_end)].date.unique()); c=set(Q[Q.fold==f.fold].date)
    emit(m,ab|c,skip_band_days=c-ab).assign(fold=f.fold).to_csv(out,index=False)
    print(f"s23 fold {f.fold:2d}: {time.time()-t0:.0f}s",flush=True)
print('DONE2')
