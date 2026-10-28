"""Task A/B QP stage in the cloud workspace (identical logic/config to fit_windows.py job_ab)."""
import sys,os,pickle,time; sys.path.insert(0,'.'); sys.path.insert(0,'/mnt/user-data/outputs/arbfree_neural/code')
import numpy as np, pandas as pd
from multiprocessing import Pool
from prep import load
import qpmix
CFG=dict(lam=1e-7,vega_floor=5e-3,n_fine=32,fine_scales=(1,4),n_coarse=50,coarse_scales=(1,4,16)); ALPHA=1.0
p=load()
F=pd.read_csv('/mnt/user-data/uploads/iv_surface/research_pipeline/05_splits/folds.csv',parse_dates=['test_start','test_end'])
def job(args):
    day,gd,ps=args; sd=[];exps=[]
    for e,g in gd.groupby('expiry',sort=True):
        v=g[~g.is_heldout]
        if len(v)<3: continue
        pad=(v.k.max()-v.k.min())*0.2; k=v.k.values; iv=v.iv.values; wt=np.ones(len(k)); q=ps[ps.expiry==e]
        if len(q): k=np.r_[k,q.k.values]; iv=np.r_[iv,q.iv.values]; wt=np.r_[wt,np.full(len(q),ALPHA)]
        sd.append(dict(k=k,iv=iv,tau=g.tau.iloc[0],kpad=(v.k.min()-pad,v.k.max()+pad),wt=wt,kvis=(v.k.min(),v.k.max()))); exps.append(e)
    sl=qpmix.fit_day_joint2(sd,**CFG)
    S=dict(expiries=exps,taus=[s['tau'] for s in sd],slices=[(s.z,s.v,s.pi) for s in sl],kvis=[s['kvis'] for s in sd])
    rows=[]
    for j,e in enumerate(exps):
        h=gd[(gd.expiry==e)&gd.is_heldout]
        if len(h)==0: continue
        P=qpmix.surface_otm(sl,j,h.k.values)
        rows.append(pd.DataFrame({'date':day,'expiry':e,'strike':h.strike.values,'k':h.k.values,'tau':h.tau.values,'iv':h.iv.values,
                                  'in_range':h.in_range.values,'pred':np.sqrt(qpmix.implied_w(P,h.k.values)/h.tau.iloc[0])}))
    return day,(pd.concat(rows) if rows else None),S
if __name__=='__main__':
    os.makedirs('full_v2',exist_ok=True)
    while True:
        todo=[f for f in sorted(F.itertuples(),key=lambda r:-r.fold) if f.fold>=12 and not os.path.exists(f"full_v2/fold{f.fold}.pkl") and not os.path.exists(f"skip_{f.fold}")]
        if not todo: break
        f=todo[0]
        if not os.path.exists(f'pseudo/fold{f.fold}.csv'): time.sleep(30); continue
        t0=time.time(); ps=pd.read_csv(f'pseudo/fold{f.fold}.csv',parse_dates=['date','expiry'])
        te=p[(p.date>=f.test_start)&(p.date<f.test_end)]
        jobs=[(d,g,ps[ps.date==d]) for d,g in te.groupby('date')]
        with Pool(2) as pool: res=pool.map(job,jobs,chunksize=2)
        preds=pd.concat([r[1] for r in res if r[1] is not None]); preds['fold']=f.fold
        pickle.dump(dict(preds=preds,surf={r[0]:r[2] for r in res}),open(f'full_v2/fold{f.fold}.pkl','wb'))
        i=preds[preds.in_range]; o=preds[~preds.in_range]
        print(f"AB fold {f.fold:2d}: {time.time()-t0:.0f}s days={len(jobs)} in={np.sqrt(np.mean(((i.pred-i.iv)*100)**2)):.3f} oor={np.sqrt(np.nanmean(((o.pred-o.iv)*100)**2)):.3f}",flush=True)
    print('AB_DONE')
