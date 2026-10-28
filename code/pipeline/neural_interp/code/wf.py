"""Walk-forward evaluation of the learned neural interpolator over the paper's 18 folds.
Train: leave-one-out on VISIBLE strikes of all slices dated < test_start (held-out strikes never used).
Test : the paper's fixed Task-A held-out strikes (in visible range) of slices dated in [test_start, test_end).
Config frozen from the 2022Q4 validation sweep: base features, MLP 256-256-128 GELU, Huber, wd 1e-3, ~37 epochs, 5 seeds."""
import sys,time; sys.path.insert(0,'.')
from load import load; import feats2, mlp, numpy as np, pandas as pd
p=load(); p['date']=pd.to_datetime(p.date)
folds=pd.read_csv('/mnt/user-data/uploads/iv_surface/research_pipeline/05_splits/folds.csv',parse_dates=['train_start','train_end','test_start','test_end'])
ncol=None; base=[]; c=0
for side in range(2):
    for m in range(4):
        base+=[c,c+1,c+2]; c+=3
        if m<2: c+=2
base+=list(range(c,c+6))
SEEDS=[0,1,2,3,4]
out=[]
for f in folds.itertuples():
    t0=time.time()
    tr=p[p.date<f.test_start]; te=p[(p.date>=f.test_start)&(p.date<f.test_end)]
    Xtr,Ltr,Ytr,_=feats2.build(tr,'loo'); Xte,Lte,Yte,Ite=feats2.build(te,'ho')
    ytr=(Ytr-Ltr)*100
    steps=int(min(6000,max(1500,round(len(Xtr)*37/2048))))
    preds=[]
    for s in SEEDS:
        model=mlp.train(Xtr[:,base],ytr,steps=steps,hidden=(256,256,128),wd=1e-3,loss='huber',verbose=False,seed=s)
        preds.append(mlp.predict(model,Xte[:,base]))
    P=np.array(preds)
    d=p.loc[Ite,['date','expiry','strike','k','tau','iv']].copy()
    d['fold']=f.fold; d['lin']=Lte; d['nn']=Lte+P.mean(0)/100
    for i,s in enumerate(SEEDS): d[f'nn_s{s}']=Lte+P[i]/100
    out.append(d)
    e_l=(d.lin-d.iv)*100; e_n=(d.nn-d.iv)*100
    print(f"fold {f.fold:2d} ntrain={len(Xtr)} steps={steps} ntest={len(d)} lin={np.sqrt(np.mean(e_l**2)):.4f} nn={np.sqrt(np.mean(e_n**2)):.4f} ({time.time()-t0:.0f}s)",flush=True)
    pd.concat(out).to_parquet('wf_preds.parquet') if False else pd.concat(out).to_csv('wf_preds.csv',index=False)
print('DONE')
