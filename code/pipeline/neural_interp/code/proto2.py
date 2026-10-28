import sys,time; sys.path.insert(0,'.')
from load import load; import feats2, mlp, numpy as np, pandas as pd
from scipy.interpolate import Akima1DInterpolator
p=load(); p['date']=pd.to_datetime(p.date)
tr=p[p.date<'2022-10-01']; va=p[(p.date>='2022-10-01')&(p.date<'2023-01-01')]; te=p[(p.date>='2023-01-01')&(p.date<'2024-01-01')]
t0=time.time()
Xtr,Ltr,Ytr,_=feats2.build(tr,'loo'); Xva,Lva,Yva,_=feats2.build(va,'ho'); Xte,Lte,Yte,Ite=feats2.build(te,'ho')
print('built',Xtr.shape,Xte.shape,f'{time.time()-t0:.0f}s',flush=True)
ytr=(Ytr-Ltr)*100; yva=(Yva-Lva)*100; yte=(Yte-Lte)*100
def rep(name,e): print(f"{name:28s} rmse={np.sqrt(np.mean(e**2)):.4f} mae={np.mean(np.abs(e)):.4f} med={np.median(np.abs(e)):.4f}",flush=True)
rep('linear',-yte)
# feature-group ablation: base(no liq,no adj) / +liq / +adj / all
ncol=Xtr.shape[1]
base_idx=[]; liq_idx=[]; c=0
for side in range(2):
    for m in range(4):
        base_idx+= [c,c+1,c+2]; c+=3
        if m<2: liq_idx+=[c,c+1]; c+=2
base_idx+=list(range(c,c+6)); c+=6; adj_idx=list(range(c,ncol))
sets={'base':base_idx,'+liq':base_idx+liq_idx,'+adj':base_idx+adj_idx,'all':list(range(ncol))}
preds={}
for name,cols in sets.items():
    t0=time.time(); model=mlp.train(Xtr[:,cols],ytr,steps=6000,loss='huber',verbose=False)
    pt=mlp.predict(model,Xte[:,cols]); preds[name]=pt
    rep(f'NN {name} ({time.time()-t0:.0f}s)',pt-yte)
# 3-seed ensemble of 'all'
ens=[preds['all']]
for s in [1,2]:
    model=mlp.train(Xtr,ytr,steps=6000,loss='huber',verbose=False,seed=s); ens.append(mlp.predict(model,Xte))
rep('NN all, 3-seed ensemble',np.mean(ens,0)-yte)
np.save('proto2_idx.npy',Ite); np.save('proto2_pred.npy',(Lte+np.mean(ens,0)/100))
