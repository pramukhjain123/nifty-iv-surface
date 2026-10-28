import sys,time; sys.path.insert(0,'.')
from load import load; from feats import build; import mlp, numpy as np, pandas as pd
p=load(); p['date']=pd.to_datetime(p.date)
tr=p[p.date<'2022-10-01']; va=p[(p.date>='2022-10-01')&(p.date<'2023-01-01')]; te=p[(p.date>='2023-01-01')&(p.date<'2024-01-01')]
t0=time.time()
Xtr,Ltr,Ytr,_=build(tr,'loo'); Xva,Lva,Yva,_=build(va,'ho'); Xte,Lte,Yte,Ite=build(te,'ho')
print('built',Xtr.shape,Xva.shape,Xte.shape,f'{time.time()-t0:.0f}s',flush=True)
ytr=(Ytr-Ltr)*100; yva=(Yva-Lva)*100; yte=(Yte-Lte)*100
print('lin rmse val/test', np.sqrt(np.mean(yva**2)), np.sqrt(np.mean(yte**2)))
# clip extreme training targets (noisy quotes) for stability
for clipv in [None,5.0]:
  yt=ytr if clipv is None else np.clip(ytr,-clipv,clipv)
  for loss in ['mse','huber']:
    t0=time.time(); model=mlp.train(Xtr,yt,Xva,yva,steps=6000,loss=loss,verbose=False)
    pv=mlp.predict(model,Xva); pt=mlp.predict(model,Xte)
    print(f"clip={clipv} loss={loss}: val rmse={np.sqrt(np.mean((pv-yva)**2)):.4f} test rmse={np.sqrt(np.mean((pt-yte)**2)):.4f} mae={np.mean(np.abs(pt-yte)):.4f} med={np.median(np.abs(pt-yte)):.4f} ({time.time()-t0:.0f}s)",flush=True)
