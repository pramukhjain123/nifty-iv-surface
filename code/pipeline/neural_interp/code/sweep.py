import sys,time,itertools; sys.path.insert(0,'.')
from load import load; import feats2, mlp, numpy as np, pandas as pd
p=load(); p['date']=pd.to_datetime(p.date)
tr=p[p.date<'2022-10-01']; va=p[(p.date>='2022-10-01')&(p.date<'2023-01-01')]; te=p[(p.date>='2023-01-01')&(p.date<'2024-01-01')]
Xtr,Ltr,Ytr,_=feats2.build(tr,'loo'); Xva,Lva,Yva,_=feats2.build(va,'ho'); Xte,Lte,Yte,Ite=feats2.build(te,'ho')
ytr=(Ytr-Ltr)*100; yva=(Yva-Lva)*100; yte=(Yte-Lte)*100
ncol=Xtr.shape[1]; base=[]; c=0
for side in range(2):
    for m in range(4):
        base+=[c,c+1,c+2]; c+=3
        if m<2: c+=2
base+=list(range(c,c+6))
dv_cols=[base[i] for i in range(0,24) if i%3==1]+[base[-1]]
def tf(X,cols,asinh):
    X=X[:,cols].copy()
    if asinh:
        for j,cc in enumerate(cols):
            if cc in dv_cols: X[:,j]=np.arcsinh(X[:,j])
    return X
def m(e): return np.sqrt(np.mean(e**2)),np.mean(np.abs(e)),np.median(np.abs(e))
print(f"linear: val {m(-yva)[0]:.4f} test rmse/mae/med {m(-yte)}")
rows=[]
for asinh,hid,steps,wd in itertools.product([False,True],[(64,64),(128,128),(256,256,128)],[1500,4000],[1e-5,1e-3]):
    t0=time.time()
    model=mlp.train(tf(Xtr,base,asinh),ytr,steps=steps,hidden=hid,wd=wd,loss='huber',verbose=False)
    pv=mlp.predict(model,tf(Xva,base,asinh)); pt=mlp.predict(model,tf(Xte,base,asinh))
    r=dict(asinh=asinh,hid=str(hid),steps=steps,wd=wd,val_rmse=m(pv-yva)[0],val_med=m(pv-yva)[2],test_rmse=m(pt-yte)[0],test_mae=m(pt-yte)[1],test_med=m(pt-yte)[2],s=time.time()-t0)
    rows.append(r); print(r,flush=True)
df=pd.DataFrame(rows).sort_values('val_rmse'); print(df.to_string()); df.to_csv('sweep.csv',index=False)
