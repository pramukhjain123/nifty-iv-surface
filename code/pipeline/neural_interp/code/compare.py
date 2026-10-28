import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, duckdb, glob
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from load import load
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
wf=pd.read_csv('wf_preds.csv',parse_dates=['date','expiry'])
key=['date','expiry','strike']
# classical per-slice interpolants on the same quotes
p=load(); p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
rows=[]
for (d,e),g in p[p.date>=wf.date.min()].groupby(['date','expiry'],sort=False):
    v=g[~g.is_heldout]; h=g[g.is_heldout & g.in_range]
    if len(h)==0 or len(v)<3: continue
    a=Akima1DInterpolator(v.k.values,v.iv.values,method='makima')(h.k.values)
    pc=PchipInterpolator(v.k.values,v.iv.values)(h.k.values)
    rows.append(pd.DataFrame({'date':d,'expiry':e,'strike':h.strike.values,'Akima':a,'PCHIP':pc}))
cl=pd.concat(rows)
df=wf.merge(cl,on=key,how='left')
b=duckdb.connect().execute(f"select method,date,expiry,strike,iv_pred from '{R}/results/baseline_predictions_long_fixed.parquet'").df()
b['date']=pd.to_datetime(b.date); b['expiry']=pd.to_datetime(b.expiry)
bw=b.pivot_table(index=key,columns='method',values='iv_pred').reset_index()
df=df.merge(bw,on=key,how='left')
fe=pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/results/fengler_chunks/fengler_pred_*.csv')])
fe['date']=pd.to_datetime(fe.date); fe['expiry']=pd.to_datetime(fe.expiry)
df=df.merge(fe[key+['iv_pred']].rename(columns={'iv_pred':'Fengler'}),on=key,how='left')
df=df.rename(columns={'lin':'Linear','nn':'NeuralInterp'})
methods=['NeuralInterp','Linear','Akima','PCHIP','ThinPlate','Fengler','CubicSpline','SVI','SSVI']
df.to_csv('matched.csv',index=False)
# common subset: all methods have a prediction
common=df.dropna(subset=methods)
def stats(d,m):
    e=(d[m]-d.iv)*100; return dict(method=m,n=len(e),rmse=np.sqrt(np.mean(e**2)),mae=np.mean(np.abs(e)),median=np.median(np.abs(e)),p99=np.quantile(np.abs(e),.99))
print(f"all test held-out in-range quotes: {len(df)};  common subset (every method has a prediction): {len(common)}")
T=pd.DataFrame([stats(common,m) for m in methods]).sort_values('rmse'); print(T.to_string(index=False,float_format='%.4f'))
# per fold RMSE
pf=common.groupby('fold').apply(lambda d: pd.Series({m:np.sqrt(np.mean(((d[m]-d.iv)*100)**2)) for m in methods}))
print(pf.to_string(float_format='%.3f'))
best_classical=[m for m in methods if m!='NeuralInterp']
wins={m:int((pf.NeuralInterp<pf[m]).sum()) for m in best_classical}; print('folds where NN beats method (of',len(pf),'):',wins)
# Diebold-Mariano on squared errors vs each classical method, day-clustered (Newey-West on daily mean loss differential)
def dm(d,m):
    l=((d.NeuralInterp-d.iv)**2-(d[m]-d.iv)**2)*1e4
    s=pd.Series(l.values,index=d.date.values).groupby(level=0).mean()
    x=s.values; n=len(x); mu=x.mean(); L=int(np.floor(4*(n/100)**(2/9)))
    g0=np.mean((x-mu)**2); v=g0+2*sum((1-k/(L+1))*np.mean((x[k:]-mu)*(x[:-k]-mu)) for k in range(1,L+1))
    return mu/np.sqrt(v/n)
print('DM t-stats (negative = NN lower squared error), day-clustered NW:',{m:round(dm(common,m),2) for m in best_classical})
T.to_csv('summary_common.csv',index=False); pf.to_csv('per_fold_rmse.csv')
