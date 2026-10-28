"""Score the improved neural interpolator with the paper's own Table 6/7 metric:
mean over the 17 blind folds (fold 16 excluded) of per-fold test-window held-out RMSE (and MAE/median/10%-trimmed mean)."""
import numpy as np, pandas as pd, duckdb, glob
from scipy.stats import trim_mean
R='/mnt/user-data/uploads/iv_surface/research_pipeline'; key=['date','expiry','strike']
folds=pd.read_csv(f'{R}/05_splits/folds.csv',parse_dates=['test_start','test_end'])
b=duckdb.connect().execute(f"select method,date,expiry,strike,iv_actual,iv_pred,in_range from '{R}/results/baseline_predictions_long_fixed.parquet'").df()
fe=pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/results/fengler_chunks/fengler_pred_*.csv')])[['method','date','expiry','strike','iv_actual','iv_pred']]
vr=duckdb.connect().execute(f"select * from '{R}/05_splits/visible_k_range.parquet'").df()
pk=duckdb.connect().execute(f"select date,expiry,strike,k from '{R}/04_iv/panel.parquet'").df()
fe=fe.assign(date=pd.to_datetime(fe.date),expiry=pd.to_datetime(fe.expiry))
for d in (vr,pk,b): d['date']=pd.to_datetime(d.date); d['expiry']=pd.to_datetime(d.expiry)
fe=fe.merge(pk,on=key).merge(vr,on=['date','expiry']); fe['in_range']=(fe.k>=fe.k_vis_min)&(fe.k<=fe.k_vis_max)
allp=pd.concat([b,fe[b.columns]])
# full held-out universe with fold labels
def tag(df):
    df=df.copy(); df['fold']=-1
    for f in folds.itertuples(): df.loc[(df.date>=f.test_start)&(df.date<f.test_end),'fold']=f.fold
    return df[df.fold>=0]
allp=tag(allp)
wf=pd.read_csv('wf_preds.csv',parse_dates=['date','expiry'])
# Our model: in-range = NN; out-of-range needs a fallback. Build hybrids.
univ=allp[allp.method=='ThinPlate'][key+['iv_actual','in_range','fold']]   # thin-plate covers ~all held-out quotes
nn=univ.merge(wf[key+['nn']],on=key,how='left')
fb=allp.pivot_table(index=key,columns='method',values='iv_pred').reset_index()
nn=nn.merge(fb[key+['Fengler','ThinPlate']],on=key,how='left')
# flat extrapolation fallback: nearest visible quote's IV (purely nonparametric, no classical model)
p=duckdb.connect().execute(f"select p.date,p.expiry,p.k,p.iv,h.is_heldout from '{R}/04_iv/panel.parquet' p join '{R}/05_splits/held_out_strikes.parquet' h using(date,expiry,strike)").df()
p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
vis=p[~p.is_heldout]
lo=vis.loc[vis.groupby(['date','expiry']).k.idxmin(),['date','expiry','iv']].rename(columns={'iv':'iv_lo'})
hi=vis.loc[vis.groupby(['date','expiry']).k.idxmax(),['date','expiry','iv']].rename(columns={'iv':'iv_hi'})
nn=nn.merge(pk,on=key).merge(vr,on=['date','expiry']).merge(lo,on=['date','expiry']).merge(hi,on=['date','expiry'])
nn['flat']=np.where(nn.k<nn.k_vis_min,nn.iv_lo,nn.iv_hi)
nn['NN+flat']=np.where(nn.in_range,nn.nn,nn.flat)
nn['NN+Fengler']=np.where(nn.in_range,nn.nn,nn.Fengler)
def agg(df,col,truth='iv_actual'):
    rows=[]
    for f,g in df.groupby('fold'):
        g=g.dropna(subset=[col]); e=np.abs(g[col]-g[truth])*100
        rows.append(dict(fold=f,rmse=np.sqrt(np.mean(e**2)),mae=e.mean(),med=np.median(e),trim=trim_mean(e,0.1)))
    r=pd.DataFrame(rows)
    if len(r)==0: return pd.Series(dtype=float),0
    r=r[r.fold!=16]
    return r[['rmse','mae','med','trim']].mean(), len(r)
print("=== Paper's Table 6/7 metric: test-window held-out, mean over 17 blind folds (fold 16 excluded), ALL held-out strikes ===")
out=[]
for m in ['ThinPlate','Fengler','SSVI','SVI','CubicSpline']:
    s,n=agg(allp[allp.method==m].rename(columns={'iv_pred':m}),m); out.append(dict(method=m,**s,folds=n))
for m in ['NN+flat','NN+Fengler']:
    s,n=agg(nn,m); out.append(dict(method=m,**s,folds=n))
T=pd.DataFrame(out); print(T.to_string(index=False,float_format='%.3f'))
print("\n=== same metric, IN-RANGE held-out only ===")
out=[]
allp["in_range"]=allp.in_range.fillna(False).astype(bool); inr=allp[allp.in_range]
for m in ['ThinPlate','Fengler','SSVI','SVI','CubicSpline']:
    s,n=agg(inr[inr.method==m].rename(columns={'iv_pred':m}),m); out.append(dict(method=m,**s))
s,n=agg(nn[nn.in_range],'nn'); out.append(dict(method='NN (ours)',**s))
print(pd.DataFrame(out).to_string(index=False,float_format='%.3f'))
print("\nout-of-range share of test held-out:", 1-nn.in_range.mean())
oor=nn[~nn.in_range]
for c in ['flat','Fengler','ThinPlate']:
    e=(oor[c]-oor.iv_actual).dropna()*100; print(f"  OOR {c}: n={len(e)} rmse={np.sqrt(np.mean(e**2)):.3f}")

print("\n=== full-coverage hybrid (no dropped quotes) ===")
nn['NN+ThinPlate']=np.where(nn.in_range,nn.nn,nn.ThinPlate)
nn['NN+Fengler/TP']=np.where(nn.in_range,nn.nn,nn.Fengler.fillna(nn.ThinPlate))
for c in ['NN+ThinPlate','NN+Fengler/TP']:
    s,n=agg(nn,c); print(c, s.round(3).to_dict(), 'coverage', nn[c].notna().mean())
s,n=agg(allp[allp.method=='ThinPlate'].rename(columns={'iv_pred':'TP'}),'TP'); print('ThinPlate', s.round(3).to_dict())
# per-fold wins vs ThinPlate & Fengler on the paper metric
def perfold(df,col):
    return df.dropna(subset=[col]).groupby('fold').apply(lambda g: np.sqrt(np.mean(((g[col]-g.iv_actual)*100)**2)))
a=perfold(nn,'NN+ThinPlate'); t=perfold(allp[allp.method=='ThinPlate'].rename(columns={'iv_pred':'TP'}),'TP'); f=perfold(allp[allp.method=='Fengler'].rename(columns={'iv_pred':'FE'}),'FE')
blind=[i for i in a.index if i!=16]
print('blind folds where NN+ThinPlate beats ThinPlate:',int((a[blind]<t[blind]).sum()),'/17;  beats Fengler:',int((a[blind]<f[blind]).sum()),'/17')

clean=[i for i in a.index if i not in (4,5,6,7,8,16)]
print(f"\n12 fully clean folds (also excluding 4-8 used in my design choices): NN+ThinPlate={a[clean].mean():.3f} ThinPlate={t[clean].mean():.3f} Fengler={f[clean].mean():.3f}")
ai=perfold(nn[nn.in_range],'nn'); ti=perfold(allp[(allp.method=='ThinPlate')&allp.in_range].rename(columns={'iv_pred':'TP'}),'TP'); fi=perfold(allp[(allp.method=='Fengler')&allp.in_range].rename(columns={'iv_pred':'FE'}),'FE')
print(f"in-range, 12 clean folds: NN={ai[clean].mean():.3f} ThinPlate={ti[clean].mean():.3f} Fengler={fi[clean].mean():.3f};  in-range wins vs ThinPlate {int((ai[blind]<ti[blind]).sum())}/17, vs Fengler {int((ai[blind]<fi[blind]).sum())}/17")
