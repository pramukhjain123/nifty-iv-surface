"""Score the arbitrage-free neural surface with the paper's Table 6/7 metric and pooled Task A in/out-of-range."""
import pickle,glob, numpy as np, pandas as pd, duckdb
from scipy.stats import trim_mean
R='/mnt/user-data/uploads/iv_surface/research_pipeline'; key=['date','expiry','strike']
import os
src={}
for f in sorted(glob.glob('full_v2/fold*.pkl'))+sorted(glob.glob('/mnt/user-data/uploads/iv_surface/research_pipeline/arbfree_neural/ab/fold*.pkl')): src[os.path.basename(f)]=f
ours=pd.concat([pickle.load(open(f,'rb'))['preds'] for f in src.values()])
print('folds:',sorted(ours.fold.unique()))
ours['date']=pd.to_datetime(ours.date); ours['expiry']=pd.to_datetime(ours.expiry)
b=duckdb.connect().execute(f"select method,date,expiry,strike,iv_pred from '{R}/results/baseline_predictions_long_fixed.parquet'").df()
fe=pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/results/fengler_chunks/fengler_pred_*.csv')])[['date','expiry','strike','iv_pred']].assign(method='Fengler')
b=pd.concat([b,fe]); b['date']=pd.to_datetime(b.date); b['expiry']=pd.to_datetime(b.expiry)
bw=b.pivot_table(index=key,columns='method',values='iv_pred').reset_index()
nn=pd.read_csv('../nn2/wf_preds.csv',parse_dates=['date','expiry'])[key+['nn']].rename(columns={'nn':'NN-interp (no arb. guarantee)'})
d=ours.rename(columns={'pred':'ArbFree-Neural (new)'}).merge(bw,on=key,how='left').merge(nn,on=key,how='left')
M=['ArbFree-Neural (new)','ThinPlate','Fengler','SSVI','SVI','CubicSpline']
def per_fold(df,m):
    r=[]
    for f,g in df.groupby('fold'):
        g=g.dropna(subset=[m]); e=np.abs(g[m]-g.iv)*100
        if len(e): r.append(dict(fold=f,rmse=np.sqrt(np.mean(e**2)),mae=e.mean(),med=np.median(e),trim=trim_mean(e,0.1),cov=len(g)))
    return pd.DataFrame(r)
def table(df,ms,title):
    print(f"\n=== {title} ===")
    rows=[]
    for m in ms:
        pf=per_fold(df,m); pf=pf[pf.fold!=16]
        rows.append(dict(method=m,folds=len(pf),rmse=pf.rmse.mean(),mae=pf.mae.mean(),median=pf.med.mean(),trim10=pf.trim.mean()))
    T=pd.DataFrame(rows); print(T.to_string(index=False,float_format='%.3f')); return T
T6=table(d,M,"Paper Table 6/7 metric: test-window held-out, mean over 17 blind folds (fold 16 excl.), ALL held-out")
Ti=table(d[d.in_range],M+['NN-interp (no arb. guarantee)'],"Same metric, IN-RANGE held-out only")
To=table(d[~d.in_range],M,"Same metric, OUT-OF-RANGE held-out only")
# per-fold wins
a=per_fold(d,'ArbFree-Neural (new)').set_index('fold').rmse
for m in ['ThinPlate','Fengler','SSVI']:
    o=per_fold(d,m).set_index('fold').rmse; bl=[f for f in a.index if f!=16]
    print(f"blind folds where new model beats {m}: {(a[bl]<o[bl]).sum()}/{len(bl)}")
# pooled (all test-window quotes) in/oor
print("\npooled over all test-window held-out quotes:")
for lab,sub in (('in-range',d[d.in_range]),('out-of-range',d[~d.in_range])):
    print(' ',lab,{m:round(float(np.sqrt(np.nanmean(((sub[m]-sub.iv)*100)**2))),3) for m in M})
print('coverage of new model:',d['ArbFree-Neural (new)'].notna().mean(), ' n=',len(d))
# paired daily-loss tests (paper Sec VI-F style): per-date mean squared loss, new vs each baseline
from scipy.stats import wilcoxon, binomtest
bl=d[d.fold!=16]
for m in ['ThinPlate','Fengler','SSVI']:
    x=bl.dropna(subset=[m,'ArbFree-Neural (new)']).groupby('date').apply(lambda g: pd.Series({'new':np.mean((g['ArbFree-Neural (new)']-g.iv)**2),'base':np.mean((g[m]-g.iv)**2)}))
    wins=int((x.new<x.base).sum()); n=len(x)
    print(f"daily paired vs {m}: new wins {wins}/{n} days, sign-test p={binomtest(wins,n).pvalue:.2e}, Wilcoxon p={wilcoxon(x.new,x.base).pvalue:.2e}")
os.makedirs('final',exist_ok=True); d.to_csv('final/final_predictions.csv',index=False); T6.to_csv('final/table6_metric.csv',index=False); Ti.to_csv('final/table6_inrange.csv',index=False); To.to_csv('final/table6_oor.csv',index=False)
