"""Task C aggregate exactly like the paper's 12d (mean over folds of per-fold RMSE/MAE/median, fold 16 excluded)."""
import glob, numpy as np, pandas as pd
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
ours=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/arbfree_neural/c/fold*.csv'))]).assign(method='ArbFree-Neural (new)')
base=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/results/phase6_taskC_fold*_raw_d0_1000000000_m30.csv'))]).drop_duplicates(['fold','window','date','expiry','k','method'])
df=pd.concat([base[['fold','window','date','expiry','k','method','iv_actual','iv_pred']],ours[['fold','window','date','expiry','k','method','iv_actual','iv_pred']]])
df=df[df.fold.isin(ours.fold.unique())]
rows=[]
for (w,m),g in df[df.fold!=16].groupby(['window','method']):
    pf=[]
    for f,gf in g.groupby('fold'):
        s=gf.dropna(subset=['iv_pred']); e=(s.iv_pred-s.iv_actual).values*100
        if len(e): pf.append((np.sqrt(np.mean(e**2)),np.mean(np.abs(e)),np.median(np.abs(e)),len(s)/len(gf)))
    pf=np.array(pf); rows.append(dict(window=w,method=m,folds=len(pf),rmse=pf[:,0].mean(),mae=pf[:,1].mean(),median=pf[:,2].mean(),coverage=pf[:,3].mean()))
T=pd.DataFrame(rows).sort_values(['window','rmse']); print(T.to_string(index=False,float_format='%.3f')); T.to_csv('final/taskc_table11.csv',index=False)
