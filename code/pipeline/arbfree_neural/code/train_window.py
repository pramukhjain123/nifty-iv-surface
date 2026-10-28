"""Table 6 'Train hold-out' column for the new model: held-out strikes on days inside each blind fold's training
window. Days < 2021-10-01 use fold-0's prior (trained on visible strikes of dates < 2021-10-01, which include those
days' visible quotes, never their held-out ones); later days use the out-of-time prediction from the fold whose
test window contains them (that prior never saw the day at all)."""
import glob,pickle,os, numpy as np, pandas as pd
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
src={}
for f in sorted(glob.glob('full_v2/fold*.pkl'))+sorted(glob.glob(f'{R}/arbfree_neural/ab/fold*.pkl')): src[os.path.basename(f)]=f
ab=pd.concat([pickle.load(open(f,'rb'))['preds'] for f in src.values()])
pre=pd.concat([pickle.load(open(f,'rb'))['preds'] for f in sorted(glob.glob('pre/chunk*.pkl'))])
allp=pd.concat([pre,ab.drop(columns='fold')]); allp['date']=pd.to_datetime(allp.date)
print('days covered:',allp.date.nunique(),' quotes:',len(allp))
F=pd.read_csv(f'{R}/05_splits/folds.csv',parse_dates=['train_start','train_end'])
r=[]
for f in F.itertuples():
    if f.fold==16: continue
    s=allp[(allp.date>=f.train_start)&(allp.date<f.train_end)]; r.append(np.sqrt(np.mean(((s.pred-s.iv)*100)**2)))
print(f"NEW model train hold-out (mean over 17 blind folds): {np.mean(r):.3f}   [paper: ThinPlate 0.947, Fengler 0.833, SSVI 2.791, SVI 3.946, v4 4.14, v3 6.68]")
pre_e=(pre.pred-pre.iv)*100; print(f"pre-period pooled: in-range {np.sqrt(np.mean(pre_e[pre.in_range]**2)):.3f}  oor {np.sqrt(np.mean(pre_e[~pre.in_range]**2)):.3f}")
os.makedirs('final',exist_ok=True); pd.DataFrame({'fold':[f.fold for f in F.itertuples() if f.fold!=16],'train_holdout_rmse':r}).to_csv('final/table6_train_window.csv',index=False)
