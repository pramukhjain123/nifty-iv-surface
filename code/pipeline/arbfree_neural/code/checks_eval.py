"""Ablation (no neural prior) and seed-variance scoring with the paper's Table 6/7 metric."""
import glob,pickle,os,sys, numpy as np, pandas as pd, duckdb
from scipy.stats import trim_mean, wilcoxon, binomtest
R='/mnt/user-data/uploads/iv_surface/research_pipeline'; key=['date','expiry','strike']
def loadpk(paths):
    src={}
    for f in paths: src[os.path.basename(f)]=f
    d=pd.concat([pickle.load(open(f,'rb'))['preds'] for f in src.values()]); d['date']=pd.to_datetime(d.date); d['expiry']=pd.to_datetime(d.expiry); return d
main=loadpk(sorted(glob.glob('full_v2/fold*.pkl'))+sorted(glob.glob(f'{R}/arbfree_neural/ab/fold*.pkl')))
nop=loadpk(sorted(glob.glob(f'{R}/arbfree_neural/ab_noprior/fold*.pkl')))
d=main.rename(columns={'pred':'with_prior'}).merge(nop[key+['pred']].rename(columns={'pred':'no_prior'}),on=key,how='left')
seedp=sorted(glob.glob('full_s23/fold*.pkl'))+sorted(glob.glob(f'{R}/arbfree_neural/ab_s23/fold*.pkl'))
if seedp:
    s=loadpk(seedp); d=d.merge(s[key+['pred']].rename(columns={'pred':'seed23'}),on=key,how='left')
def t6(df,col):
    r=[]
    for f,g in df.groupby('fold'):
        if f==16: continue
        g=g.dropna(subset=[col]); e=np.abs(g[col]-g.iv)*100
        r.append((np.sqrt(np.mean(e**2)),e.mean(),np.median(e),trim_mean(e,0.1)))
    return np.array(r).mean(0), len(r)
cols=['with_prior','no_prior']+(['seed23'] if 'seed23' in d else [])
for lab,sub in (('ALL held-out',d),('IN-RANGE',d[d.in_range]),('OUT-OF-RANGE',d[~d.in_range])):
    print(f"\n{lab} (Table 6 metric, 17 blind folds)")
    for c in cols:
        if c=='seed23':
            folds=set(d.dropna(subset=['seed23']).fold)
            v,n=t6(sub[sub.fold.isin(folds)],c); vw,_=t6(sub[sub.fold.isin(folds)],'with_prior')
            print(f"  {c:11s} rmse={v[0]:.3f} mae={v[1]:.3f} med={v[2]:.3f}   (same {n} folds, seeds 0/1: rmse={vw[0]:.3f})")
        else:
            v,n=t6(sub,c); print(f"  {c:11s} rmse={v[0]:.3f} mae={v[1]:.3f} med={v[2]:.3f} trim={v[3]:.3f}  folds={n}")
bl=d[d.fold!=16]
pf=bl.groupby('fold').apply(lambda g: pd.Series({c:np.sqrt(np.mean(((g[c]-g.iv)*100)**2)) for c in ['with_prior','no_prior']}))
print("\nfolds where prior helps:",int((pf.with_prior<pf.no_prior).sum()),"/",len(pf))
x=bl.groupby('date').apply(lambda g: pd.Series({'a':np.mean((g.with_prior-g.iv)**2),'b':np.mean((g.no_prior-g.iv)**2)}))
w=int((x.a<x.b).sum()); print(f"daily paired: with-prior wins {w}/{len(x)} days, sign p={binomtest(w,len(x)).pvalue:.1e}, Wilcoxon p={wilcoxon(x.a,x.b).pvalue:.1e}")
if 'seed23' in d:
    s=d.dropna(subset=['seed23']); s=s[s.fold!=16]
    pf2=s.groupby('fold').apply(lambda g: pd.Series({c:np.sqrt(np.mean(((g[c]-g.iv)*100)**2)) for c in ['with_prior','seed23']}))
    diff=pf2.seed23-pf2.with_prior; print(f"seed variance: folds={len(pf2)}  mean |RMSE(seeds 2/3) - RMSE(seeds 0/1)| = {diff.abs().mean():.4f}  max {diff.abs().max():.4f}; across-seed sd of fold-mean = {np.std([pf2.with_prior.mean(),pf2.seed23.mean()],ddof=1):.4f}")
    pf2.to_csv('final/seed_variance_by_fold.csv')
os.makedirs('final',exist_ok=True); pf.to_csv('final/ablation_by_fold.csv')
