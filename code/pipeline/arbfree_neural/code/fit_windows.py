"""Arbitrage-free neural IV surface -- QP stage (runs on the 16-core machine).
usage: python fit_windows.py ab <folds...> | c <folds...> | arb
  ab : Task A/B -- fit every test-window day of each fold, predict its fixed 20% held-out strikes, store slices
  c  : Task C   -- on the paper's exact Task C quote set, fit each day's NON-band quotes, reconstruct band slices
  arb: paper-rule dense-grid butterfly / calendar check on all stored Task A/B surfaces
Frozen config (chosen on development days < 2021-10-01 only): two-resolution dictionary (fine 32 atoms x
kernels (1,4) within +-4 ATM std; coarse 50 atoms x kernels (1,4,16) over the padded range), lam=1e-7,
vega_floor=5e-3, calendar grid 100 pts, pseudo-quote weight alpha=1.0."""
import sys,os,glob,pickle,time
from pathlib import Path
import numpy as np, pandas as pd, duckdb
from multiprocessing import Pool
sys.path.insert(0,str(Path(__file__).parent))
import qpmix, surface
ROOT=Path(r"E:\Research\iv_surface\research_pipeline"); OUT=ROOT/"arbfree_neural"
CFG=dict(lam=1e-7,vega_floor=5e-3,n_fine=32,fine_scales=(1,4),n_coarse=50,coarse_scales=(1,4,16))
ALPHA=float(os.environ.get("ALPHA","1.0")); NPROC=int(os.environ.get("NPROC","3"))
PSEUDO=os.environ.get("PSEUDO_DIR","pseudo"); TAG=os.environ.get("OUT_TAG","")   # TAG e.g. "_noprior" -> ab_noprior/, c_noprior/
def load_panel():
    con=duckdb.connect()
    p=con.execute(f"""select p.date,p.expiry,p.strike,p.k,p.tau,p.iv,h.is_heldout,vr.k_vis_min,vr.k_vis_max
      from '{ROOT/'04_iv'/'panel.parquet'}' p join '{ROOT/'05_splits'/'held_out_strikes.parquet'}' h using(date,expiry,strike)
      join '{ROOT/'05_splits'/'visible_k_range.parquet'}' vr using(date,expiry) order by date,expiry,k""").df()
    p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
    p['in_range']=(p.k>=p.k_vis_min)&(p.k<=p.k_vis_max)
    p['dte']=(p.tau*365).round().astype(int); p['in_band']=(p.dte>=30)&(p.dte<=60)
    return p
def slices_data(gd,ps,band_excluded=False):
    sd=[];exps=[]
    for e,g in gd.groupby('expiry',sort=True):
        if band_excluded and g.in_band.iloc[0]: continue
        v=g[~g.is_heldout]
        if len(v)<3: continue
        pad=(v.k.max()-v.k.min())*0.2; k=v.k.values; iv=v.iv.values; wt=np.ones(len(k))
        q=ps[ps.expiry==e] if ps is not None and len(ps) else None
        if q is not None and len(q): k=np.r_[k,q.k.values]; iv=np.r_[iv,q.iv.values]; wt=np.r_[wt,np.full(len(q),ALPHA)]
        sd.append(dict(k=k,iv=iv,tau=g.tau.iloc[0],kpad=(v.k.min()-pad,v.k.max()+pad),wt=wt,kvis=(v.k.min(),v.k.max())))
        exps.append(e)
    return sd,exps
def job_ab(args):
    day,gd,ps=args
    sd,exps=slices_data(gd,ps)
    sl=qpmix.fit_day_joint2(sd,**CFG)
    S=dict(expiries=exps,taus=[s['tau'] for s in sd],slices=[(s.z,s.v,s.pi) for s in sl],kvis=[s['kvis'] for s in sd])
    rows=[]
    for j,e in enumerate(exps):
        h=gd[(gd.expiry==e)&gd.is_heldout]
        if len(h)==0: continue
        P=qpmix.surface_otm(sl,j,h.k.values)
        rows.append(pd.DataFrame({'date':day,'expiry':e,'strike':h.strike.values,'k':h.k.values,'tau':h.tau.values,'iv':h.iv.values,
                                  'in_range':h.in_range.values,'pred':np.sqrt(qpmix.implied_w(P,h.k.values)/h.tau.iloc[0])}))
    return day,(pd.concat(rows) if rows else None),S
def job_c(args):
    day,gd,ps,qd=args
    sd,exps=slices_data(gd,ps,band_excluded=True)
    if len(sd)<2: return qd.assign(iv_pred=np.nan)
    sl=qpmix.fit_day_joint2(sd,**CFG)
    S=dict(taus=[s['tau'] for s in sd],slices=[(s.z,s.v,s.pi) for s in sl])
    out=[]
    for (w,e),qe in qd.groupby(['window','expiry']):
        try: pr=surface.iv(S,qe.k.values,qe.tau.iloc[0])
        except ValueError: pr=np.full(len(qe),np.nan)
        out.append(qe.assign(iv_pred=pr))
    return pd.concat(out)
def run_ab(folds):
    p=load_panel(); F=pd.read_csv(ROOT/'05_splits'/'folds.csv',parse_dates=['test_start','test_end'])
    for f in F.itertuples():
        if f.fold not in folds: continue
        out=OUT/('ab'+TAG)/f'fold{f.fold}.pkl'
        if out.exists(): continue
        t0=time.time(); ps=pd.read_csv(OUT/PSEUDO/f'fold{f.fold}.csv',parse_dates=['date','expiry']) if ALPHA>0 else pd.DataFrame(columns=['date','expiry','k','iv'])
        te=p[(p.date>=f.test_start)&(p.date<f.test_end)]
        jobs=[(d,g,ps[ps.date==d]) for d,g in te.groupby('date')]
        with Pool(NPROC) as pool: res=pool.map(job_ab,jobs,chunksize=1)
        preds=pd.concat([r[1] for r in res if r[1] is not None]); preds['fold']=f.fold
        pickle.dump(dict(preds=preds,surf={r[0]:r[2] for r in res}),open(out,'wb'))
        i=preds[preds.in_range]; o=preds[~preds.in_range]
        print(f"AB fold {f.fold:2d}: {time.time()-t0:.0f}s days={len(jobs)} in={np.sqrt(np.mean(((i.pred-i.iv)*100)**2)):.3f} oor={np.sqrt(np.nanmean(((o.pred-o.iv)*100)**2)):.3f}",flush=True)
def run_c(folds):
    p=load_panel()
    Q=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT/'results'/'phase6_taskC_fold*_raw_d0_1000000000_m30.csv')))])
    Q=Q[Q.method=='Fengler+PCHIP'].drop_duplicates(['fold','window','date','expiry','k'])[['fold','window','date','expiry','tau','k','iv_actual']]
    Q['date']=pd.to_datetime(Q.date); Q['expiry']=pd.to_datetime(Q.expiry)
    for f in folds:
        out=OUT/('c'+TAG)/f'fold{f}.csv'
        if out.exists(): continue
        t0=time.time(); ps=pd.read_csv(OUT/PSEUDO/f'fold{f}.csv',parse_dates=['date','expiry']) if ALPHA>0 else pd.DataFrame(columns=['date','expiry','k','iv'])
        qf=Q[Q.fold==f]
        jobs=[(d,p[p.date==d],ps[ps.date==d],qd) for d,qd in qf.groupby('date')]
        with Pool(NPROC) as pool: res=pool.map(job_c,jobs,chunksize=1)
        d=pd.concat(res); d.to_csv(out,index=False)
        r=lambda x: np.sqrt(np.nanmean(((x.iv_pred-x.iv_actual)*100)**2))
        print(f"C fold {f:2d}: {time.time()-t0:.0f}s c1={r(d[d.window=='c1']):.3f} c2={r(d[d.window=='c2']):.3f} cov={d.iv_pred.notna().mean():.4f}",flush=True)
H=1e-3
def _w(S,m,k): return qpmix.implied_w(surface.otm_listed(S,m,k),k)
def check_day(S):
    out=dict(b_slices=0,b_any=0,b_obs=0,b_pad=0,b_pts=0,b_npts=0,c_pairs=0,c_any=0,c_min=np.inf,tb_pts=0,tb_viol=0,tc_pts=0,tc_viol=0,tc_min=np.inf,undef=0)
    M=len(S['taus'])
    for m in range(M):
        lo,hi=S['kvis'][m]; pad=0.2*(hi-lo); kg=np.linspace(lo-pad,hi+pad,61)
        w=_w(S,m,kg); wp=(_w(S,m,kg+H)-_w(S,m,kg-H))/(2*H); wpp=(_w(S,m,kg+H)-2*w+_w(S,m,kg-H))/H**2
        g=(1-kg*wp/(2*w))**2-wp**2/4*(1/w+0.25)+wpp/2; obs=(kg>=lo)&(kg<=hi); v=g<0
        out['undef']+=int((~np.isfinite(g)).sum()); out['b_slices']+=1; out['b_any']+=int(v.any()); out['b_obs']+=int(v[obs].any()); out['b_pad']+=int(v[~obs].any())
        out['b_pts']+=int(v.sum()); out['b_npts']+=len(kg)
    lo=min(x[0] for x in S['kvis']); hi=max(x[1] for x in S['kvis']); pad=0.2*(hi-lo); kd=np.linspace(lo-pad,hi+pad,121)
    W=[_w(S,m,kd) for m in range(M)]
    for m in range(M-1):
        d=W[m+1]-W[m]; out['c_pairs']+=1; out['c_any']+=int((d<0).any()); out['c_min']=min(out['c_min'],float(d.min()))
    taus=np.array(S['taus']); tg=np.linspace(min(7/365,taus[0]),taus[-1],25); Wt=[]
    for t in tg:
        P=surface.otm(S,kd,t); Wt.append(qpmix.implied_w(P,kd))
        x=np.exp(kd); C=np.where(kd>=0,P,P+1-x); sl=np.diff(C)/np.diff(x)
        out['tb_pts']+=len(sl)-1; out['tb_viol']+=int((np.diff(sl)< -1e-12).sum())
    dt=np.diff(np.array(Wt),axis=0); out['tc_pts']+=dt.size; out['tc_viol']+=int((dt<0).sum()); out['tc_min']=float(dt.min())
    return out
def run_arb():
    surfs=[]
    for f in sorted(glob.glob(str(OUT/('ab'+TAG)/'fold*.pkl'))): surfs+=list(pickle.load(open(f,'rb'))['surf'].values())
    with Pool(NPROC) as pool: res=pool.map(check_day,surfs,chunksize=4)
    tot={}
    for r in res:
        for k_,v in r.items(): tot[k_]=min(tot.get(k_,np.inf),v) if k_ in ('c_min','tc_min') else tot.get(k_,0)+v
    lines=[f"days={len(surfs)}",
     f"BUTTERFLY (61-pt grid, visible range +-20% padding, Durrleman g by central differences h=1e-3): slices={tot['b_slices']} with any violation: all={100*tot['b_any']/tot['b_slices']:.4f}% observed={100*tot['b_obs']/tot['b_slices']:.4f}% padding={100*tot['b_pad']/tot['b_slices']:.4f}% grid pts={100*tot['b_pts']/tot['b_npts']:.5f}% undefined={tot['undef']}",
     f"CALENDAR (consecutive expiries, shared per-day k grid 121 pts): pairs={tot['c_pairs']} with any violation={100*tot['c_any']/tot['c_pairs']:.4f}% min dw={tot['c_min']:.3e}",
     f"BETWEEN MATURITIES (25 tau x 121 k per day): calendar viol {tot['tc_viol']}/{tot['tc_pts']} (min step dw {tot['tc_min']:.3e}); convexity viol {tot['tb_viol']}/{tot['tb_pts']}"]
    print('\n'.join(lines)); (OUT/f'arb_summary{TAG}.txt').write_text('\n'.join(lines))
if __name__=='__main__':
    for d in ('ab'+TAG,'c'+TAG): (OUT/d).mkdir(parents=True,exist_ok=True)
    mode=sys.argv[1]; folds=[int(x) for x in sys.argv[2:]] or list(range(18))
    {'ab':lambda: run_ab(folds),'c':lambda: run_c(folds),'arb':run_arb}[mode]()
