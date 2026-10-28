"""Same arbitrage repair applied to every method: per slice, take the completed smile (visible market quotes +
method's held-out predictions), and project normalized call prices onto the discrete no-arbitrage set
(decreasing, convex in strike, within bounds) by vega-weighted least squares (QP). Then re-score held-out IV error.
After repair every method's completed smile is butterfly-free on the strike grid by construction."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, quadprog
from scipy.stats import norm
from scipy.optimize import brentq
from load import load
df=pd.read_csv('matched.csv',parse_dates=['date','expiry'])
methods=['NeuralInterp','Akima','Linear','ThinPlate','Fengler']
p=load(); p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
vis=p[(~p.is_heldout)&(p.date>=df.date.min())][['date','expiry','k','tau','iv']]
vg=dict(tuple(vis.groupby(['date','expiry'])))
def call(k,s,tau):
    v=np.maximum(s,1e-6)*np.sqrt(tau); d1=(-k+0.5*v*v)/v; return norm.cdf(d1)-np.exp(k)*norm.cdf(d1-v)
def vega(k,s,tau):
    v=np.maximum(s,1e-6)*np.sqrt(tau); d1=(-k+0.5*v*v)/v; return norm.pdf(d1)*np.sqrt(tau)
def ivinv(c,k,tau):
    lo=max(1-np.exp(k),0)
    if c<=lo+1e-14: return np.nan
    return brentq(lambda s: call(k,s,tau)-c,1e-4,5.0)
def project(k,s,tau):
    n=len(k); x=np.exp(k); c0=call(k,s,tau); w=1/np.maximum(vega(k,s,tau),1e-4)**2
    G=np.diag(w+1e-12); a=w*c0
    C=[];b=[]
    for i in range(n-1):  # decreasing: c_i - c_{i+1} >= 0
        r=np.zeros(n); r[i]=1; r[i+1]=-1; C.append(r); b.append(0.0)
    for i in range(1,n-1):  # convex
        h1=x[i]-x[i-1]; h2=x[i+1]-x[i]; r=np.zeros(n); r[i-1]=1/h1; r[i]=-(1/h1+1/h2); r[i+1]=1/h2; C.append(r); b.append(0.0)
    lb=np.maximum(1-x,0)
    for i in range(n):
        r=np.zeros(n); r[i]=1; C.append(r); b.append(lb[i]+1e-10)
    C=np.array(C).T; b=np.array(b)
    # scale objective for numerical stability
    sc=1/np.mean(w)
    sol=quadprog.solve_qp(G*sc,a*sc,C,b)[0]
    return sol
rows=[]; fail={m:0 for m in methods}; ns=0
for (d,e),h in df.groupby(['date','expiry']):
    if h[methods].isna().any().any() or (d,e) not in vg: continue
    v=vg[(d,e)]; tau=v.tau.values[0]; ns+=1
    for m in methods:
        k=np.r_[v.k.values,h.k.values]; s=np.r_[v.iv.values,h[m].values]; isp=np.r_[np.zeros(len(v)),np.ones(len(h))].astype(bool)
        o=np.argsort(k); k,s,isp=k[o],s[o],isp[o]
        try:
            c=project(k,s,tau)
            ivr=np.array([ivinv(ci,ki,tau) for ci,ki in zip(c[isp],k[isp])])
            if np.isnan(ivr).any(): raise ValueError
        except Exception:
            fail[m]+=1; ivr=s[isp]
        hh=h.set_index('k').loc[k[isp]]
        rows.append(pd.DataFrame({'method':m,'fold':hh.fold.values,'iv':hh.iv.values,'raw':s[isp],'rep':ivr}))
R=pd.concat(rows)
print(f"slices={ns}  projection failures (kept unrepaired): {fail}")
out=[]
for m,g in R.groupby('method',sort=False):
    er=(g.raw-g.iv)*100; ep=(g.rep-g.iv)*100
    out.append(dict(method=m,n=len(g),rmse_raw=np.sqrt(np.mean(er**2)),rmse_repaired=np.sqrt(np.mean(ep**2)),mae_raw=np.mean(np.abs(er)),mae_repaired=np.mean(np.abs(ep))))
T=pd.DataFrame(out).sort_values('rmse_repaired'); print(T.to_string(index=False,float_format='%.4f')); T.to_csv('repair_summary.csv',index=False)
