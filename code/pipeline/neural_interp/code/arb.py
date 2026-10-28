"""Discrete static-arbitrage check of each method's completed smile.
For every test slice: visible market quotes + the method's predictions at the held-out strikes.
Normalized Black-76 call prices c(x)=Black(F=1,K=x=e^k,sigma,tau). A butterfly violation is a
consecutive strike triple (involving >=1 predicted point) where c is not convex in K (tolerance 1e-7).
'Market' = the actual held-out quotes, i.e. the arbitrage content of the real data itself."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy.stats import norm
from load import load
df=pd.read_csv('matched.csv',parse_dates=['date','expiry'])
methods=['NeuralInterp','Linear','Akima','PCHIP','ThinPlate','Fengler','CubicSpline','SVI','SSVI']
df['Market']=df.iv
methods=['Market']+methods
p=load(); p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
p=p[p.date>=df.date.min()]
vis=p[~p.is_heldout][['date','expiry','k','tau','iv']]
def call(k,s,tau):
    x=np.exp(k); v=np.maximum(s,1e-6)*np.sqrt(tau)
    d1=(-k+0.5*v*v)/v; return norm.cdf(d1)-x*norm.cdf(d1-v)
cnt={m:[0,0] for m in methods}; slc={m:[0,0] for m in methods}
vg=dict(tuple(vis.groupby(['date','expiry'])))
for (d,e),h in df.groupby(['date','expiry']):
    if h[methods].isna().any().any() or (d,e) not in vg: continue
    v=vg[(d,e)]; tau=v.tau.values[0]
    for m in methods:
        k=np.r_[v.k.values,h.k.values]; s=np.r_[v.iv.values,h[m].values]; isp=np.r_[np.zeros(len(v)),np.ones(len(h))].astype(bool)
        o=np.argsort(k); k,s,isp=k[o],s[o],isp[o]
        c=call(k,s,tau); x=np.exp(k)
        x1,x2,x3=x[:-2],x[1:-1],x[2:]; c1,c2,c3=c[:-2],c[1:-1],c[2:]
        chord=((x3-x2)*c1+(x2-x1)*c3)/(x3-x1)
        viol=c2>chord+1e-7
        inv=isp[:-2]|isp[1:-1]|isp[2:]
        cnt[m][0]+=int((viol&inv).sum()); cnt[m][1]+=int(inv.sum())
        slc[m][0]+=int((viol&inv).any()); slc[m][1]+=1
out=pd.DataFrame([dict(method=m,butterfly_triples_viol_pct=100*cnt[m][0]/cnt[m][1],slices_with_any_viol_pct=100*slc[m][0]/slc[m][1],n_slices=slc[m][1]) for m in methods])
print(out.to_string(index=False,float_format='%.2f')); out.to_csv('arb_summary.csv',index=False)
