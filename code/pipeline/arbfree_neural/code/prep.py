import duckdb, numpy as np, pandas as pd
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
def load():
    con=duckdb.connect()
    p=con.execute(f"""select p.date,p.expiry,p.strike,p.k,p.tau,p.iv,h.is_heldout,vr.k_vis_min,vr.k_vis_max
      from '{R}/04_iv/panel.parquet' p join '{R}/05_splits/held_out_strikes.parquet' h using(date,expiry,strike)
      join '{R}/05_splits/visible_k_range.parquet' vr using(date,expiry) order by date,expiry,k""").df()
    p['date']=pd.to_datetime(p.date); p['expiry']=pd.to_datetime(p.expiry)
    p['in_range']=(p.k>=p.k_vis_min)&(p.k<=p.k_vis_max)
    return p
def pack(p, extra=None):
    """pad per day. visible quotes for fitting; returns dict of arrays + per-day expiry tau table"""
    days=sorted(p.date.unique()); D=len(days)
    exps=p.groupby('date').expiry.unique()
    M=max(len(e) for e in exps)
    vis=p[~p.is_heldout]
    Q=vis.groupby('date').size().max()
    k=np.zeros((D,Q)); tau=np.ones((D,Q))*0.1; m=np.zeros((D,Q),int); iv=np.ones((D,Q))*0.2; mask=np.zeros((D,Q))
    taus=np.ones((D,M)); nM=np.zeros(D,int); exp_index={}
    for d,day in enumerate(days):
        e=sorted(exps[day]); g=p[p.date==day]
        tl=[g[g.expiry==x].tau.iloc[0] for x in e]
        taus[d,:len(e)]=tl; taus[d,len(e):]=tl[-1]+np.arange(1,M-len(e)+1)*1e-3; nM[d]=len(e)
        for j,x in enumerate(e): exp_index[(day,x)]=(d,j)
        v=g[~g.is_heldout]; n=len(v)
        k[d,:n]=v.k.values; tau[d,:n]=v.tau.values; iv[d,:n]=v.iv.values; mask[d,:n]=1
        m[d,:n]=[exp_index[(day,x)][1] for x in v.expiry]
    return dict(days=days,k=k,tau=tau,m=m,iv=iv,mask=mask,taus=taus,nM=nM,exp_index=exp_index)
