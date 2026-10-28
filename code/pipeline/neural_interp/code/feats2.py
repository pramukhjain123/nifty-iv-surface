import numpy as np, pandas as pd
M=4
def _lin(kc,vc,x):
    return np.interp(x,kc,vc,left=np.nan,right=np.nan)
def adj_feats(adj, zq, zL, zR):
    """adj: (z_vis, iv_vis) of another expiry same day. returns curvature residual at zq relative to chord zL-zR, plus ok flag"""
    if adj is None: z=np.zeros_like(zq); return [z,z,z]
    za,va=adj
    a_q=_lin(za,va,zq); a_L=_lin(za,va,zL); a_R=_lin(za,va,zR)
    t=(zq-zL)/np.maximum(zR-zL,1e-9)
    r=(a_q-(a_L+(a_R-a_L)*t))*100
    ok=np.isfinite(r)
    # also local slope of adjacent smile (vol pts per unit z)
    s=(a_R-a_L)*100
    return [np.where(ok,r,0.0), np.where(ok,s,0.0), ok.astype(float)]
def _feat(kc, ivc, liq, kq, iL, iR, tau, atm, adjs):
    n=len(kc); kL=kc[iL]; kR=kc[iR]; h=kR-kL
    t=(kq-kL)/h
    lin=ivc[iL]+(ivc[iR]-ivc[iL])*t
    cols=[]
    for side,start,step in (('L',iL,-1),('R',iR,1)):
        for m in range(M):
            idx=start+step*m
            ok=(idx>=0)&(idx<n); ic=np.clip(idx,0,n-1)
            cols += [np.where(ok,(kc[ic]-kq)/h,0.0)/10.0, np.where(ok,(ivc[ic]-lin)*100,0.0), ok.astype(float)]
            if m<2: cols += [np.where(ok,liq[0][ic],0.0), np.where(ok,liq[1][ic],0.0)]
    sq=np.sqrt(tau)
    cols += [t, h/sq*10, np.full_like(kq,sq), kq/sq, np.full_like(kq,atm*10), (ivc[iR]-ivc[iL])*100]
    zq,zL,zR=kq/sq,kL/sq,kR/sq
    for a in adjs: cols += adj_feats(a,zq,zL,zR)
    return np.stack(cols,1).astype(np.float32), lin
def day_examples(gd, mode):
    """gd: all slices of one date. returns list of (X,lin,y,idx)"""
    vis={}
    slices=[]
    for e,g in gd.groupby('expiry',sort=True):
        v=g[~g.is_heldout.values]
        tau=g.tau.values[0]
        vis[e]=(v.k.values/np.sqrt(tau), v.iv.values) if len(v)>=3 else None
        slices.append((e,tau,g))
    out=[]
    for i,(e,tau,g) in enumerate(slices):
        v=g[~g.is_heldout.values]; kc=v.k.values; ivc=v.iv.values; n=len(kc)
        if n<3: continue
        liq=(np.log1p(v.volume.values)/10, np.log1p(v.oi.values)/10)
        atm=ivc[np.argmin(np.abs(kc))]
        adjs=[vis[slices[i-1][0]] if i>0 else None, vis[slices[i+1][0]] if i+1<len(slices) else None]
        if mode=='loo':
            j=np.arange(1,n-1); iL=j-1; iR=j+1; kq=kc[j]; y=ivc[j]; idx=v.index.values[j]
        else:
            h=g[g.is_heldout.values & g.in_range.values]
            if len(h)==0: continue
            kq=h.k.values; y=h.iv.values; idx=h.index.values
            iR=np.clip(np.searchsorted(kc,kq,side='left'),1,n-1); iL=iR-1
        X,lin=_feat(kc,ivc,liq,kq,iL,iR,tau,atm,adjs)
        out.append((X,lin,y,idx))
    return out
def build(p, mode):
    R=[]
    for _,gd in p.groupby('date',sort=False): R+=day_examples(gd,mode)
    return tuple(np.concatenate([r[i] for r in R]) for i in range(4))
