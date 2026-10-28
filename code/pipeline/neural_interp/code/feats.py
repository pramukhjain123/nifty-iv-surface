import numpy as np, pandas as pd
M=4  # neighbors per side
def _feat(kc, ivc, kq, iL, iR, tau, atm):
    n=len(kc); kL=kc[iL]; kR=kc[iR]; h=kR-kL
    t=(kq-kL)/h
    lin=ivc[iL]+(ivc[iR]-ivc[iL])*t
    cols=[]
    for side,start,step in (('L',iL,-1),('R',iR,1)):
        for m in range(M):
            idx=start+step*m
            ok=(idx>=0)&(idx<n); ic=np.clip(idx,0,n-1)
            dk=np.where(ok,(kc[ic]-kq)/h,0.0)
            dv=np.where(ok,(ivc[ic]-lin)*100,0.0)
            cols += [dk/10.0, dv, ok.astype(float)]
    sq=np.sqrt(tau)
    cols += [t, h/sq*10, np.full_like(kq,sq), kq/sq, np.full_like(kq,atm*10), (ivc[iR]-ivc[iL])*100]
    return np.stack(cols,1).astype(np.float32), lin
def slice_examples(g, mode):
    """g: one (date,expiry) slice sorted by k. mode 'loo' (visible-only targets) or 'ho' (held-out in-range targets)."""
    v=g[~g.is_heldout.values]; kc=v.k.values; ivc=v.iv.values; n=len(kc)
    if n<3: return None
    tau=g.tau.values[0]; atm=ivc[np.argmin(np.abs(kc))]
    if mode=='loo':
        j=np.arange(1,n-1); iL=j-1; iR=j+1; kq=kc[j]; y=ivc[j]; idx=v.index.values[j]
    else:
        h=g[g.is_heldout.values & g.in_range.values]
        if len(h)==0: return None
        kq=h.k.values; y=h.iv.values; idx=h.index.values
        iR=np.searchsorted(kc,kq,side='left'); iR=np.clip(iR,1,n-1); iL=iR-1
    X,lin=_feat(kc,ivc,kq,iL,iR,tau,atm)
    return X, lin, y, idx
def build(p, mode):
    Xs=[];L=[];Y=[];I=[]
    for _,g in p.groupby(['date','expiry'],sort=False):
        r=slice_examples(g,mode)
        if r is None: continue
        Xs.append(r[0]);L.append(r[1]);Y.append(r[2]);I.append(r[3])
    return np.concatenate(Xs),np.concatenate(L),np.concatenate(Y),np.concatenate(I)
