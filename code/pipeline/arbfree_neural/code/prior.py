"""Neural prior: learned interior interpolator + learned edge extrapolator, trained on VISIBLE strikes only
(leave-out of visible points), used to generate pseudo-quotes at generic locations (midpoints between visible
strikes, and 1-2 strike spacings beyond each edge). Never sees held-out strikes."""
import numpy as np, pandas as pd, mlp
M=4
def interior_feats(kc,ivc,kq,iL,iR,tau,atm):
    n=len(kc); kL=kc[iL]; kR=kc[iR]; h=kR-kL; t=(kq-kL)/h; lin=ivc[iL]+(ivc[iR]-ivc[iL])*t; cols=[]
    for start,step in ((iL,-1),(iR,1)):
        for m in range(M):
            idx=start+step*m; ok=(idx>=0)&(idx<n); ic=np.clip(idx,0,n-1)
            cols+=[np.where(ok,(kc[ic]-kq)/h,0)/10,np.where(ok,(ivc[ic]-lin)*100,0),ok.astype(float)]
    sq=np.sqrt(tau); cols+=[t,h/sq*10,np.full_like(kq,sq),kq/sq,np.full_like(kq,atm*10),(ivc[iR]-ivc[iL])*100]
    return np.stack(cols,1).astype(np.float32),lin
def exterior_feats(kc,ivc,kq,side,tau,atm):
    """kc sorted visible (remaining); side=+1 right edge, -1 left edge; kq beyond the edge"""
    n=len(kc); e=n-1 if side>0 else 0; ke=kc[e]; ive=ivc[e]
    h=abs(kc[e]-kc[e-side]); u=side*(kq-ke)/h; cols=[]
    for m in range(5):
        idx=e-side*m; ok=(idx>=0)&(idx<n); ic=np.clip(idx,0,n-1)
        cols+=[np.full_like(kq,np.where(ok,side*(kc[ic]-ke)/h,0)/10),np.full_like(kq,np.where(ok,(ivc[ic]-ive)*100,0)),np.full_like(kq,float(ok))]
    sq=np.sqrt(tau); cols+=[u,np.full_like(kq,h/sq*10),np.full_like(kq,sq),kq/sq,np.full_like(kq,atm*10),np.full_like(kq,float(side))]
    return np.stack(cols,1).astype(np.float32),np.full_like(kq,ive)
def _slices(df):
    for (d,e),g in df.groupby(['date','expiry'],sort=False):
        v=g[~g.is_heldout]
        if len(v)>=3: yield d,e,g.tau.iloc[0],v.k.values,v.iv.values
def build_train(df):
    XI=[];YI=[];XE=[];YE=[]
    for d,e,tau,kc,ivc in _slices(df):
        n=len(kc); atm=ivc[np.argmin(np.abs(kc))]
        j=np.arange(1,n-1); X,lin=interior_feats(kc,ivc,kc[j],j-1,j+1,tau,atm); XI.append(X); YI.append((ivc[j]-lin)*100)
        if n>=6:
            for side in (1,-1):
                for mrem in (1,2):
                    if side>0: kr,ir=kc[:n-mrem],ivc[:n-mrem]; qk,qv=kc[n-mrem:],ivc[n-mrem:]
                    else: kr,ir=kc[mrem:],ivc[mrem:]; qk,qv=kc[:mrem],ivc[:mrem]
                    X,base=exterior_feats(kr,ir,qk,side,tau,ir[np.argmin(np.abs(kr))]); XE.append(X); YE.append((qv-base)*100)
    return np.concatenate(XI),np.concatenate(YI),np.concatenate(XE),np.concatenate(YE)
def train(df,seeds=(0,1)):
    XI,YI,XE,YE=build_train(df)
    si=int(min(6000,max(1500,round(len(XI)*37/2048)))); se=int(min(6000,max(1500,round(len(XE)*37/2048))))
    mi=[mlp.train(XI,YI,steps=si,hidden=(256,256,128),wd=1e-3,loss='huber',verbose=False,seed=s) for s in seeds]
    me=[mlp.train(XE,YE,steps=se,hidden=(128,128),wd=1e-3,loss='huber',verbose=False,seed=s) for s in seeds]
    return mi,me
def pseudo_for_slice(models,tau,kc,ivc,n_out=2,klim=(-0.35,0.25)):
    """pseudo-quotes: midpoints between visible strikes + n_out spacings beyond each edge"""
    mi,me=models; n=len(kc); atm=ivc[np.argmin(np.abs(kc))]; K=[];V=[];T=[]
    if n>=2:
        kq=0.5*(kc[:-1]+kc[1:]); X,lin=interior_feats(kc,ivc,kq,np.arange(n-1),np.arange(1,n),tau,atm)
        K.append(kq); V.append(lin+np.mean([mlp.predict(m,X) for m in mi],0)/100); T.append(np.zeros(len(kq)))
    if n>=3:
        for side in (1,-1):
            e=n-1 if side>0 else 0; h=abs(kc[e]-kc[e-side])
            kq=kc[e]+side*h*np.arange(1,n_out+1); kq=kq[(kq>=klim[0])&(kq<=klim[1])]
            if len(kq)==0: continue
            X,base=exterior_feats(kc,ivc,kq,side,tau,atm)
            K.append(kq); V.append(base+np.mean([mlp.predict(m,X) for m in me],0)/100); T.append(np.ones(len(kq)))
    if not K: return np.array([]),np.array([]),np.array([])
    return np.concatenate(K),np.maximum(np.concatenate(V),0.01),np.concatenate(T)
