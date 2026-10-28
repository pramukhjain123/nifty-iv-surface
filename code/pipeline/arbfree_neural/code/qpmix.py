"""Butterfly-free slice by construction: lognormal-kernel mixture density with non-negative weights.
Y = mu_I * LN(1, v),  mu_i = exp(z_i) on a fixed grid,  pi >= 0, sum pi = 1, sum pi*mu = 1 (forward / martingale).
OTM price(k) = sum_i pi_i * mu_i * Black_side(k - z_i, v)  -- LINEAR in pi -> one small QP per slice, global optimum.
Any feasible pi gives a genuine positive density -> call prices convex & within bounds -> g(k) > 0 for all k."""
import numpy as np, quadprog
from scipy.special import ndtr
from scipy.stats import norm
def call_put(kk,v):
    sv=np.sqrt(v); d1=(-kk+0.5*v)/sv; d2=d1-sv
    co=ndtr(d1)-np.exp(kk)*ndtr(d2); po=np.exp(kk)*ndtr(-d2)-ndtr(-d1)
    call=np.where(kk>=0,co,po+1-np.exp(kk)); put=np.where(kk<0,po,co-1+np.exp(kk))
    return call,put
def black_otm(k,v):
    c,p=call_put(k,v); return np.where(k>=0,c,p)
def design(k,z,v):
    """A[j,i] = mu_i * OTM-side price of component i at k_j  (v scalar or per-atom array)"""
    kk=k[:,None]-z[None,:]; c,p=call_put(kk,np.broadcast_to(np.atleast_1d(v)[None,:] if np.ndim(v) else v,kk.shape))
    return np.where(k[:,None]>=0,c,p)*np.exp(z)[None,:]
def implied_w(P,k,lo=1e-10,hi=8.0,it=100):
    a=np.full(P.shape,np.log(lo)); b=np.full(P.shape,np.log(hi))
    for _ in range(it):
        mid=0.5*(a+b); f=black_otm(k,np.exp(mid))-P; a=np.where(f>0,a,mid); b=np.where(f>0,mid,b)
    return np.exp(0.5*(a+b))
class Slice:
    def __init__(self,z,v,pi): self.z,self.v,self.pi=z,v,pi
    def otm(self,k): return design(np.atleast_1d(k),self.z,self.v)@self.pi
def fit_slice(k,iv,tau,wt=None,kpad=(None,None),n_atoms=40,kern=1.0,lam=1e-3,vega_floor=1e-3,margin=4.0,scales=None,upper=None):
    """k, iv: quotes to fit (visible [+pseudo]); wt: extra weights. Returns Slice."""
    k=np.asarray(k,float); iv=np.asarray(iv,float); n=len(k)
    wt=np.ones(n) if wt is None else np.asarray(wt,float)
    i0=np.argmin(np.abs(k)); s=iv[i0]*np.sqrt(tau)                 # ATM total std
    lo=min(k.min(),kpad[0] if kpad[0] is not None else k.min())-margin*s
    hi=max(k.max(),kpad[1] if kpad[1] is not None else k.max())+margin*s
    z0=np.linspace(lo,hi,n_atoms); dz=z0[1]-z0[0]
    widths=[kern] if scales is None else [kern*f for f in scales]
    z=np.concatenate([z0]*len(widths)); v=np.concatenate([np.full(n_atoms,max((w_*dz)**2,1e-8)) for w_ in widths])
    nA=len(z)
    A=design(k,z,v)
    vv=iv**2*tau; P=black_otm(k,vv); vega=norm.pdf((-k+0.5*vv)/np.sqrt(vv))*np.sqrt(tau)
    W=wt/np.maximum(vega,vega_floor)**2
    G=(A*W[:,None]).T@A; a=(A*W[:,None]).T@P
    D2b=np.diff(np.eye(n_atoms),2,axis=0); D2=np.kron(np.eye(len(widths)),D2b); scale=np.trace(G)/nA
    G=G+lam*scale*(D2.T@D2)+1e-10*scale*np.eye(nA)
    mu=np.exp(z)
    C=np.vstack([np.ones(nA),mu,np.eye(nA)]).T; b=np.r_[1.0,1.0,np.zeros(nA)]
    if upper is not None:                       # calendar sweep: this slice's OTM price <= next-longer slice's, on a grid
        kg,ug=upper; Ag=design(kg,z,v); rs=1.0/np.maximum(ug,1e-12)  # row-scale for conditioning
        C=np.hstack([C,(-Ag*rs[:,None]).T]); b=np.r_[b,-ug*rs]
    sc=1.0/np.max(np.abs(G))
    pi=None
    for Cc,bb in ((C,b),(C[:,:nA+2],b[:nA+2])):  # fall back to no sweep constraint if infeasible
        try:
            pi=quadprog.solve_qp(G*sc,a*sc,Cc,bb,meq=2)[0]; break
        except ValueError:
            try:
                G2=G+1e-6*scale*np.eye(nA); s2=1.0/np.max(np.abs(G2)); pi=quadprog.solve_qp(G2*s2,a*s2,Cc,bb,meq=2)[0]; break
            except ValueError: continue
    pi=np.maximum(pi,0); pi=_renorm(pi,mu)
    return Slice(z,v,pi)
def _renorm(pi,mu):
    """restore the two equality constraints exactly after clipping tiny negatives (keeps pi>=0)"""
    # solve for a,b>=... simple: scale to sum 1 then fix mean by moving mass between the two extreme atoms
    pi=pi/pi.sum(); err=pi@mu-1.0
    if abs(err)>1e-12:
        j=np.argmax(mu) if err<0 else np.argmin(mu); i=np.argmin(mu) if err<0 else np.argmax(mu)
        # move mass t from i to j: change in mean = t*(mu_j-mu_i) = -err
        t=-err/(mu[j]-mu[i]); t=min(max(t,0),pi[i]); pi[i]-=t; pi[j]+=t
    return pi
def surface_otm(slices,m,k):
    """running max over listed expiries j<=m of slice OTM prices (calendar-free by construction)"""
    return np.max(np.stack([slices[j].otm(k) for j in range(m+1)]),axis=0)

def fit_day(slices_data, cfg, sweep=True, grid_n=150):
    """slices_data: list (sorted by maturity) of dicts with k, iv, tau, kpad[, wt]. Longest first, each shorter
    constrained below the next-longer on a dense common k grid. Returns list of Slice (by maturity)."""
    M=len(slices_data); out=[None]*M
    klo=min(s['kpad'][0] for s in slices_data); khi=max(s['kpad'][1] for s in slices_data)
    kg=np.linspace(klo-0.05,khi+0.05,grid_n)
    for j in range(M-1,-1,-1):
        s=slices_data[j]; up=None
        if sweep and j<M-1: up=(kg,out[j+1].otm(kg))
        out[j]=fit_slice(s['k'],s['iv'],s['tau'],wt=s.get('wt'),kpad=s['kpad'],upper=up,**cfg)
    return out

def fit_day_joint(slices_data, n_atoms=60, scales=(1,4,16), kern=1.0, lam=1e-6, vega_floor=5e-3, margin=4.0, grid_n=100, calendar=True):
    """One QP per day: all maturities jointly, common atom grid, calendar ordering OTM_j(k_g) <= OTM_{j+1}(k_g) on a grid.
    Each slice remains a genuine mixture density (butterfly-free); the running max in surface_otm guarantees
    calendar everywhere (between grid points too)."""
    M=len(slices_data)
    klo=min(min(s['k'].min(),s['kpad'][0]) for s in slices_data); khi=max(max(s['k'].max(),s['kpad'][1]) for s in slices_data)
    Z=[];Vv=[];MU=[]
    for s in slices_data:                      # per-slice atom grid sized to that maturity
        o=np.argsort(s['k']); sj=np.interp(0,s['k'][o],s['iv'][o])*np.sqrt(s['tau'])
        lo=min(s['k'].min(),s['kpad'][0])-margin*sj; hi=max(s['k'].max(),s['kpad'][1])+margin*sj
        z0=np.linspace(lo,hi,n_atoms); dz=z0[1]-z0[0]
        Z.append(np.concatenate([z0]*len(scales))); Vv.append(np.concatenate([np.full(n_atoms,max((kern*f*dz)**2,1e-8)) for f in scales])); MU.append(np.exp(Z[-1]))
    nA=len(Z[0]); smax=max(np.sqrt(np.max(v)) for v in Vv)
    D2b=np.diff(np.eye(n_atoms),2,axis=0); D2=np.kron(np.eye(len(scales)),D2b)
    N=M*nA; G=np.zeros((N,N)); a=np.zeros(N)
    for j,s in enumerate(slices_data):
        z=Z[j]; v=Vv[j]
        k=s['k']; iv=s['iv']; tau=s['tau']; wt=s.get('wt',np.ones(len(k)))
        A=design(k,z,v); vv=iv**2*tau; P=black_otm(k,vv); vega=norm.pdf((-k+0.5*vv)/np.sqrt(vv))*np.sqrt(tau)
        W=wt/np.maximum(vega,vega_floor)**2/len(k)
        Gj=(A*W[:,None]).T@A; sc=np.trace(Gj)/nA
        Gj=Gj+lam*sc*(D2.T@D2)+1e-10*sc*np.eye(nA)
        G[j*nA:(j+1)*nA,j*nA:(j+1)*nA]=Gj; a[j*nA:(j+1)*nA]=(A*W[:,None]).T@P
    eqs=[];beq=[]
    for j in range(M):
        r=np.zeros(N); r[j*nA:(j+1)*nA]=1; eqs.append(r); beq.append(1.0)
        r=np.zeros(N); r[j*nA:(j+1)*nA]=MU[j]; eqs.append(r); beq.append(1.0)
    ineq=[np.eye(N)]; bin_=[np.zeros(N)]
    if calendar and M>1:
        kg=np.linspace(klo,khi,grid_n); Ags=[design(kg,Z[j],Vv[j]) for j in range(M)]
        for j in range(M-1):
            ref=np.maximum(Ags[j+1].sum(1)/nA,1e-10); rs=1.0/np.maximum(np.abs(Ags[j+1]).max(1),1e-12)
            R=np.zeros((grid_n,N)); R[:,(j+1)*nA:(j+2)*nA]=Ags[j+1]*rs[:,None]; R[:,j*nA:(j+1)*nA]=-Ags[j]*rs[:,None]
            ineq.append(R); bin_.append(np.zeros(grid_n))
    C=np.vstack(eqs+ineq).T; b=np.r_[beq,np.concatenate(bin_)]
    s_=1.0/np.max(np.abs(G)); x=None
    for Cc,bb in ((C,b),(C[:,:2*M+N],b[:2*M+N])):
        try: x=quadprog.solve_qp(G*s_,a*s_,Cc,bb,meq=2*M)[0]; break
        except ValueError:
            try:
                G2=G+1e-7*np.eye(N)*np.mean(np.diag(G)); s2=1.0/np.max(np.abs(G2)); x=quadprog.solve_qp(G2*s2,a*s2,Cc,bb,meq=2*M)[0]; break
            except ValueError: continue
    out=[]
    for j in range(M):
        pi=np.maximum(x[j*nA:(j+1)*nA],0); out.append(Slice(Z[j],Vv[j],_renorm(pi,MU[j])))
    return out

def slice_dictionary(s, n_fine=32, fine_half=4.0, fine_scales=(1,4), n_coarse=32, coarse_scales=(1,4,16), margin=4.0, kern=1.0):
    """two-resolution atom dictionary for one slice: fine atoms (spacing ~ ATM std/4) within +-fine_half ATM std of
    the forward, coarse atoms across the full padded range. Returns z, v, list of (start, n) blocks for smoothing."""
    o=np.argsort(s['k']); sj=np.interp(0,s['k'][o],s['iv'][o])*np.sqrt(s['tau'])
    lo=min(s['k'].min(),s['kpad'][0])-margin*sj; hi=max(s['k'].max(),s['kpad'][1])+margin*sj
    zs=[];vs=[];blocks=[];start=0
    for (a,b,n,scs) in ((-fine_half*sj,fine_half*sj,n_fine,fine_scales),(lo,hi,n_coarse,coarse_scales)):
        z0=np.linspace(a,b,n); dz=z0[1]-z0[0]
        for f in scs:
            zs.append(z0); vs.append(np.full(n,max((kern*f*dz)**2,1e-10))); blocks.append((start,n)); start+=n
    return np.concatenate(zs),np.concatenate(vs),blocks
def fit_day_joint2(slices_data, lam=1e-7, vega_floor=5e-3, grid_n=100, calendar=True, **dict_kw):
    """joint per-day QP (as fit_day_joint) on the two-resolution dictionary"""
    M=len(slices_data)
    klo=min(min(s['k'].min(),s['kpad'][0]) for s in slices_data); khi=max(max(s['k'].max(),s['kpad'][1]) for s in slices_data)
    Z=[];Vv=[];B=[]
    for s in slices_data:
        z,v,bl=slice_dictionary(s,**dict_kw); Z.append(z); Vv.append(v); B.append(bl)
    nA=len(Z[0]); N=M*nA; G=np.zeros((N,N)); a=np.zeros(N)
    D2=np.zeros((0,nA))
    for st,n in B[0]:
        blk=np.zeros((n-2,nA)); blk[:,st:st+n]=np.diff(np.eye(n),2,axis=0); D2=np.vstack([D2,blk])
    for j,s in enumerate(slices_data):
        z=Z[j]; v=Vv[j]; k=s['k']; iv=s['iv']; tau=s['tau']; wt=s.get('wt',np.ones(len(k)))
        A=design(k,z,v); vv=iv**2*tau; P=black_otm(k,vv); vega=norm.pdf((-k+0.5*vv)/np.sqrt(vv))*np.sqrt(tau)
        W=wt/np.maximum(vega,vega_floor)**2/len(k)
        Gj=(A*W[:,None]).T@A; sc=np.trace(Gj)/nA
        G[j*nA:(j+1)*nA,j*nA:(j+1)*nA]=Gj+lam*sc*(D2.T@D2)+1e-10*sc*np.eye(nA); a[j*nA:(j+1)*nA]=(A*W[:,None]).T@P
    eqs=[];beq=[]
    for j in range(M):
        r=np.zeros(N); r[j*nA:(j+1)*nA]=1; eqs.append(r); beq.append(1.0)
        r=np.zeros(N); r[j*nA:(j+1)*nA]=np.exp(Z[j]); eqs.append(r); beq.append(1.0)
    ineq=[np.eye(N)]; bin_=[np.zeros(N)]
    if calendar and M>1:
        kg=np.linspace(klo,khi,grid_n); Ags=[design(kg,Z[j],Vv[j]) for j in range(M)]
        for j in range(M-1):
            rs=1.0/np.maximum(np.abs(Ags[j+1]).max(1),1e-12)
            R=np.zeros((grid_n,N)); R[:,(j+1)*nA:(j+2)*nA]=Ags[j+1]*rs[:,None]; R[:,j*nA:(j+1)*nA]=-Ags[j]*rs[:,None]
            ineq.append(R); bin_.append(np.zeros(grid_n))
    C=np.vstack(eqs+ineq).T; b=np.r_[beq,np.concatenate(bin_)]
    s_=1.0/np.max(np.abs(G)); x=None
    for Cc,bb in ((C,b),(C[:,:2*M+N],b[:2*M+N])):
        try: x=quadprog.solve_qp(G*s_,a*s_,Cc,bb,meq=2*M)[0]; break
        except ValueError:
            try:
                G2=G+1e-7*np.eye(N)*np.mean(np.diag(G)); s2=1.0/np.max(np.abs(G2)); x=quadprog.solve_qp(G2*s2,a*s2,Cc,bb,meq=2*M)[0]; break
            except ValueError: continue
    return [Slice(Z[j],Vv[j],_renorm(np.maximum(x[j*nA:(j+1)*nA],0),np.exp(Z[j]))) for j in range(M)]
