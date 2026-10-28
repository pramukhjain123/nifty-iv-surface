"""The full (k, tau) surface from a day's fitted slices.
Listed maturities: running max of per-slice mixture prices (butterfly- and calendar-free).
Between listed maturities: P(k,tau) = P_lo(k) + a(tau) * (P_hi(k) - P_lo(k)) with a(tau) in [0,1] nondecreasing,
chosen so ATM total variance is linear in tau ('variance-time'). Convex combination of convex call curves with a
weight nondecreasing in tau and P_hi >= P_lo  =>  still butterfly- and calendar-free at every tau."""
import numpy as np, qpmix
def _slices(S): return [qpmix.Slice(*x) for x in S['slices']]
def otm_listed(S,m,k): return qpmix.surface_otm(_slices(S),m,np.atleast_1d(k))
def otm(S,k,tau):
    k=np.atleast_1d(np.asarray(k,float)); taus=np.array(S['taus']); sl=_slices(S)
    j=int(np.searchsorted(taus,tau-1e-12))
    if j>=len(taus): raise ValueError('tau beyond last listed maturity (no extrapolation)')
    hi=qpmix.surface_otm(sl,j,k); th_hi=qpmix.implied_w(qpmix.surface_otm(sl,j,np.zeros(1)),np.zeros(1))[0]
    if abs(tau-taus[j])<1e-12: return hi
    if j==0: lo=np.zeros_like(k); th_lo=0.0; t_lo=0.0
    else: lo=qpmix.surface_otm(sl,j-1,k); th_lo=qpmix.implied_w(qpmix.surface_otm(sl,j-1,np.zeros(1)),np.zeros(1))[0]; t_lo=taus[j-1]
    lin=(tau-t_lo)/(taus[j]-t_lo); th_t=th_lo+lin*(th_hi-th_lo)
    B=lambda th: float(qpmix.black_otm(np.zeros(1),np.full(1,max(th,1e-16)))[0]) if th>0 else 0.0
    den=B(th_hi)-B(th_lo); a=lin if den<=0 else (B(th_t)-B(th_lo))/den
    a=min(max(a,0.0),1.0)
    return lo+a*(hi-lo)
def iv(S,k,tau):
    k=np.atleast_1d(np.asarray(k,float)); return np.sqrt(qpmix.implied_w(otm(S,k,tau),k)/tau)
