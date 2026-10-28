"""Paper-rule dense-grid arbitrage check (identical check_day as fit_windows.py) over every Task A/B surface."""
import sys,glob,pickle,os; sys.path.insert(0,'/mnt/user-data/outputs/arbfree_neural/code')
import numpy as np
from multiprocessing import Pool
import fit_windows as FW
if __name__=='__main__':
    src={}
    for f in sorted(glob.glob('full_v2/fold*.pkl'))+sorted(glob.glob('/mnt/user-data/uploads/iv_surface/research_pipeline/arbfree_neural/ab/fold*.pkl')): src[os.path.basename(f)]=f
    surfs=[]
    for f in src.values(): surfs+=list(pickle.load(open(f,'rb'))['surf'].values())
    with Pool(2) as pool: res=pool.map(FW.check_day,surfs,chunksize=8)
    tot={}
    for r in res:
        for k_,v in r.items(): tot[k_]=min(tot.get(k_,np.inf),v) if k_ in ('c_min','tc_min') else tot.get(k_,0)+v
    lines=[f"folds: {sorted(src)}  days={len(surfs)}",
     f"BUTTERFLY (paper rule: 61 pts/maturity over visible range +-20% padding; Durrleman g by central differences h=1e-3): slices={tot['b_slices']}; slices with any violation: all={100*tot['b_any']/tot['b_slices']:.4f}%  observed={100*tot['b_obs']/tot['b_slices']:.4f}%  padding={100*tot['b_pad']/tot['b_slices']:.4f}%; grid points violating={100*tot['b_pts']/tot['b_npts']:.5f}% of {tot['b_npts']}; undefined w={tot['undef']}",
     f"CALENDAR (consecutive expiries, shared per-day 121-pt k grid): pairs={tot['c_pairs']}; pairs with any violation={100*tot['c_any']/tot['c_pairs']:.4f}%; min w(next)-w(prev)={tot['c_min']:.3e}",
     f"BETWEEN LISTED MATURITIES (25 tau x 121 k per day): calendar violations {tot['tc_viol']}/{tot['tc_pts']} (min step {tot['tc_min']:.3e}); price-convexity violations {tot['tb_viol']}/{tot['tb_pts']}"]
    os.makedirs('final',exist_ok=True); open('final/arb_summary.txt','w').write('\n'.join(lines)); print('\n'.join(lines))
