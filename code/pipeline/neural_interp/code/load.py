import duckdb, pandas as pd, numpy as np
R='/mnt/user-data/uploads/iv_surface/research_pipeline'
def load():
    con=duckdb.connect()
    p=con.execute(f"""select p.date,p.expiry,p.strike,p.k,p.tau,p.iv,p.volume,p.oi,h.is_heldout,vr.k_vis_min,vr.k_vis_max
      from '{R}/04_iv/panel.parquet' p join '{R}/05_splits/held_out_strikes.parquet' h using(date,expiry,strike)
      join '{R}/05_splits/visible_k_range.parquet' vr using(date,expiry) order by date,expiry,k""").df()
    p['in_range']=(p.k>=p.k_vis_min)&(p.k<=p.k_vis_max)
    return p
