import palantir
import scanpy as sc
import pandas as pd
import sys
import os
import traceback

def calc_dc(ad):
    sc.tl.pca(ad)
    components = pd.DataFrame(ad.obsm["X_pca"], index=ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
    return dc

if __name__ == "__main__":
    ad_path, out_path, lock_path, bk = sys.argv[1:5]
    tmp = out_path + ".tmp"
    
    try:
        print(f"calc dc for {ad_path}, {out_path}")
        ad = sc.read_h5ad(ad_path)
        dc = calc_dc(ad)
        
        dc.to_csv(tmp)
        os.rename(tmp, out_path)

    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        if os.path.exists(out_path):
            os.remove(out_path)
        raise

    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)
