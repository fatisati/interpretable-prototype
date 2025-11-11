import palantir
import scanpy as sc
import pandas as pd
import sys
import os

def calc_dc(ad):
    sc.tl.pca(ad)
    components = pd.DataFrame(ad.obsm["X_pca"], index=ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
    return dc

if __name__ == "__main__":
    ad_path, save_path, lock_path, bk = sys.argv[1:5]
    print(f"calc dc for {ad_path}, {save_path}")
    ad = sc.read_h5ad(ad_path)
    dc = calc_dc(ad)
    dc.to_csv(save_path)
    print("done")
    try:
        os.remove(ad_path)
        os.remove(lock_path)
        print('both files removed')
    except FileNotFoundError:
        print(f'couldnt remove files, {ad_path}, {lock_path}')
