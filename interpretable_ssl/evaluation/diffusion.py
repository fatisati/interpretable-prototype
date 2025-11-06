import palantir
import scanpy as sc
import pandas as pd
import sys
import os

def calc_dc(ad, bk):
    dfs = []
    for b, sub in ad.obs.groupby(bk):
        sub_ad = ad[ad.obs[bk] == b].copy()
        sc.tl.pca(sub_ad)
        components = pd.DataFrame(sub_ad.obsm["X_pca"], index=sub_ad.obs_names)
        dm_res = palantir.utils.run_diffusion_maps(components)
        dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
        dfs.append(dc)
    return pd.concat(dfs)

if __name__ == "__main__":
    ad_path, save_path, lock_path, bk = sys.argv[1:5]
    print(f"calc dc for {ad_path}, {save_path}")
    ad = sc.read_h5ad(ad_path)
    dc = calc_dc(ad, bk)
    dc.to_csv(save_path)
    print("done")
    try:
        os.remove(ad_path)
        os.remove(lock_path)
        print('both files removed')
    except FileNotFoundError:
        print(f'couldnt remove files, {ad_path}, {lock_path}')
