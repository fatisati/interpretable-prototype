import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np
from sklearn.neighbors import NearestNeighbors


def spatial_compactness(ad, spatial_key="spatial", mc_key="SEACell", bk="batch"):
    if spatial_key not in ad.obsm:
        return None

    X = ad.obsm[spatial_key]
    mc = ad.obs[mc_key].astype(str).values
    b = ad.obs[bk].values
    mcs = ad.obs[mc_key].astype(str).unique()

    vals = []
    for m in mcs:
        idx = np.where(mc == m)[0]
        if len(idx) == 1:
            vals.append(1.0)
            continue

        coords = X[idx]
        batches = b[idx]
        coh = []
        for u in np.unique(batches):
            cu = coords[batches == u]
            if len(cu) == 1:
                coh.append(1.0)
            else:
                d = pdist(cu)
                coh.extend(np.exp(-d))

        vals.append(np.mean(coh) if coh else 1.0)

    return pd.Series(vals, index=mcs, name="spatial_compactness")



def calc_purity(df, label_key, mc_key="SEACell", return_per_mc=False):
    if label_key not in df.columns:
        return None

    groups = df.groupby(mc_key)
    mcs = groups.groups.keys()

    per_mc = [
        sub[label_key].value_counts(normalize=True).max()
        for _, sub in groups
    ]

    if return_per_mc:
        return pd.Series(per_mc, index=pd.Index(mcs, dtype=str), name="purity")

    return float(np.mean(per_mc))



def compute_mc_compactness_and_separation(X, mc_ids, batches):
    compact_vals = []
    for mc in np.unique(mc_ids):
        cell_idx = np.where(mc_ids == mc)[0]
        X_mc = X[cell_idx]
        b_mc = batches[cell_idx]
        batch_means = {u: X_mc[b_mc == u].mean(0) for u in np.unique(b_mc)}
        X_centered = np.vstack([X_mc[b_mc == u] - batch_means[u] for u in batch_means])
        compact_vals.append((X_centered**2).sum() / len(X_centered))
    compactness = np.array(compact_vals)

    batch_means_global = {u: X[batches == u].mean(0) for u in np.unique(batches)}
    X_global = np.zeros_like(X)
    for u in batch_means_global:
        X_global[batches == u] = X[batches == u] - batch_means_global[u]

    mc_centroids = {mc: X_global[mc_ids == mc].mean(0) for mc in np.unique(mc_ids)}
    C = np.vstack(list(mc_centroids.values()))
    nn = NearestNeighbors(n_neighbors=2).fit(C)
    dists, _ = nn.kneighbors(C)
    separation = dists[:, 1]

    return compactness, separation



