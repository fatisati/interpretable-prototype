import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np
from sklearn.neighbors import NearestNeighbors
from interpretable_ssl.evaluation.de_helper import *
from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import *
import time
import subprocess
from interpretable_ssl.evaluation.dropout_recovery import *
from interpretable_ssl.evaluation.nsc import *
import os, sys, uuid
from interpretable_ssl.evaluation.niche_recovery import *

spatial_labels = ["niches_2D", "niches_3D", "fibroblast_subclusters", "EMT_niche"]


def spatial_compactness(
    ad, spatial_key="spatial", mc_key="SEACell", bk="batch", return_size=False
):

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
                sigma = np.median(d) + 1e-9
                coh.extend(np.exp(-(d**2) / (2 * sigma**2)))

        vals.append(np.mean(coh) if coh else 1.0)

    comp = pd.Series(vals, index=mcs, name="spatial_compactness")

    if not return_size:
        return comp

    # compute sizes
    sizes = pd.Series([np.sum(mc == m) for m in mcs], index=mcs, name="size")

    # return DataFrame
    return pd.DataFrame({"spatial_compactness": comp, "size": sizes})


def compute_aff_dc_compactness(aff, mc_ids, batches):
    """Compute per-(metacell, batch) compactness in diffusion component space.

    aff: scipy sparse CSR affinity matrix (n_cells x n_cells), diagonal=0 is fine —
         we set it to 1 internally before running diffusion maps.
    mc_ids: array of metacell assignment per cell
    batches: array of batch label per cell

    Returns
    -------
    comp_df : pd.DataFrame  shape (n_metacells, n_batches)
        compactness[m, b] = mean squared dist of cells in (metacell m, batch b)
        from their centroid. NaN if fewer than 2 cells.
    counts_df : pd.DataFrame  shape (n_metacells, n_batches)
        number of cells per (metacell, batch). 0 if absent.
    """
    import palantir

    K = aff.tocsr(copy=True)
    K.setdiag(1)

    dm_res = palantir.utils.diffusion_maps_from_kernel(K, n_components=10)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
    X = dc.values

    unique_mcs = np.unique(mc_ids)
    unique_batches = np.unique(batches)

    comp_rows = {}
    count_rows = {}
    for mc in unique_mcs:
        comp_row = {}
        count_row = {}
        for b in unique_batches:
            idx = np.where((mc_ids == mc) & (batches == b))[0]
            count_row[b] = len(idx)
            if len(idx) == 0:
                comp_row[b] = np.nan       # batch absent from metacell — exclude
            elif len(idx) == 1:
                comp_row[b] = 0.0          # single cell: zero variance, CVAE-aligned
            else:
                X_mb = X[idx]
                mu = X_mb.mean(0)
                comp_row[b] = float(((X_mb - mu) ** 2).sum() / len(X_mb))
        comp_rows[mc] = comp_row
        count_rows[mc] = count_row

    comp_df = pd.DataFrame(comp_rows, dtype=float).T   # (n_metacells, n_batches)
    counts_df = pd.DataFrame(count_rows, dtype=float).T
    return comp_df, counts_df


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


def calc_purity(
    df, label_key, mc_key="SEACell", return_per_mc=False, return_major_label=False
):
    if label_key not in df.columns:
        return None

    groups = df.groupby(mc_key)

    pur = []
    maj = []

    for _, sub in groups:
        vc = sub[label_key].value_counts(normalize=True)
        pur.append(vc.max())
        maj.append(vc.idxmax())

    if return_per_mc:
        purity = pd.Series(
            pur, index=pd.Index(groups.groups.keys(), dtype=str), name="purity"
        )
        if return_major_label:
            major = pd.Series(
                maj, index=pd.Index(groups.groups.keys(), dtype=str), name="major_label"
            )
            return purity, major
        return purity

    return float(np.mean(pur))


def calc_batch_entropy(df, batch_key, mc_key="SEACell", return_per_mc=False):
    if batch_key not in df.columns:
        return None

    groups = df.groupby(mc_key)
    ents = []
    for _, sub in groups:
        p = sub[batch_key].value_counts(normalize=True).values
        ents.append(float(-np.sum(p * np.log(p + 1e-10))))

    if return_per_mc:
        return pd.Series(ents, index=pd.Index(groups.groups.keys(), dtype=str), name="batch_entropy")
    return float(np.mean(ents))


def calc_modularity_per_batch(A, assignments, batch_labels):
    """Compute modularity for each batch on an edge-filtered subgraph.

    For each batch: keep all edges where at least one endpoint belongs to the
    batch, then run standard Newman weighted modularity on that graph.

    Args:
        A:             scipy sparse adjacency (n_cells x n_cells). Symmetrized
                       internally — pass the raw affinity matrix.
        assignments:   integer cluster labels, shape (n_cells,).
        batch_labels:  batch label per cell, shape (n_cells,).

    Returns:
        pd.Series indexed by batch name, values are modularity scores.
        Returns None if A has no edges.
    """
    import scipy.sparse as sp

    A = sp.csr_matrix(A)
    A = (A + A.T) / 2
    A_coo = A.tocoo()

    n = A.shape[0]
    cluster_ids = np.unique(assignments)
    batch_ids = np.unique(batch_labels)
    batch_mod_vals = {}

    for batch in batch_ids:
        in_batch = np.zeros(n, dtype=bool)
        in_batch[batch_labels == batch] = True

        keep = in_batch[A_coo.row] | in_batch[A_coo.col]
        A_b = sp.coo_matrix(
            (A_coo.data[keep], (A_coo.row[keep], A_coo.col[keep])),
            shape=(n, n),
        ).tocsr()

        degrees_b = np.array(A_b.sum(axis=1)).ravel()
        two_m_b = float(degrees_b.sum())
        if two_m_b == 0:
            batch_mod_vals[str(batch)] = 0.0
            continue

        Q_b = 0.0
        for c in cluster_ids:
            mask_c = (assignments == c)
            e_k = float(A_b[mask_c][:, mask_c].sum())
            d_k = float(degrees_b[mask_c].sum())
            Q_b += (e_k - d_k * d_k / two_m_b) / two_m_b
        batch_mod_vals[str(batch)] = float(Q_b)

    s = pd.Series(batch_mod_vals, name='modularity')
    s.index.name = 'batch'
    return s


def mc_label_purity(ad, lk, mc_key="SEACell", name="model", save_path=None):
    g = ad.obs.groupby(mc_key)[lk]
    maj = g.agg(lambda x: x.value_counts().idxmax())
    pur = g.apply(lambda x: (x == x.value_counts().idxmax()).mean())
    ct_avg = pur.groupby(maj).mean()
    ct_avg = ct_avg.reindex(ad.obs[lk].unique()).fillna(0)
    out = ct_avg.to_frame().T
    out["avg"] = ct_avg.mean()
    out.index = [name]
    if save_path is not None:
        out.to_csv(save_path + f"/avg_purity_{lk}.csv")
    return out


def get_rare(ad, label_key, thr=0.25):
    freq = ad.obs[label_key].value_counts(normalize=True)
    thr = freq.quantile(0.25)
    rare = freq[freq < thr].index.tolist()
    return rare


def summarize_metacell_quality(ad, bk, label_key, save_dir, model_name):
    avg_metrics, _ = avg_mc_quality_metrics(ad, bk, label_key)

    summary = {}
    for col in avg_metrics.columns:
        vals = pd.to_numeric(avg_metrics[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        center = (
            np.mean(vals)
            if ("purity" in col or col == "spatial_compactness")
            else np.median(vals)
        )
        q25, q75 = np.percentile(vals, [25, 75])
        summary[f"{col}_center"] = round(center, 3)
        summary[f"{col}_summary"] = f"{center:.3f} ± {(q75 - q25):.3f}"

    freq = ad.obs[label_key].value_counts(normalize=True)
    thr = freq.quantile(0.25)
    rare = freq[freq < thr].index.tolist()

    major = avg_metrics[label_key]
    purity = avg_metrics[f"{label_key}_purity"]
    summary[f"rare_{label_key}_purity"] = round(
        float(purity[major.isin(rare)].mean()), 3
    )

    df_summary = pd.DataFrame([summary], index=[model_name])
    df_summary.to_csv(f"{save_dir}/mc_quality_summary.csv")
    avg_metrics.to_csv(f"{save_dir}/mc_quality.csv")

    return df_summary, avg_metrics


def avg_mc_quality_metrics(ad, bk, lk):
    mc = ad.obs["SEACell"].astype(str).values
    batches = ad.obs[bk].values
    out = {}

    keys = [lk] + spatial_labels
    for k in keys:
        if k in ad.obs:
            out[f"{k}_purity"], out[k] = calc_purity(
                ad.obs, k, return_per_mc=True, return_major_label=True
            )

    if "spatial" in ad.obsm:
        out["spatial_compactness"] = spatial_compactness(ad, "spatial", "SEACell", bk)

    if "dc" not in ad.obsm:
        import palantir

        sc.tl.pca(ad)
        print("computing diffusion components using palantir...")
        components = pd.DataFrame(ad.obsm["X_pca"], index=ad.obs_names)
        dm_res = palantir.utils.run_diffusion_maps(components)
        dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
        ad.obsm["dc"] = dc.values

    X = ad.obsm["dc"]
    comp, sep = compute_mc_compactness_and_separation(X, mc, batches)
    out["compactness"] = pd.Series(comp, index=np.unique(mc))
    out["separation"] = pd.Series(sep, index=np.unique(mc))

    df = pd.concat(out, axis=1)
    df.columns = [c if isinstance(c, str) else c[1] for c in df.columns]
    return df, out


def compute_dc(ad, batch_key, out_dir="./", base=None, remove_dc=True):
    if base is None:
        base = f"{out_dir}/{len(ad)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ad_path, dc_path, lock_path = base + ".h5ad", base + ".csv", base + ".lock"

    ad.write(ad_path)
    have_lock = False
    if not os.path.exists(dc_path):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            have_lock = True
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "interpretable_ssl.evaluation.diffusion",
                    ad_path,
                    dc_path,
                    lock_path,
                    batch_key,
                ]
            )
        except FileExistsError:
            proc = None

        while True:
            if os.path.exists(dc_path):
                break
            if proc is not None and proc.poll() is not None and proc.returncode != 0:
                raise RuntimeError("Diffusion failed")
            time.sleep(0.1)

    df = pd.read_csv(dc_path, index_col=0)

    if have_lock and os.path.exists(ad_path):
        os.remove(ad_path)
    if remove_dc:
        if os.path.exists(dc_path):
            os.remove(dc_path)
    if have_lock and os.path.exists(lock_path):
        os.remove(lock_path)

    return df


def get_seacell_path(ds_id):
    from interpretable_ssl.configs.paths import get_seacell_model_dir
    return get_seacell_model_dir(ds_id)


def save_df(df, save_path, append=False):
    if append and os.path.exists(save_path):
        df_ = pd.read_csv(save_path, index_col=0)
        df_ = pd.concat([df_, df], axis=0)
    else:
        df_ = df
    df_.to_csv(save_path)


def save_mc_stats(ad, mc_ad, lk, bk, name, save_path, epsilon, append=False):
    res = pd.DataFrame(index=[name])
    res["epsilon"] = epsilon
    unused = len(mc_ad) - ad.obs["SEACell"].nunique()
    res["unused_proto"] = unused
    res["unused_proto_ratio"] = unused / len(mc_ad)
    def merge_labels(adata, l1, l2):
        adata.obs[f"{l1}_{l2}"] = (
            adata.obs[l1].astype(str) + "_" + adata.obs[l2].astype(str)
        )

    if "niches_2D" in ad.obs:
        merge_labels(ad, lk, "niches_2D")
        merge_labels(mc_ad, lk, "niches_2D")

    for k in spatial_labels + [lk, bk, f"{lk}_niches_2D"]:
        if k in ad.obs and k in mc_ad.obs:
            res[f"cov_{k}"] = (
                len(
                    set(mc_ad.obs[k].dropna().unique())
                    & set(ad.obs[k].dropna().unique())
                )
                / ad.obs[k].nunique()
            )
    if save_path is not None:
        save_df(res, f"{save_path}/stats.csv", append)
    return res

def purity_stats(ad, label_keys, save_path, name):
    stats = {}

    for lk in label_keys:
        if lk not in ad.obs:
            continue
        tab = pd.crosstab(ad.obs['SEACell'], ad.obs[lk])
        purity = tab.div(tab.sum(axis=1), axis=0).max(axis=1)

        stats[f'{lk}_mean']   = purity.mean()
        stats[f'{lk}_median'] = purity.median()
        stats[f'{lk}_std']    = purity.std()

    df = pd.DataFrame([stats])
    df.index = [name]
    df.to_csv(save_path + '/purity_stats.csv')
    
def save_all_mc_metrics(
    ad,
    mc_ad,
    lk,
    bk,
    save_path,
    epsilon=None,
    mc_key="SEACell",
    name="seacell",
    append=False,
    z = None
):
    
    purity_stats(ad, spatial_labels + [lk], save_path, name)
    save_mc_stats(ad, mc_ad, lk, bk, name, save_path, epsilon, append)
    for k in [lk] + spatial_labels:
        if k in ad.obs:
            mc_label_purity(ad, k, mc_key, name, save_path)

    if "metacell" in ad.layers:
        de_mc = ad.copy()
        de_mc.X = ad.layers["metacell"]
    else:
        de_mc = mc_ad
    if bk is not None and ad.obs[bk].nunique() > 1:
        compute_dge_consistency(de_mc, ad, lk, bk, name=name, save_path=save_path)

    # I feel maybe this hsould be like so i can pass sim when i had it (maybe from scproto)
    gene_recovery(ad, mc_ad, mc_key, name, lk, save_path)
    eval_mc_labeling(ad, lk, name, path=save_path)
    eval_mc_labeling_v2(ad, mc_ad, lk, name, save_path)

    mc_ad = mc_ad[mc_ad.obs[lk].notna()].copy()

    if "spatial" in ad.obsm:
        if z is not None:
            niche_ct_silhouette_df(z, ad.obs['niches_2D'], ad.obs[lk], name, save_path)
        all_purities(ad, lk, name, save_path, mc_key)
        celltype_niche_dge(ad, mc_ad, lk, name, save_path)
        evaluate_markers(ad, mc_ad, lk, name, mc_key, save_path)
        eval_niches(mc_ad, lk, "niches_2D", name, save_path)

    if f"{name}_mc_pca" not in mc_ad.obsm:
        sc.tl.pca(mc_ad)
        mc_ad.obsm[f"{name}_mc_pca"] = mc_ad.obsm["X_pca"]
    obsm_keys = [f"{name}_mc_pca"]

    if name == "seacell" and bk is not None and mc_ad.obs[bk].nunique() > 1:
        sce.pp.harmony_integrate(
            mc_ad,
            key=bk,  # your batch column
            basis=f"{name}_mc_pca",  # which embedding to correct
            adjusted_basis=f"{name}_mc_pca_harmoney",  # where to store corrected PCs
        )
        obsm_keys.append(f"{name}_mc_pca_harmoney")

    get_metacell_metrics(ad, mc_ad, obsm_keys, bk, lk, save_path)
    if "dc" not in ad.obsm:
        ad.obsm["dc"] = compute_dc(ad, bk).values
    summarize_metacell_quality(ad, bk, lk, save_path, name)


def compute_modularity(ad, mc_idx):
    """Newman weighted modularity using ad.obsp['connectivities']."""
    import scipy.sparse as sp
    if 'connectivities' not in ad.obsp:
        sc.pp.neighbors(ad, use_rep='X_pca')
    A = sp.csr_matrix(ad.obsp['connectivities'])
    A = (A + A.T) / 2
    degrees = np.array(A.sum(axis=1)).ravel()
    two_m = degrees.sum()
    if two_m == 0:
        return {'modularity': 0.0, 'per_cluster_contribution': {}}
    Q = 0.0
    per_cluster = {}
    for c in np.unique(mc_idx):
        mask = (mc_idx == c)
        e_k = A[mask][:, mask].sum()
        d_k = degrees[mask].sum()
        contrib = float((e_k - d_k * d_k / two_m) / two_m)
        per_cluster[int(c)] = contrib
        Q += contrib
    return {'modularity': float(Q), 'per_cluster_contribution': per_cluster}


def compute_task1_metrics(
    ad,
    mc_idx,
    lk,
    bk,
    nk,
    save_path,
    ds_id,
    method_tag,
    n_components=50,
    k_neighbors=50,
    affinity_type='arbf',
    graph_dir='./graphs',
):
    """Compute and save task-1 metacell quality metrics for any metacell method.

    Args:
        ad:            Original adata with obs containing label/batch columns.
                       Must have obsp['connectivities'] or X_pca for modularity.
        mc_idx:        Integer array (n_cells,) of metacell assignments.
        lk:            Label (cell-type) key in ad.obs.
        bk:            Batch key in ad.obs, or None.
        nk:            Niche key in ad.obs, or None.
        save_path:     Directory to write CSVs and metrics.json.
        ds_id:         Dataset ID string (used for affinity path lookup).
        method_tag:    Short name for print prefixes, e.g. 'seacell' or 'metaq'.
        n_components:  Affinity graph n_components (default 50).
        k_neighbors:   Affinity graph k_neighbors (default 50).
        affinity_type: Affinity graph type (default 'arbf').
        graph_dir:     Directory where affinity .pkl files live (default './graphs').

    Returns:
        dict with keys purity, niche_purity, batch_entropy, modularity,
        n_unused_protos, unused_proto_ratio.
    """
    import json
    import pickle
    import scipy.sparse as sp
    from interpretable_ssl.configs.paths import get_affinity_path

    os.makedirs(save_path, exist_ok=True)
    obs = ad.obs.copy()
    obs['_mc'] = mc_idx

    # --- unused protos ---
    n_total_mc = int(mc_idx.max()) + 1
    n_unused = int(n_total_mc - len(np.unique(mc_idx)))
    unused_ratio = n_unused / n_total_mc
    print(f"[{method_tag}] unused protos: {n_unused}/{n_total_mc} ({unused_ratio:.2%})")

    mc_sizes = obs['_mc'].astype(str).value_counts().rename('size')
    mc_sizes.index.name = 'metacell'
    mc_sizes.to_csv(os.path.join(save_path, 'size_per_mc.csv'))

    # --- cell-type purity ---
    purity_per_mc = calc_purity(obs, label_key=lk, mc_key='_mc', return_per_mc=True)
    weighted_mean_purity = weighted_std_purity = None
    if purity_per_mc is not None:
        purity_per_mc.index.name = 'metacell'
        purity_per_mc.to_csv(os.path.join(save_path, 'purity_per_mc.csv'))
        w = mc_sizes.reindex(purity_per_mc.index).fillna(0)
        wsum = w.sum()
        weighted_mean_purity = float((purity_per_mc * w).sum() / wsum)
        weighted_std_purity = float(np.sqrt(((purity_per_mc - weighted_mean_purity) ** 2 * w).sum() / wsum))
        print(f"[{method_tag}] mean cell-type purity: {purity_per_mc.mean():.4f}  "
              f"(size-weighted: {weighted_mean_purity:.4f} ± {weighted_std_purity:.4f})")
    else:
        print(f"[{method_tag}] cell-type purity: label key '{lk}' not in obs, skipped")

    # --- niche purity ---
    niche_purity_per_mc = calc_purity(obs, label_key=nk, mc_key='_mc', return_per_mc=True) if nk else None
    if niche_purity_per_mc is not None:
        niche_purity_per_mc.index.name = 'metacell'
        niche_purity_per_mc.to_csv(os.path.join(save_path, 'niche_purity_per_mc.csv'))
        w = mc_sizes.reindex(niche_purity_per_mc.index).fillna(0)
        wsum = w.sum()
        wm = float((niche_purity_per_mc * w).sum() / wsum)
        ws = float(np.sqrt(((niche_purity_per_mc - wm) ** 2 * w).sum() / wsum))
        print(f"[{method_tag}] mean niche purity: {niche_purity_per_mc.mean():.4f}  "
              f"(size-weighted: {wm:.4f} ± {ws:.4f})")

    # --- batch entropy ---
    entropy_per_mc = None
    weighted_mean_entropy = weighted_std_entropy = None
    if bk is not None:
        entropy_per_mc = calc_batch_entropy(obs, batch_key=bk, mc_key='_mc', return_per_mc=True)
    if entropy_per_mc is not None:
        entropy_per_mc.index.name = 'metacell'
        entropy_per_mc.to_csv(os.path.join(save_path, 'batch_entropy_per_mc.csv'))
        w = mc_sizes.reindex(entropy_per_mc.index).fillna(0)
        wsum = w.sum()
        weighted_mean_entropy = float((entropy_per_mc * w).sum() / wsum)
        weighted_std_entropy = float(np.sqrt(((entropy_per_mc - weighted_mean_entropy) ** 2 * w).sum() / wsum))
        print(f"[{method_tag}] mean batch entropy: {entropy_per_mc.mean():.4f}  "
              f"(size-weighted: {weighted_mean_entropy:.4f} ± {weighted_std_entropy:.4f})")
    else:
        print(f"[{method_tag}] batch entropy: batch key not found, skipped")

    # --- coverage ---
    coverage = None
    if lk in ad.obs.columns:
        majority_labels = obs.groupby('_mc')[lk].agg(lambda x: x.mode()[0])
        coverage = majority_labels.nunique() / ad.obs[lk].nunique()
        print(f"[{method_tag}] coverage: {coverage:.4f}")

    # --- modularity ---
    mod_result = compute_modularity(ad, mc_idx)
    print(f"[{method_tag}] modularity: {mod_result['modularity']:.4f}")

    def _convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError(f'Not serializable: {type(o)}')

    with open(os.path.join(save_path, 'modularity.json'), 'w') as f:
        json.dump(mod_result, f, indent=2, default=_convert)

    # --- per-batch modularity ---
    mean_batch_mod = std_batch_mod = None
    if bk is not None and bk in ad.obs.columns:
        A = sp.csr_matrix(ad.obsp['connectivities'])
        batch_mod_s = calc_modularity_per_batch(A, mc_idx, ad.obs[bk].values)
        batch_mod_s.to_csv(os.path.join(save_path, 'modularity_per_batch.csv'))
        mean_batch_mod = float(batch_mod_s.mean())
        std_batch_mod = float(batch_mod_s.std())
        print(f"[{method_tag}] per-batch modularity: mean={mean_batch_mod:.4f}, std={std_batch_mod:.4f}")

    # --- aff-DC compactness ---
    metrics_aff = {}
    try:
        graph_path = get_affinity_path(ds_id, len(ad), n_components, k_neighbors, affinity_type, graph_dir)
        print(f"[aff_dc_compactness] looking for graph at: {graph_path}")
        if not os.path.exists(graph_path):
            print("[aff_dc_compactness] graph not found, skipping.")
        else:
            with open(graph_path, 'rb') as f:
                aff = pickle.load(f)
            aff = sp.csr_matrix(aff)
            aff_for_dc = aff.copy()
            aff_for_dc.setdiag(1)
            batches_arr = ad.obs[bk].values if bk is not None else np.zeros(len(ad), dtype=str)
            comp_df, counts_df = compute_aff_dc_compactness(aff_for_dc, mc_idx.astype(str), batches_arr)
            valid_counts = counts_df.where(comp_df.notna(), 0)
            per_mc_mean = (comp_df.fillna(0) * valid_counts).sum(axis=1) / valid_counts.sum(axis=1)
            per_batch_mean = (comp_df.fillna(0) * valid_counts).sum(axis=0) / valid_counts.sum(axis=0)
            out_df = comp_df.copy()
            out_df['weighted_mean'] = per_mc_mean
            csv_path = os.path.join(save_path, 'aff_dc_compactness.csv')
            out_df.to_csv(csv_path)
            metrics_aff['aff_compactness_mean'] = float(per_mc_mean.mean())
            metrics_aff['aff_compactness_per_batch'] = {str(b): float(v) for b, v in per_batch_mean.items()}
            print(f"[aff_dc_compactness] mean={metrics_aff['aff_compactness_mean']:.4f} | saved to {csv_path}")
    except Exception as e:
        import traceback
        print(f"Warning: aff_dc_compactness failed: {e}")
        traceback.print_exc()

    # --- metrics.json (read-then-update to preserve task2 metrics) ---
    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    if purity_per_mc is not None:
        metrics['mean_cell_type_purity'] = float(purity_per_mc.mean())
        metrics['weighted_mean_cell_type_purity'] = weighted_mean_purity
        metrics['weighted_std_cell_type_purity'] = weighted_std_purity
    if niche_purity_per_mc is not None:
        metrics['mean_niche_purity'] = float(niche_purity_per_mc.mean())
    if entropy_per_mc is not None:
        metrics['mean_batch_entropy'] = float(entropy_per_mc.mean())
        metrics['weighted_mean_batch_entropy'] = weighted_mean_entropy
        metrics['weighted_std_batch_entropy'] = weighted_std_entropy
    if coverage is not None:
        metrics['coverage'] = float(coverage)
    metrics['modularity'] = mod_result['modularity']
    metrics['n_unused_protos'] = n_unused
    metrics['unused_proto_ratio'] = unused_ratio
    if mean_batch_mod is not None:
        metrics['mean_modularity_batch'] = mean_batch_mod
        metrics['std_modularity_batch'] = std_batch_mod
    metrics.update(metrics_aff)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[{method_tag}] saved metrics to {save_path}")

    return {
        'purity': purity_per_mc,
        'niche_purity': niche_purity_per_mc,
        'batch_entropy': entropy_per_mc,
        'modularity': mod_result,
        'coverage': coverage,
        'n_unused_protos': n_unused,
        'unused_proto_ratio': unused_ratio,
    }


def calc_task2_metrics(ad, mc_ad, lk, bk, obsm_keys, name, save_path):
    """
    Task 2: Metacell representation quality.
      - Coverage:        fraction of cell types in ad represented by at least one metacell
      - DGE consistency: agreement between metacell DE and per-batch single-cell DE
      - scGraph:         how well metacell embeddings recover single-cell consensus structure

    Returns:
        dict with scalar summaries for logging:
            {'coverage': float, 'dge_jaccard_avg': float, 'scgraph_corr_avg': float}
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. Coverage
    coverage = len(set(mc_ad.obs[lk].dropna().unique())) / ad.obs[lk].nunique()
    pd.DataFrame({"coverage": [coverage]}, index=[name]).to_csv(
        f"{save_path}/coverage.csv"
    )

    # 2. DGE consistency (saves dge_consistency.csv)
    df_rbo, df_kt, df_jac = compute_dge_consistency(mc_ad, ad, lk, bk, name=name, save_path=save_path)
    dge_rbo_avg     = float(df_rbo["avg global"].iloc[0]) if df_rbo is not None else None
    dge_kendall_avg = float(df_kt["avg global"].iloc[0])  if df_kt  is not None else None
    dge_jaccard_avg = float(df_jac["avg global"].iloc[0]) if df_jac is not None else None

    # 3. scGraph
    scg = get_mc_scg(ad, mc_ad, bk, lk, obsm_keys)
    scg.to_csv(f"{save_path}/scgraph.csv")
    scgraph_corr_avg = float(scg["Corr-PCA"].mean()) if scg is not None else None

    return {
        "coverage":         coverage,
        "dge_rbo_avg":      dge_rbo_avg,
        "dge_kendall_avg":  dge_kendall_avg,
        "dge_jaccard_avg":  dge_jaccard_avg,
        "scgraph_corr_avg": scgraph_corr_avg,
    }
