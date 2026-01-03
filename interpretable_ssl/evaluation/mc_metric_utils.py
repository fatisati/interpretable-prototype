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
    return f"/ictstr01/home/icb/fatemehs.hashemig/models/{ds_id}/seacell/"


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
):
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
