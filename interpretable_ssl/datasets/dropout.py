import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm
from interpretable_ssl.datasets.dataset_configs import *
import scipy.sparse as sp

from sklearn.neighbors import BallTree


def pseudobulk_sc(ad, lk, bk):
    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    labs = ad.obs[lk].values
    bks = ad.obs[bk].values
    genes = ad.var_names.to_list()

    groups = pd.DataFrame({"ct": labs, "bk": bks})
    uniq = groups.drop_duplicates()

    mats, obs_rows = [], []
    for _, r in uniq.iterrows():
        g = np.where((labs == r.ct) & (bks == r.bk))[0]
        if len(g):
            mats.append(np.asarray(X[g].sum(axis=0)).ravel())
            obs_rows.append({lk: r.ct, bk: r.bk})

    return AnnData(
        np.vstack(mats),
        obs=pd.DataFrame(obs_rows),
        var=pd.DataFrame(index=genes),
    )


def scanpy_markers(ad, group_key="ct", pval_thr=0.01, logfc_thr=0.5):
    if ad.X > 20:
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)

    # pb.obs[group_key] = [i.split("|")[0] for i in pb.obs.index]

    sc.tl.rank_genes_groups(ad, groupby=group_key, method="t-test", key_added="dge")
    df = sc.get.rank_genes_groups_df(ad, group=None, key="dge")

    df = df[(df["pvals_adj"] < pval_thr) & (df["logfoldchanges"] > logfc_thr)]

    out = {ct: df[df["group"] == ct]["names"].values for ct in df["group"].unique()}
    return out


def mask_dropout(X, labels, cell_frac=0.1, alpha=1.0, beta=1.0, seed=0, markers=None):
    rng = np.random.default_rng(seed)
    libs = np.asarray(X.sum(axis=1)).ravel()
    libs_norm = libs / libs.mean()
    expr = np.log1p(X)

    n_cells, n_genes = X.shape
    X_masked = X.copy()
    masked_counts = np.zeros(n_cells, dtype=int)

    # ---- masking per cell type ----
    for ct in tqdm(np.unique(labels)):
        idx = np.where(labels == ct)[0]
        k = max(1, int(len(idx) * cell_frac))

        # weight: low-lib cells get masked more often
        valid = libs[idx] > 0
        idx_valid = idx[valid]

        if len(idx_valid) == 0:
            continue

        w = 1.0 / libs[idx_valid]
        w = w / w.sum()

        k_eff = min(k, len(idx_valid))
        chosen = rng.choice(idx_valid, size=k_eff, replace=False, p=w)

        expr_sel = expr[chosen].toarray() if sp.issparse(expr) else expr[chosen]
        libs_sel = libs_norm[chosen]

        # dropout probability
        p_sel = np.exp(-alpha * expr_sel) * np.exp(-beta * libs_sel[:, None])
        p_sel = np.clip(p_sel, 0, 1)
        if markers is not None:
            genes_ct = markers.get(ct, [])
            if len(genes_ct) > 0:
                gene_mask = np.zeros(p_sel.shape[1], dtype=bool)
                gene_mask[genes_ct] = True
                p_sel[:, ~gene_mask] = 0

        # ensure zero-count genes can never be masked
        p_sel[expr_sel == 0] = 0

        # sample Bernoulli mask
        M = rng.random(p_sel.shape) < p_sel

        # apply mask
        for i, c in enumerate(chosen):
            genes = np.where(M[i])[0]
            X_masked[c, genes] = 0
            masked_counts[c] = len(genes)

    # ---- summary statistics ----
    df = pd.DataFrame({"label": labels, "genes_masked": masked_counts})
    grp = df.groupby("label")

    out = pd.DataFrame(
        {
            "n_cells_masked": grp.apply(lambda g: (g.genes_masked > 0).sum()),
            "prop_masked_cells": grp.apply(lambda g: (g.genes_masked > 0).mean()),
            "avg_genes_masked": grp.apply(
                lambda g: (
                    g.genes_masked[g.genes_masked > 0].mean()
                    if (g.genes_masked > 0).any()
                    else 0.0
                )
            ),
        }
    )

    return X_masked, df, out


# maybe correct in future, for now taking too much time, and not that much important
# maybe also for each batch, yu find ct markers, and mask is for each batch separte
def mask_markers(ad, bk, lk):
    pb_ad = pseudobulk_sc(ad, lk, bk)
    ct_counts = pb_ad.obs[lk].value_counts()

    multi_ct = ct_counts[ct_counts > 1].index
    single_ct = ct_counts[ct_counts == 1].index
    pb_multi = pb_ad[pb_ad.obs[lk].isin(multi_ct)].copy()
    markers_multi = scanpy_markers(pb_multi, lk)
    marker_single = {}
    for ct in single_ct:
        sub = ad[
            (ad.obs[lk] == ct)
            | (ad.obs[bk] == ad.obs.loc[ad.obs[lk] == ct, bk].iloc[0])
        ].copy()
        marker_single[ct] = (
            sc.tl.rank_genes_groups(
                sub,
                groupby=lk,
                groups=[ct],
                reference="rest",
                method="t-test",
                key_added="dge",
            )
            or sc.get.rank_genes_groups_df(sub, key="dge")
            .query("group == @ct")["names"]
            .values
        )

    markers = markers_multi | marker_single
    gene_to_idx = {g: i for i, g in enumerate(ad.var_names)}
    markers_idx = {
        ct: [gene_to_idx[g] for g in gs if g in gene_to_idx]
        for ct, gs in markers.items()
    }

    return mask_dropout(ad.layers["counts"], ad.obs[lk].values, markers=markers_idx)


def celltype_similarity_ratio(aff, labels):
    labels = np.array(labels)
    uniq = np.unique(labels)
    out = {}

    for ct in uniq:
        idx = np.where(labels == ct)[0]
        sims = aff[idx]
        ct_dict = {}
        for ct2 in uniq:
            jdx = np.where(labels == ct2)[0]
            ct_dict[ct2] = sims[:, jdx].mean()
        out[ct] = ct_dict

    df = pd.DataFrame(out).T
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=1)
    return df


import matplotlib.pyplot as plt


def plot_heatmap(df):
    plt.imshow(df.values, aspect="auto")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.colorbar()
    plt.show()


def normalize_if_needed(ad):
    if ad.X.max() > 20:
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    return ad


def save_gt_v0(ad):
    ad.layers["counts_gt"] = ad.layers["counts"].copy()
    ad = normalize_if_needed(ad)
    ad.layers["lognorm_gt"] = ad.X.copy()


def save_gt(ad):
    X_obs = ad.layers["counts"].copy()
    ad.layers["counts_gt"] = X_obs
    ad_obs = ad.copy()
    ad_obs.X = X_obs
    sc.pp.normalize_total(ad_obs, target_sum=1e4)
    sc.pp.log1p(ad_obs)
    ad.layers["lognorm_gt"] = ad_obs.X.copy()


def save_masked(ad):
    X_obs = ad.layers["counts"].copy()
    ad_obs = ad.copy()
    ad_obs.X = X_obs
    sc.pp.normalize_total(ad_obs, target_sum=1e4)
    sc.pp.log1p(ad_obs)
    ad.layers["lognorm"] = ad_obs.X.copy()


def report(ad):
    print(ad.layers["counts_gt"].max(), ad.layers["lognorm_gt"].max())
    print(ad.layers["counts"].max(), ad.layers["lognorm"].max())


def compute_markers(input_ad, lk, min_genes=20, pval=0.01, logfc=0.5):
    ad = input_ad.copy()
    ad.X = ad.layers['lognorm_gt']
    # sc.pp.normalize_total(ad)
    # sc.pp.log1p(ad)
    sc.tl.rank_genes_groups(ad, groupby=lk, method="t-test", key_added="dge")
    df = sc.get.rank_genes_groups_df(ad, group=None, key="dge")
    df = df[(df.pvals_adj < pval) & (df.logfoldchanges > logfc)]
    gene_to_idx = {g: i for i, g in enumerate(ad.var_names)}
    out = {}
    for ct, g in df.groupby("group"):
        genes = [gene_to_idx[x] for x in g.names if x in gene_to_idx]
        if len(genes) >= min_genes:
            out[ct] = genes
    return out

def apply_dropout(ad, lk, save_path=None, **kwargs):
    save_gt(ad)
    
    X_masked, df, out = mask_dropout(ad.layers["counts"], ad.obs[lk], **kwargs)
    ad.layers["counts"] = X_masked
    save_masked(ad)
    report(ad)
    if save_path is not None:
        ad.write(save_path)
    return out


def apply_batch_mask(ds_id, b, save_path=None):
    ad, bk, lk, n = load_ds(ds_id)
    save_gt(ad)
    mask_idx = ad.obs[bk] == b
    b_ad = ad[mask_idx]
    X_masked, df, out = mask_dropout(b_ad.layers["counts"], b_ad.obs[lk])

    X = ad.X.toarray()  # dense
    X[mask_idx] = X_masked.toarray() if sp.issparse(X_masked) else X_masked
    ad.layers["counts"] = sp.csr_matrix(X)

    save_masked(ad)

    report(ad)
    ad.write(save_path)
    return out


def sample_spill_cells(X, coords, radius, thr=0.1):

    # X = adata.X.toarray()
    n = X.shape[0]

    tree = BallTree(coords)
    neighbors = tree.query_radius(coords, r=radius, count_only=False)
    neighbor_count = np.array([len(nei) - 1 for nei in neighbors])
    crowding = neighbor_count / (neighbor_count.max() + 1e-12)

    lib = X.sum(1)
    low_rna = np.percentile(lib, 30) - lib
    low_rna = np.clip(low_rna, 0, None)
    low_rna = low_rna / (low_rna.max() + 1e-12)

    risk = 0.7 * crowding + 0.3 * low_rna
    risk[lib == 0] = 0.0
    spillover_prob = (risk + 1e-6) / (risk.sum() + 1e-12)
    spillover_prob = spillover_prob / spillover_prob.sum()
    # adata.obs["spillover_risk"] = risk
    # adata.obs["spillover_prob"] = spillover_prob
    m = int(thr * n)

    spill_cells = np.random.choice(n, size=m, replace=False, p=spillover_prob)
    return spill_cells


def get_spill_cnt(X, coords, radius, spill_cells):

    alpha = 0.1

    tree = BallTree(coords)
    neighbors = tree.query_radius(coords, r=radius)

    Xn = X.copy()

    no_nei = 0

    for i in spill_cells:
        nei = neighbors[i]
        if len(nei) == 1:
            no_nei += 1
            continue
        d = np.linalg.norm(coords[nei] - coords[i], axis=1)
        w = np.exp(-(d**2) / (np.median(d) ** 2 + 1e-12))
        w = w / w.sum()
        Xn[i] = (1 - alpha) * X[i] + alpha * (w @ X[nei])

    lib = X.sum(1).astype(int)
    Xn_int = X.copy()

    for i in spill_cells:
        nei = neighbors[i]
        nei = nei[nei != i]
        if len(nei) == 0:
            continue
        p = Xn[i].astype(np.float64)
        p = np.clip(p, 0, None)
        p = p / (p.sum() + 1e-12)
        s = p[:-1].sum()
        if s >= 1.0:
            p[:-1] = p[:-1] / (s + 1e-12)
            p[-1] = 0.0
        else:
            p[-1] = 1.0 - s
        Xn_int[i] = np.random.multinomial(lib[i], p)

    Xn = Xn_int

    # adata.layers["neighbor_spillover"] = Xn
    delta_l1 = np.abs(Xn - X).sum(1) / (X.sum(1) + 1e-12)
    spill_mask = np.zeros(X.shape[0], dtype=bool)
    spill_mask[spill_cells] = True
    print(no_nei, delta_l1[spill_mask].mean(), delta_l1[~spill_mask].mean())
    return Xn


def get_counts(adata):
    X = adata.layers["counts"]
    return X.toarray() if hasattr(X, "toarray") else X


def apply_spill(ad, save_path):
    save_gt(ad)
    coords = ad.obsm["spatial"][:, :2]
    radius = np.percentile(BallTree(coords).query(coords, k=2)[0][:, 1], 70)
    X = get_counts(ad)
    spill_cells = sample_spill_cells(X, coords, radius)
    ad.layers["counts"] = get_spill_cnt(X, coords, radius, spill_cells)
    save_masked(ad)
    report(ad)
    ad.write(save_path)
