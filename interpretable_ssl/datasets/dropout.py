import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm


def pseudobulk_sc(ad, label_key, batch_key):
    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    labs = ad.obs[label_key].values
    bks = ad.obs[batch_key].values
    genes = ad.var_names.to_list()

    groups = pd.DataFrame({"ct": labs, "bk": bks})
    uniq = groups.drop_duplicates()

    mats, idxs = [], []
    for _, r in uniq.iterrows():
        g = np.where((labs == r.ct) & (bks == r.bk))[0]
        if len(g):
            mats.append(np.asarray(X[g].sum(axis=0)).ravel())
            idxs.append(f"{r.ct}|{r.bk}")

    pb = np.vstack(mats)
    return AnnData(pb, obs=pd.DataFrame(index=idxs), var=pd.DataFrame(index=genes))


def scanpy_markers(pb, group_key="ct", pval_thr=0.01, logfc_thr=0.5):
    sc.pp.normalize_total(pb)
    sc.pp.log1p(pb)

    pb.obs[group_key] = [i.split("|")[0] for i in pb.obs.index]

    sc.tl.rank_genes_groups(pb, groupby=group_key, method="t-test", key_added="dge")
    df = sc.get.rank_genes_groups_df(pb, group=None, key="dge")

    df = df[(df["pvals_adj"] < pval_thr) & (df["logfoldchanges"] > logfc_thr)]

    out = {ct: df[df["group"] == ct]["names"].values for ct in df["group"].unique()}
    return out


def mask_dropout(X, labels, cell_frac=0.1, alpha=1.0, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    libs = X.sum(axis=1)
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
        w = 1 / libs[idx]
        w = w / w.sum()
        chosen = rng.choice(idx, size=k, replace=False, p=w)

        expr_sel = expr[chosen]
        libs_sel = libs_norm[chosen]

        # dropout probability
        p_sel = np.exp(-alpha * expr_sel) * np.exp(-beta * libs_sel[:, None])
        p_sel = np.clip(p_sel, 0, 1)

        # ensure zero-count genes can never be masked
        p_sel[X[chosen] == 0] = 0

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
