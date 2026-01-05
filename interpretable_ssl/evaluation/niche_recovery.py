import numpy as np
import pandas as pd
import scanpy as sc
import ast
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if len(a | b) else np.nan


def rbo(l1, l2, p=0.9):
    s1, s2 = [], []
    rbo_sum = 0.0
    for i in range(1, min(len(l1), len(l2)) + 1):
        s1.append(l1[i - 1])
        s2.append(l2[i - 1])
        overlap = len(set(s1) & set(s2))
        rbo_sum += overlap / i * (p ** (i - 1))
    return (1 - p) * rbo_sum


def metacell_niche_validation_df(
    mc_ad,
    sig_niches,
    celltype_key,
    niche_key,
    topk=10,
    n_pca=50,
    kmeans_k=10,
):
    rows = []

    for ct in mc_ad.obs[celltype_key].unique():
        ct_df = sig_niches[sig_niches["cell_type"] == ct]

        ct_mc = mc_ad[mc_ad.obs[celltype_key] == ct].copy()
        n_mc, n_feat = ct_mc.n_obs, ct_mc.n_vars
        max_pc = min(n_mc - 1, n_feat - 1)
        if max_pc < 1:
            continue

        n_pc = min(n_pca, max_pc)
        sc.pp.pca(ct_mc, n_comps=n_pc, svd_solver="auto")

        if n_mc < 2:
            continue

        k_eff = min(kmeans_k, n_mc - 1)
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(
            ct_mc.obsm["X_pca"]
        )
        ct_mc.obs["kmeans"] = km.labels_.astype(str)

        all_niches = set(ct_mc.obs[niche_key].unique()) | set(ct_df["niche"].unique())

        for niche in all_niches:
            pos = ct_mc.obs[niche_key] == niche
            npos, ncont = pos.sum(), (~pos).sum()

            if npos < 1 or ncont < 1:
                rows.append(
                    {
                        "cell_type": ct,
                        "niche": niche,
                        "npos": int(npos),
                        "ncont": int(ncont),
                        "mc_pval": np.nan,
                        "jaccard": np.nan,
                        "rbo": np.nan,
                        "shared_genes": [],
                        "mc_dge": None,
                    }
                )
                continue

            g = np.where(pos, "pos", "ctrl")
            tab = pd.crosstab(g, ct_mc.obs["kmeans"])
            mc_pval = (
                chi2_contingency(tab)[1]
                if tab.shape[0] > 1 and tab.shape[1] > 1
                else np.nan
            )

            mc_genes, mc_dge = [], None
            if npos >= 2 and ncont >= 2:
                tmp = ct_mc.copy()
                tmp.obs["group"] = g
                sc.tl.rank_genes_groups(
                    tmp, "group", groups=["pos"], reference="ctrl", method="wilcoxon"
                )
                rg_mc = sc.get.rank_genes_groups_df(tmp, group="pos")
                mc_genes = rg_mc["names"].head(topk).tolist()
                mc_dge = rg_mc.head(topk)[
                    ["names", "logfoldchanges", "pvals_adj"]
                ].to_dict("records")

            if niche in ct_df["niche"].values:
                ref_dge = ct_df.loc[ct_df["niche"] == niche, "dge"].iloc[0]
                ref_dge = (
                    ast.literal_eval(ref_dge) if isinstance(ref_dge, str) else ref_dge
                )
                ref_genes = (
                    [g["names"] for g in ref_dge][:topk]
                    if isinstance(ref_dge, list)
                    else []
                )
                shared = list(set(mc_genes) & set(ref_genes))
                jac = jaccard(mc_genes, ref_genes)
                rb = rbo(mc_genes, ref_genes) if mc_genes and ref_genes else np.nan
            else:
                shared, jac, rb = [], np.nan, np.nan

            rows.append(
                {
                    "cell_type": ct,
                    "niche": niche,
                    "npos": int(npos),
                    "ncont": int(ncont),
                    "mc_pval": mc_pval,
                    "jaccard": jac,
                    "rbo": rb,
                    "shared_genes": shared,
                    "mc_dge": mc_dge,
                    "is_known_niche": niche in ct_df["niche"].values,
                }
            )

    return pd.DataFrame(rows)


def discovered_niches_df(
    df,
    p_thr=1e-2,
    rbo_thr=0.2,
    jac_thr=0.25,
):
    return df[
        (df["npos"] >= 2)
        & (
            (df["mc_pval"] < p_thr)
            | (df["rbo"] >= rbo_thr)
            | (df["jaccard"] >= jac_thr)
        )
    ].reset_index(drop=True)


def niche_metrics_dfs(
    df,
    sig_niches,
    name,
    p_thr=1e-2,
    min_npos=2,
    min_ncont=2,
):
    m1, m2, m3 = {}, {}, {}

    cts = set(df["cell_type"]) | set(sig_niches["cell_type"])

    for ct in cts:
        known = (
            sig_niches.loc[sig_niches["cell_type"] == ct, "niche"]
            .astype(str)
            .unique()
            .tolist()
        )
        known = [k for k in known if k != "Excluded"]

        sub = df[df["cell_type"] == ct].copy()
        sub["niche"] = sub["niche"].astype(str)

        if len(known) == 0:
            m1[ct] = np.nan
            m2[ct] = np.nan
        else:
            sub_known = sub[sub["niche"].isin(known)]
            m1[ct] = (
                (sub_known["mc_pval"] < p_thr) & (sub_known["jaccard"] > 0)
            ).sum() / len(known)
            m2[ct] = (sub_known["mc_pval"] < p_thr).sum() / len(known)

        new = sub[~sub["niche"].isin(known)]
        new = new[(new["npos"] >= min_npos) & (new["ncont"] >= min_ncont)]
        m3[ct] = int((new["mc_pval"] < p_thr).sum())

    return {
        # known validated, known recovered, new founded niches
        "kv_niches": pd.DataFrame([m1], index=[name]),
        "kr_niches": pd.DataFrame([m2], index=[name]),
        "new_niches": pd.DataFrame([m3], index=[name]),
    }


def save_df(df, name, path, save_name):
    df.index = [name]
    if path is None:
        return
    df.to_csv(f"{path}/{save_name}")


def eval_niches(mc_ad, lk, nk, name, save_path):
    p = "/home/icb/fatemehs.hashemig/data/spatial/nsc_sig_niches.csv"
    sig_niches = pd.read_csv(p, index_col=0)
    res_df = metacell_niche_validation_df(
        mc_ad, sig_niches, celltype_key=lk, niche_key=nk, topk=50
    )
    res_df = discovered_niches_df(res_df)
    res_df.to_csv(f"{save_path}/enriched_niches.csv")
    dfs = niche_metrics_dfs(
        res_df,
        sig_niches,
        name,
        p_thr=1e-2,
        min_npos=2,
        min_ncont=2,
    )
    for k, df in dfs.items():
        save_df(df, name, save_path, f"{k}.csv")


def mc_majority_label(ad, mc_idx, label_key):
    tab = pd.crosstab(ad.obs[mc_idx], ad.obs[label_key])
    lab = tab.idxmax(axis=1)
    pur = tab.max(axis=1) / tab.sum(axis=1)
    return lab, pur


def metacells_of_ct(ad, mc_idx, ct_key, ct):
    tab = pd.crosstab(ad.obs[mc_idx], ad.obs[ct_key])
    return tab.idxmax(axis=1).loc[lambda x: x == ct].index


def valid_niches(ad, ct_key, niche_key, ct, min_cells, drop_niches=("Excluded",)):
    s = ad.obs.loc[
        (ad.obs[ct_key] == ct) & (~ad.obs[niche_key].isin(drop_niches)), niche_key
    ].value_counts()
    return s[s >= min_cells].index.tolist()


def niche_f1_for_niche_ct(ad, mc_idx, ct_key, niche_key, mc_lab, ct, niche):
    mcs = mc_lab[mc_lab == niche].index
    if len(mcs) == 0:
        return 0.0
    df = ad.obs[(ad.obs[mc_idx].isin(mcs)) & (ad.obs[ct_key] == ct)]
    y_true = (df[niche_key] == niche).to_numpy()
    y_pred = np.ones(len(df), dtype=bool)
    tp = y_true.sum()
    fp = (~y_true).sum()
    fn = 0
    return 0.0 if tp == 0 else (2 * tp) / (2 * tp + fp + fn)


def eval_ct_niche_grouping(ad, mc_idx, ct_key, niche_key, ct, min_cells):
    vn = valid_niches(ad, ct_key, niche_key, ct, min_cells)
    mc_ids = metacells_of_ct(ad, mc_idx, ct_key, ct)

    mc_lab, mc_pur = mc_majority_label(ad, mc_idx, niche_key)
    mc_lab = mc_lab.loc[mc_ids]
    mc_pur = mc_pur.loc[mc_ids]

    rows = []
    for n in vn:
        mcs = mc_lab[mc_lab == n].index

        cells = ad.obs[(ad.obs[ct_key] == ct) & (ad.obs[niche_key] == n)]

        purity = mc_pur.loc[mcs].mean() if len(mcs) else 0.0
        f1 = niche_f1_for_niche_ct(ad, mc_idx, ct_key, niche_key, mc_lab, ct, n)

        rows.append(
            {
                "niche": n,
                "n_cells": len(cells),
                "n_metacells": len(mcs),
                "niche_purity": purity,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows).set_index("niche")


def macro_scores_by_ct(
    ad, mc_idx, ct_key, niche_key, min_cells, name, save_path, drop_na=True
):
    f1_out = {}
    pur_out = {}

    for ct in ad.obs[ct_key].unique():
        df = eval_ct_niche_grouping(ad, mc_idx, ct_key, niche_key, ct, min_cells)
        if df.empty:
            f1_out[ct] = np.nan
            pur_out[ct] = np.nan
            continue

        f1 = df["f1"]
        pur = df["niche_purity"]

        if drop_na:
            f1 = f1.dropna()
            pur = pur.dropna()

        f1_out[ct] = f1.mean() if len(f1) else np.nan
        pur_out[ct] = pur.mean() if len(pur) else np.nan

    df_f1 = pd.DataFrame([f1_out])
    df_pur = pd.DataFrame([pur_out])
    save_df(df_f1, name, save_path, "/ct_niche_macrof1.csv")
    save_df(df_pur, name, save_path, "/ct_niche_macro_pur.csv")
    return df
