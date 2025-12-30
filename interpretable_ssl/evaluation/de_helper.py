import scanpy as sc
import pandas as pd
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import kendalltau


def get_rare(ad=None, label_key=None, thr=0.25, labels=None):
    if labels is None:
        labels = ad.obs[label_key]

    freq = labels.value_counts(normalize=True)
    cutoff_thr = freq.quantile(thr)
    rare = freq[freq < cutoff_thr].index.tolist()
    return rare


def get_gene_map(ad):
    return ad.var["feature_name"].to_dict()


def calc_dge_filtered(ad, label_key, filter_thr=0.05):
    group_cnt = ad.obs[label_key].value_counts()
    valid_groups = group_cnt[group_cnt > 1].index
    ad_valid = ad[ad.obs[label_key].isin(valid_groups)]
    sc.tl.rank_genes_groups(
        ad_valid, groupby=label_key, method="t-test", key_added="rank_genes"
    )
    dge_df = sc.get.rank_genes_groups_df(ad_valid, key="rank_genes", group=None)
    dge_df = dge_df[dge_df["pvals_adj"] < filter_thr]
    # dge_df["group"] = dge_df["group"].map(lambda x: x.lower())
    return dge_df


def calculate_avg_jaccard(mc_de, batch_de_dict, col_name=None):
    if col_name is None:
        col_name = "avg_jaccard_index"
    else:
        col_name += "_jaccard"
    row_dict = {"index": col_name}

    for ct in mc_de["group"].unique():
        jaccard_indices = []
        mc_genes = set(mc_de[mc_de["group"] == ct]["names"])

        for b_de in batch_de_dict.values():
            b_genes = set(b_de[b_de["group"] == ct]["names"])
            intersection = len(mc_genes & b_genes)
            union = len(mc_genes | b_genes)
            jaccard_indices.append(intersection / union if union > 0 else 0)

        # directly store as column
        row_dict[ct] = sum(jaccard_indices) / len(jaccard_indices)

    # create DataFrame with one row, index = col_name
    return pd.DataFrame([row_dict]).set_index("index")


def get_mc_jaccard(mc_adata, ad, lk, bk, thr, name=None):
    if ad.X.max() > 20:
        ad = ad.copy()
        ad.X = ad.layers["lognorm"]
    mc_de = calc_dge_filtered(mc_adata, lk, thr)
    batch_de_dict = {}
    for b in ad.obs[bk].unique():
        b_ad = ad[ad.obs[bk] == b].copy()
        batch_de_dict[b] = calc_dge_filtered(b_ad, lk, thr)
    return calculate_avg_jaccard(mc_de, batch_de_dict, name)


# -------------------- Helpers --------------------


def rbo(l1, l2, p=0.9):
    """Minimal RBO implementation."""
    s, L = set(), set()
    rbo_sum = 0.0
    for i, (x, y) in enumerate(zip(l1, l2), 1):
        s.add(x)
        L.add(y)
        overlap = len(s & L)
        rbo_sum += (overlap / i) * (p ** (i - 1))
    return (1 - p) * rbo_sum


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if len(a | b) else 0.0


def calc_dge(ad, label_key):
    """Compute DGE without filtering."""
    group_cnt = ad.obs[label_key].value_counts()
    valid = group_cnt[group_cnt > 1].index
    ad = ad[ad.obs[label_key].isin(valid)]
    sc.tl.rank_genes_groups(
        ad, groupby=label_key, method="t-test", key_added="rank_genes"
    )
    df = sc.get.rank_genes_groups_df(ad, key="rank_genes", group=None)
    # df["group"] = df["group"].str.lower()
    return df


def prepare_topK(dge_df, ct, thr, K):
    """Filter by p-value, sort by score, return top-K gene names."""
    df = dge_df.query("group == @ct and pvals_adj < @thr")
    df = df.sort_values("scores", ascending=False)
    return df["names"].tolist()[:K]


def safe_kendall(list1, list2):
    """Compute Kendall only on shared genes."""
    shared = list(set(list1) & set(list2))
    if len(shared) < 2:
        return 0
    r1 = [list1.index(g) for g in shared]
    r2 = [list2.index(g) for g in shared]
    tau, _ = kendalltau(r1, r2)
    return tau


def add_celltype_percent_to_cols(df, ad, lk):
    pct = (ad.obs[lk].value_counts(normalize=True) * 100).to_dict()
    # pct = {k: v for k, v in pct.items()}
    rename = {ct: f"{ct} ({pct.get(ct, 0):.1f}%)" for ct in df.columns}
    return df.rename(columns=rename)


def get_batch_K(ad, lk, bk, thr=0.01, K_cap=200, fc_thr=0.5, fixed_K=200):
    batch_de = {b: calc_dge(ad[ad.obs[bk] == b], lk) for b in ad.obs[bk].unique()}

    if fixed_K is not None:
        K_dict = {ct: fixed_K for ct in ad.obs[lk].unique()}
        return batch_de, K_dict

    K_dict = {}
    for ct in ad.obs[lk].unique():
        vals = []
        for b, df in batch_de.items():
            n_b = (
                df["group"].eq(ct)
                & (df["pvals_adj"] < thr)
                & (df["logfoldchanges"] > fc_thr)
            ).sum()
            if n_b > 0:
                vals.append(n_b)
        K_dict[ct] = 0 if len(vals) == 0 else min(int(np.median(vals)), K_cap)

    return batch_de, K_dict


# -------------------- Main Function --------------------
def compute_dge_consistency(
    mc_ad,
    ad,
    lk,
    bk,
    thr=0.01,
    name=None,
    K_cap=200,
    rbo_p=0.9,
    save_path=None,
):
    def _ensure_lognorm(a):
        if a.X.max() > 20:
            print("error no lognorm")
        return a

    if "lognorm_gt" in ad.layers:
        ad.X = ad.layers["lognorm_gt"]
    ad = _ensure_lognorm(ad)
    mc_ad = _ensure_lognorm(mc_ad)

    mc_de = calc_dge(mc_ad, lk)
    batch_de, K_dict = get_batch_K(ad, lk, bk, thr, K_cap)

    rows_rbo, rows_kt, rows_jac = {}, {}, {}

    for ct in ad.obs[lk].unique():
        K = K_dict.get(ct, 0)
        if K == 0:
            rows_rbo[ct] = rows_kt[ct] = rows_jac[ct] = 0.0
            continue

        mc_top = prepare_topK(mc_de, ct, thr, K)
        rbo_vals, kt_vals, jac_vals = [], [], []

        for df in batch_de.values():
            bt = prepare_topK(df, ct, thr, K)
            if len(bt):
                rbo_vals.append(rbo(mc_top, bt, p=rbo_p))
                kt_vals.append(safe_kendall(mc_top, bt))
                jac_vals.append(jaccard(mc_top, bt))

        rows_rbo[ct] = float(np.nanmean(rbo_vals)) if len(rbo_vals) else 0.0
        rows_kt[ct] = float(np.nanmean(kt_vals)) if len(kt_vals) else 0.0
        rows_jac[ct] = float(np.nanmean(jac_vals)) if len(jac_vals) else 0.0

    rare_ct = get_rare(ad, lk)

    def _to_df(rows, metric):
        df = pd.DataFrame([rows], index=[name or metric])
        df["avg global"] = df.mean(axis=1)
        df["avg rare"] = df[rare_ct].mean(axis=1)
        rename = {}
        for ct in ad.obs[lk].unique():
            frac = (ad.obs[lk] == ct).mean() * 100
            rename[ct] = f"{ct} ({frac:.1f}%, K={K_dict.get(ct,0)})"
        return df.rename(columns=rename)

    df_rbo = _to_df(rows_rbo, "avg_rbo")
    df_kt = _to_df(rows_kt, "avg_kendall")
    df_jac = _to_df(rows_jac, "avg_jaccard")

    if save_path is not None:
        df_rbo.to_csv(f"{save_path}/dge_rbo.csv")
        df_kt.to_csv(f"{save_path}/dge_kendall.csv")
        df_jac.to_csv(f"{save_path}/dge_jaccard.csv")
    if "lognorm_gt" in ad.layers:
        ad.X = ad.layers["lognorm"]
    return df_rbo, df_kt, df_jac


def topk_per_niche(ad, k=10):
    groups = ad.uns["rank_genes_groups"]["names"].dtype.names
    return {
        g: sc.get.rank_genes_groups_df(ad, group=g)["names"].head(k).tolist()
        for g in groups
    }


def get_niche_markers(ad, ct, ct_key, niche_key, k=50):
    ad_ct = ad[ad.obs[ct_key] == ct].copy()
    ad_ct = ad_ct[ad_ct.obs[niche_key] != "Excluded"].copy()

    valid = ad_ct.obs[niche_key].value_counts()
    valid = valid[valid > 1].index.tolist()

    if len(valid) == 0:
        return {}

    sc.tl.rank_genes_groups(
        ad_ct,
        groupby=niche_key,
        groups=valid,
        reference="rest",
        method="wilcoxon",
        use_raw=False,
    )

    return topk_per_niche(ad_ct, k)


def summerize_dict(d, thr):
    # , np.mean(np.array(list(d.values())))
    return np.mean(np.array(list(d.values())) > thr)


def compare_niche_dge(ad, mc_ad, lk, ct, thr):
    sc_markers = get_niche_markers(ad, ct, lk, "niches_2D")
    mc_markers = get_niche_markers(mc_ad, ct, lk, "niches_2D")
    rbo_dict = {
        n: rbo(sc_markers[n], mc_markers.get(n, []), p=0.95) for n in sc_markers
    }
    jaccard_dict = {
        n: jaccard(sc_markers[n], mc_markers.get(n, [])) for n in sc_markers
    }
    return summerize_dict(rbo_dict, thr), summerize_dict(jaccard_dict, thr)


def celltype_niche_dge(ad, mc_ad, lk, name, save_path):
    r_df = pd.DataFrame(index=["rbo"])
    j_df = pd.DataFrame(index=["jaccard"])

    for ct in ad.obs[lk].unique():
        r, j = compare_niche_dge(ad, mc_ad, lk, ct, 0.01)
        r_df[ct] = r
        j_df[ct] = j
    r_df.index = [name]
    j_df.index = [name]
    r_df.to_csv(save_path + "/ct_niche_rbo.csv")
    j_df.to_csv(save_path + "/ct_niche_jaccard.csv")
    return r_df, j_df
