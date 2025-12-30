import numpy as np
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score


def masked_corr_per_ct(X_true, X_hat, labels, mask, name):
    rare_ct = get_rare(labels=labels)

    labels = np.array(labels)
    uniq = np.unique(labels)
    out = {}
    for ct in uniq:
        idx = np.where(labels == ct)[0]
        m = mask[idx] > 0
        if not m.any():
            out[ct] = np.nan
            continue
        t = X_true[idx][m]
        h = X_hat[idx][m]
        if t.std() == 0 or h.std() == 0:
            out[ct] = np.nan
        else:
            out[ct] = np.corrcoef(t, h)[0, 1]
    avg = np.nanmean(list(out.values()))
    out["global avg"] = avg
    df = pd.DataFrame([out], index=[name])

    df["rare avg"] = df[rare_ct].mean(axis=1)

    # ---- micro-average (all cells together) ----
    m_all = mask > 0
    if m_all.any():
        t_all = X_true[m_all]
        h_all = X_hat[m_all]
        if t_all.std() == 0 or h_all.std() == 0:
            micro_avg = np.nan
        else:
            micro_avg = np.corrcoef(t_all, h_all)[0, 1]
    else:
        micro_avg = np.nan
    df["micro avg"] = micro_avg
    return df


def get_rare(ad=None, label_key=None, thr=0.25, labels=None):
    if labels is None:
        labels = ad.obs[label_key]
    labels = labels.astype(str)
    freq = labels.value_counts(normalize=True)
    thr = freq.quantile(0.25)
    rare = freq[freq < thr].index.tolist()
    return rare


import numpy as np
import scipy.sparse as sp


def dropout_mask(X_gt, X_obs):
    # ensure same representation
    if sp.issparse(X_gt) or sp.issparse(X_obs):
        X_gt = X_gt.tocsr() if sp.issparse(X_gt) else sp.csr_matrix(X_gt)
        X_obs = X_obs.tocsr() if sp.issparse(X_obs) else sp.csr_matrix(X_obs)

        return (X_gt > 0).multiply(X_obs == 0)

    else:
        return (X_gt > 0) & (X_obs == 0)


def changed_mask(X_gt, X_obs):
    if sp.issparse(X_gt) or sp.issparse(X_obs):
        X_gt = X_gt.tocsr() if sp.issparse(X_gt) else sp.csr_matrix(X_gt)
        X_obs = X_obs.tocsr() if sp.issparse(X_obs) else sp.csr_matrix(X_obs)

        return X_gt != X_obs

    else:
        return X_gt != X_obs


def to_dense(X):
    return X.toarray() if sp.issparse(X) else X


def gene_recovery(
    masked_ad,
    mc_ad,
    mc_key,
    name,
    lk,
    save_path=None,
    gt_layer="lognorm_gt",
):
    if gt_layer not in masked_ad.layers:
        print("no gt layer")
        return

    mask = changed_mask(masked_ad.layers["counts_gt"], masked_ad.layers["counts"])

    if mask.sum() == 0:
        return

    if "metacell" not in masked_ad.layers:
        mc_ids = masked_ad.obs[mc_key].values
        idx = mc_ad.obs_names.get_indexer(mc_ids)
        X_hat = mc_ad.X[idx]
    else:
        print("using obsm key")
        X_hat = masked_ad.layers["metacell"]

    X_true = to_dense(masked_ad.layers[gt_layer])
    X_hat = to_dense(X_hat)
    mask = to_dense(mask).astype(bool)
    snr_corr_df(
        masked_ad,
        X_true,
        X_hat,
        [lk, "niches_2D", "niches_3D", "fibroblast_subclusters", "EMT_niche"],
        name,
        save_path,
    )

    df = masked_corr_per_ct(X_true, X_hat, masked_ad.obs[lk], mask, name)
    if save_path is None:
        return df
    df.to_csv(save_path + f"/gene_recovery.csv")


def mc_label_majority_assigned(ad, mc_assign, lk, n_mc):
    out = {}
    for m in range(n_mc):
        idx = np.where(mc_assign == m)[0]
        out[m] = (
            Counter(ad.obs[lk].values[idx]).most_common(1)[0][0] if len(idx) else None
        )
    return out


def mc_label_topk(sim, ad, lk, k):
    out = {}
    for m in range(sim.shape[1]):
        idx = np.argsort(-sim[:, m])[:k]
        out[m] = Counter(ad.obs[lk].values[idx]).most_common(1)[0][0]
    return out


def cell_labels_from_mc(mc_assign, mc_labels):
    return np.array([mc_labels[m] for m in mc_assign])


def f1_per_ct_df(y_true, y_pred, name):
    rare_ct = get_rare(labels=y_true)
    y_true = y_true.values
    cts = np.unique(y_true)
    vals = {ct: f1_score(y_true == ct, y_pred == ct) for ct in cts}
    vals["global avg"] = np.mean(list(vals.values()))
    df = pd.DataFrame([vals], index=[name])
    df["rare avg"] = df[rare_ct].mean(axis=1)
    return df


def masked_f1(ad, y_true, y_pred, name):
    if "counts_gt" in ad.layers:
        gene_mask = dropout_mask(ad.layers["counts_gt"], ad.layers["counts"])
        cell_mask = np.asarray(gene_mask.sum(axis=1)).ravel() > 0

        if cell_mask.sum() > 0:
            return f1_per_ct_df(
                y_true[cell_mask],
                y_pred[cell_mask],
                name,
            )
    return None


def eval_mc_labeling(ad, lk, name, sim=None, path=None, mc_key="SEACell", k=10):
    if sim is None:
        mc_assign, _ = pd.factorize(ad.obs[mc_key].values)
        n_mc = mc_assign.max() + 1
    else:
        mc_assign = np.argmax(sim, axis=1)
        n_mc = sim.shape[1]

    mc_lab_assign = mc_label_majority_assigned(ad, mc_assign, lk, n_mc)
    y_true = ad.obs[lk]
    y_pred_a = cell_labels_from_mc(mc_assign, mc_lab_assign)
    assign_majority = f1_per_ct_df(y_true, y_pred_a, name)
    masked_df = masked_f1(ad, y_true, y_pred_a, name)

    if sim is not None:
        mc_lab_topk = mc_label_topk(sim, ad, lk, k)
        y_pred_b = cell_labels_from_mc(mc_assign, mc_lab_topk)
        topk_majority = f1_per_ct_df(y_true, y_pred_b, name)
    else:
        topk_majority = None

    if path is not None:
        assign_majority.to_csv(path + "/majority_f1.csv")
        if topk_majority is not None:
            topk_majority.to_csv(path + "/topk_f1.csv")
        if masked_df is not None:
            masked_df.to_csv(path + "/masked_f1.csv")
    return assign_majority, topk_majority


def calc_snr_per_cell_masked(gt, pred, mask):
    if sp.issparse(gt):
        gt = gt.toarray()
    if sp.issparse(pred):
        pred = pred.toarray()
    mask = np.asarray(mask)

    out = np.full(gt.shape[0], np.nan)
    for i in range(gt.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        diff = gt[i, m] - pred[i, m]
        signal = np.mean(gt[i, m] ** 2)
        noise = np.mean(diff**2)
        out[i] = 10 * np.log10(signal / (noise + 1e-12))
    return out


def corr_per_cell(gt, pred):
    gt = gt.toarray() if sp.issparse(gt) else gt
    pred = pred.toarray() if sp.issparse(pred) else pred

    return np.array(
        [
            (
                np.corrcoef(gt[i], pred[i])[0, 1]
                if gt[i].std() > 0 and pred[i].std() > 0
                else np.nan
            )
            for i in range(gt.shape[0])
        ]
    )


def snr_corr_df(ad, gt, pred, lks, name, path=None):
    mask = changed_mask(ad.layers["counts_gt"], ad.layers["counts"])
    changed = np.asarray(mask.sum(axis=1)).ravel() > 0

    snr = calc_snr_per_cell_masked(gt, pred, mask.toarray())
    corr = corr_per_cell(gt[changed], pred[changed])

    dfs = {}

    for lk in lks:
        if lk not in ad.obs:
            continue

        labels_all = ad.obs[lk].values
        labels_corr = labels_all[changed]

        for metric, vals, labels in [
            ("snr", snr, labels_all),
            ("corr", corr, labels_corr),
        ]:
            df = pd.DataFrame({metric: vals, lk: labels}).dropna()

            grp = df.groupby(lk)[metric].mean()
            micro_avg = np.nanmean(vals)
            macro_avg = grp.mean()
            rare_ct = get_rare(ad, lk)
            rare_avg = grp.loc[grp.index.isin(rare_ct)].mean()

            out = grp.to_frame().T
            out["micro_avg"] = micro_avg
            out["macro_avg"] = macro_avg
            out["rare_avg"] = rare_avg
            out.index = [name]

            dfs[f"{lk}_{metric}"] = out

            if path is not None:
                out.to_csv(f"{path}/{lk}_{metric}.csv")

    return dfs


def save_df(df, name, path, save_name):
    df.index = [name]
    if path is None:
        return
    df.to_csv(f"{path}/{save_name}")


def proto_f1(ad, mc_ad, lk, name, save_path):
    mc_idx = ad.obs["mc_idx"].values
    proto_labels = mc_ad.obs[lk].values
    pred = proto_labels[mc_idx]
    gt = ad.obs[lk].values
    classes = np.unique(np.concatenate([gt, pred]))

    f1_per_class = f1_score(gt, pred, labels=classes, average=None, zero_division=0)

    out = {c: f for c, f in zip(classes, f1_per_class)}

    out["f1_micro"] = f1_score(gt, pred, average="micro", zero_division=0)
    out["f1_macro"] = f1_score(gt, pred, average="macro", zero_division=0)
    # out["f1_weighted"] = f1_score(gt, pred, average="weighted", zero_division=0)

    df = pd.DataFrame([out])
    save_df(df, name, save_path, f"{lk}_f1.csv")
    return f"{lk}_f1", df


def niche_macro_f1_per_celltype(
    ad,
    mc_ad,
    name,
    save_path,
    exclude_niches=None,
):
    mc_idx = ad.obs["mc_idx"].values
    if exclude_niches is None:
        exclude_niches = []

    # prototype → niche labels
    proto_niche = mc_ad.obs["niches_2D"].values
    pred_niche = proto_niche[mc_idx]

    gt_niche = ad.obs["niches_2D"].values
    gt_ct = ad.obs["celltypes"].values

    out = {}

    for ct in np.unique(gt_ct):
        # cell-type subset
        mask = gt_ct == ct

        # exclude GT niche labels
        if exclude_niches:
            mask &= ~np.isin(gt_niche, exclude_niches)

        if mask.sum() < 2:
            continue

        out[ct] = f1_score(
            gt_niche[mask],
            pred_niche[mask],
            average="macro",
            zero_division=0,
        )
    df = pd.DataFrame([out])
    df["avg"] = df.mean(axis=1)
    save_df(df, name, save_path, "ct_niche_f1.csv")
    return "ct_niche_f1", df


def eval_mc_labeling_v2(ad, mc_ad, lk, name, save_path):
    dfs = {}
    labels = [lk, "niches_2D", "niches_3D", "fibroblast_subclusters", "EMT_niche"]
    for lk in labels:
        if lk in ad.obs and lk in mc_ad.obs:
            k, df = proto_f1(ad, mc_ad, lk, name, save_path)
            dfs[k] = df
    if "spatial" in ad.obsm:
        k, df = niche_macro_f1_per_celltype(ad, mc_ad, name, save_path, ["Excluded"])
        dfs[k] = df
    return dfs
