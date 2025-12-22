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


def make_mask(X_gt, X_obs):
    if sp.issparse(X_gt) or sp.issparse(X_obs):
        gt_pos = X_gt > 0
        obs_zero = X_obs == 0
        return gt_pos.multiply(obs_zero)
    else:
        return (X_gt > 0) & (X_obs == 0)


def to_dense(X):
    return X.toarray() if sp.issparse(X) else X


def dropout_recovery(
    masked_ad, mc_ad, mc_key, name, lk, save_path=None, gt_layer="lognorm_gt"
):
    if gt_layer not in masked_ad.layers:
        return
    mask = make_mask(masked_ad.layers[gt_layer], masked_ad.X)

    if mask.sum() == 0:
        return

    if "metacell" not in masked_ad.layers:
        mc_ids = masked_ad.obs[mc_key].values
        idx = mc_ad.obs_names.get_indexer(mc_ids)
        X_hat = mc_ad.X[idx]
    else:
        print("usingobsm key")
        X_hat = masked_ad.layers["metacell"]

    X_true = to_dense(masked_ad.layers[gt_layer])
    X_hat = to_dense(X_hat)
    mask = to_dense(mask).astype(bool)

    df = masked_corr_per_ct(X_true, X_hat, masked_ad.obs[lk], mask, name)
    if save_path is None:
        return df
    df.to_csv(save_path + "/dropout_recovery.csv")


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
    if "lognorm_gt" in ad.layers:
        gt = ad.layers["lognorm_gt"]
        obs = ad.X

        gene_mask = make_mask(gt, obs)
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
