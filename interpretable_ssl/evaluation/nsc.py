import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

celltype_markers_mapped = {
    "Tumor cells": ["KRT19", "EPCAM", "KRT8", "KRT18"],
    "Fibroblasts": ["COL1A1", "COL3A1", "LUM", "DCN"],
    "Cytotoxic T cells": ["NKG7", "GNLY", "GZMB"],
    "Cycling immune cells": [
        "MKI67",
        "TOP2A",
    ],  # ⚠️ cycling state, not a main annotated cell type
    "Macrophages": ["LST1", "CD68", "CTSD"],
    "Monocytes": ["LST1"],  # ⚠️ not separated from macrophages as a distinct class
    "Regulatory T cells": ["FOXP3", "IL2RA"],
    "Vascular endothelium": ["PECAM1", "VWF"],
    "Pericytes": ["RGS5", "PDGFRB"],
    "Lymphatic endothelial cells": ["PDPN", "LYVE1"],
    "Basal epithelial cells": ["KRT5", "KRT14"],
    "Plasma cells": ["MZB1", "IGHG1"],
    "B cells": ["MS4A1"],  # ⚠️ very low abundance, not emphasized
    "Smooth muscle cells": ["ACTA2", "TAGLN"],
    "Mast cells": ["TPSAB1"],  # ⚠️ present at very low frequency, not discussed
    "Dendritic cells": ["IDO1", "FCER1A", "CCR7"],
    "Alveolar cells": ["SFTPA1", "SFTPC"],
    "Respiratory epithelium": ["KRT19", "SCGB1A1"],
}


def marker_expression_coverage(ad, lk, layer = None):
    rows = {}
    X = ad.layers[layer] if layer is not None else ad.X
    genes = ad.var_names.to_numpy()
    labels = ad.obs[lk].to_numpy()

    for ct, markers in celltype_markers_mapped.items():
        idx = labels == ct
        if idx.sum() == 0:
            rows[ct] = 0
            continue

        valid_markers = [m for m in markers if m in genes]
        print(ct, valid_markers)
        if len(valid_markers) == 0:
            rows[ct] = -1
            continue

        m_idx = np.isin(genes, valid_markers)
        prop = (X[idx][:, m_idx] > 0).mean(axis=0)
        rows[ct] = prop.mean()

    return pd.DataFrame([rows])


def marker_f1_wide(
    ad, markers, ct_key, layer="lognorm", batch_key=None, train_batches=None
):
    vals = {}
    cols = {}

    for ct, genes in markers.items():
        genes = [g for g in genes if g in ad.var_names]
        if len(genes) == 0:
            continue

        X = ad[:, genes].layers[layer]
        y = (ad.obs[ct_key].values == ct).astype(int)
        if y.sum() < 2 or y.sum() == len(y):
            continue

        if batch_key is not None and train_batches is not None:
            tr = ad.obs[batch_key].isin(train_batches).values
            te = ~tr
        else:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(y))
            k = int(0.8 * len(y))
            tr = idx[:k]
            te = idx[k:]

        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )

        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])

        col = f"{ct} ({len(genes)})"
        cols[ct] = col
        vals[col] = f1_score(y[te], pred)

    return pd.DataFrame([vals], columns=vals.keys())

def get_rare(ad=None, label_key=None, thr=0.25, labels=None):
    if labels is None:
        labels = ad.obs[label_key]
    labels = labels.astype(str)
    freq = labels.value_counts(normalize=True)
    thr = freq.quantile(0.25)
    rare = freq[freq < thr].index.tolist()
    return rare

def evaluate_markers(
    ad,
    mc_ad,
    lk,
    name,
    mc_key="SEACell",
    save_path=None,
):
    mc_covarage = marker_expression_coverage(mc_ad, lk)
    mc_ad.layers["lognorm"] = mc_ad.X
    mc_f1 = marker_f1_wide(mc_ad, celltype_markers_mapped, lk)

    mc_ids = ad.obs[mc_key].values
    idx = mc_ad.obs_names.get_indexer(mc_ids)
    ad.layers["mc"] = mc_ad.X[idx]
    sample_cov = marker_expression_coverage(ad, lk, layer = 'mc')
    ad.layers["lognorm"] = ad.X
    sample_f1 = marker_f1_wide(ad, celltype_markers_mapped, lk, layer = 'mc')
    dfs = [mc_covarage, mc_f1, sample_cov, sample_f1]
    for df in dfs:
        df.index = [name]
        df["global avg"] = df.mean(axis=1)
        rare_ct = get_rare(labels = ad.obs[lk])
        df["rare avg"] = df[
            [col for col in df.columns if any(ct in col for ct in rare_ct)]
        ].mean(axis=1)

    if save_path is not None:
        mc_covarage.to_csv(save_path + "/mc_marker_coverage.csv")
        mc_f1.to_csv(save_path + "/mc_marker_f1.csv")
        sample_cov.to_csv(save_path + "/sample_marker_coverage.csv")
        sample_f1.to_csv(save_path + "/sample_marker_f1.csv")
    return dfs
