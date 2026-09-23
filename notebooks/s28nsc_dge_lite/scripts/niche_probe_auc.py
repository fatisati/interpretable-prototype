"""Per-(cell_type, niche) linear-probe AUC -- how much niche information is in gene
expression at all, and is it reachable by a bounded decoder?

WHY THIS EXISTS (vs niche_sep.py / sep2.py / sep3.py, which already probe PCA50/z):

  1. Those probe `X_pca[:, :50]`. PCA-50 is ITSELF a bottleneck -- a niche program living
     in low-variance directions is discarded before the probe ever sees it. To ask "is the
     info in gene expression AT ALL" the probe must run on the full 960-gene panel, which
     is exactly what the encoder receives (this dataset has no HVG subsetting).
  2. No control task. `chance = 1/n_niches` is the THEORETICAL floor; with 960 features and
     a few hundred positives a probe can fit noise. Hewitt & Liang (2019) require a
     random-label control, and the number you report is selectivity = real - control.
  3. Multiclass balanced accuracy per cell type cannot be joined to the DGE ground truth,
     which is per (cell_type, niche). This does one-vs-rest AUC per pair so the two tables
     merge on (cell_type, niche).
  4. Linear only. The linear/nonlinear gap separates "not linearly decodable" from
     "not decodable" -- different diagnoses, different fixes.

METHOD (standard linear-probing / linear-evaluation protocol):
    Alain & Bengio 2016, arXiv:1610.01644     -- linear classifier probes
    Hewitt & Liang 2019, EMNLP                -- control tasks, selectivity
    Voita & Titov 2020, EMNLP                 -- MDL probing (capacity confound)
    Xu et al. 2020, ICLR                      -- V-information / usable information

    for each cell_type:
      for each niche in that cell_type (excluding 'Excluded'):
          y      = (niche_of_cell == this_niche)        one-vs-rest, within cell type
          X      = feature set
          auc      = 5-fold CV ROC-AUC of an L2 logistic regression
          auc_ctrl = same, with y permuted (control task)
          selectivity = auc - auc_ctrl

Outputs one row per (cell_type, niche) to tables/s28nsc_niche_probe_auc.csv.
Load-if-exists per CLAUDE.md sec4; set FORCE_RERUN=True to recompute.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import h5py
from anndata._io.specs.registry import _REGISTRY, IOSpec


@_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
def _read_null(elem, _reader):
    return None


import anndata as ad
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- config
FORCE_RERUN = False

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(HERE)
OUT_CSV = os.path.join(NB_DIR, "tables", "s28nsc_niche_probe_auc.csv")

H5AD = os.environ.get(
    "PROBE_H5AD",
    r"G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data\spatial\s28nsc.h5ad",
)
DGE_GT = os.environ.get(
    "PROBE_DGE_GT",
    r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes"
    r"\interpretable-prototype\files\celltype_niches_full_s28nsc.csv",
)

CT_KEY, NICHE_KEY = "celltypes", "niches_2D"
EXCLUDE_NICHES = ("Excluded",)

MIN_POS = 30        # min cells of this cell type in this niche
MIN_CTRL = 30       # min cells of this cell type in other niches
MAX_CELLS_PER_CT = 6000   # subsample big cell types; keeps the sweep ~30 min not ~3 h
N_FOLDS = 5
SEED = 0


def probe_auc(X, y, estimator, n_folds=N_FOLDS, seed=SEED):
    """Cross-validated one-vs-rest ROC-AUC of `estimator` predicting binary y from X."""
    if y.sum() < n_folds or (~y).sum() < n_folds:
        return np.nan
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        est = estimator()
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def _linear():
    return LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=4, learning_rate=0.1, random_state=SEED
    )


def dge_gene_counts(path, ct_key="cell_type", niche_key="niche", alpha=0.05):
    """n significant DE genes per (cell_type, niche) from the saved ground truth."""
    if not os.path.exists(path):
        print(f"  DGE ground truth not found at {path} -- skipping the join")
        return None
    usecols = [ct_key, niche_key, "logfoldchanges", "pvals_adj"]
    df = pd.read_csv(path, usecols=usecols)
    df["sig"] = df["pvals_adj"] < alpha
    g = df.groupby([ct_key, niche_key])
    out = pd.DataFrame(
        {
            "n_dge_genes": g["sig"].sum(),
            "max_abs_logfc": g.apply(
                lambda d: d.loc[d["sig"], "logfoldchanges"].abs().max()
                if d["sig"].any()
                else np.nan
            ),
        }
    ).reset_index()
    return out.rename(columns={ct_key: "cell_type", niche_key: "niche"})


def main():
    if os.path.exists(OUT_CSV) and not FORCE_RERUN:
        d = pd.read_csv(OUT_CSV)
        print(f"loaded from {OUT_CSV}  ({len(d)} pairs)")
        print(d.round(3).to_string(index=False))
        return d

    t0 = time.time()
    print(f"reading {H5AD}")
    a = ad.read_h5ad(H5AD)

    xmax = float(a.X.max())
    print(f"  shape={a.shape}  X.max()={xmax:.2f}  -> "
          f"{'log-normalized' if xmax < 30 else 'RAW COUNTS (check!)'}")

    X_expr = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X_pca = np.asarray(a.obsm["X_pca"][:, :50])
    CT = a.obs[CT_KEY].astype(str).to_numpy()
    NI = a.obs[NICHE_KEY].astype(str).to_numpy()
    del a

    rng = np.random.default_rng(SEED)
    rows = []

    for ct in pd.unique(CT):
        m = np.flatnonzero(CT == ct)
        if len(m) > MAX_CELLS_PER_CT:
            m = rng.choice(m, MAX_CELLS_PER_CT, replace=False)
        y_all = NI[m]

        niches = [n for n in pd.unique(y_all) if n not in EXCLUDE_NICHES]
        keep_mask = ~np.isin(y_all, EXCLUDE_NICHES)
        m, y_all = m[keep_mask], y_all[keep_mask]
        if len(m) < 2 * MIN_POS:
            continue

        Xe = StandardScaler().fit_transform(X_expr[m])
        Xp = StandardScaler().fit_transform(X_pca[m])

        for niche in niches:
            y = y_all == niche
            npos, nctrl = int(y.sum()), int((~y).sum())
            if npos < MIN_POS or nctrl < MIN_CTRL:
                continue

            y_perm = rng.permutation(y)

            r = {
                "cell_type": ct,
                "niche": niche,
                "n_pos": npos,
                "n_ctrl": nctrl,
                "prevalence": npos / (npos + nctrl),
                "auc_expr960": probe_auc(Xe, y, _linear),
                "auc_expr960_ctrl": probe_auc(Xe, y_perm, _linear),
                "auc_pca50": probe_auc(Xp, y, _linear),
                "auc_pca50_ctrl": probe_auc(Xp, y_perm, _linear),
                "auc_pca50_gbm": probe_auc(Xp, y, _gbm),
                "auc_pca50_gbm_ctrl": probe_auc(Xp, y_perm, _gbm),
            }
            r["sel_expr960"] = r["auc_expr960"] - r["auc_expr960_ctrl"]
            r["sel_pca50"] = r["auc_pca50"] - r["auc_pca50_ctrl"]
            r["sel_pca50_gbm"] = r["auc_pca50_gbm"] - r["auc_pca50_gbm_ctrl"]
            rows.append(r)
            print(
                f"  {ct[:26]:<26} {niche[:24]:<24} n+={npos:<5} "
                f"expr960={r['auc_expr960']:.3f} (ctrl {r['auc_expr960_ctrl']:.3f})  "
                f"pca50={r['auc_pca50']:.3f}  gbm={r['auc_pca50_gbm']:.3f}",
                flush=True,
            )

    d = pd.DataFrame(rows)

    gt = dge_gene_counts(DGE_GT)
    if gt is not None:
        d = d.merge(gt, on=["cell_type", "niche"], how="left")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    d.to_csv(OUT_CSV, index=False)
    print(f"\ncomputed and saved to {OUT_CSV}  ({len(d)} pairs, {time.time()-t0:.0f}s)")
    return d


if __name__ == "__main__":
    main()
