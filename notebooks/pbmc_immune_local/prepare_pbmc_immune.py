"""Step 1: figshare Immune_ALL_human.h5ad -> data/scpoli/Immune_ALL_human_hvg.h5ad
         (4,000 HVGs, X log-normalized as shipped, layers['counts'] kept)
Step 2: (cell type x batch)-stratified 20% node-heldout split, TAG arbf_ctb20_seed0,
         exactly as modal_scripts/make_ctb_split.py (seed 0, MIN_FOR_TEST 2)
Step 3: cached train affinity graph (arbf, k=50, 50 PCs) in ./graphs

Load-if-exists at every step; FORCE_RERUN=True recomputes.

ASSUMPTION (not recorded anywhere in the repo): HVGs = scanpy highly_variable_genes
(seurat flavor, n_top_genes=4000, no batch_key) on the figshare log-normalized X, and X is
kept as shipped. Checked 2026-10-31 against files/notes/lambda_calibration.md: the density
(7.27%) matches exactly, so the gene set is very likely the same, but per-cell sum(X) is 338
vs 267 and max 11.21 vs 9.46, so the original X had a different normalization that none of
{figshare X, log1p(counts/total*1e3|1e4|median)} reproduces. Lambdas are re-measured locally,
so the recipe holds; absolute metric values are NOT directly comparable to the Modal arms.
files/notes/lambda_calibration.md confirms 4,000 genes, 33,506 cells, CSR X,
per-cell sum(X) ~267, counts integer -- we check those here.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp

FORCE_RERUN = False
t0 = time.time()

# ---------------------------------------------------------------- step 1: HVG file
if os.path.exists(HVG_H5) and not FORCE_RERUN:
    ad = sc.read_h5ad(HVG_H5); print(f"[1] loaded {HVG_H5}")
else:
    ad = sc.read_h5ad(RAW_H5)
    print(f"[1] raw: {ad}")
    print(f"    X max={ad.X.max():.3f}  layers={list(ad.layers)}  obs={list(ad.obs.columns)}")
    assert 'counts' in ad.layers, "expected layers['counts'] in the figshare file"
    assert ad.X.max() < 30, "X should already be log-normalized in the figshare file"
    sc.pp.highly_variable_genes(ad, n_top_genes=4000)
    ad = ad[:, ad.var['highly_variable']].copy()
    ad.X = sp.csr_matrix(ad.X).astype(np.float32)
    ad.layers['counts'] = sp.csr_matrix(ad.layers['counts']).astype(np.float32)
    os.makedirs(os.path.dirname(HVG_H5), exist_ok=True)
    ad.write_h5ad(HVG_H5); print(f"[1] wrote {HVG_H5}")
rs = np.asarray(ad.X.sum(1)).ravel()
print(f"[1] {ad.n_obs:,} cells x {ad.n_vars:,} genes | density {ad.X.nnz/ad.X.shape[0]/ad.X.shape[1]:.2%} "
      f"| mean per-cell sum(X) {rs.mean():.0f} (ref 267) | X max {ad.X.max():.2f} (ref 9.46) "
      f"| counts integer: {bool(np.allclose(ad.layers['counts'].data, np.round(ad.layers['counts'].data)))}")
print(f"    batches: {ad.obs[BATCH_KEY].value_counts().to_dict()}")
print(f"    {ad.obs[LABEL_KEY].nunique()} cell types")

# ---------------------------------------------------------------- step 2: split
os.makedirs(SPLIT_DIR, exist_ok=True)
if os.path.exists(TRAIN_H5) and os.path.exists(SPLIT_NPZ) and not FORCE_RERUN:
    d = np.load(SPLIT_NPZ, allow_pickle=True); ad_tr = sc.read_h5ad(TRAIN_H5)
    print(f"[2] loaded split: train {len(d['train_ids']):,} test {len(d['test_ids']):,}")
else:
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
    adp = get_preprocessed_adata(DS)        # same loader the Modal split used
    n = adp.n_obs
    cts = np.asarray(adp.obs[LABEL_KEY]).astype(str)
    bts = np.asarray(adp.obs[BATCH_KEY]).astype(str)
    names = np.asarray(adp.obs_names).astype(str)
    TEST_FRAC, SEED, MIN_FOR_TEST = 0.20, 0, 2
    rng = np.random.RandomState(SEED)
    test_idx, rows = [], []
    for key, idx in pd.Series(range(n)).groupby([pd.Series(cts), pd.Series(bts)]).groups.items():
        idx = np.asarray(list(idx)); m = len(idx)
        k = 0 if m < MIN_FOR_TEST else min(max(1, int(round(TEST_FRAC * m))), m - 1)
        pick = rng.choice(idx, size=k, replace=False) if k else np.array([], int)
        test_idx.append(pick)
        rows.append({"cell_type": key[0], "batch": key[1], "n": m, "n_test": k, "n_train": m - k})
    test_idx = np.sort(np.concatenate(test_idx))
    test_mask = np.zeros(n, bool); test_mask[test_idx] = True
    df = pd.DataFrame(rows).sort_values(["cell_type", "batch"])
    big = df[df.n >= MIN_FOR_TEST]
    assert (big.n_test >= 1).all() and (big.n_train >= 1).all()
    print(f"[2] {len(df)} (type,batch) groups; test {test_mask.sum():,} ({100*test_mask.mean():.2f}%) train {(~test_mask).sum():,}")
    np.savez(SPLIT_NPZ, train_ids=names[~test_mask].astype(object), test_ids=names[test_mask].astype(object))
    ad_tr = adp[~test_mask].copy()
    ad_tr.layers["lognorm"] = ad_tr.X.copy()
    if "conditions_combined" not in ad_tr.obs:
        ad_tr.obs["conditions_combined"] = ad_tr.obs[BATCH_KEY].astype(str).astype("category")
    ad_tr.write_h5ad(TRAIN_H5)
    df.to_csv(os.path.join(ROOT, 'notebooks', 'pbmc_immune_local', 'tables', 'ctb_split_groups.csv'), index=False)
    print(f"[2] wrote {TRAIN_H5} ({ad_tr.n_obs:,} cells) and {SPLIT_NPZ}")

# ---------------------------------------------------------------- step 3: train affinity
from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity
os.makedirs(os.path.join(ROOT, 'graphs'), exist_ok=True)
ds_name = f"{DS}_{TAG}_train"
if 'X_pca' not in ad_tr.obsm:
    sc.tl.pca(ad_tr, n_comps=50)   # same as adata_augmenter.py:233 does inside the trainer
aff = load_or_build_affinity(ad_tr, ds_name, "arbf", bk=BATCH_KEY, graph_dir=os.path.join(ROOT, 'graphs'))
A = sp.csr_matrix(aff); A.setdiag(0); A.eliminate_zeros(); co = A.tocoo()
b = pd.factorize(np.asarray(ad_tr.obs[BATCH_KEY]).astype(str))[0]
same = b[co.row] == b[co.col]
print(f"[3] train affinity shape={aff.shape} nnz={aff.nnz:,} | within-batch {100*same.mean():.2f}% cross-batch {100*(~same).mean():.2f}%")
print(f"done in {time.time()-t0:.0f}s")
