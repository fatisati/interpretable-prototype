"""[D59] Build a (cell type x batch)-stratified node-heldout split for pbmc-immune,
plus its train affinity graph.

WHY A NEW SPLIT
---------------
The existing `arbf_nodeheldout20_seed0` split is stratified by BATCH only. The two
metrics that now carry the diagnosis -- within-batch purity and cross-batch
agreement -- are both computed per (metacell, batch) group, and the rare metrics
average over only 5 batches. Stratifying by (cell type x batch) guarantees every
(type, batch) combination is represented in BOTH halves, which is what makes those
decompositions stable.

STRUCTURE MATCHED TO THE EXISTING TEMPLATE (verified, not assumed):
  - split npz holds exactly {train_ids, test_ids}  (object arrays of obs_names)
  - the train h5ad carries layers['lognorm'], which is a BIT-EXACT copy of X
    (checked: max|X - lognorm| = 0.0)
  - obs['conditions_combined'] equals obs['study'] for this dataset (5 categories)

Also force-builds the TRAIN affinity graph here, once, on CPU -- otherwise each of
the three training runs would rebuild the same graph on a GPU.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/make_ctb_split.py
"""
import modal

app = modal.App("scproto-ctb-split")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in ["lung", "lung_fetal_donor", "lung_fetal_organoid", "brain",
                     "breast", "heart", "eye", "gut_fetal", "skin", "COVID",
                     "pancreas"]
    ])
)

volume = modal.Volume.from_name("scproto-data")

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY = "study", "final_annotation"
TAG = "arbf_ctb20_seed0"          # (cell type x batch)-stratified, 20% test, seed 0
TEST_FRAC = 0.20
SEED = 0
MIN_FOR_TEST = 2                  # a (type,batch) group needs >=2 cells to donate one


def _env():
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


@app.function(image=image, cpu=8, memory=32768, timeout=7200,
              volumes={"/vol": volume})
def build():
    import os
    _env()
    import numpy as np
    import pandas as pd
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata

    ad = get_preprocessed_adata(DS)
    n = ad.n_obs
    cts = np.asarray(ad.obs[LABEL_KEY]).astype(str)
    bts = np.asarray(ad.obs[BATCH_KEY]).astype(str)
    names = np.asarray(ad.obs_names).astype(str)
    print(f"[ctb] {DS}: {n:,} cells, {len(set(cts))} cell types, {len(set(bts))} batches")

    # --- stratify by (cell type, batch) ---
    rng = np.random.RandomState(SEED)
    test_idx = []
    rows = []
    for key, idx in pd.Series(range(n)).groupby(
            [pd.Series(cts), pd.Series(bts)]).groups.items():
        idx = np.asarray(list(idx))
        m = len(idx)
        if m < MIN_FOR_TEST:
            k = 0                                   # too small to split; keep in train
        else:
            k = max(1, int(round(TEST_FRAC * m)))   # at least one test cell
            k = min(k, m - 1)                       # never take the whole group
        pick = rng.choice(idx, size=k, replace=False) if k else np.array([], int)
        test_idx.append(pick)
        rows.append({"cell_type": key[0], "batch": key[1], "n": m,
                     "n_test": k, "n_train": m - k})
    test_idx = np.sort(np.concatenate(test_idx)) if test_idx else np.array([], int)
    test_mask = np.zeros(n, bool); test_mask[test_idx] = True

    df = pd.DataFrame(rows).sort_values(["cell_type", "batch"])
    print(f"[ctb] {len(df)} (type,batch) groups; "
          f"{(df.n_test == 0).sum()} too small to donate a test cell "
          f"(total {df.loc[df.n_test == 0, 'n'].sum()} cells)")
    print(f"[ctb] test {test_mask.sum():,} ({100*test_mask.mean():.2f}%)  "
          f"train {(~test_mask).sum():,}")
    print(df.to_string(index=False))

    # every (type,batch) group with >=MIN_FOR_TEST cells must appear in BOTH halves
    big = df[df.n >= MIN_FOR_TEST]
    assert (big.n_test >= 1).all() and (big.n_train >= 1).all(), \
        "a (type,batch) group is missing from one half -- stratification failed"

    outdir = "/vol/data/node_heldout"
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, f"{DS}_{TAG}_split.npz"),
             train_ids=names[~test_mask].astype(object),
             test_ids=names[test_mask].astype(object))
    print(f"[ctb] wrote {DS}_{TAG}_split.npz")

    # --- train h5ad, matching the existing template ---
    ad_tr = ad[~test_mask].copy()
    ad_tr.layers["lognorm"] = ad_tr.X.copy()            # template: lognorm == X exactly
    if "conditions_combined" not in ad_tr.obs:
        ad_tr.obs["conditions_combined"] = ad_tr.obs[BATCH_KEY].astype(str).astype("category")
    tr_path = os.path.join(outdir, f"{DS}_{TAG}_train.h5ad")
    ad_tr.write_h5ad(tr_path)
    print(f"[ctb] wrote {tr_path}  ({ad_tr.n_obs:,} cells)")
    volume.commit()

    # --- force-build the TRAIN affinity graph once, on CPU ---
    from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity
    ds_name = f"{DS}_{TAG}_train"
    os.makedirs("/vol/code/graphs", exist_ok=True)
    aff = load_or_build_affinity(ad_tr, ds_name, "arbf", bk=BATCH_KEY,
                                 graph_dir="/vol/code/graphs")
    print(f"[ctb] train affinity: shape={aff.shape} nnz={aff.nnz:,}")
    volume.commit()

    # how much cross-batch signal does the new TRAIN graph carry?
    import scipy.sparse as sp
    A = sp.csr_matrix(aff); A.setdiag(0); A.eliminate_zeros(); co = A.tocoo()
    b = pd.factorize(np.asarray(ad_tr.obs[BATCH_KEY]).astype(str))[0]
    same = b[co.row] == b[co.col]
    ct = np.asarray(ad_tr.obs[LABEL_KEY]).astype(str)
    print(f"[ctb] train graph: within-batch {100*same.mean():.2f}%  "
          f"cross-batch {100*(~same).mean():.2f}%")
    if (~same).sum():
        print(f"[ctb]   of cross-batch edges, same cell type: "
              f"{100*(ct[co.row[~same]] == ct[co.col[~same]]).mean():.1f}%")
    return {"n": int(n), "n_test": int(test_mask.sum()), "n_train": int((~test_mask).sum()),
            "groups": int(len(df)), "nnz_train_graph": int(aff.nnz)}


@app.local_entrypoint()
def main():
    print("[D59] building (cell type x batch)-stratified split + train graph")
    print(build.remote())
