"""Shared pipeline utilities used by more than one baseline.

Not a baseline adapter itself. Split creation, the affinity graph, and the
cluster count are all decided once by evaluation/prepare_eval.py and saved;
everything here that touches them is read-only (see docs/architecture.md).
The one exception is `sweep_leiden_to_k`, which is shared *code* (every
embedding/graph-output baseline needs the same sweep logic), not shared
*data* -- there's nothing to "read" for that one, each baseline runs it on
its own embedding.
"""
import json
import platform
from importlib.metadata import PackageNotFoundError, version
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sklearn.model_selection import train_test_split


def looks_like_raw_counts(values, threshold=0.01):
    """True if `values` (a matrix or array) looks like real integer counts,
    rather than already-normalized data stored in what might look like a
    counts layer (see datasets/pancreas.md for why this check exists at all
    -- trusting a layer's name here has silently gone wrong before).
    """
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return False
    frac_noninteger = float(np.mean(np.abs(nonzero - np.round(nonzero)) > 1e-3))
    return frac_noninteger < threshold


def compute_pca(adata, layer, n_comps):
    """PCA needs log-normalized data, not raw counts. Some datasets' X is
    already log-normalized, some (heart atlas) ship real raw counts that
    still need normalizing first -- checked directly, not assumed, same as
    scvi_baseline.py's gene_likelihood choice.

    Shared by prepare_eval.py and any baseline (e.g. SEACells) that needs
    the exact same PCA a saved affinity graph was built from -- not a
    second, independently-computed one that merely uses the same settings.
    """
    ad = adata.copy()
    values = ad.layers[layer] if layer else ad.X
    if looks_like_raw_counts(values):
        if layer:
            ad.X = ad.layers[layer].copy()
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    elif layer:
        ad.X = ad.layers[layer].copy()

    sc.pp.pca(ad, n_comps=n_comps)
    return ad.obsm["X_pca"]


def stratified_split(adata, label_key, batch_key, test_size, seed):
    """Split so every (cell type, batch) group keeps its proportion on both
    sides -- a plain random split can wipe out a rare group by chance alone.
    Groups with only one cell can't be split (sklearn needs >=2 per group
    to stratify), so those single cells go to train.

    Used only by prepare_eval.py -- the one place allowed to create a split,
    and only when the dataset has no frozen split yet (or --refresh-split).
    """
    groups = adata.obs[label_key].astype(str) + "||" + adata.obs[batch_key].astype(str)
    positions = np.arange(adata.n_obs)

    group_sizes = groups.value_counts()
    splittable = groups.isin(group_sizes[group_sizes >= 2].index).to_numpy()

    train_pos = list(positions[~splittable])
    test_pos = []
    if splittable.any():
        split_train, split_test = train_test_split(
            positions[splittable],
            test_size=test_size,
            stratify=groups[splittable],
            random_state=seed,
        )
        train_pos += list(split_train)
        test_pos += list(split_test)

    return np.array(sorted(train_pos)), np.array(sorted(test_pos))


def load_split(adata, label_key, batch_key, test_size, seed, split_path):
    """Read-only: errors if `split_path` is missing or was built with
    different parameters, instead of silently creating one. Splits are
    created only by evaluation/prepare_eval.py and committed under
    datasets/frozen/<dataset>/ (see evaluation/frozen.py) -- see
    docs/architecture.md for why every baseline must read the same one
    instead of making its own.
    """
    split_path = Path(split_path)
    meta_path = split_path.parent / "split.json"  # split.csv or split.csv.gz
    requested_meta = {
        "seed": seed, "test_size": test_size, "label_key": label_key, "batch_key": batch_key,
    }

    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} does not exist -- run `python evaluation/prepare_eval.py` for "
            f"this dataset first. Baselines only read the split, they don't create it."
        )

    saved_meta = json.loads(meta_path.read_text())
    saved_params = {k: saved_meta[k] for k in requested_meta}
    if saved_params != requested_meta:
        raise ValueError(
            f"{split_path} was created with {saved_params}, but this call asked for "
            f"{requested_meta} -- they must match, since a different seed or test_size "
            f"would produce a different split. Re-run prepare_eval.py with the parameters "
            f"you actually want, or pass the ones already on disk."
        )
    saved = pd.read_csv(split_path).set_index("cell_id")["split"]
    if set(saved.index) != set(adata.obs_names):
        raise ValueError(
            f"{split_path} does not match this dataset's cells "
            f"({len(saved.index)} saved vs {adata.n_obs} in adata) -- "
            f"check you have the right h5ad file, or re-run prepare_eval.py."
        )
    print(f"[split] loaded {split_path}")
    return saved.loc[adata.obs_names].to_numpy()


def load_cluster_config(cluster_config_path):
    """Read-only: the target cluster count (K) is decided once by
    prepare_eval.py and saved here. Errors if missing rather than letting
    a baseline derive its own -- see docs/architecture.md's Clustering
    section for why every baseline must hit the same K.
    """
    cluster_config_path = Path(cluster_config_path)
    if not cluster_config_path.exists():
        raise FileNotFoundError(
            f"{cluster_config_path} does not exist -- run `python evaluation/prepare_eval.py` "
            f"for this dataset first."
        )
    return json.loads(cluster_config_path.read_text())


# Both are scanpy's defaults; written out so the seed is visible and recorded in
# config.json. A fixed seed repeats only on the same machine and package versions:
# the same embedding + same seeds gave different clusters on Windows vs Mac
# (ARI 0.78, docs/decisions/0003 step 7), so official clusters are frozen as files.
NEIGHBORS_SEED = 0  # sc.pp.neighbors: its neighbour search is approximate (pynndescent)
LEIDEN_SEED = 0


def sweep_leiden_to_k(adata, use_rep, target_k, resolutions=None, max_refine_steps=12):
    """Leiden doesn't take a target cluster count directly the way k-means
    would, and its resolution parameter doesn't map cleanly onto cluster
    count -- the count can stay flat across a range of resolution values,
    then jump. So: sweep a small range of resolution values and keep
    whichever one's cluster count is closest to target_k, instead of
    solving for an exact count.

    Shared by every baseline whose method only produces an embedding or a
    graph (scVI, Harmony, ComBat, BBKNN) -- SEACells doesn't call this at
    all, since clustering is intrinsic to that method (see
    docs/architecture.md's Clustering section).

    Builds the neighbor graph once, then sweeps resolution on top of it --
    the expensive step only runs once, not once per candidate resolution.
    After the coarse grid, bisects between the two grid values that bracket
    target_k (see step 2 below) so the result lands close to K, not just
    on whichever grid value happens to be nearest.
    """
    if resolutions is None:
        # Log-spaced: target_k here can be a handful of cell-type-level
        # clusters or (n_cells // 75) metacell-level clusters in the
        # hundreds, and resolution doesn't scale linearly with cluster
        # count -- calibrated directly on this project's data (heart
        # atlas, ~18.6k cells): resolution ~15-20 was needed to reach
        # ~200-250 clusters, nowhere close to reachable below resolution 2.
        resolutions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0,
                       7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    ad = adata.copy()
    t0 = time.perf_counter()
    sc.pp.neighbors(ad, use_rep=use_rep, random_state=NEIGHBORS_SEED)
    print(f"[cluster] neighbour graph on {ad.n_obs:,} cells: {time.perf_counter() - t0:.0f}s")

    tried = {}  # resolution -> (n_clusters, labels)

    def try_resolution(resolution):
        t = time.perf_counter()
        sc.tl.leiden(ad, resolution=resolution, key_added="_sweep", flavor="igraph",
                     random_state=LEIDEN_SEED)
        labels = ad.obs["_sweep"].to_numpy()
        tried[resolution] = (len(set(labels)), labels.copy())
        # One line per attempt, so a long search shows it is moving and where.
        print(f"[cluster]   try {len(tried):2d}: resolution={resolution:.4g} -> "
              f"{tried[resolution][0]} clusters (target {target_k}), "
              f"{time.perf_counter() - t:.0f}s")

    # 1. Coarse grid: find roughly where target_k sits.
    for resolution in resolutions:
        try_resolution(resolution)

    # 2. Refine: the coarse grid can jump right over target_k (heart atlas:
    # 15 -> 193 clusters, 20 -> 265, target 248). Bisect between the
    # closest grid value below target_k and the closest above, on a log
    # scale, until exact or out of tries. Cluster count isn't perfectly
    # monotone in resolution, so the bracket is re-read from everything
    # tried so far each step, not assumed.
    for _ in range(max_refine_steps):
        if any(k == target_k for k, _ in tried.values()):
            break
        below = [r for r, (k, _) in tried.items() if k < target_k]
        above = [r for r, (k, _) in tried.items() if k > target_k]
        if not below or not above:
            break  # target_k outside the grid's reach -- keep the closest
        lo, hi = max(below), min(above)
        if lo >= hi:
            break  # non-monotone region, no clean bracket
        mid = float(np.sqrt(lo * hi))
        if mid in tried or hi / lo < 1.001:
            break
        try_resolution(mid)

    best_resolution = min(tried, key=lambda r: (abs(tried[r][0] - target_k), r))
    best_k, best_labels = tried[best_resolution]
    print(f"[cluster] target_k={target_k} -> resolution={best_resolution:.4g} gave {best_k} clusters "
          f"({len(tried)} resolutions tried)")
    return best_labels, best_resolution


def environment_versions():
    """Exact versions of everything that can change a baseline's output
    (repo rule: see CLAUDE.md), saved into config.json next to the result."""
    versions = {"python": platform.python_version()}
    for pkg in ["scvi-tools", "torch", "scanpy", "anndata", "leidenalg", "igraph",
                "pynndescent", "numpy", "pandas"]:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def cluster_embedding_file(run_dir):
    """The one clustering code path for every embedding-output baseline
    (scVI, Harmony, ComBat). Reads <run_dir>/embedding.csv (latent_*
    columns + split), clusters twice to the shared K from the dataset's
    cluster_config.json, and rewrites in place:
      - embedding.csv's cluster_train_run / cluster_full_run columns
      - config.json's clustering fields (and environment versions)
    Training scripts call this right after saving the embedding;
    evaluation/recluster.py calls it alone, so clustering can change
    without retraining -- and both paths run identical code.

    cluster_train_run: train cells only (that run's whole population),
    test cells get NA. cluster_full_run: every cell -- cohesion for a test
    cell needs every other cell sharing its cluster, train cells included.
    Which rows count for *assignment* scoring is the metrics step's call.
    """
    run_dir = Path(run_dir)
    emb_path, config_path = run_dir / "embedding.csv", run_dir / "config.json"
    df = pd.read_csv(emb_path)
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError(f"{emb_path} has no latent_* columns -- nothing to cluster")
    cluster_config = load_cluster_config(run_dir.parent / "cluster_config.json")

    # float32: what the model produced before the CSV round-trip.
    ad = anndata.AnnData(obs=pd.DataFrame(index=df["cell_id"].astype(str).to_numpy()))
    ad.obsm["X_emb"] = df[latent_cols].to_numpy(dtype=np.float32)
    train_mask = (df["split"] == "train").to_numpy()

    train_labels, resolution_train = sweep_leiden_to_k(
        ad[train_mask], use_rep="X_emb", target_k=cluster_config["k_train"]
    )
    full_labels, resolution_full = sweep_leiden_to_k(
        ad, use_rep="X_emb", target_k=cluster_config["k_full"]
    )

    df["cluster_train_run"] = pd.NA
    df.loc[train_mask, "cluster_train_run"] = train_labels
    df["cluster_full_run"] = full_labels
    df.to_csv(emb_path, index=False)

    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update({
        "k_train_target": cluster_config["k_train"],
        "k_full_target": cluster_config["k_full"],
        "resolution_train": resolution_train,
        "resolution_full": resolution_full,
        "n_clusters_train": int(pd.Series(train_labels).nunique()),
        "n_clusters_full": int(pd.Series(full_labels).nunique()),
        "neighbors_seed": NEIGHBORS_SEED,
        "leiden_seed": LEIDEN_SEED,
        "environment": environment_versions(),
    })
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[cluster] wrote cluster columns to {emb_path}")
    return config
