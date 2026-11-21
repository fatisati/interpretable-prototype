"""Shared pipeline utilities used by more than one baseline.

Not a baseline adapter itself. Split creation, the affinity graph, and the
cluster count are all decided once by baselines/prepare_eval.py and saved;
everything here that touches them is read-only (see docs/architecture.md).
The one exception is `sweep_leiden_to_k`, which is shared *code* (every
embedding/graph-output baseline needs the same sweep logic), not shared
*data* -- there's nothing to "read" for that one, each baseline runs it on
its own embedding.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

    Used only by prepare_eval.py -- the one place allowed to create a split.
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
    created only by baselines/prepare_eval.py -- see docs/architecture.md
    for why every baseline must read the same one instead of making its own.
    """
    split_path = Path(split_path)
    meta_path = split_path.with_suffix(".json")
    requested_meta = {
        "seed": seed, "test_size": test_size, "label_key": label_key, "batch_key": batch_key,
    }

    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} does not exist -- run `python baselines/prepare_eval.py` for "
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
            f"{cluster_config_path} does not exist -- run `python baselines/prepare_eval.py` "
            f"for this dataset first."
        )
    return json.loads(cluster_config_path.read_text())


def sweep_leiden_to_k(adata, use_rep, target_k, resolutions=None):
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
    sc.pp.neighbors(ad, use_rep=use_rep)

    best_labels, best_gap, best_resolution, best_k = None, None, None, None
    for resolution in resolutions:
        sc.tl.leiden(ad, resolution=resolution, key_added="_sweep", flavor="igraph")
        labels = ad.obs["_sweep"].to_numpy()
        n_clusters = len(set(labels))
        gap = abs(n_clusters - target_k)
        if best_gap is None or gap < best_gap:
            best_gap, best_labels, best_resolution, best_k = gap, labels.copy(), resolution, n_clusters

    print(f"[cluster] target_k={target_k} -> resolution={best_resolution} gave {best_k} clusters")
    return best_labels, best_resolution
