"""Rare-cell-type node holdout: can a method place a rare population it never saw?

The node-holdout experiment (`notebooks/node_heldout_modularity.ipynb`) removes 20% of
cells uniformly at random. This one removes them where it actually hurts: a **subset of
the rare cell types** is deleted from the training data entirely -- absent from the
affinity graph, from Stage-1 pretraining, from Stage-2 training and from early stopping
-- and the model is then asked to place those cells through a frozen encoder.

Only *some* rare types are held out (default half, never all), so the model still sees
rare populations during training; what it has never seen is *these* ones. That is the
question the reviewer's rare-cell claim really rests on: is scProto's rare-cell
advantage memorisation of populations it was trained on, or structure it can impose on
a population it meets for the first time?

Arms, all trained on the *same* reduced training set:

    scVI (default) + Leiden     kNN placement of test cells in the scVI latent
    scVI (default) + SEACells   same
    scVI + scProto Stage 2      native: frozen encoder -> prototype argmax

This fixes an asymmetry the node-holdout notebook had to document as a caveat: there,
the two-step baselines' encoders had been trained on every test cell's expression in
their original full-data run. Here every arm's encoder sees the reduced training set and
nothing else.

Placement of unseen cells is the one thing the baselines cannot do natively -- Leiden
and SEACells are not inductive. Rather than exclude them (the node-holdout notebook's
choice), they are given the standard extension: a held-out cell inherits the majority
cluster of its k nearest training cells *in their own scVI latent*. That is the
strongest reasonable version of these baselines, and it is stated as part of the method
rather than presented as if they were inductive.

Metrics, all restricted to held-out cells and reported per batch (mean ± std, paired
one-sided Wilcoxon vs. the scProto arm):
  * modularity on the untouched full-dataset graph, restricted to edges touching a
    held-out cell -- same statistic as Table 1 (`calc_modularity_per_batch`)
  * recovery, homogeneity and concentration of the held-out rare types
  * same-cluster rate on rare-touching edges

Everything is cached: the split, the train-only h5ad, the trained models and the
baselines all reload if present.
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.dataset_configs import DATASETS, register_dataset
from interpretable_ssl.evaluation.batch_correct_baselines import get_canonical_affinity
from interpretable_ssl.evaluation.mc_metric_utils import calc_modularity_per_batch, get_rare


# ---------------------------------------------------------------------------
# selection + split
# ---------------------------------------------------------------------------


def select_heldout_rare_types(ad, label_key, frac_types=0.5, max_types=None,
                              min_cells=20, seed=0):
    """Choose which rare cell types to hold out.

    Rare types come from the codebase's own definition (`get_rare`: bottom-quartile
    global frequency), so "rare" means here exactly what it means in every other table.
    Types with fewer than `min_cells` cells are skipped -- with a handful of cells the
    per-batch metrics are noise, and removing them barely changes training.

    Deliberately holds out only a fraction: the model must still see *some* rare
    populations, otherwise the experiment would measure "can it handle rarity at all"
    rather than "can it place a rare population it has not met".

    Selection is by ascending frequency (rarest first), not random -- reproducible, and
    it targets the hardest cases rather than letting a seed decide how hard the test is.
    """
    rare = get_rare(ad, label_key)
    freq = ad.obs[label_key].value_counts()
    eligible = [t for t in rare if freq.get(t, 0) >= min_cells]
    if not eligible:
        raise ValueError(
            f"no rare cell type has >= {min_cells} cells (rare types: {rare}); "
            f"lower min_cells or pick the held-out types by hand."
        )

    eligible = sorted(eligible, key=lambda t: freq[t])
    n = max_types if max_types is not None else max(1, int(round(frac_types * len(eligible))))
    n = min(n, len(eligible) - 1) if len(eligible) > 1 else 1   # never hold out every rare type
    chosen = eligible[:n]

    print(f"rare types ({len(rare)}): {rare}")
    print(f"eligible (>= {min_cells} cells): {eligible}")
    print(f"holding out {len(chosen)}/{len(eligible)}: "
          + ", ".join(f"{t} (n={freq[t]})" for t in chosen))
    return chosen


def split_rare_heldout(ad, label_key, batch_key, heldout_types, cell_fraction=1.0,
                       min_train_per_batch=50, seed=0):
    """Hold out `cell_fraction` of the cells belonging to `heldout_types`.

    cell_fraction=1.0 (default) removes those types completely -- the model never sees a
    single cell of them. Lower it to leave a few behind if you want "severely
    under-represented" rather than "absent".

    A batch is never reduced below `min_train_per_batch` training cells. That guard
    matters for more than sample size: every held-out cell's batch value must still be
    one the trained model has seen, otherwise the frozen-forward-pass assumption breaks
    and the encoder path would silently fall back to a query-mapping retrain, leaking
    training signal into an "unseen cell" evaluation.

    Returns (train_idx, test_idx) as positional index arrays.
    """
    rng = np.random.default_rng(seed)
    n = ad.n_obs
    labels = ad.obs[label_key].astype(str).values
    batches = ad.obs[batch_key].astype(str).values if batch_key else np.zeros(n, dtype=str)

    heldout_types = [str(t) for t in heldout_types]
    candidate = np.isin(labels, heldout_types)
    test_mask = np.zeros(n, dtype=bool)

    for b in pd.unique(batches):
        idx_b = np.where((batches == b) & candidate)[0]
        if len(idx_b) == 0:
            continue
        n_b_total = int((batches == b).sum())
        n_take = int(round(cell_fraction * len(idx_b)))
        n_take = min(n_take, max(0, n_b_total - min_train_per_batch))
        if n_take < len(idx_b):
            print(f"  batch {b!r}: holding out {n_take}/{len(idx_b)} candidate cells "
                  f"(capped to keep >= {min_train_per_batch} train cells)")
        chosen = idx_b if n_take >= len(idx_b) else rng.choice(idx_b, size=n_take, replace=False)
        test_mask[chosen] = True

    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    print(f"held-out: {len(test_idx)}/{n} cells ({test_mask.mean():.2%}), "
          f"{len(train_idx)} train cells remain")
    for t in heldout_types:
        tot = int((labels == t).sum())
        held = int((labels[test_mask] == t).sum())
        print(f"    {t:<30} {held}/{tot} held out"
              + ("  (type fully absent from training)" if held == tot else ""))
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------


def load_full_counts_adata(ds_id):
    """Full dataset with raw counts in X -- the reference AnnData every arm's
    assignments are aligned to, and the source the train-only subset is written from."""
    ad = SingleCellDataset(name=ds_id, use_counts=True, **DATASETS[ds_id]).adata
    print(f"[{ds_id}] full data: {ad.n_obs} cells x {ad.n_vars} genes")
    return ad


def write_train_dataset(ds_id, ad_full, train_idx, tag, data_dir=None, K=None):
    """Write the train-only AnnData and register it as its own dataset.

    Registering rather than filtering in-place is what makes the holdout real: the
    trainer builds its affinity graph from this file, and the graph's filename encodes
    this dataset's own name and cell count, so held-out cells cannot appear as a node in
    it, cannot enter the edge sampler, and contribute no gradient anywhere.
    """
    data_dir = data_dir or os.path.join(os.environ.get("CODE_DIR", "."), "rare_heldout_data")
    os.makedirs(data_dir, exist_ok=True)

    train_ds_id = f"{ds_id}_{tag}"
    path = os.path.join(data_dir, f"{train_ds_id}.h5ad")
    if not os.path.exists(path):
        ad_full[train_idx].copy().write_h5ad(path)
        print(f"[{ds_id}] wrote train-only AnnData ({len(train_idx)} cells): {path}")
    else:
        print(f"[{ds_id}] reusing train-only AnnData: {path}")

    cfg = DATASETS[ds_id]
    register_dataset(
        train_ds_id, path,
        batch_key=cfg.get("batch_key"),
        label_key=cfg["label_key"],
        num_prototypes=K or cfg["num_prototypes"],
    )
    return train_ds_id, path


def load_or_make_split(ds_id, ad_full, label_key, batch_key, tag, heldout_types,
                       cell_fraction, min_train_per_batch, seed, data_dir=None):
    """Split, cached to disk by cell id.

    Cached rather than recomputed so a resumed session scores the checkpoint against the
    split it was actually trained on, instead of trusting a reseeded recomputation to
    land in the same place.
    """
    data_dir = data_dir or os.path.join(os.environ.get("CODE_DIR", "."), "rare_heldout_data")
    os.makedirs(data_dir, exist_ok=True)
    split_path = os.path.join(data_dir, f"{ds_id}_{tag}_split.npz")

    if os.path.exists(split_path):
        d = np.load(split_path, allow_pickle=True)
        train_idx = ad_full.obs_names.get_indexer(d["train_ids"])
        test_idx = ad_full.obs_names.get_indexer(d["test_ids"])
        assert (train_idx >= 0).all() and (test_idx >= 0).all(), (
            f"cached split {split_path} references cells absent from the current data -- "
            f"delete it and re-run."
        )
        print(f"[{ds_id}] loaded cached split: {split_path}")
        return train_idx, test_idx, list(d["heldout_types"])

    train_idx, test_idx = split_rare_heldout(
        ad_full, label_key, batch_key, heldout_types,
        cell_fraction=cell_fraction, min_train_per_batch=min_train_per_batch, seed=seed,
    )
    np.savez(split_path,
             train_ids=ad_full.obs_names.values[train_idx],
             test_ids=ad_full.obs_names.values[test_idx],
             heldout_types=np.array(heldout_types, dtype=object))
    print(f"[{ds_id}] saved split: {split_path}")
    return train_idx, test_idx, heldout_types


# ---------------------------------------------------------------------------
# placing unseen cells
# ---------------------------------------------------------------------------


def prepare_test_adata(ad_full, test_idx, batch_key):
    """Held-out cells in the model's own input space: X = log1p(counts), counts kept in
    a layer, plus the `conditions_combined` column scarches' dataset layer requires."""
    from interpretable_ssl.trainers.scvi_proto import log1p_matrix
    from interpretable_ssl.trainers.scpoli_helpers import add_condition_combined

    ad_test = ad_full[test_idx].copy()
    if "counts" not in ad_test.layers:
        ad_test.layers["counts"] = ad_test.X.copy()
    ad_test.X = log1p_matrix(ad_test.layers["counts"])
    add_condition_combined(ad_test, [batch_key])
    return ad_test


def assign_unseen_scproto(t, ad_test):
    """Prototype assignment for cells the model has never seen, via a frozen forward
    pass -- scProto's native inductive path (`_get_assignments`' 'proto' branch, applied
    to an arbitrary AnnData).

    The compatibility assertion is not decorative: if a held-out cell carried a batch
    value absent from training, `encode_adata` would take the query-mapping branch and
    retrain, which would leak training signal into an unseen-cell evaluation.
    """
    assert t.check_conditions_compatible(t.model, ad_test), (
        "held-out cells carry batch values the model never saw -- encode_adata would "
        "retrain instead of doing a frozen forward pass. Raise min_train_per_batch."
    )
    with torch.no_grad():
        z = t.encode_adata(ad_test, t.model, z_idx=1)
        return t.model.prototypes(z).argmax(dim=1).cpu().numpy(), z.cpu().numpy()


def assign_unseen_knn(z_train, labels_train, z_test, k=15):
    """Majority cluster of the k nearest training cells in the scVI latent.

    Leiden and SEACells are transductive -- they label the cells they were run on and
    have no mechanism for a new one. This is the standard way to extend them, and it is
    a genuinely strong baseline here: it gets the same encoder, the same latent space,
    and full access to every training cell's cluster label.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k, len(z_train))).fit(z_train)
    _, idx = nn.kneighbors(z_test)
    neigh = np.asarray(labels_train)[idx]                       # (n_test, k)
    out = np.empty(len(z_test), dtype=int)
    for i, row in enumerate(neigh):
        vals, counts = np.unique(row, return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def subgraph_touching(aff, mask, require="any"):
    """Edges of `aff` touching (`any`) or contained in (`all`) the masked node set,
    returned as a matrix of the same shape with every other edge zeroed -- so node
    indices stay aligned with the assignment arrays."""
    A = sp.csr_matrix(aff).tocoo()
    mask = np.asarray(mask)
    keep = (mask[A.row] | mask[A.col]) if require == "any" else (mask[A.row] & mask[A.col])
    out = sp.coo_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape).tocsr()
    return (out + out.T) / 2


def same_cluster_rate(aff, assignments, cell_mask=None, require="any"):
    """Fraction of edges whose endpoints share a cluster. Stays interpretable on small,
    sparse edge subsets where Newman modularity's global-degree null term gets noisy."""
    A = sp.csr_matrix(aff)
    A = ((A + A.T) / 2).tocoo()
    upper = A.row < A.col
    rows, cols = A.row[upper], A.col[upper]
    if cell_mask is not None:
        cell_mask = np.asarray(cell_mask)
        em = (cell_mask[rows] | cell_mask[cols]) if require == "any" else (cell_mask[rows] & cell_mask[cols])
        rows, cols = rows[em], cols[em]
    if len(rows) == 0:
        return {"rate": float("nan"), "n_edges": 0}
    a = np.asarray(assignments)
    return {"rate": float((a[rows] == a[cols]).mean()), "n_edges": int(len(rows))}


# ---------------------------------------------------------------------------
# held-out rare-type metrics
# ---------------------------------------------------------------------------


def heldout_rare_metrics_per_batch(obs, mc_col, label_key, batch_key, heldout_types,
                                   test_mask):
    """Per-batch metrics for the held-out types, computed on held-out cells only.

    `obs` must cover ALL cells (train + test) with their assignments, because metacell
    composition is a property of the whole partition; only the *evaluation* is
    restricted to held-out cells.

    recovery    fraction of a held-out type's test cells whose metacell has that type as
                its majority label -- i.e. did the type get metacells of its own. Same
                quantity the paper's rare-cell F1 calls recall.
    homogeneity mean over held-out cells of the fraction of their metacell sharing their
                label. Same formula as the rare table's homogeneity.
    concentration share of a type's test cells falling in its single most-used metacell
                -- did they land together, or scatter across the partition.
    n_metacells number of distinct metacells the type's test cells landed in
                (lower = more concentrated), normalised by the type's test-cell count.

    Returned as per-batch lists so they can be paired across methods for a Wilcoxon
    test, matching how every other rare-cell number in this rebuttal is tested.
    """
    obs = obs.copy()
    obs["_test"] = np.asarray(test_mask)
    heldout_types = [str(t) for t in heldout_types]

    mc_sizes = obs[mc_col].value_counts()
    mc_label_counts = obs.groupby([mc_col, label_key], observed=True).size()
    mc_majority = obs.groupby(mc_col, observed=True)[label_key].agg(
        lambda s: s.value_counts().idxmax()
    )

    out = {"recovery": [], "homogeneity": [], "concentration": [], "n_mc_per_cell": []}
    batches = obs[batch_key].astype(str).values if batch_key else np.zeros(len(obs), dtype=str)

    for b in pd.unique(batches):
        sub = obs[(batches == b) & obs["_test"].values & obs[label_key].astype(str).isin(heldout_types)]
        if sub.empty:
            continue

        rec, hom, con, nmc = [], [], [], []
        for ct, cells in sub.groupby(sub[label_key].astype(str), observed=True):
            mcs = cells[mc_col]
            rec.append(float((mcs.map(mc_majority) == ct).mean()))

            frac = np.array([
                mc_label_counts.get((m, ct), 0) / mc_sizes[m] for m in mcs
            ], dtype=float)
            hom.append(float(frac.mean()))

            counts = mcs.value_counts()
            con.append(float(counts.iloc[0] / len(mcs)))
            nmc.append(float(len(counts) / len(mcs)))

        out["recovery"].append(float(np.mean(rec)))
        out["homogeneity"].append(float(np.mean(hom)))
        out["concentration"].append(float(np.mean(con)))
        out["n_mc_per_cell"].append(float(np.mean(nmc)))

    return out


def paired_significance(per_batch_by_method, ref_name, metric_name):
    """Paired one-sided Wilcoxon (ref > other) with win counts and Bonferroni
    correction -- the same convention as `rare_metric_significance_paired`, applied to
    the held-out metrics computed here."""
    from scipy.stats import wilcoxon

    def stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    ref = np.asarray(per_batch_by_method.get(ref_name, []), dtype=float)
    others = [m for m in per_batch_by_method if m != ref_name]
    rows = []
    for m in [ref_name] + others:
        arr = np.asarray(per_batch_by_method[m], dtype=float)
        row = {"metric": metric_name, "method": m, "n": len(arr),
               "mean": round(float(arr.mean()), 3) if len(arr) else np.nan,
               "std": round(float(arr.std()), 3) if len(arr) else np.nan}
        if m != ref_name and len(arr) == len(ref) and len(arr):
            row["n_wins"] = int((ref > arr).sum())
            try:
                _, p = wilcoxon(ref, arr, alternative="greater")
                row["p_raw"] = round(float(p), 4)
                row["p_adj"] = round(min(float(p) * max(len(others), 1), 1.0), 4)
                row["sig"] = stars(row["p_adj"])
            except ValueError:
                pass
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------


def run_rare_holdout_experiment(ds_id, frac_types=0.5, max_types=None, cell_fraction=1.0,
                                min_train_per_batch=50, min_cells=20, seed=0, knn_k=15,
                                scvi_epochs=50, scvi_n_latent=10, stage2_max_epochs=20,
                                eval_freq=3, patience=6, batch_size=1024,
                                umap_steps_per_epoch=500, skip_if_exists=True,
                                data_dir=None):
    """Full experiment for one dataset. Returns (results_df, significance_df, info)."""
    from interpretable_ssl.experiments.scvi_stage2 import (
        SCVI_DEF_TAG, run_scvi_stage2_experiment,
    )

    cfg = DATASETS[ds_id]
    lk, bk = cfg["label_key"], cfg.get("batch_key")
    K = cfg["num_prototypes"]

    # --- 1. full data + the untouched graph everything is scored against ---
    ad_full = load_full_counts_adata(ds_id)
    aff_full = sp.csr_matrix(get_canonical_affinity(ds_id, len(ad_full)))

    # --- 2. pick the held-out types and split ---
    heldout_types = select_heldout_rare_types(
        ad_full, lk, frac_types=frac_types, max_types=max_types,
        min_cells=min_cells, seed=seed,
    )
    tag = f"rareho{int(round(cell_fraction * 100))}_t{len(heldout_types)}_seed{seed}"
    train_idx, test_idx, heldout_types = load_or_make_split(
        ds_id, ad_full, lk, bk, tag, heldout_types, cell_fraction,
        min_train_per_batch, seed, data_dir=data_dir,
    )
    test_mask = np.zeros(len(ad_full), dtype=bool)
    test_mask[test_idx] = True

    # --- 3. train every arm on the reduced data ---
    train_ds_id, _ = write_train_dataset(ds_id, ad_full, train_idx, tag, data_dir=data_dir, K=K)
    res = run_scvi_stage2_experiment(
        train_ds_id, scvi_epochs=scvi_epochs, scvi_n_latent=scvi_n_latent,
        stage2_max_epochs=stage2_max_epochs, eval_freq=eval_freq, patience=patience,
        batch_size=batch_size, umap_steps_per_epoch=umap_steps_per_epoch,
        K=K, skip_if_exists=skip_if_exists,
    )
    t = res["trainer"]

    # --- 4. place the held-out cells ---
    ad_test = prepare_test_adata(ad_full, test_idx, bk)
    proto_test, _z_test_norm = assign_unseen_scproto(t, ad_test)
    z_test = t.get_latent(adata=ad_test, l2norm=False)
    z_train = res["z_scvi"]

    train_names = t.train_ds.adata.obs_names
    proto_train = pd.Series(t._get_assignments()[0], index=train_names)

    methods = {"scProto Stage 2 (scVI)": pd.concat(
        [proto_train, pd.Series(proto_test, index=ad_test.obs_names)]
    ).reindex(ad_full.obs_names).values.astype(int)}

    for arm, label in (("seacells", "SEACells (scVI) + kNN"), ("leiden", "Leiden (scVI) + kNN")):
        info = res["baselines"].get(arm)
        if info is None or info.get("save_path") is None:
            print(f"  {label}: no run on disk, skipping")
            continue
        path = os.path.join(info["save_path"], "cell_assignments.csv")
        if not os.path.exists(path):
            print(f"  {label}: cell_assignments.csv missing in {info['save_path']}, skipping")
            continue
        train_assign = pd.read_csv(path).set_index("cell_id")["metacell_id"].reindex(train_names)
        if train_assign.isna().any():
            print(f"  {label}: assignments missing for {int(train_assign.isna().sum())} train cells, skipping")
            continue
        test_assign = assign_unseen_knn(z_train, train_assign.values.astype(int), z_test, k=knn_k)
        methods[label] = pd.concat(
            [train_assign.astype(int), pd.Series(test_assign, index=ad_test.obs_names)]
        ).reindex(ad_full.obs_names).values.astype(int)

    # --- 5. score, on the untouched full graph ---
    test_edges = subgraph_touching(aff_full, test_mask, require="any")
    rare_mask = ad_full.obs[lk].astype(str).isin([str(x) for x in heldout_types]).values
    print(f"[{ds_id}] {int(test_mask.sum())} held-out cells; "
          f"{test_edges.nnz // 2} edges touch one")

    records, per_batch = [], {}
    for name, assign in methods.items():
        obs = ad_full.obs[[c for c in (lk, bk) if c]].copy()
        obs["_mc"] = assign

        full_mod = calc_modularity_per_batch(aff_full, assign, ad_full.obs[bk].values)
        test_mod = calc_modularity_per_batch(test_edges, assign, ad_full.obs[bk].values)
        rare_metrics = heldout_rare_metrics_per_batch(obs, "_mc", lk, bk, heldout_types, test_mask)

        per_batch[name] = {
            "heldout_modularity": list(test_mod.values),
            **{k: v for k, v in rare_metrics.items()},
        }
        records.append({
            "dataset": ds_id, "method": name,
            "full_graph_modularity_mean": float(full_mod.mean()),
            "full_graph_modularity_std": float(full_mod.std()),
            "heldout_modularity_mean": float(test_mod.mean()),
            "heldout_modularity_std": float(test_mod.std()),
            **{f"{k}_mean": float(np.mean(v)) if v else np.nan for k, v in rare_metrics.items()},
            **{f"{k}_std": float(np.std(v)) if v else np.nan for k, v in rare_metrics.items()},
            "heldout_edge_same_cluster_rate": same_cluster_rate(test_edges, assign)["rate"],
            "rare_edge_same_cluster_rate": same_cluster_rate(
                test_edges, assign, cell_mask=rare_mask, require="any")["rate"],
        })

    df = pd.DataFrame.from_records(records)

    sig_rows = []
    ref = "scProto Stage 2 (scVI)"
    for metric in ("heldout_modularity", "recovery", "homogeneity", "concentration"):
        vals = {m: per_batch[m][metric] for m in per_batch}
        sig_rows += [{**r, "dataset": ds_id} for r in paired_significance(vals, ref, metric)]
    sig_df = pd.DataFrame(sig_rows)

    info = {
        "dataset": ds_id, "tag": tag, "train_ds_id": train_ds_id,
        "heldout_types": [str(x) for x in heldout_types],
        "n_heldout_cells": int(test_mask.sum()), "n_train_cells": int(len(train_idx)),
        "K": int(K), "run_dir": res["run_dir"], "per_batch": per_batch,
    }
    return df, sig_df, info


def run_all_datasets(ds_ids, **kwargs):
    """One dataset at a time; a failure is reported and skipped so the rest still
    produce results. Every stage reloads on a re-run."""
    dfs, sigs, infos = [], [], {}
    for ds_id in ds_ids:
        print(f"\n{'=' * 70}\n=== {ds_id} ===\n{'=' * 70}")
        try:
            df, sig, info = run_rare_holdout_experiment(ds_id, **kwargs)
            dfs.append(df)
            sigs.append(sig)
            infos[ds_id] = info
            print(f"\n----- [{ds_id}] held-out rare types: {info['heldout_types']} -----")
            print(df[["method", "heldout_modularity_mean", "recovery_mean",
                      "homogeneity_mean", "concentration_mean"]].to_string(index=False))
        except Exception as e:  # noqa: BLE001 -- keep the other datasets
            import traceback
            print(f"\n!!! [{ds_id}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    sig_all = pd.concat(sigs, ignore_index=True) if sigs else pd.DataFrame()
    return df_all, sig_all, infos


def save_summary(df, sig_df, infos, out_dir, filename="scvi_rare_heldout_summary.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump({"results": df.to_dict(orient="records"),
                   "significance": sig_df.to_dict(orient="records"),
                   "info": infos}, f, indent=2, default=str)
    print("saved", path)
    return path
