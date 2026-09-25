"""Score every saved baseline partition on the NODE-HOLDOUT protocol's test cells.

Protocol (user decision 2026-09-24): node-holdout is THE evaluation setup. 20% of
cells (stratified by batch, split cached in node_heldout_data/<ds>_*_split.npz by
the rebuttal's node_heldout_modularity.ipynb) are the test set; every metric below
is computed restricted to those cells, against the FULL affinity graph:

  coh_aff / coh_gram   cohesion_per_cell (V[b] semantics, self=1) -> test cells only
  modularity           modularity_per_batch(node_mask=test) = per-batch Newman
                       modularity over edges TOUCHING a test cell (the rebuttal's
                       test_node_edges statistic), mean +/- std across batches
  batch entropy        per metacell over its TEST members only (unw + size-wtd)
  ct purity            per metacell over its TEST members only (unw + size-wtd)
  rare_*               paper's batch-local definitions on the test cells
  coverage             fraction of (batch, celltype) pairs among test cells covered

HONESTY CAVEAT, printed and to be repeated in any table: the baselines here were
fitted on ALL cells (they saw the test cells during clustering) -- an advantage in
their favor. Only scProto node-holdout runs are leak-free (trained on the 80%
train graph, test cells placed by a frozen-encoder pass). If scProto wins anyway,
the comparison understates the win.

Run:  DATA_DIR=<dir> python modal_scripts/score_nodeheldout.py pancreas
"""
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modal_scripts.score_all_baselines import (  # noqa: E402
    DATASETS, BASELINES, MODELS, VOLUME_RUNS, load_labels)
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    cohesion_per_cell, modularity_per_batch, rare_metrics_paper,
    load_affinity, read_obs)

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "node_heldout_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "files", "tables", "node_heldout_scoreboard")


def read_obs_names(h5_path):
    with h5py.File(h5_path, "r") as f:
        idx_col = f["obs"].attrs.get("_index", "index")
        if isinstance(idx_col, bytes):
            idx_col = idx_col.decode()
        v = f["obs"][idx_col][:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def test_mask_for(ds, h5_path, n):
    d = np.load(os.path.join(SPLIT_DIR, f"{ds}_arbf_nodeheldout20_seed0_split.npz"),
                allow_pickle=True)
    names = read_obs_names(h5_path)
    assert len(names) == n
    pos = {nm: i for i, nm in enumerate(names)}
    test_ids = [str(x) for x in d["test_ids"]]
    idx = np.array([pos[t] for t in test_ids])
    mask = np.zeros(n, bool)
    mask[idx] = True
    return mask


def _per_mc_over_mask(labels, values, mask, min_cells=1):
    """For each metacell: rows of `values` restricted to mask. Yields (mc, subset)."""
    for mc in np.unique(labels[mask]):
        sub = values[(labels == mc) & mask]
        if len(sub) >= min_cells:
            yield mc, sub


def purity_entropy_on_mask(labels, cts, bts, mask):
    pur, ent, sizes = [], [], []
    for _, sub_ct in _per_mc_over_mask(labels, cts, mask):
        _, cnt = np.unique(sub_ct, return_counts=True)
        pur.append(cnt.max() / cnt.sum())
        sizes.append(cnt.sum())
    for _, sub_bt in _per_mc_over_mask(labels, bts, mask):
        _, cnt = np.unique(sub_bt, return_counts=True)
        p = cnt / cnt.sum()
        ent.append(float(-(p * np.log(p)).sum()))
    pur, ent, sizes = np.array(pur), np.array(ent), np.array(sizes, float)
    w = sizes / sizes.sum()
    wmean_p = float((pur * w).sum())
    wstd_p = float(np.sqrt(((pur - wmean_p) ** 2 * w).sum()))
    wmean_e = float((ent * w).sum())
    wstd_e = float(np.sqrt(((ent - wmean_e) ** 2 * w).sum()))
    return {
        "ct_purity_mean": float(pur.mean()), "ct_purity_std": float(pur.std()),
        "ct_purity_wmean": wmean_p, "ct_purity_wstd": wstd_p,
        "batch_entropy_mean": float(ent.mean()),
        "batch_entropy_std": float(ent.std()),
        "batch_entropy_wmean": wmean_e, "batch_entropy_wstd": wstd_e,
        "n_mc_with_test": int(len(pur)),
        "median_test_cells_per_mc": float(np.median(sizes)),
    }


def coverage_on_mask(labels, cts, mask):
    """score_partition's coverage (majority-represented types / all types),
    restricted to test cells: majorities taken over each metacell's TEST members.
    Same formula as metacell_scoreboard.score_partition, not a new definition."""
    majors = set()
    for _, sub_ct in _per_mc_over_mask(labels, cts, mask):
        vals, cnt = np.unique(sub_ct, return_counts=True)
        majors.add(vals[np.argmax(cnt)])
    return float(len(majors) / len(np.unique(cts[mask])))


def run(ds):
    n, h5, bk, lk, nk, K = DATASETS[ds]
    t0 = time.time()
    h5_path = os.path.join(os.environ["DATA_DIR"], h5)
    # GRAPHS_DIR override: the Drive-synced graphs/ dir fails h5py/pickle
    # reads on large files (Errno 22); fetch from the Modal volume instead
    _gd = os.environ.get("GRAPHS_DIR", "graphs")
    A_fu = load_affinity(f"{_gd}/affinity_{ds}{n}_ncomp50_kneighbors50_arbf.pkl")
    obs = read_obs(h5_path, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    bcodes = pd.factorize(bts)[0]
    mask = test_mask_for(ds, h5_path, n)
    print(f"\n=== {ds} NODE-HOLDOUT: N={n:,} test={int(mask.sum()):,} "
          f"batches={len(set(bts))} ({time.time()-t0:.0f}s)")
    print("    caveat: baselines below were FITTED ON ALL CELLS (advantage them); "
          "only scProto node-holdout rows are leak-free")

    roots = [os.path.join(MODELS, ds), os.path.join(VOLUME_RUNS, ds)]
    dirs = []
    for r in roots:
        if os.path.isdir(r):
            dirs += [(r, e) for e in sorted(os.listdir(r))
                     if os.path.isdir(os.path.join(r, e))]
    cache = {}
    rows = []
    for label, pat in BASELINES:
        hits = [(r, d) for (r, d) in dirs if re.search(pat, d)]
        if not hits:
            print(f"   --  {label:<26} MISSING")
            continue
        for r, h in hits:
            lab = load_labels(os.path.join(r, h), n)
            if lab is None:
                print(f"   --  {label:<26} {h[:40]}: no usable per-cell labels")
                continue
            name = label if len(hits) == 1 else f"{label} [{h[:28]}]"
            row = {"method": name, "n_mc": int(len(np.unique(lab)))}
            ca = cohesion_per_cell(lab, A_fu, bcodes, sim="aff", _cache=cache)
            cg = cohesion_per_cell(lab, A_fu, bcodes, sim="gram", _cache=cache)
            row["coh_aff_mean"] = float(ca[mask].mean())
            row["coh_aff_std"] = float(ca[mask].std())
            row["coh_gram_mean"] = float(cg[mask].mean())
            row["coh_gram_std"] = float(cg[mask].std())
            mods = np.array(list(
                modularity_per_batch(lab, A_fu, bcodes, node_mask=mask).values()))
            row["modularity_test_mean"] = float(mods.mean())
            row["modularity_test_std"] = float(mods.std())
            row.update(purity_entropy_on_mask(lab, cts, bts, mask))
            row["coverage"] = coverage_on_mask(lab, cts, mask)
            rare = rare_metrics_paper(lab[mask], cts[mask], bts[mask])
            row.update({k: v for k, v in rare.items()
                        if not isinstance(v, (list, str))})
            row["rare_types_seen"] = rare.get("rare_types_seen")
            rows.append(row)
            print(f"   OK  {name:<40} coh_aff={row['coh_aff_mean']:.4f} "
                  f"mod={row['modularity_test_mean']:.4f} "
                  f"pur_w={row['ct_purity_wmean']:.4f} "
                  f"be={row['batch_entropy_mean']:.4f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{ds}_scoreboard.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"   saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["pancreas"]):
        run(ds)
