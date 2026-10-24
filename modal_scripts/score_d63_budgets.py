"""[D61] Score the two proto_recon-mix3 arms (+ d60_leid_noc and epoch-0 leiden against their SAME-INIT epoch-0 baselines, with the
full metric cross-product the user specified:

  cohesion (aff, gram)         x {train, test} x {rare, common, all}
  purity {within, cross, pooled} x {train, test} x {rare, common, all}
  rare metrics (homogeneity, F1, coverage) on test (and train for reference)

Per-cell indicator definitions (so rare/common slicing is exact):
  within  = 1 if the cell's type == the majority type of its (metacell, batch)
            group (>=3 cells), computed within the train or test half
  pooled  = 1 if the cell's type == its metacell's GLOBAL majority type
  cross   = 1 if the cell's (metacell, batch) majority == the metacell's global
            majority -- the cell-level version of cross_agree
  rare    = the cell's type is in the bottom quartile of global type frequencies

Baseline pairing per the D60 ADR: d60_leid_noc vs d60_epoch0_leid;
d60_wayp_noc and d60_wayp_comm vs d59_epoch0 (waypoint).

usage:
  DATA_DIR=<scratch>/data/ GRAPHS_DIR=<scratch>/graphs \
    python modal_scripts/score_d60_ctb.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    cohesion_per_cell, rare_metrics_paper, load_affinity, read_obs)

DS = "pbmc-immune"
TAG = "arbf_ctb20_seed0"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
OUT = os.path.join(REPO, "files", "tables", "d63_budgets")
MIN_CELLS = 3

# arm -> (dir, its epoch-0 baseline dir or None if it IS a baseline)
ARMS = [
    ("epoch0_leid", "d60_epoch0_leid", None),
    ("d60_leid_noc", "d60_leid_noc", "d60_epoch0_leid"),
    ("d61_pr3_sg", "d61_pr3_sg", "d60_epoch0_leid"),
    ("d61_pr3_lv", "d61_pr3_lv", "d60_epoch0_leid"),
    ("d61_pr3_sgdec", "d61_pr3_sgdec", "d60_epoch0_leid"),
    ("d61_pr3_lvdec", "d61_pr3_lvdec", "d60_epoch0_leid"),
    ("d61_pr3_sgdec_nr", "d61_pr3_sgdec_nr", "d60_epoch0_leid"),
    ("d61_pr3_lvdec_nr", "d61_pr3_lvdec_nr", "d60_epoch0_leid"),
    ("leiden_st1", "d59_leiden_ctb", None),
    ("d62_pr16", "d62_pr16", "d60_epoch0_leid"),
    ("d62_pr8", "d62_pr8", "d60_epoch0_leid"),
    ("d63_mse-rec", "d63_mse-rec", "d60_epoch0_leid"),
    ("d63_mse-norec", "d63_mse-norec", "d60_epoch0_leid"),
]


def per_cell_indicators(lab, cts, bts, half_mask, min_cells=MIN_CELLS):
    """within / pooled / cross as per-cell 0/1 (nan where group too small),
    computed inside one half (train or test)."""
    n = len(lab)
    within = np.full(n, np.nan)
    pooled = np.full(n, np.nan)
    cross = np.full(n, np.nan)
    idx = np.where(half_mask)[0]
    l, c, b = lab[idx], cts[idx], bts[idx]
    for mc in np.unique(l):
        m = l == mc
        if m.sum() < min_cells:
            continue
        vals, cnt = np.unique(c[m], return_counts=True)
        gmaj = vals[np.argmax(cnt)]
        pooled[idx[m]] = (c[m] == gmaj).astype(float)
        for bb in np.unique(b[m]):
            s = m & (b == bb)
            if s.sum() < min_cells:
                continue
            v2, c2 = np.unique(c[s], return_counts=True)
            bmaj = v2[np.argmax(c2)]
            within[idx[s]] = (c[s] == bmaj).astype(float)
            cross[idx[s]] = float(bmaj == gmaj)
    return within, pooled, cross


def main():
    n, h5, bk, lk, nk, K = DATASETS[DS]
    h5p = os.path.join(os.environ["DATA_DIR"], h5)
    gd = os.environ.get("GRAPHS_DIR", "graphs")
    A = load_affinity(f"{gd}/affinity_{DS}{n}_ncomp50_kneighbors50_arbf.pkl")
    obs = read_obs(h5p, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    bcodes = pd.factorize(bts)[0]
    import anndata
    names = np.asarray(anndata.read_h5ad(h5p, backed="r").obs_names).astype(str)
    d = np.load(os.path.join(REPO, "node_heldout_data",
                             f"{DS}_{TAG}_split.npz"), allow_pickle=True)
    tid = set(str(x) for x in d["test_ids"])
    test = np.array([x in tid for x in names])
    train = ~test

    freq = pd.Series(cts).value_counts()
    qcut = np.quantile(freq.values, 0.25)
    rare = np.isin(cts, list(freq[freq <= qcut].index))
    sl = {"all": np.ones(n, bool), "rare": rare, "common": ~rare}
    halves = {"train": train, "test": test}
    print(f"{DS}/{TAG}: N={n:,} train={train.sum():,} test={test.sum():,} "
          f"rare cells={rare.sum():,} ({100*rare.mean():.1f}%)")

    cache, rows = {}, []
    for arm, dirname, _base in ARMS:
        d_ = os.path.join(RUNS, dirname)
        lab = load_labels(d_, n)
        if lab is None:
            print(f"  --  {arm}: no labels in {dirname}")
            continue
        r = {"arm": arm, "n_mc": int(len(np.unique(lab)))}
        ca = cohesion_per_cell(lab, A, bcodes, sim="aff", _cache=cache)
        cg = cohesion_per_cell(lab, A, bcodes, sim="gram", _cache=cache)
        for hname, hm in halves.items():
            for sname, sm in sl.items():
                m = hm & sm
                r[f"coh_aff_{hname}_{sname}"] = float(ca[m].mean())
                r[f"coh_gram_{hname}_{sname}"] = float(cg[m].mean())
            w, p, x = per_cell_indicators(lab, cts, bts, hm)
            for sname, sm in sl.items():
                for mname, vec in (("within", w), ("pooled", p), ("cross", x)):
                    v = vec[hm & sm]
                    r[f"{mname}_{hname}_{sname}"] = float(np.nanmean(v))
            rm = rare_metrics_paper(lab[hm], cts[hm], bts[hm])
            for k in ("rare_homogeneity_mean", "rare_f1_mean", "rare_coverage_mean"):
                if k in rm:
                    r[f"{k}_{hname}"] = float(rm[k])
        rows.append(r)
        print(f"  OK  {arm:<14} coh_te {r['coh_aff_test_all']:.4f} "
              f"(rare {r['coh_aff_test_rare']:.4f}) "
              f"within_te {r['within_test_all']:.4f} "
              f"cross_te {r['cross_test_all']:.4f} "
              f"pooled_te {r['pooled_test_all']:.4f}")

    df = pd.DataFrame(rows).set_index("arm")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{DS}_d63_full.csv")
    df.to_csv(p)
    print(f"\nsaved {p}")

    base_of = {a: b for a, d_, b in ARMS if b}
    base_name = {"d60_epoch0_leid": "epoch0_leid"}
    KEY = ["coh_aff_train_all", "coh_aff_test_all",
           "coh_aff_test_rare", "coh_aff_test_common",
           "within_test_all", "within_test_rare",
           "cross_train_all", "cross_test_all",
           "pooled_test_all", "pooled_test_rare",
           "rare_homogeneity_mean_test", "rare_f1_mean_test",
           "rare_coverage_mean_test"]
    KEY = [k for k in KEY if k in df.columns]
    print("\n=== KEY METRICS ===")
    print(df[KEY].round(4).to_string())
    print("\n=== DELTA vs SAME-INIT epoch-0 (positive = trained arm better) ===")
    for arm, b in base_of.items():
        bn = base_name[b]
        if arm in df.index and bn in df.index:
            delta = df.loc[arm, KEY].astype(float) - df.loc[bn, KEY].astype(float)
            print(f"\n{arm}  (vs {bn}):")
            print(delta.round(4).to_string())


if __name__ == "__main__":
    main()
