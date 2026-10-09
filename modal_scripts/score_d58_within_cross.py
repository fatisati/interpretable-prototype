"""[D58] Decompose purity into WITHIN-batch and CROSS-batch, per arm.

The aggregate ct_purity pools a metacell's members across batches, so it cannot
distinguish the two mechanisms the model is supposed to implement:

  mechanism 1  cohesion groups cells correctly WITHIN each bio batch
  mechanism 2  the same group is matched to its counterpart ACROSS batches

Three metrics, all on the same saved labels (no GPU, no cost):

  within_purity   for each (metacell, batch) cell with >= min_cells members:
                  max label count / members.  Size-weighted.
                  -> "inside one batch, is this metacell one cell type?"
                     This is mechanism 1.

  pooled_purity   for each metacell over ALL its members (= the existing
                  ct_purity_wmean). Low pooled with high within means the
                  metacell is one type per batch but a DIFFERENT type in
                  different batches.

  cross_agree     for each metacell, take the majority label per batch, then the
                  size-weighted fraction of (metacell, batch) cells whose
                  majority equals the metacell's GLOBAL majority.
                  -> "are we matching like with like across batches?"
                     This is mechanism 2, and it is the metric the aggregate
                     numbers hide.

Also reports, from each arm's grad_probe.jsonl, whether each term's VALUE was
still improving at the end of training -- i.e. whether 10 epochs undertrains.

usage:
  DATA_DIR=<scratch>/data/ python modal_scripts/score_d58_within_cross.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from modal_scripts.score_nodeheldout import test_mask_for  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs  # noqa: E402

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
OUT = os.path.join(REPO, "files", "tables", "d58_ablation")
ORDER = ["base_s31", "base_s32", "gram0", "aff0", "comm0", "recon0"]
TERMS = ("cohesion_gram", "cohesion_aff", "umap", "recon")
MIN_CELLS = 3            # a (metacell, batch) cell needs this many to be scored


def decompose(lab, cts, bts, mask, min_cells=MIN_CELLS):
    """within_purity, pooled_purity, cross_agree -- all size-weighted."""
    l, c, b = lab[mask], cts[mask], bts[mask]
    wp_num = wp_den = 0.0           # within-batch purity, weighted
    pp_num = pp_den = 0.0           # pooled purity, weighted
    ca_num = ca_den = 0.0           # cross-batch majority agreement, weighted
    n_cells_scored = 0
    for mc in np.unique(l):
        m = l == mc
        if m.sum() < min_cells:
            continue
        # pooled over all batches
        vals, cnt = np.unique(c[m], return_counts=True)
        glob_major = vals[np.argmax(cnt)]
        pp_num += cnt.max()
        pp_den += m.sum()
        # per batch inside this metacell
        for bb in np.unique(b[m]):
            sel = m & (b == bb)
            if sel.sum() < min_cells:
                continue
            v2, c2 = np.unique(c[sel], return_counts=True)
            wp_num += c2.max()
            wp_den += sel.sum()
            ca_num += sel.sum() * float(v2[np.argmax(c2)] == glob_major)
            ca_den += sel.sum()
            n_cells_scored += int(sel.sum())
    return {
        "within_purity": wp_num / wp_den if wp_den else np.nan,
        "pooled_purity": pp_num / pp_den if pp_den else np.nan,
        "cross_agree": ca_num / ca_den if ca_den else np.nan,
        "cells_scored": n_cells_scored,
    }


def convergence(arm):
    """Was each term's VALUE still improving at the end of training?
    Compares the median of the first 10% of probed steps to the last 10%."""
    p = os.path.join(RUNS, "d58_" + arm, "grad_probe.jsonl")
    if not os.path.exists(p):
        return {}
    rows = [json.loads(x) for x in open(p)]
    rows = [r for r in rows if any(isinstance(v, dict) for v in r.values())]
    if len(rows) < 20:
        return {}
    out = {"probe_rows": len(rows)}
    q = max(2, len(rows) // 10)
    for t in TERMS:
        v = [r[t]["value"] for r in rows if t in r and r[t]["lambda"] > 0]
        if len(v) < 20:
            continue
        first, last = float(np.median(v[:q])), float(np.median(v[-q:]))
        out[t + "_first"] = first
        out[t + "_last"] = last
        out[t + "_pct_change"] = 100.0 * (last - first) / abs(first) if first else np.nan
    return out


def main():
    n, h5, bk, lk, nk, K = DATASETS[DS]
    h5_path = os.path.join(os.environ["DATA_DIR"], h5)
    obs = read_obs(h5_path, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    mask = test_mask_for(DS, h5_path, n)
    allm = np.ones(n, bool)
    print("{}: N={:,}  test={:,}  batches={}  min_cells_per_(mc,batch)={}".format(
        DS, n, int(mask.sum()), len(set(bts)), MIN_CELLS))

    rows = []
    for arm in ORDER:
        d = os.path.join(RUNS, "d58_" + arm)
        if not os.path.isdir(d):
            continue
        lab = load_labels(d, n)
        if lab is None:
            print("  -- {}: no labels".format(arm))
            continue
        te = decompose(lab, cts, bts, mask)
        fu = decompose(lab, cts, bts, allm)
        r = {"arm": arm,
             "within_test": te["within_purity"], "pooled_test": te["pooled_purity"],
             "cross_agree_test": te["cross_agree"],
             "within_full": fu["within_purity"], "pooled_full": fu["pooled_purity"],
             "cross_agree_full": fu["cross_agree"]}
        r.update(convergence(arm))
        rows.append(r)
        print("  OK  {:<10} within={:.4f} pooled={:.4f} cross_agree={:.4f}  (test)".format(
            arm, r["within_test"], r["pooled_test"], r["cross_agree_test"]))

    if not rows:
        print("nothing scored")
        return
    df = pd.DataFrame(rows).set_index("arm")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "{}_within_cross.csv".format(DS))
    df.to_csv(p)

    show = ["within_test", "pooled_test", "cross_agree_test",
            "within_full", "pooled_full", "cross_agree_full"]
    print("\n=== WITHIN vs POOLED vs CROSS-BATCH AGREEMENT ===")
    print(df[show].round(4).to_string())

    if "base_s31" in df.index:
        print("\n=== DELTA vs base_s31 ===")
        print((df[show].astype(float) - df.loc["base_s31", show].astype(float))
              .drop(index="base_s31", errors="ignore").round(4).to_string())

    conv = [c for c in df.columns if c.endswith("_pct_change")]
    if conv:
        print("\n=== CONVERGENCE: % change in each term's VALUE, first 10% -> last 10%")
        print("    (large negative = still improving fast = 10 epochs UNDERTRAINS)")
        print(df[["probe_rows"] + conv].round(2).to_string())
    print("\nsaved {}".format(p))


if __name__ == "__main__":
    main()
