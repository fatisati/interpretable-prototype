"""[D58] Score the ablation arms on the NODE-HOLDOUT protocol.

Why a separate script rather than score_nodeheldout.py: that one iterates its
BASELINES regex list, which has no d58_* pattern, so the arms would simply be
skipped. Every metric here calls the SAME functions in the same way as
score_nodeheldout.py:145-168 -- no new metric definitions.

Metrics (the set frozen in my-docs/doing/ablation-plan.txt):
  MAIN       rare homogeneity / F1 / coverage, cross-batch homogeneity,
             batch entropy, cell-type purity (TEST and FULL, size-weighted),
             test cohesion (aff + gram)
  SECONDARY  test modularity, n metacells used

usage:
  DATA_DIR=<scratch>/data/ GRAPHS_DIR=<scratch>/graphs \
    python modal_scripts/score_d58_arms.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from modal_scripts.score_nodeheldout import (  # noqa: E402
    test_mask_for, purity_entropy_on_mask, coverage_on_mask)
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    cohesion_per_cell, modularity_per_batch, rare_metrics_paper,
    load_affinity, read_obs)

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
OUT = os.path.join(REPO, "files", "tables", "d58_ablation")
# display order; base first so deltas read against it
ORDER = ["base_s31", "base_s32", "base_s33",
         "gram0", "aff0", "coh0", "comm0", "recon0",
         "comm2x", "recon2x", "gram2x", "aff2x"]


def main():
    n, h5, bk, lk, nk, K = DATASETS[DS]
    h5_path = os.path.join(os.environ["DATA_DIR"], h5)
    gd = os.environ.get("GRAPHS_DIR", "graphs")
    # FULL graph, not the train-only one: held-out cells' true neighbours only
    # exist there (CLAUDE.md 7a)
    A_fu = load_affinity(f"{gd}/affinity_{DS}{n}_ncomp50_kneighbors50_arbf.pkl")
    obs = read_obs(h5_path, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    bcodes = pd.factorize(bts)[0]
    mask = test_mask_for(DS, h5_path, n)
    all_mask = np.ones(n, bool)
    print("{} NH: N={:,}  test={:,}  batches={}".format(
        DS, n, int(mask.sum()), len(set(bts))))

    found = [a for a in ORDER if os.path.isdir(os.path.join(RUNS, "d58_" + a))]
    print("arms present: {}\n".format(found))
    cache, rows = {}, []
    for arm in found:
        lab = load_labels(os.path.join(RUNS, "d58_" + arm), n)
        if lab is None:
            print("   --  {:<10} no usable per-cell labels".format(arm))
            continue
        r = {"arm": arm, "n_mc": int(len(np.unique(lab)))}
        ca = cohesion_per_cell(lab, A_fu, bcodes, sim="aff", _cache=cache)
        cg = cohesion_per_cell(lab, A_fu, bcodes, sim="gram", _cache=cache)
        r["coh_aff_test"] = float(ca[mask].mean())
        r["coh_aff_test_std"] = float(ca[mask].std())
        r["coh_gram_test"] = float(cg[mask].mean())
        r["coh_gram_test_std"] = float(cg[mask].std())
        mods = np.array(list(
            modularity_per_batch(lab, A_fu, bcodes, node_mask=mask).values()))
        r["modularity_test"] = float(mods.mean())
        r["modularity_test_std"] = float(mods.std())
        # TEST-only purity / batch entropy (size-weighted over metacells)
        te = purity_entropy_on_mask(lab, cts, bts, mask)
        r["purity_test_w"] = te["ct_purity_wmean"]
        r["purity_test_wstd"] = te["ct_purity_wstd"]
        r["entropy_test_w"] = te["batch_entropy_wmean"]
        r["entropy_test_wstd"] = te["batch_entropy_wstd"]
        # FULL-dataset purity / batch entropy, same function, mask = everything
        fu = purity_entropy_on_mask(lab, cts, bts, all_mask)
        r["purity_full_w"] = fu["ct_purity_wmean"]
        r["purity_full_wstd"] = fu["ct_purity_wstd"]
        r["entropy_full_w"] = fu["batch_entropy_wmean"]
        r["coverage_test"] = coverage_on_mask(lab, cts, mask)
        rare = rare_metrics_paper(lab[mask], cts[mask], bts[mask])
        for k, v in rare.items():
            if not isinstance(v, (list, str)):
                r[k] = v
        rows.append(r)
        print("   OK  {:<10} rare_homog={:.4f}  purity_test={:.4f}  "
              "entropy={:.4f}  coh_aff={:.4f}  mod={:.4f}".format(
                  arm, r.get("rare_homogeneity_mean", float("nan")),
                  r["purity_test_w"], r["entropy_test_w"],
                  r["coh_aff_test"], r["modularity_test"]))

    if not rows:
        print("nothing scored")
        return
    df = pd.DataFrame(rows).set_index("arm")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "{}_d58_ablation.csv".format(DS))
    df.to_csv(p)
    print("\nsaved {}".format(p))

    MAIN = ["rare_homogeneity_mean", "rare_f1_mean", "rare_coverage_mean",
            "rare_cross_batch_homogeneity_mean", "entropy_test_w",
            "purity_test_w", "purity_full_w", "coh_aff_test", "coh_gram_test",
            "modularity_test", "n_mc"]
    MAIN = [c for c in MAIN if c in df.columns]
    print("\n=== MAIN METRICS ===")
    print(df[MAIN].round(4).to_string())

    if "base_s31" in df.index:
        print("\n=== DELTA vs base_s31 (negative = the arm is WORSE) ===")
        d = df[MAIN].astype(float) - df.loc["base_s31", MAIN].astype(float)
        print(d.drop(index="base_s31", errors="ignore").round(4).to_string())
        seeds = [a for a in ("base_s31", "base_s32", "base_s33") if a in df.index]
        if len(seeds) > 1:
            print("\n=== NOISE FLOOR: spread across {} base seeds {} ===".format(
                len(seeds), seeds))
            print("(any delta above smaller than this is NOT interpretable)")
            print(df.loc[seeds, MAIN].astype(float).agg(["min", "max", "std"])
                  .round(4).to_string())
        else:
            print("\nNOISE FLOOR: only 1 base seed present -- deltas are "
                  "UNBOUNDED until a second seed lands.")


if __name__ == "__main__":
    main()
