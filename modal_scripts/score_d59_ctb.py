"""[D59] The report: cohesion and purity, TRAIN and TEST, within-batch and
cross-batch, for the three arms on the (cell type x batch)-stratified split.

Answers, in one table, what each term is actually for:
  cohesion       should raise cohesion and within-batch purity
  community      the only term that sees cross-batch pairs
  recon          aligns z across batches but not the ASSIGNMENT
  proto_recon    aimed at the assignment -- the candidate fix for cross-batch

Metrics (all on the SAME full affinity graph, same code as score_nodeheldout.py):
  coh_aff / coh_gram      per cell, reported on TRAIN and on TEST separately
  within_purity           per (metacell, batch) group, size-weighted
  pooled_purity           per metacell over all its members
  cross_agree             fraction of (metacell, batch) groups whose majority
                          label equals the metacell's GLOBAL majority
                          -> this is the cross-batch alignment metric

For the scProto arms the test half is scored under BOTH placement rules:
  argmax        nearest PROTOTYPE VECTOR (what scProto has always done)
  latent 15-NN  15-NN cosine among TRAIN cells in z, majority of their train
                labels -- exactly what the Leiden baseline uses, so this is the
                matched comparison. Reads no test labels.

note: score_nodeheldout.test_mask_for() hardcodes the OLD split tag
(nodeheldout20_seed0), so the mask is built here from the new split npz.

usage:
  DATA_DIR=<scratch>/data/ GRAPHS_DIR=<scratch>/graphs \
    python modal_scripts/score_d59_ctb.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from modal_scripts.score_d58_within_cross import decompose  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    cohesion_per_cell, load_affinity, read_obs)

DS = "pbmc-immune"
TAG = "arbf_ctb20_seed0"
SPLIT = os.path.join(REPO, "node_heldout_data", f"{DS}_{TAG}_split.npz")
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
OUT = os.path.join(REPO, "files", "tables", "d59_ctb")
ARMS = [("scpoli_leiden", "d59_leiden_ctb"),
        ("scproto_base", "d59_scproto_base"),
        ("scproto_protorecon", "d59_scproto_protorecon")]


def latent_vote(lab, z, train, test, k=15):
    """Re-place TEST cells by 15-NN cosine among TRAIN cells in z -- the
    baseline's rule. Uses only TRAIN assignments, so no test label is read."""
    from sklearn.neighbors import NearestNeighbors
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(zn[train])
    _, idx = nn.kneighbors(zn[test])
    ltr = lab[train]
    out = lab.copy()
    out[test] = [np.bincount(ltr[r]).argmax() for r in idx]
    return out


def main():
    n, h5, bk, lk, nk, K = DATASETS[DS]
    h5p = os.path.join(os.environ["DATA_DIR"], h5)
    gd = os.environ.get("GRAPHS_DIR", "graphs")
    A = load_affinity(f"{gd}/affinity_{DS}{n}_ncomp50_kneighbors50_arbf.pkl")
    obs = read_obs(h5p, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    bcodes = pd.factorize(bts)[0]

    # mask from the NEW split (test_mask_for hardcodes the old tag)
    d = np.load(SPLIT, allow_pickle=True)
    test_ids = set(str(x) for x in d["test_ids"])
    names = np.asarray(
        read_obs(h5p, [])["_index"] if "_index" in read_obs(h5p, []) else [])
    if names.size != n:                      # fall back to anndata for obs_names
        import anndata
        names = np.asarray(anndata.read_h5ad(h5p, backed="r").obs_names).astype(str)
    test = np.array([nm in test_ids for nm in names])
    train = ~test
    assert test.sum() == len(test_ids), \
        f"mask {test.sum()} != split {len(test_ids)} -- obs_names mismatch"
    print(f"{DS} / {TAG}: N={n:,}  train={train.sum():,}  test={test.sum():,}  "
          f"batches={len(set(bts))}")

    cache, rows = {}, []
    for label, dirname in ARMS:
        d_ = os.path.join(RUNS, dirname)
        if not os.path.isdir(d_):
            print(f"  --  {label}: no dir {dirname}")
            continue
        lab = load_labels(d_, n)
        if lab is None:
            print(f"  --  {label}: no usable labels")
            continue
        zp = os.path.join(d_, "latent_z_full.npy")
        variants = [("argmax" if label != "scpoli_leiden" else "15-NN", lab)]
        if label != "scpoli_leiden" and os.path.exists(zp):
            variants.append(("15-NN", latent_vote(lab, np.load(zp), train, test)))
        for vname, L in variants:
            ca = cohesion_per_cell(L, A, bcodes, sim="aff", _cache=cache)
            cg = cohesion_per_cell(L, A, bcodes, sim="gram", _cache=cache)
            tr = decompose(L, cts, bts, train)
            te = decompose(L, cts, bts, test)
            rows.append({
                "arm": label, "placement": vname, "n_mc": int(len(np.unique(L))),
                "coh_aff_TRAIN": float(ca[train].mean()),
                "coh_aff_TEST": float(ca[test].mean()),
                "coh_gram_TRAIN": float(cg[train].mean()),
                "coh_gram_TEST": float(cg[test].mean()),
                "within_TRAIN": tr["within_purity"], "within_TEST": te["within_purity"],
                "pooled_TRAIN": tr["pooled_purity"], "pooled_TEST": te["pooled_purity"],
                "cross_TRAIN": tr["cross_agree"], "cross_TEST": te["cross_agree"],
            })
            print(f"  OK  {label:<20} [{vname:<6}] "
                  f"coh_aff tr/te {rows[-1]['coh_aff_TRAIN']:.4f}/"
                  f"{rows[-1]['coh_aff_TEST']:.4f}  "
                  f"within tr/te {tr['within_purity']:.4f}/{te['within_purity']:.4f}  "
                  f"cross tr/te {tr['cross_agree']:.4f}/{te['cross_agree']:.4f}")

    if not rows:
        print("nothing scored")
        return
    df = pd.DataFrame(rows).set_index(["arm", "placement"])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{DS}_d59_ctb.csv")
    df.to_csv(p)

    print("\n=== COHESION (the training objective), TRAIN and TEST ===")
    print(df[["coh_aff_TRAIN", "coh_aff_TEST",
              "coh_gram_TRAIN", "coh_gram_TEST"]].round(4).to_string())
    print("\n=== WITHIN-BATCH purity ===")
    print(df[["within_TRAIN", "within_TEST"]].round(4).to_string())
    print("\n=== CROSS-BATCH agreement  <- the deficit D58 isolated ===")
    print(df[["cross_TRAIN", "cross_TEST"]].round(4).to_string())
    print("\n=== POOLED purity (pools a metacell's members across batches) ===")
    print(df[["pooled_TRAIN", "pooled_TEST", "n_mc"]].round(4).to_string())
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
