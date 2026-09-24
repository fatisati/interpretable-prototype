"""Score every metric-ready run (baselines + scProto) on the frozen scoreboard.

Runs LOCALLY -- the per-cell assignments live on the Drive model dir and the
graphs are in ./graphs, so nothing needs a GPU or a volume round-trip.

Both sides go through interpretable_ssl.evaluation.metacell_scoreboard, so the
comparison is the same code on the same graphs. Every run is scored on:

  * the TRAIN graph  = arbf_heldout20_seed0        (the visible 80%)
  * the TEST graph   = arbf_heldout20_seed0_testedges  (the hidden 20%)

The test numbers are the de-circularised ones: no model had a training signal
for any of those edges.

Usage:
  DATA_DIR=... python modal_scripts/score_all_baselines.py
  DATA_DIR=... python modal_scripts/score_all_baselines.py pancreas
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    load_affinity, read_obs, score_partition)

MODELS = r"G:/.shortcut-targets-by-id/1-0FnoSFlnEGxgKJQCAbspCuHc3osVs25/models"

# ds -> (n_cells, h5ad rel to DATA_DIR, batch_key, label_key, niche_key, K)
DATASETS = {
    "pancreas":    (16382, "pancreas_hvg.h5ad", "tech", "celltype", None, 220),
    "lung":        (32472, "lung_hvg.h5ad", "batch", "cell_type", None, 300),
    "pbmc-immune": (33506, "scpoli/Immune_ALL_human_hvg.h5ad", "study",
                    "final_annotation", None, 300),
}

# Display name -> regex matching the run-dir name. Ordered as the table should read.
# Only unambiguous, canonical runs; dim variants are kept separate on purpose
# because a matched-dim comparison is the whole point for Harmony/ComBat/BBKNN.
BASELINES = [
    ("SEACells",                 r"^seacell$"),
    ("SEACells (masked graph)",  r"^seacell_arbf_heldout20_seed0$"),
    ("SuperCell",                r"^supercell$"),
    ("MetaCell2",                r"^metacell2$"),
    ("MetaQ",                    r"^metaq$"),
    ("SURE",                     r"^sure$"),
    ("Leiden + scPoli-Stage1",   r"^leiden_X_stage1z_K\d+$"),
    ("Leiden + scVI",            r"^leiden_X_scvi_K\d+$"),
    ("Leiden + scVI (gauss)",    r"^leiden_X_scvi_gauss_K\d+$"),
    ("Leiden + Harmony",         r"^leiden_X_harmony_K\d+$"),
    ("Leiden + Harmony d8",      r"^leiden_X_harmony_d8_K\d+$"),
    ("Leiden + Harmony d20",     r"^leiden_X_harmony_d20_K\d+$"),
    ("Leiden + Harmony d50",     r"^leiden_X_harmony_d50_K\d+$"),
    ("Leiden + ComBat",          r"^leiden_X_combat_K\d+$"),
    ("Leiden + ComBat d8",       r"^leiden_X_combat_d8_K\d+$"),
    ("Leiden + BBKNN",           r"^leiden_X_bbknn_K\d+$"),
    ("Leiden + BBKNN d8",        r"^leiden_X_bbknn_d8_K\d+$"),
    ("SEACells + scVI",          r"^seacell_X_scvi$"),
    ("SEACells + Harmony",       r"^seacell_X_harmony$"),
    ("SEACells + Harmony d8",    r"^seacell_X_harmony_d8$"),
    ("SEACells + ComBat",        r"^seacell_X_combat$"),
    ("SEACells + ComBat d8",     r"^seacell_X_combat_d8$"),
    ("SEACells + BBKNN",         r"^seacell_X_bbknn$"),
    ("SEACells + BBKNN d8",      r"^seacell_X_bbknn_d8$"),
    # scProto: the NEW cohesion runs carry an explicit experiment name
    ("scProto (cohesion, D42)",  r"^d42_[a-z\-]+_"),
]


def load_labels(run_dir, n_cells):
    """Per-cell metacell labels from a run dir, or None."""
    csv = os.path.join(run_dir, "umap_cells.csv")
    if os.path.exists(csv):
        try:
            d = pd.read_csv(csv, usecols=["metacell_id"])
            if len(d) == n_cells:
                return d["metacell_id"].to_numpy()
        except Exception:
            pass
    npz = os.path.join(run_dir, "clusters.npz")
    if os.path.exists(npz):
        try:
            z = np.load(npz, allow_pickle=True)
            for k in ("assignments", "labels", "clusters", "arr_0"):
                if k in z:
                    v = np.asarray(z[k]).ravel()
                    if v.shape[0] == n_cells:
                        return v
        except Exception:
            pass
    return None


def run(ds):
    n, h5, bk, lk, nk, K = DATASETS[ds]
    t0 = time.time()
    A_tr = load_affinity(
        f"graphs/affinity_{ds}{n}_ncomp50_kneighbors50_arbf_heldout20_seed0.pkl")
    A_te = load_affinity(
        f"graphs/affinity_{ds}{n}_ncomp50_kneighbors50_arbf_heldout20_seed0"
        f"_testedges.pkl")
    obs = read_obs(os.path.join(os.environ["DATA_DIR"], h5),
                   [bk, lk] + ([nk] if nk else []))
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    nis = np.asarray(obs[nk]).astype(str) if nk and obs.get(nk) is not None else None
    print(f"\n=== {ds}: N={n:,}  train-edges={A_tr.nnz//2:,}  "
          f"test-edges={A_te.nnz//2:,}  batches={len(set(bts))}  "
          f"celltypes={len(set(cts))}  ({time.time()-t0:.0f}s)")

    root = os.path.join(MODELS, ds)
    dirs = [e for e in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, e))]
    cache_tr, cache_te = {}, {}
    rows = []
    for label, pat in BASELINES:
        hits = [d for d in dirs if re.search(pat, d)]
        if not hits:
            print(f"   --  {label:<26} MISSING")
            continue
        for h in hits:
            lab = load_labels(os.path.join(root, h), n)
            if lab is None:
                print(f"   --  {label:<26} {h[:40]}: no usable per-cell labels")
                continue
            name = label if len(hits) == 1 else f"{label} [{h[:28]}]"
            r = score_partition(lab, cts, bts, A_train=A_tr, A_test=A_te,
                                niches=nis, n_proto=None,
                                _cache=cache_tr, _cache_test=cache_te)
            r["method"] = name
            r["run_dir"] = h
            rows.append(r)
            print(f"   OK  {name:<34} K={r['n_mc']:<5} "
                  f"coh_aff_test={r.get('coh_aff_test_mean', float('nan')):.5f} "
                  f"mod_test={r.get('modularity_test_mean', float('nan')):.4f}")
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("method")
    outdir = "files/tables/d42_scoreboard"
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/{ds}_scoreboard.csv")
    print(f"   saved {outdir}/{ds}_scoreboard.csv  ({time.time()-t0:.0f}s)")
    return df


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DATASETS)
    for ds in targets:
        try:
            run(ds)
        except Exception as e:
            import traceback
            print(f"\n{ds}: FAILED -- {e}")
            traceback.print_exc()
