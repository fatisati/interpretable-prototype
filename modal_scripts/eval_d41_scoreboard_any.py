"""[D41] The frozen scoreboard for any dataset: cohesion arms vs SEACells.

Both sides go through interpretable_ssl.evaluation.metacell_scoreboard, so the
comparison is the same code on the same graph -- the only way it means anything.

Metrics and the claim each answers (see the module docstring for definitions):
  1 preserves within-batch communities -> coh_aff/coh_gram, per-batch AND global
  2 combines batches                   -> batch_entropy (UNNORMALISED nats)
  3 bio proof                          -> ct_purity, coverage, and for spatial
                                          datasets joint_purity + covered states
  4 useful for rare types              -> the PAPER's batch-local rare metrics,
                                          especially cross_batch_homogeneity,
                                          the only one immune to the
                                          "each batch gets its own private
                                          cluster" loophole

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run \
      modal_scripts/eval_d41_scoreboard_any.py --datasets pancreas,pbmc-immune
"""
import modal

app = modal.App("scproto-d41-scoreboard")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pandas", "h5py", "scikit-learn")
)

volume = modal.Volume.from_name("scproto-data")

# ds -> (affinity pkl, h5ad under /vol/data, batch_key, label_key, niche_key,
#        K, seacells run dir under /vol/models)
DATASETS = {
    "pancreas": ("graphs/affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl",
                 "pancreas_hvg.h5ad", "tech", "celltype", None, 220,
                 "pancreas/seacell"),
    "pbmc-immune": ("graphs/affinity_pbmc-immune33506_ncomp50_kneighbors50_arbf.pkl",
                    "scpoli/Immune_ALL_human_hvg.h5ad", "study",
                    "final_annotation", None, 300, "pbmc-immune/seacell"),
    "lung": ("graphs/affinity_lung32472_ncomp50_kneighbors50_arbf.pkl",
             "lung_hvg.h5ad", "batch", "cell_type", None, 300, "lung/seacell"),
    "crcx": ("graphs/affinity_crcx50130_ncomp50_kneighbors50_arbf.pkl",
             "spatial/crc_xenium/crc_xenium_prepped.h5ad", "patient_id",
             "celltype", "Niche", 670, "crcx/seacell_arbf"),
}

ARMS = ["A_aff", "B_gram", "C_both", "D_ctrl"]


@app.function(image=image, timeout=7200, volumes={"/vol": volume},
              cpu=8, memory=65536)
def evaluate(ds: str):
    import glob
    import os
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    from interpretable_ssl.evaluation.metacell_scoreboard import (
        load_affinity, read_obs, score_partition)

    pkl, h5rel, bkey, lkey, nkey, K, sc_dir = DATASETS[ds]
    A = load_affinity(pkl)
    N = A.shape[0]
    obs = read_obs(os.path.join("/vol/data", h5rel),
                   [bkey, lkey] + ([nkey] if nkey else []))
    batches = obs[bkey].astype(str)
    cts = obs[lkey].astype(str)
    niches = obs[nkey].astype(str) if nkey and obs.get(nkey) is not None else None
    print(f"[{ds}] N={N:,} edges={A.nnz//2:,} batches={len(set(batches))} "
          f"celltypes={len(set(cts))}"
          + (f" niches={len(set(niches))}" if niches is not None else ""))

    cache = {}
    rows = []

    for arm in ARMS:
        hits = sorted(glob.glob(f"/vol/models/**/*d41_{ds}_{arm}*/clusters.npz",
                                recursive=True))
        if not hits:
            print(f"  [{arm}] no clusters.npz -- skipped")
            continue
        z = np.load(hits[0], allow_pickle=True)
        key = next((k for k in ("assignments", "labels", "clusters", "arr_0")
                    if k in z), None)
        lab = np.asarray(z[key]).ravel()
        if lab.shape[0] != N:
            print(f"  [{arm}] SKIPPED: {lab.shape[0]} labels vs {N} cells")
            continue
        print(f"  [{arm}] {len(np.unique(lab))} live prototypes")
        r = score_partition(lab, cts, batches, A=A, niches=niches,
                            n_proto=K, _cache=cache)
        r["method"] = arm
        rows.append(r)

    # --- SEACells on the identical graph -----------------------------------
    sc_csv = f"/vol/models/{sc_dir}/umap_cells.csv"
    if os.path.exists(sc_csv):
        d = pd.read_csv(sc_csv)
        if len(d) == N:
            print(f"  [SEACells] {d.metacell_id.nunique()} metacells")
            r = score_partition(d["metacell_id"].to_numpy(), cts, batches,
                                A=A, niches=niches, _cache=cache)
            r["method"] = "SEACells"
            rows.append(r)
        else:
            print(f"  [SEACells] SKIPPED: {len(d)} rows vs {N} cells")
    else:
        print(f"  [SEACells] {sc_csv} not on the volume")

    if not rows:
        return ds, {}
    df = pd.DataFrame(rows).set_index("method")
    outdir = "/vol/code/files/tables/d41_scoreboard"
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/{ds}_scoreboard.csv")
    volume.commit()
    return ds, df.to_dict()


BLOCKS = [
    ("1. WITHIN-BATCH COMMUNITIES (the objective itself)",
     ["coh_aff_perbatch", "coh_gram_perbatch", "coh_aff_global",
      "coh_gram_global"]),
    ("2. COMBINES BATCHES (higher better; unnormalised nats)",
     ["batch_entropy_mean", "batch_entropy_std", "batch_entropy_wmean"]),
    ("3. BIO PROOF",
     ["ct_purity_mean", "ct_purity_std", "ct_purity_wmean", "coverage",
      "joint_purity", "n_covered_states", "n_states_total", "state_coverage"]),
    ("4. RARE CELL TYPES (paper definitions, batch-local, mean over batches)",
     ["rare_cross_batch_homogeneity_mean", "rare_homogeneity_mean",
      "rare_f1_mean", "rare_coverage_mean", "rare_recall_mean",
      "rare_precision_mean", "rare_repr_macro_mean", "rare_purity_mean",
      "n_rare_types"]),
    ("housekeeping", ["n_dead", "n_mc", "median_size"]),
]


@app.local_entrypoint()
def main(datasets: str = "pancreas"):
    import pandas as pd
    order = ARMS + ["SEACells"]
    for ds, res in evaluate.map([d.strip() for d in datasets.split(",")
                                 if d.strip()]):
        if not res:
            print(f"\n[{ds}] nothing to report")
            continue
        df = pd.DataFrame(res).T
        df = df[[c for c in order if c in df.columns]]
        pd.set_option("display.width", 220)
        print("\n" + "=" * 104)
        print(f"[D41] {ds.upper()} SCOREBOARD -- cohesion arms vs SEACells, "
              f"same graph, same code")
        print("=" * 104)
        for title, keys in BLOCKS:
            sub = df.reindex(keys).dropna(how="all")
            if sub.empty:
                continue
            print(f"\n--- {title}")
            print(sub.to_string(float_format=lambda v: f"{v:.4f}"))
        if "C_both" in df.columns and "D_ctrl" in df.columns:
            print("\n--- ATTRIBUTION: C_both(locl) vs D_ctrl(off), "
                  "identical except coh_batch")
            for k in ("batch_entropy_mean", "rare_cross_batch_homogeneity_mean",
                      "rare_recall_mean", "rare_coverage_mean", "rare_f1_mean",
                      "ct_purity_mean", "rare_precision_mean"):
                if k in df.index:
                    c, d_ = df.loc[k, "C_both"], df.loc[k, "D_ctrl"]
                    print(f"    {k:<38} C {c:>8.4f}   D {d_:>8.4f}   "
                          f"delta {c-d_:+.4f}")
        print(f"\nsaved: files/tables/d41_scoreboard/{ds}_scoreboard.csv")
