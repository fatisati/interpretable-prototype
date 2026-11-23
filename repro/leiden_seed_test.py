"""Rerun ONLY the Leiden step of an embedding baseline with other seeds.

Experiment tool for docs/decisions/0003. Same embedding, same neighbour graph (built
exactly as common.sweep_leiden_to_k does), same resolution as the official run (read from
its config.json) -- only Leiden's random_state changes. Seed 0 must reproduce the
official clusters exactly (sanity check, printed). Each seed gets its own folder with an
embedding.csv that evaluation/compute_metrics.py can score.

Usage (from repo root):
    python repro/leiden_seed_test.py runs/rare_cell_preservation/heart_cell_atlas/pca \
        runs/seed_test/heart_cell_atlas 0 1 2 3
"""
import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

run_dir, out_root, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
df = pd.read_csv(run_dir / "embedding.csv")
cfg = json.loads((run_dir / "config.json").read_text())
latent = [c for c in df.columns if c.startswith("latent_")]
train = (df["split"] == "train").to_numpy()


def graph(mask):
    ad = anndata.AnnData(obs=pd.DataFrame(index=df.loc[mask, "cell_id"].astype(str).to_numpy()))
    ad.obsm["X_emb"] = df.loc[mask, latent].to_numpy(dtype=np.float32)
    sc.pp.neighbors(ad, use_rep="X_emb")  # default random_state=0, as in the pipeline
    return ad


runs = {"train": (graph(train), train, cfg["resolution_train"]),
        "full": (graph(np.ones(len(df), bool)), np.ones(len(df), bool), cfg["resolution_full"])}

for seed in seeds:
    out = df.copy()
    for name, (ad, mask, res) in runs.items():
        sc.tl.leiden(ad, resolution=res, key_added="s", flavor="igraph", random_state=seed)
        labels = ad.obs["s"].to_numpy()
        official = df.loc[mask, f"cluster_{name}_run"].astype(str).to_numpy()
        print(f"seed {seed} {name}: {len(set(labels))} clusters, "
              f"ARI vs official (seed 0) = {adjusted_rand_score(official, labels):.4f}")
        col = f"cluster_{name}_run"
        out[col] = pd.NA
        out.loc[mask, col] = labels
    d = out_root / f"pca_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "embedding.csv", index=False)
