"""Which step makes clusters differ across machines: the neighbour graph, Leiden, or both?

Experiment tool for docs/decisions/0003. Both machines start from the SAME embedding file
(the Mac's committed PCA embedding), so the embedding cannot be the cause.

  build    on machine 1: build the neighbour graph exactly as evaluation/common.py's
           sweep_leiden_to_k does (sc.pp.neighbors defaults, graph seed 0), run Leiden with
           seed 0 at the resolution in config.json, save graph + clusters.
  compare  on machine 2: build its own graph the same way and compare against machine 1's:
           (1) graph: are the neighbour lists and edge weights identical?
           (2) Leiden: run Leiden seed 0 on MACHINE 1's graph -> same clusters as machine 1?
           (3) end to end: Leiden seed 0 on its OWN graph -> same clusters as machine 1?

Reading the result:
  graph identical, Leiden identical     -> no difference (end to end should match too)
  graph differs,   Leiden identical     -> the graph is the source; Leiden is portable
  graph identical, Leiden differs       -> Leiden is the source
  graph differs,   Leiden differs       -> both

Usage (from repo root):
    E=repro/evidence/mac-studio/heart_cell_atlas/pca
    python repro/graph_leiden_check.py build   $E/embedding.csv.gz $E/config.json repro/evidence/windows/graph_leiden_check
    python repro/graph_leiden_check.py compare $E/embedding.csv.gz $E/config.json repro/evidence/windows/graph_leiden_check
"""
import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

SEED = 0


def load(emb_path, cfg_path):
    df = pd.read_csv(emb_path)
    cfg = json.loads(Path(cfg_path).read_text())
    latent = [c for c in df.columns if c.startswith("latent_")]
    train = (df["split"] == "train").to_numpy()
    jobs = {"train": (train, cfg["resolution_train"]),
            "full": (np.ones(len(df), bool), cfg["resolution_full"])}
    return df, latent, jobs


def build_graph(df, latent, mask):
    ad = anndata.AnnData(obs=pd.DataFrame(index=df.loc[mask, "cell_id"].astype(str).to_numpy()))
    ad.obsm["X_emb"] = df.loc[mask, latent].to_numpy(dtype=np.float32)
    sc.pp.neighbors(ad, use_rep="X_emb")  # same call as evaluation/common.py (graph seed 0)
    return ad


def leiden(ad, resolution):
    sc.tl.leiden(ad, resolution=resolution, key_added="c", flavor="igraph", random_state=SEED)
    return ad.obs["c"].astype(str).to_numpy()


def ad_from_saved(npz_path, cell_ids):
    z = np.load(npz_path, allow_pickle=False)
    ad = anndata.AnnData(obs=pd.DataFrame(index=cell_ids))
    conn = sp.csr_matrix((z["conn_data"], z["conn_indices"], z["conn_indptr"]), shape=tuple(z["shape"]))
    dist = sp.csr_matrix((z["dist_data"], z["dist_indices"], z["dist_indptr"]), shape=tuple(z["shape"]))
    ad.obsp["connectivities"], ad.obsp["distances"] = conn, dist
    ad.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                           "params": {"method": "umap"}}
    return ad, z


def save_graph(ad, path):
    c, d = ad.obsp["connectivities"].tocsr(), ad.obsp["distances"].tocsr()
    np.savez_compressed(path, shape=np.array(c.shape),
                        conn_data=c.data, conn_indices=c.indices, conn_indptr=c.indptr,
                        dist_data=d.data, dist_indices=d.indices, dist_indptr=d.indptr,
                        cell_ids=np.array(ad.obs_names, dtype=str))


def neighbour_sets(dist):
    dist = dist.tocsr()
    return [frozenset(dist.indices[dist.indptr[i]:dist.indptr[i + 1]]) for i in range(dist.shape[0])]


def cmd_build(emb, cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, latent, jobs = load(emb, cfg)
    clusters = pd.DataFrame({"cell_id": df["cell_id"].astype(str)})
    for name, (mask, res) in jobs.items():
        ad = build_graph(df, latent, mask)
        save_graph(ad, out / f"graph_{name}.npz")
        labels = leiden(ad, res)
        clusters[f"cluster_{name}"] = pd.NA
        clusters.loc[mask, f"cluster_{name}"] = labels
        mac = df.loc[mask, f"cluster_{name}_run"].astype(str).to_numpy()
        print(f"{name}: resolution {res:.4f}, {len(set(labels))} clusters; "
              f"ARI vs the clusters in the embedding file = {adjusted_rand_score(mac, labels):.4f}")
    clusters.to_csv(out / "clusters_seed0.csv", index=False)


def cmd_compare(emb, cfg, theirs):
    theirs = Path(theirs)
    df, latent, jobs = load(emb, cfg)
    their_clusters = pd.read_csv(theirs / "clusters_seed0.csv", dtype=str)
    for name, (mask, res) in jobs.items():
        ids = df.loc[mask, "cell_id"].astype(str).to_numpy()
        their_ad, z = ad_from_saved(theirs / f"graph_{name}.npz", ids)
        assert list(z["cell_ids"]) == list(ids), "cell order differs from the embedding file"
        mine = build_graph(df, latent, mask)
        target = their_clusters.loc[mask, f"cluster_{name}"].to_numpy()

        # (1) graph
        a, b = neighbour_sets(mine.obsp["distances"]), neighbour_sets(their_ad.obsp["distances"])
        same_lists = np.mean([x == y for x, y in zip(a, b)])
        diff = abs(mine.obsp["connectivities"] - their_ad.obsp["connectivities"])
        max_w = diff.max() if diff.nnz else 0.0
        # (2) Leiden on their graph, (3) Leiden on my graph
        on_theirs = leiden(their_ad, res)
        on_mine = leiden(mine, res)
        ari_l, ari_e = adjusted_rand_score(target, on_theirs), adjusted_rand_score(target, on_mine)
        print(f"\n== {name} job (resolution {res:.4f})")
        print(f"(1) graph:      {same_lists:.2%} of cells have the same 15-neighbour list; "
              f"max edge-weight difference {max_w:.2e}")
        print(f"(2) Leiden:     seed 0 on THEIR graph -> {len(set(on_theirs))} clusters, "
              f"ARI vs theirs {ari_l:.4f}")
        print(f"(3) end to end: seed 0 on MY graph    -> {len(set(on_mine))} clusters, "
              f"ARI vs theirs {ari_e:.4f}")
        g = "identical" if same_lists == 1 and max_w < 1e-6 else "DIFFERS"
        l = "identical" if ari_l == 1 else "DIFFERS"
        print(f"    -> graph {g}, Leiden {l}")


if __name__ == "__main__":
    mode, emb, cfg, d = sys.argv[1:5]
    {"build": cmd_build, "compare": cmd_compare}[mode](emb, cfg, d)
