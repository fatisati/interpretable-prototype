"""Per (cell_type, niche) state: does each method give it its own metacell(s), and how big?

Answers, with numbers rather than arithmetic hand-waving:
  - how many single cells are in each (cell_type, niche) state
  - how many metacells each method assigns to that state (majority vote)
  - the SIZES of those metacells
  - how the prototype budget is spent across cell types (is there room to split niches?)

scProto and SEACells here use the SAME affinity (bk08), so any difference is the
partitioning method, not the graph.

Output: tables/s28nsc_niche_mc_coverage.csv  (+ prototype allocation printed)
Load-if-exists per CLAUDE.md sec4; FORCE_RERUN=True to recompute.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import h5py
from anndata._io.specs.registry import _REGISTRY, IOSpec


@_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
def _read_null(elem, _reader):
    return None


import anndata as ad

FORCE_RERUN = False

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(HERE)
OUT_CSV = os.path.join(NB_DIR, "tables", "s28nsc_niche_mc_coverage.csv")

H5AD = os.environ.get(
    "PROBE_H5AD",
    r"G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data\spatial\s28nsc.h5ad",
)
MODELS = r"G:\.shortcut-targets-by-id\1-0FnoSFlnEGxgKJQCAbspCuHc3osVs25\models\s28nsc"

PROTO_RUN = (
    "proto_bk08_np200_ema15_d1d2_g0.1_ds-s28n_NP200_prtInit-wayp_aff-bk08_lprec0.01"
    "_usim-prot_cvae_e50_ecal1_lpu0.1_pum-ema_lna1_nagg-max_nadl-cpm_nadn-pair"
    "_nagm0.1_upm-dotp_v31"
)
# NOTE: the scProto run folder name pushes the full path past Windows' 260-char MAX_PATH,
# so os.path.exists() returns False even though the file is there (CLAUDE.md sec11). Copy it
# to a short path first and point SCPROTO_ASSIGN at the copy.
METHODS = {
    "scproto": os.environ.get(
        "SCPROTO_ASSIGN", os.path.join(MODELS, PROTO_RUN, "cell_assignments.csv")
    ),
    "seacell": os.environ.get(
        "SEACELL_ASSIGN", os.path.join(MODELS, "seacell_bk08", "cell_assignments.csv")
    ),
}

CT_KEY, NICHE_KEY = "celltypes", "niches_2D"
EXCLUDE_NICHES = ("Excluded",)


def load_assignments(path, obs_names):
    """cell -> metacell id, reindexed onto obs_names. Tolerates a few column spellings."""
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    cell_col = next(
        (cols[c] for c in ("cell_id", "cell", "index", "unnamed: 0", "obs_names") if c in cols),
        df.columns[0],
    )
    mc_col = next(
        (cols[c] for c in ("metacell_id", "metacell", "seacell", "cluster", "label") if c in cols),
        df.columns[1],
    )
    s = df.set_index(df[cell_col].astype(str))[mc_col]
    return s.reindex(pd.Index(obs_names).astype(str))


def main():
    if os.path.exists(OUT_CSV) and not FORCE_RERUN:
        d = pd.read_csv(OUT_CSV)
        print(f"loaded from {OUT_CSV}")
        return d

    a = ad.read_h5ad(H5AD)
    obs = a.obs[[CT_KEY, NICHE_KEY]].copy()
    obs.columns = ["cell_type", "niche"]
    obs = obs.astype(str)
    n_cells_total = len(obs)
    del a

    assign = {}
    for name, path in METHODS.items():
        s = load_assignments(path, obs.index)
        if s is None:
            continue
        assign[name] = s
        n_mc = s.dropna().nunique()
        cov = s.notna().mean()
        sizes = s.dropna().value_counts()
        print(
            f"{name:<10} metacells={n_mc:<5} coverage={cov:.1%}  "
            f"size mean={sizes.mean():.0f} median={sizes.median():.0f} "
            f"min={sizes.min()} max={sizes.max()} cv={sizes.std()/sizes.mean():.2f}"
        )

    obs_v = obs[~obs["niche"].isin(EXCLUDE_NICHES)]
    states = obs_v.groupby(["cell_type", "niche"]).size().rename("n_cells").reset_index()

    for name, s in assign.items():
        mc_ct = pd.crosstab(s, obs["cell_type"]).idxmax(axis=1)
        mc_ni = pd.crosstab(s, obs["niche"]).idxmax(axis=1)
        mc_size = s.dropna().value_counts()

        owner = pd.DataFrame({"cell_type": mc_ct, "niche": mc_ni, "size": mc_size})
        g = owner.groupby(["cell_type", "niche"])["size"]
        agg = pd.DataFrame(
            {
                f"{name}_n_mc": g.size(),
                f"{name}_mc_sizes": g.apply(lambda x: ",".join(map(str, sorted(x, reverse=True)))),
                f"{name}_cells_in_mc": g.sum(),
            }
        ).reset_index()
        states = states.merge(agg, on=["cell_type", "niche"], how="left")

        # prototype budget per cell type
        per_ct = owner.groupby("cell_type").size().sort_values(ascending=False)
        n_niche_per_ct = obs_v.groupby("cell_type")["niche"].nunique()
        alloc = pd.DataFrame({"n_mc": per_ct, "n_niches": n_niche_per_ct}).dropna()
        alloc["mc_per_niche"] = alloc["n_mc"] / alloc["n_niches"]
        print(f"\n--- {name}: prototype budget per cell type ---")
        print(alloc.sort_values("n_mc", ascending=False).to_string())

    for name in assign:
        states[f"{name}_n_mc"] = states[f"{name}_n_mc"].fillna(0).astype(int)
        states[f"{name}_mc_sizes"] = states[f"{name}_mc_sizes"].fillna("")

    # join the probe AUC / DGE table if present
    probe = os.path.join(NB_DIR, "tables", "s28nsc_niche_probe_auc.csv")
    if os.path.exists(probe):
        p = pd.read_csv(probe)
        p["auc_best"] = p[["auc_expr960", "auc_pca50", "auc_pca50_gbm"]].max(axis=1)
        states = states.merge(
            p[["cell_type", "niche", "auc_best", "n_dge_genes"]],
            on=["cell_type", "niche"],
            how="left",
        )

    states = states.sort_values("n_cells", ascending=False)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    states.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}  ({len(states)} states, {n_cells_total} cells)")
    return states


if __name__ == "__main__":
    main()
