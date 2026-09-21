"""DGE head-to-head on s28nsc/bk02 -- the decisive metric the run_mc_task eval does not compute.

WHY
---
Every comparison this session has used purity / niche purity / graph modularity, because
that is what run_mc_task returns. But the references the work is judged against are
`pearson_r` from evaluation/niche_dge.py, which measures how well each metacell partition
recovers CELLTYPE x NICHE differential expression:

    Leiden-modularity K=200   DGE 0.856      SEACells bk02        DGE 0.849
    scProto best (CPM)        DGE 0.842      modularity lambda=10 DGE 0.822
    Leiden-CPM K=200          DGE 0.630      modularity lambda=1  DGE 0.672

None of today's new arms has a DGE number, so none of them can yet be compared to those.
This computes it for the arms that matter, from each run's saved cell_assignments.csv.

CPU ONLY -- the DE tests are the cost, there is no GPU work here.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/eval_dge_headtohead.py
"""
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git", extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in [
            "lung", "lung_fetal_donor", "lung_fetal_organoid", "brain", "breast",
            "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas",
        ]
    ])
)

volume = modal.Volume.from_name("scproto-data")

# substrings that uniquely identify each run folder under /vol/models/s28nsc
# Round 2: the arms WITHOUT an off-diagonal, which are the only ones with a healthy size
# distribution (rt2_A median 136 vs 13-15 for every arm carrying one). If their DGE is
# decent then cohesion is fine and the off-diagonal was the entire problem.
ARMS = {
    "pub_A  EXACT published":        ["pub_A_published"],
    "pub_C  corrected diagonal":     ["pub_C_diagonal"],
    "pub_D  both corrections":       ["pub_D_both"],
    "mch_C  modularity + cohesion":  ["mch_C_modu_coh20"],
    "SEACells bk02 K=200":           ["seacell_X_bklam02_K200"],
}


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes={"/vol": volume})
def run():
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass

    import numpy as np
    import pandas as pd
    import anndata as ad_io
    from interpretable_ssl.evaluation import niche_dge as nd

    CT_KEY, NICHE_KEY = "celltypes", "niches_2D"
    base = "/vol/models/s28nsc"
    adata = ad_io.read_h5ad("/vol/data/spatial/NSCLC_3D_section_28.h5ad")
    print(f"[data] {adata.shape}")

    dirs = os.listdir(base)

    def find(subs):
        c = [d for d in dirs if all(s in d for s in subs)]
        if not c:
            raise FileNotFoundError(f"no run matching {subs}")
        c.sort()
        return os.path.join(base, c[-1])

    labels = {}
    for name, subs in ARMS.items():
        try:
            d = find(subs)
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
            continue
        p = os.path.join(d, "cell_assignments.csv")
        if not os.path.exists(p):
            print(f"  SKIP {name}: no cell_assignments.csv in {os.path.basename(d)}")
            continue
        df = pd.read_csv(p)
        lab = df.set_index("cell_id")["metacell_id"].reindex(adata.obs_names.to_numpy()).dropna()
        labels[name] = lab
        print(f"  {name:<28} K_live={lab.nunique():>4}  cells={len(lab):,}  "
              f"({os.path.basename(d)[:40]}...)")

    out = "/vol/code/files/tables/s28nsc_dge_headtohead"
    os.makedirs(out, exist_ok=True)
    gt = nd.get_ground_truth(
        adata, "/vol/code/files/celltype_niches_full_s28nsc.csv",
        ct_key=CT_KEY, niche_key=NICHE_KEY)
    print(f"\n[ground truth] {len(gt)} (cell type, niche) pairs\n")

    res = {}
    for name, lab in labels.items():
        print(f"=== DGE: {name} ===")
        res[name] = nd.run_niche_dge(name, lab, adata, gt, out,
                                     ct_key=CT_KEY, niche_key=NICHE_KEY)
    summary = nd.summary_table(res)
    summary.to_csv(f"{out}/dge_summary.csv")
    volume.commit()

    print("\n" + "=" * 96)
    print("DGE HEAD-TO-HEAD -- s28nsc / bk02, K=200")
    print("=" * 96)
    print(summary.round(4).to_string())
    print("\nreferences from EXPERIMENTS / decisions.md (same metric, pearson_r):")
    print("  Leiden-modularity 0.856   SEACells 0.849   scProto best (CPM) 0.842")
    print("  modularity lambda=10 0.822   modularity lambda=1 0.672   Leiden-CPM 0.630")
    return summary.to_dict()


@app.local_entrypoint()
def main():
    run.remote()
