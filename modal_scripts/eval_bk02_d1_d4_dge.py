"""Collapse diagnostics + niche-DGE recovery for the bk02 (BANKSY lambda=0.2) arms on
s28nsc, K=200 -- run AFTER train_bk02_d1_offdiag_min.py has finished.

Arms compared (all on the SAME affinity_type='bk02' graph, so any gap is attributable to
the loss, not to a different spatial context):
  - D1_only     : nassoc_diag_norm='pair' only (nassoc_offdiag_norm left at default 'sqrt')
                  -- the already-trained 'proto_bk02_..._d1pair_modal_...' run.
  - D1_D4       : nassoc_diag_norm='pair' + nassoc_offdiag_norm='min'
                  -- the new 'proto_bk02_..._d1_offdmin_modal_...' run.
  - SEACells_bklam02 : transcriptomics + BANKSY(lambda=0.2) archetypes, zero scProto loss
                  -- the reference for what a healthy size/tightness distribution looks
                  like on this exact embedding.

Run directories are auto-discovered by substring match on os.listdir(MODEL_DIR/'s28nsc')
rather than the library's extract_model_key resolver, so this script does not need to
know the exact auto-generated suffix (NP200_prtInit-wayp_..._v31 etc.) in advance.

Writes (then pull back locally with `modal volume get`):
  /vol/code/files/celltype_niches_full_s28nsc.csv   ground truth (shared, computed once)
  /vol/code/files/tables/s28nsc_bk02_d1d4/*.csv
  /vol/code/files/figures/s28nsc_bk02_d1d4/*.png

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/eval_bk02_d1_d4_dge.py
CPU-only -- no GPU needed for eval.
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
)

volume = modal.Volume.from_name("scproto-data")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/vol": volume})
def eval_bk02_arms():
    import os
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"

    import sys
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    import pandas as pd
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        joint_purity_per_metacell, collapse_summary, plot_collapse, metacell_sizes,
    )
    from interpretable_ssl.evaluation import niche_dge as nd

    DS_ID, CT_KEY, NICHE_KEY = "s28nsc", "celltypes", "niches_2D"

    ds_conf = DATASETS[DS_ID]
    adata = sc.read_h5ad(ds_conf["path"])
    print(adata)

    affinity = load_or_build_affinity(
        adata, DS_ID, affinity_type="bk02", k=50, k_neighbors=50,
        bk=ds_conf.get("batch_key"))
    print("affinity:", affinity.shape)

    model_base = os.path.join(os.environ["MODEL_DIR"], DS_ID)
    run_dirs = os.listdir(model_base)

    def find_run(require, exclude=()):
        cands = [d for d in run_dirs if all(r in d for r in require) and not any(e in d for e in exclude)]
        if not cands:
            raise FileNotFoundError(f"no run dir matching require={require} exclude={exclude} under {model_base}")
        cands.sort()
        if len(cands) > 1:
            print(f"WARNING: {len(cands)} run dirs matched require={require} exclude={exclude}, using last: {cands}")
        return os.path.join(model_base, cands[-1])

    run_d1_only = find_run(require=["aff-bk02", "nadn-pair"], exclude=["naon"])
    run_d1_d4 = find_run(require=["aff-bk02", "nadn-pair", "naon"])
    run_seacells = os.path.join(model_base, "seacell_X_bklam02_K200")

    print("D1_only run   :", run_d1_only)
    print("D1_D4 run     :", run_d1_d4)
    print("SEACells run  :", run_seacells)

    def load_labels(run_dir):
        df = pd.read_csv(os.path.join(run_dir, "cell_assignments.csv"))
        return df.set_index("cell_id")["metacell_id"].reindex(adata.obs_names.to_numpy())

    labels_by_run = {
        "D1_only_bk02": load_labels(run_d1_only),
        "D1_D4_bk02": load_labels(run_d1_d4),
        "SEACells_bklam02": load_labels(run_seacells),
    }
    for name, lab in labels_by_run.items():
        n_na = int(lab.isna().sum())
        if n_na:
            print(f"{name}: {n_na} cells unassigned, excluding from that arm's metacells")
        labels_by_run[name] = lab.dropna()

    OUT_DIR = "/vol/code/files/tables/s28nsc_bk02_d1d4"
    FIG_DIR = "/vol/code/files/figures/s28nsc_bk02_d1d4"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n=== collapse summary ===")
    summaries = {}
    for name, lab in labels_by_run.items():
        pur = joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
        summaries[name] = collapse_summary(lab, affinity=affinity, name=name, purity=pur)
        plot_collapse(lab, affinity, name=name)
        plt.tight_layout(); plt.savefig(f"{FIG_DIR}/{name}.png", dpi=120, bbox_inches="tight")
        plt.close()
    collapse_df = pd.DataFrame(list(summaries.values())).set_index("run")
    collapse_df["n_metacells"] = pd.Series({n: lab.nunique() for n, lab in labels_by_run.items()})
    collapse_df["median_size"] = pd.Series({n: metacell_sizes(lab).median() for n, lab in labels_by_run.items()})
    collapse_df.to_csv(f"{OUT_DIR}/collapse_summary.csv")
    print(collapse_df.round(4).T)

    print("\n=== ground truth (niche DE) ===")
    DGE_GT_PATH = "/vol/code/files/celltype_niches_full_s28nsc.csv"
    ground_truth = nd.get_ground_truth(adata, DGE_GT_PATH, ct_key=CT_KEY, niche_key=NICHE_KEY)
    print(f"{len(ground_truth)} (cell type, niche) pairs")

    print("\n=== DGE per arm ===")
    dge_results = {}
    for name, lab in labels_by_run.items():
        dge_results[name] = nd.run_niche_dge(
            name, lab, adata, ground_truth, OUT_DIR, ct_key=CT_KEY, niche_key=NICHE_KEY)

    dge_summary = nd.summary_table(dge_results)
    dge_summary.to_csv(f"{OUT_DIR}/dge_summary.csv")
    print(dge_summary.round(4))

    fig = nd.plot_dge_summary(
        dge_summary, highlight="D1_D4_bk02", out_path=f"{FIG_DIR}/dge_summary.png",
        title="Niche-program recovery, within cell type (s28nsc, K=200, BANKSY lambda=0.2)")
    plt.close(fig) if fig is not None else None

    dge_paired = nd.compare_arms(dge_results, reference="D1_only_bk02", metric="pearson_r")
    dge_paired.to_csv(f"{OUT_DIR}/dge_paired_vs_d1only.csv", index=False)
    print(dge_paired.round(3).head(25))

    volume.commit()
    print("\nDone. Wrote:", OUT_DIR, "and", FIG_DIR)
    return {"collapse": collapse_df.to_dict(), "dge_summary": dge_summary.to_dict()}


@app.local_entrypoint()
def main():
    result = eval_bk02_arms.remote()
    print(result)
