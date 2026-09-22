"""[D27] The d27 self-similarity arms against SEACells, on the metrics that decide it.

REPORTS, per arm:
    cohesion (gram)   the objective's own value, hard assignments, full graph. Reference
                      points on this graph: Leiden 0.0816, random 0.0048, all-one 0.0014.
    cohesion (aff)    the 1-hop version of the same. Leiden 0.0384, random 0.0039.
    joint purity      cell type AND niche together -- the thing the paper claims
    ct purity         cell type alone
    DGE pearson_r     niche-program recovery within cell type
    DGE tpr           fraction of the true DE genes recovered
    n_pairs           how many (cell type, niche) states exist in the ground truth
    n_covered         how many of them have at least one metacell that is niche-positive
    frac_covered      n_covered / n_pairs   <- "ratio of DGE states covered"
    n_metacells, median size, max/median    the size distribution

All arms share affinity_type='bk02' and K=200, so any gap is the loss, not the graph.
SEACells here is seacell_X_bklam02_K200 -- the same BANKSY(lambda=0.2) embedding, zero
scProto loss -- i.e. what a healthy partition looks like on this exact geometry.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/eval_d27_vs_seacells.py
CPU only, no GPU.
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


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": volume})
def evaluate():
    import os, sys
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

    import numpy as np, pandas as pd, scipy.sparse as sp
    import matplotlib
    matplotlib.use("Agg")

    from interpretable_ssl.datasets.dataset_configs import load_dataset
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        joint_purity_per_metacell, metacell_sizes)
    import interpretable_ssl.evaluation.niche_dge as nd
    from interpretable_ssl.trainers.graph_generator import load_or_build_affinity

    CT_KEY, NICHE_KEY = "celltypes", "niches_2D"
    DS = "s28nsc"
    adata = load_dataset(DS)
    print(f"adata {adata.shape}")

    affinity = load_or_build_affinity(adata, DS, "bk02")
    A = sp.csr_matrix(affinity).astype(np.float32)
    A = ((A + A.T) / 2).tocsr(); A.setdiag(0); A.eliminate_zeros()
    N = A.shape[0]
    # gram descriptor exactly as training builds it: diagonal 1, then rows to unit length
    B = (A + 1.0 * sp.eye(N, format="csr", dtype=np.float32)).tocsr()
    rn = np.sqrt(B.multiply(B).sum(1)).A.ravel(); rn[rn == 0] = 1.0
    R = sp.csr_matrix(sp.diags(1.0 / rn) @ B)

    def cohesion_gram(lab):
        """(1/N) * sum_k within[k]/size[k],  within[k] = ||sum_{i in k} R[i,:]||^2."""
        u = {v: i for i, v in enumerate(pd.unique(lab))}
        S = sp.csr_matrix((np.ones(len(lab), np.float32),
                           (np.arange(len(lab)), [u[v] for v in lab])),
                          shape=(len(lab), len(u)))
        P = R.T @ S
        w = np.asarray(P.multiply(P).sum(0)).ravel()
        mk = np.asarray(S.sum(0)).ravel()
        return float((w / np.maximum(mk, 1e-9)).sum() / len(lab))

    def cohesion_aff(lab, d=1.0):
        """same shape, similarity = raw affinity with sim[c][c] = d."""
        u = {v: i for i, v in enumerate(pd.unique(lab))}
        S = sp.csr_matrix((np.ones(len(lab), np.float32),
                           (np.arange(len(lab)), [u[v] for v in lab])),
                          shape=(len(lab), len(u)))
        w = np.asarray(((A @ S).multiply(S)).sum(0)).ravel()
        mk = np.asarray(S.sum(0)).ravel()
        return float(((w + d * mk) / np.maximum(mk, 1e-9)).sum() / len(lab))

    model_base = os.path.join("/vol/models", DS)
    all_dirs = os.listdir(model_base)

    def find(require, exclude=()):
        c = [d for d in all_dirs
             if all(r in d for r in require) and not any(e in d for e in exclude)]
        return os.path.join(model_base, sorted(c)[-1]) if c else None

    runs = {
        "d27_A_aff_self":  find(["d27_A_aff_self"]),
        "d27_B_gram_self": find(["d27_B_gram_self"]),
        "d27_C_both_self": find(["d27_C_both_self"]),
        "d26_A_gram_noself": find(["d26_A_gram"]),
        "SEACells_bklam02": os.path.join(model_base, "seacell_X_bklam02_K200"),
    }

    labels_by_run = {}
    for name, d in runs.items():
        if d is None or not os.path.exists(os.path.join(d, "cell_assignments.csv")):
            print(f"SKIP {name}: no cell_assignments.csv (not finished yet?)")
            continue
        lab = (pd.read_csv(os.path.join(d, "cell_assignments.csv"))
               .set_index("cell_id")["metacell_id"].reindex(adata.obs_names.to_numpy()))
        n_na = int(lab.isna().sum())
        if n_na:
            print(f"{name}: {n_na} cells unassigned, dropped")
        labels_by_run[name] = lab.dropna()
        print(f"OK   {name}: {labels_by_run[name].nunique()} metacells")

    OUT = "/vol/code/files/tables/s28nsc_d27_vs_seacells"
    os.makedirs(OUT, exist_ok=True)

    print("\n=== ground truth (niche DE) ===")
    gt = nd.get_ground_truth(adata, "/vol/code/files/celltype_niches_full_s28nsc.csv",
                             ct_key=CT_KEY, niche_key=NICHE_KEY)
    print(f"{len(gt)} (cell type, niche) pairs in the ground truth")

    rows = []
    for name, lab in labels_by_run.items():
        print(f"\n--- {name} ---")
        jp = joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
        sizes = metacell_sizes(lab)
        res = nd.run_niche_dge(name, lab, adata, gt, OUT,
                               ct_key=CT_KEY, niche_key=NICHE_KEY)
        pp, summ = res["per_pair"], res["summary"]
        # a pair is COVERED when at least one metacell is niche-positive for it, which is
        # exactly what median_pos_mc_size being defined means
        covered = int(pp["median_pos_mc_size"].notna().sum()) if "median_pos_mc_size" in pp else np.nan
        rows.append({
            "run": name,
            "cohesion_gram": cohesion_gram(lab.values),
            "cohesion_aff": cohesion_aff(lab.values),
            "joint_purity": float(np.nanmean(jp.values if hasattr(jp, "values") else jp)),
            "dge_pearson_r": float(summ.get("pearson_r", pd.Series([np.nan])).iloc[0]),
            "dge_tpr": float(summ.get("tpr", pd.Series([np.nan])).iloc[0]),
            "n_pairs": int(len(pp)),
            "n_covered": covered,
            "frac_covered": covered / max(len(pp), 1),
            "n_metacells": int(lab.nunique()),
            "median_size": float(sizes.median()),
            "max_over_median": float(sizes.max() / max(sizes.median(), 1)),
        })

    df = pd.DataFrame(rows).set_index("run")
    df.to_csv(f"{OUT}/d27_vs_seacells.csv")
    print("\n" + "=" * 110)
    print("[D27] arms vs SEACells -- s28nsc / bk02 / K=200")
    print("=" * 110)
    print(df.round(4).to_string())
    print("\ncohesion reference points on this graph (hard, full-graph):")
    print("   gram: Leiden 0.0816  bad30 0.0450  random 0.0048")
    print("   aff : Leiden 0.0384  bad30 0.0216  random 0.0039")
    volume.commit()
    return df.to_dict()


@app.local_entrypoint()
def main():
    evaluate.remote()
