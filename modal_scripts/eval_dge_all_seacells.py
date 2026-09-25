"""Every cohesion arm (d26-d30) against SEACells on the SAME bk02 graph, on exactly the
metrics that decide the design:

    dead          200 - number of metacells that actually hold cells
    joint_purity  per metacell: largest fraction of one (cell type, niche) pair;
                  MEAN OVER CELLS (size-weighted)
    ct_purity     same but cell type alone
    dge_r         niche-program recovery within cell type (pearson_r, macro over pairs)
    dge_tpr       fraction of true DE genes recovered
    n_covered     (cell type, niche) states with at least one niche-positive metacell
    coh_gram      the cohesion objective itself, hard assignments, FULL graph,
                  similarity = cosine of closed neighbourhood rows, self included,
                  mean over cells.  References: Leiden 0.0816, random 0.0048.
    coh_aff       same with similarity = raw affinity, sim[c][c]=1.  Leiden 0.0384.
    median / max_med   the size distribution

SEACells here = seacell_X_bklam02_K200: archetypes on the SAME BANKSY(lambda=0.2)
embedding, zero scProto loss -- the target the cohesion design is trying to reach.

Run (workspace fatemehs-hashemig, where all run dirs live; d30's assignment CSVs are
bridged over from helmholtz first):
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/eval_all_vs_seacells.py
CPU only.
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

# --- which graph this pass scores on -------------------------------------
# A head-to-head is only valid when BOTH sides use the SAME graph, so the
# affinity tag and the SEACells baseline fitted on that same embedding move
# together. Pairs that actually exist (checked 2026-09-23):
#     bk02 <-> seacell_X_bklam02_K200      (BANKSY lambda 0.2)
#     bk04 <-> seacell_X_bklam04_K200      (BANKSY lambda 0.4)
# lambda 0.5 / 0.8 have affinities but NO SEACells baseline yet, so they
# cannot be added here until one is fitted.
AFF_TAG = "bk02"
SEACELLS_DIR = "seacell_X_bklam02_K200"


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
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    import numpy as np, pandas as pd, scipy.sparse as sp
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")

    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity
    from interpretable_ssl.evaluation import niche_dge as nd

    DS_ID, CT_KEY, NICHE_KEY = "s28nsc", "celltypes", "niches_2D"
    ds_conf = DATASETS[DS_ID]
    adata = sc.read_h5ad(ds_conf["path"])
    print(f"adata {adata.shape}")

    affinity = load_or_build_affinity(
        adata, DS_ID, affinity_type=AFF_TAG, k=50, k_neighbors=50,
        bk=ds_conf.get("batch_key"))
    A = sp.csr_matrix(affinity).astype(np.float32)
    A = ((A + A.T) / 2).tocsr(); A.setdiag(0); A.eliminate_zeros()
    N = A.shape[0]
    B = (A + 1.0 * sp.eye(N, format="csr", dtype=np.float32)).tocsr()
    rn = np.sqrt(B.multiply(B).sum(1)).A.ravel(); rn[rn == 0] = 1.0
    R = sp.csr_matrix(sp.diags(1.0 / rn) @ B)

    def S_of(lab_codes, K):
        return sp.csr_matrix((np.ones(len(lab_codes), np.float32),
                              (np.arange(len(lab_codes)), lab_codes)),
                             shape=(len(lab_codes), K))

    def cohesions(lab):
        codes, uniq = pd.factorize(lab)
        S = S_of(codes, len(uniq))
        mk = np.asarray(S.sum(0)).ravel()
        P = R.T @ S
        w = np.asarray(P.multiply(P).sum(0)).ravel()          # gram, self incl (diag=1)
        coh_g = float((w / np.maximum(mk, 1e-9)).sum() / N)
        wa = np.asarray(((A @ S).multiply(S)).sum(0)).ravel() + mk   # aff + self=1
        coh_a = float((wa / np.maximum(mk, 1e-9)).sum() / N)
        return coh_g, coh_a

    def purities(lab):
        df = pd.DataFrame({
            "mc": np.asarray(lab),
            "ct": adata.obs[CT_KEY].to_numpy(),
            "joint": (adata.obs[CT_KEY].astype(str) + "||"
                      + adata.obs[NICHE_KEY].astype(str)).to_numpy(),
        })
        # per metacell: the largest single-label fraction; MEAN OVER CELLS =
        # size-weighted, i.e. sum over metacells of (top count) / total cells
        ct_p = df.groupby("mc")["ct"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)
        j_p = df.groupby("mc")["joint"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)
        return float(ct_p), float(j_p)

    model_base = os.path.join(os.environ["MODEL_DIR"], DS_ID)
    all_dirs = os.listdir(model_base)

    def find(sub):
        c = sorted(d for d in all_dirs if sub in d)
        return os.path.join(model_base, c[-1]) if c else None

    runs = {f"SEACells_{AFF_TAG}": os.path.join(model_base, SEACELLS_DIR)}
    # Every SEACells variant that exists, so dge_r (PEARSON) -- the deciding
    # bio signal -- is measured for ALL of them in ONE pass. joint_purity /
    # ct_purity / dge_r / n_covered are computed from assignments + labels and
    # are INDEPENDENT of AFF_TAG, so a single run gives every row. Only
    # coh_aff / coh_gram depend on the graph and are read against AFF_TAG.
    for _sc, _dir in (("SEACells_lam02", "seacell_X_bklam02_K200"),
                      ("SEACells_lam04", "seacell_X_bklam04_K200"),
                      ("SEACells_lam08", "seacell_bk08"),
                      ("SEACells_arbf",  "seacell_arbf"),
                      ("SEACells_pca",   "seacell_K200")):
        _p = os.path.join(model_base, _dir)
        if os.path.exists(os.path.join(_p, "cell_assignments.csv")):
            runs[_sc] = _p
    for sub in ("d26_A_gram", "d27_A_aff_self", "d27_B_gram_self",
                "d28_A_aff", "d28_B_gram", "d28_C_both",
                "d29_A_aff", "d29_B_gram", "d29_C_both",
                "d30_A_aff", "d30_B_gram", "d30_C_both",
                # [D31]/[D32]: age-weighted + honestly-initialized [D29] memory,
                # gram and aff arms, all vs the d30 same-similarity control above
                "d31_ctrl", "d31_init", "d31_full",
                "d31a_ainit", "d31a_afull",
                # [D39] the direction test, with the [D37] normalizer fixed:
                # A = cohesion ALONE, B = cohesion + community. A reached
                # coh_aff 0.0296 / joint_purity 0.4472 / 0 dead; B 0.0126 /
                # 0.3951 / 125 dead. This pass adds dge_r (pearson) and
                # n_covered, which run_mc_task cannot compute.
                "d39_A_cohonly", "d39_B_cohcomm",
                "d39_C_gramonly", "d39_E_bothonly",
                # [D50] the routed-community arm: D39-E cohesion + lambda_umap
                # 0.02 with umap_proto_sg=1 (encoder-only community gradient)
                "d50rt_rtA",
                # [D40] the bk02 winner retrained on bk04 -- only present, and
                # only meaningful, when AFF_TAG == "bk04"
                "d40_bk04"):
        runs[sub] = find(sub)

    # --- FREE-LOGIT arm: the exact objective optimised directly (no encoder) ----
    # Measured locally: cohesion 0.0014 -> 0.0886 in 50 iters (> SEACells 0.0866),
    # 200/200 alive, no reinit. Sharp eps is REQUIRED (soft eps=1.0 stays flat).
    import torch
    _Rt = sp.csr_matrix(R.T)
    RtT = torch.sparse_csr_tensor(
        torch.from_numpy(_Rt.indptr.astype(np.int64)),
        torch.from_numpy(_Rt.indices.astype(np.int64)),
        torch.from_numpy(_Rt.data.astype(np.float32)), size=_Rt.shape)
    torch.manual_seed(0)
    logits = (0.01 * torch.randn(N, 200)).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for it in range(150):
        S = torch.softmax(logits / 0.05, 1)
        P = torch.sparse.mm(RtT, S)
        within = (P * P).sum(0) - 0.5 * S.pow(2).sum(0)      # w = 0.5
        loss = 1.0 - (within / S.sum(0).clamp(min=1e-8)).sum() / N
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            print(f"  [free_logits] iter {it}  cohesion = {1.0 - loss.item():.4f}")
    fl_lab = torch.softmax(logits / 0.05, 1).argmax(1).numpy()
    fl_dir = os.path.join(model_base, "free_logits_B")
    os.makedirs(fl_dir, exist_ok=True)
    pd.DataFrame({"cell_id": adata.obs_names.to_numpy(),
                  "metacell_id": [f"fl_{i}" for i in fl_lab]}).to_csv(
        os.path.join(fl_dir, "cell_assignments.csv"), index=False)
    runs["free_logits_B"] = fl_dir

    print("\n=== ground truth (niche DE) ===")
    gt = nd.get_ground_truth(adata, "/vol/code/files/celltype_niches_full_s28nsc.csv",
                             ct_key=CT_KEY, niche_key=NICHE_KEY)
    print(f"{len(gt)} (cell type, niche) pairs")

    OUT = f"/vol/code/files/tables/s28nsc_all_vs_seacells_{AFF_TAG}"
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for name, d in runs.items():
        if d is None or not os.path.exists(os.path.join(d, "cell_assignments.csv")):
            print(f"SKIP {name}: no cell_assignments.csv")
            continue
        lab = (pd.read_csv(os.path.join(d, "cell_assignments.csv"))
               .set_index("cell_id")["metacell_id"]
               .reindex(adata.obs_names.to_numpy()).dropna())
        ad_l = adata[lab.index]
        # restrict every computation to the assigned cells (SEACells may drop none)
        ct_p, j_p = purities(lab.values)
        coh_g, coh_a = cohesions(lab.values) if len(lab) == N else (np.nan, np.nan)
        sizes = lab.value_counts()
        res = nd.run_niche_dge(name, lab, adata, gt, OUT, ct_key=CT_KEY, niche_key=NICHE_KEY)
        pp, summ = res["per_pair"], res["summary"]
        covered = int(pp["median_pos_mc_size"].notna().sum()) if "median_pos_mc_size" in pp else -1
        rows.append({
            "run": name,
            "dead": 200 - int(lab.nunique()),
            "joint_purity": round(j_p, 4),
            "ct_purity": round(ct_p, 4),
            "dge_r": round(float(summ["pearson_r"].iloc[0]), 4) if "pearson_r" in summ else np.nan,
            "dge_tpr": round(float(summ["tpr"].iloc[0]), 4) if "tpr" in summ else np.nan,
            "n_covered": covered,
            "coh_gram": round(coh_g, 4),
            "coh_aff": round(coh_a, 4),
            "median": int(sizes.median()),
            "max_med": round(float(sizes.max() / max(sizes.median(), 1)), 1),
        })
        print(rows[-1])

    df = pd.DataFrame(rows).set_index("run")
    df.to_csv(f"{OUT}/all_vs_seacells_{AFF_TAG}.csv")
    print("\n" + "=" * 118)
    print(f"ALL ARMS vs SEACells -- s28nsc / {AFF_TAG} / K=200")
    print("=" * 118)
    print(df.to_string())
    print("\ncohesion references (hard, full graph): gram Leiden 0.0816 random 0.0048 | "
          "aff Leiden 0.0384 random 0.0039")
    volume.commit()
    return df.to_dict()


@app.local_entrypoint()
def main():
    evaluate.remote()
