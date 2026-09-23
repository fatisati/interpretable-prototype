"""[D41] Pancreas scoreboard: the four cohesion arms vs SEACells, same graph.

THE FROZEN SCOREBOARD -- decided before the run, not chosen after seeing results.
Each metric answers one of the four claims:

  1. "we preserve good WITHIN-batch communities"
        coh_aff / coh_gram, per-batch ([D41]) -- the objective ITSELF, computed
        on hard assignments over the full graph. Reported against the measured
        reference points so the number is interpretable:
            random-220  aff 0.0096  gram 0.0277   <- FLOOR
            celltype    aff 0.0733  gram 0.1860
            leiden-220  aff 0.1169  gram 0.2824   <- practical ceiling
        Also reported under the GLOBAL denominator, so an arm cannot look good
        merely because the metric was changed to match it.

  2. "but also combine batches"
        batch_entropy (mean + std, and size-weighted). SEACells: 0.1792 / 0.1909.
        UNNORMALISED Shannon entropy of the tech distribution inside each
        metacell, in nats -- exactly mc_metric_utils.calc_batch_entropy, which
        is what the paper's published numbers use. 0 = the metacell is a single
        tech; the ceiling is log(9) = 2.197 for pancreas. Higher = better mixed.
        (This implementation was validated by reproducing SEACells' own stored
        metrics.json: purity 0.9564/0.9667 and coverage 0.8571 match to 4
        decimals, and entropy matches once the normalisation is removed.)

  3. "a bio proof"
        ct_purity (mean + std, and size-weighted): per metacell, the fraction of
        its cells carrying the most common cell type. SEACells: 0.9564 / 0.9667.

  4. "why useful? for rare cell types"
        rare_homogeneity, rare_f1_macro, rare_coverage, computed PER BATCH and
        averaged, over the cell types flagged rare by the repo's own
        _get_rare_cts (rare_mode='median_batch', Q25 threshold). On pancreas
        those are the types that actually depend on cross-tech support:
        t_cell (n=7), schwann (25), epsilon (32), mast (42).

Plus n_dead (prototypes no cell chose) and coverage (fraction of cell types that
are the majority label of at least one metacell; SEACells: 0.857).

WHY A SEPARATE POST-HOC SCRIPT
The rare-cell metrics need per-batch label tables that the training loop does
not build, and SEACells' assignments live in a different run dir entirely. Doing
it here means both sides are scored by the SAME code on the SAME graph, which is
the only way the comparison means anything.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run \
       modal_scripts/eval_d41_pancreas_vs_seacells.py
"""
import modal

app = modal.App("scproto-d41-pancreas-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy", "scipy", "pandas", "anndata", "scanpy", "h5py",
                 "scikit-learn", "leidenalg", "python-igraph")
)

volume = modal.Volume.from_name("scproto-data")

ARMS = ["d41_A_aff", "d41_B_gram", "d41_C_both", "d41_D_ctrl"]

# measured on this graph, modal_scripts/measure_cohesion_spans_pancreas.py
REFERENCE = {
    "random-220": dict(coh_aff_pb=0.009624, coh_gram_pb=0.027713),
    "celltype":   dict(coh_aff_pb=0.073253, coh_gram_pb=0.186007),
    "leiden-220": dict(coh_aff_pb=0.116908, coh_gram_pb=0.282367),
}
SEACELLS_REF = dict(ct_purity=0.9564, batch_entropy=0.1792, coverage=0.857)


@app.function(image=image, timeout=3600, volumes={"/vol": volume}, cpu=4,
              memory=16384)
def evaluate():
    import glob
    import json
    import os
    import pickle

    import h5py
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    os.chdir("/vol/code")

    # ---------------------------------------------------------- the graph ---
    PKL = "graphs/affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl"
    with open(PKL, "rb") as f:
        obj = pickle.load(f)
    A = obj["affinity"] if isinstance(obj, dict) and "affinity" in obj else obj
    if isinstance(obj, dict) and not sp.issparse(A):
        A = next(v for v in obj.values() if sp.issparse(v))
    A = sp.csr_matrix(A).astype(np.float32)
    A = ((A + A.T) / 2.0).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    N = A.shape[0]

    def read_obs_cat(h5, key):
        with h5py.File(h5, "r") as f:
            g = f[f"obs/{key}"]
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in g["categories"][:]])
            return cats[g["codes"][:]]

    H5 = "/vol/data/pancreas_hvg.h5ad"
    tech = read_obs_cat(H5, "tech")
    ct = read_obs_cat(H5, "celltype")
    _, bcode = np.unique(tech.astype(str), return_inverse=True)
    bcode = bcode.astype(np.int64)
    n_bio = int(bcode.max()) + 1

    coo = A.tocoo()
    same = bcode[coo.row] == bcode[coo.col]
    A_mask = sp.csr_matrix((coo.data[same], (coo.row[same], coo.col[same])),
                           shape=A.shape)

    def gram_rows(M, delta=1.0):
        cM = (M + delta * sp.eye(M.shape[0], format="csr",
                                 dtype=np.float32)).tocsr()
        n = np.sqrt(cM.multiply(cM).sum(axis=1)).A.ravel()
        n[n == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / n) @ cM)

    R_mask = gram_rows(A_mask, 1.0)
    R_full = gram_rows(A, 1.0)

    # --------------------------------------------------------- the metrics ---
    def cohesion(lab, sim, per_batch):
        """The objective as trained: cohesion_self=0, cohesion_diag=1."""
        _, codes = np.unique(lab, return_inverse=True)
        K = codes.max() + 1
        S = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), codes)),
                          shape=(N, K))
        Sd = S.toarray()
        M = A_mask if per_batch else A
        if sim == "aff":
            numer = np.asarray((M @ S).todense()) + Sd
        else:
            R = R_mask if per_batch else R_full
            numer = np.asarray((R @ (R.T @ S)).todense())
        numer = numer - Sd                       # cohesion_self = 0
        if per_batch:
            szb = np.zeros((n_bio, K))
            np.add.at(szb, bcode, Sd)
            den = szb[bcode]
        else:
            den = np.tile(Sd.sum(0), (N, 1))
        return float((Sd * (numer / np.maximum(den, 1.0))).sum() / N)

    def purity_entropy(lab):
        df = pd.DataFrame({"mc": lab, "ct": ct, "tech": tech})
        n_tech_all = df.tech.nunique()
        rows = []
        for mc, g in df.groupby("mc"):
            vc = g.ct.value_counts()
            p = vc.iloc[0] / len(g)
            tvc = g.tech.value_counts(normalize=True)
            # EXACTLY mc_metric_utils.calc_batch_entropy: UNNORMALISED Shannon
            # entropy in nats. Do not divide by log(n_techs) -- the paper's
            # published SEACells number (0.1792) is on this scale, and
            # normalising silently makes every number incomparable to it.
            # Verified: this reproduces the stored metrics.json to 4 decimals.
            ent = float(-(tvc * np.log(tvc + 1e-10)).sum())
            rows.append((mc, len(g), p, ent))
        r = pd.DataFrame(rows, columns=["mc", "size", "purity", "entropy"])
        w = r["size"] / r["size"].sum()
        maj = df.groupby("mc").ct.agg(lambda s: s.value_counts().index[0])
        return dict(
            ct_purity_mean=float(r.purity.mean()),
            ct_purity_std=float(r.purity.std()),
            ct_purity_wmean=float((r.purity * w).sum()),
            batch_entropy_mean=float(r.entropy.mean()),
            batch_entropy_std=float(r.entropy.std()),
            batch_entropy_wmean=float((r.entropy * w).sum()),
            coverage=float(maj.nunique() / df.ct.nunique()),
            n_mc=int(r.shape[0]),
            median_size=float(r["size"].median()),
        )

    # rare cell types: the repo's own rule (paper_figures._get_rare_cts,
    # rare_mode='median_batch') -- median per-batch frequency below its Q25.
    bf = (pd.DataFrame({"ct": ct, "tech": tech})
          .groupby(["tech", "ct"]).size()
          .groupby(level="tech", group_keys=False).apply(lambda s: s / s.sum())
          .unstack(level="tech", fill_value=0.0))
    med = bf.median(axis=1)
    rare_cts = set(med[med < med.quantile(0.25)].index)

    def rare_metrics(lab):
        """Per BATCH, over the rare cell types only, then averaged over batches.

        homogeneity  for each metacell holding any rare cells in this batch, the
                     fraction of its rare cells that are the same rare type;
                     size-weighted over those metacells.
        f1_macro     treat "metacell's majority rare type" as a prediction for
                     every rare cell; macro-F1 over the rare types present.
        coverage     fraction of the rare types present in this batch that are
                     the majority rare type of at least one metacell.
        """
        from sklearn.metrics import f1_score
        out = {k: [] for k in ("rare_homogeneity", "rare_f1_macro",
                               "rare_coverage")}
        df = pd.DataFrame({"mc": lab, "ct": ct, "tech": tech})
        for tname, g in df.groupby("tech"):
            g = g[g.ct.isin(rare_cts)]
            if g.empty:
                continue
            maj = g.groupby("mc").ct.agg(lambda s: s.value_counts().index[0])
            hom = g.groupby("mc").ct.agg(
                lambda s: s.value_counts().iloc[0] / len(s))
            sizes = g.groupby("mc").size()
            out["rare_homogeneity"].append(
                float((hom * sizes).sum() / sizes.sum()))
            pred = g.mc.map(maj)
            out["rare_f1_macro"].append(
                float(f1_score(g.ct, pred, average="macro", zero_division=0)))
            out["rare_coverage"].append(
                float(len(set(maj.values)) / g.ct.nunique()))
        return {f"{k}_mean": float(np.mean(v)) for k, v in out.items()} | \
               {f"{k}_std": float(np.std(v)) for k, v in out.items()}

    def score(lab, name, n_proto=None):
        lab = np.asarray(lab)
        m = {"method": name}
        m.update(purity_entropy(lab))
        m.update(rare_metrics(lab))
        for sim in ("aff", "gram"):
            m[f"coh_{sim}_perbatch"] = cohesion(lab, sim, True)
            m[f"coh_{sim}_global"] = cohesion(lab, sim, False)
        m["n_dead"] = (int(n_proto - len(np.unique(lab)))
                       if n_proto is not None else None)
        return m

    # ------------------------------------------------------ load the arms ---
    results = []
    for arm in ARMS:
        hits = sorted(glob.glob(f"/vol/models/*/*{arm}*/clusters.npz")) or \
               sorted(glob.glob(f"/vol/models/**/*{arm}*/clusters.npz",
                                recursive=True))
        if not hits:
            print(f"[{arm}] NO clusters.npz found -- skipped")
            continue
        z = np.load(hits[0], allow_pickle=True)
        key = next((k for k in ("assignments", "labels", "clusters", "arr_0")
                    if k in z), None)
        if key is None:
            print(f"[{arm}] clusters.npz has no usable key: {list(z.keys())}")
            continue
        lab = np.asarray(z[key]).ravel()
        if lab.shape[0] != N:
            print(f"[{arm}] SKIPPED: {lab.shape[0]} labels vs {N} cells")
            continue
        print(f"[{arm}] loaded {hits[0]}  ({len(np.unique(lab))} live protos)")
        results.append(score(lab, arm, n_proto=220))

    # --------------------------------------------------------- SEACells ------
    sc_csv = "/vol/models/pancreas/seacell/umap_cells.csv"
    if os.path.exists(sc_csv):
        d = pd.read_csv(sc_csv)
        if len(d) == N:
            print(f"[SEACells] loaded {sc_csv}")
            results.append(score(d["metacell_id"].to_numpy(), "SEACells"))
        else:
            print(f"[SEACells] SKIPPED: {len(d)} rows vs {N} cells")
    else:
        print(f"[SEACells] {sc_csv} not on the volume -- upload it to compare")

    df = pd.DataFrame(results).set_index("method")
    os.makedirs("/vol/code/files/tables/d41_pancreas", exist_ok=True)
    df.to_csv("/vol/code/files/tables/d41_pancreas/d41_pancreas_scoreboard.csv")
    volume.commit()
    return df.to_dict()


@app.local_entrypoint()
def main():
    import pandas as pd
    res = evaluate.remote()
    df = pd.DataFrame(res)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 104)
    print("[D41] PANCREAS SCOREBOARD -- cohesion-only arms vs SEACells, same graph")
    print("=" * 104)

    groups = [
        ("1. preserves within-batch communities (the objective itself)",
         ["coh_aff_perbatch", "coh_gram_perbatch",
          "coh_aff_global", "coh_gram_global"]),
        ("2. combines batches",
         ["batch_entropy_mean", "batch_entropy_std", "batch_entropy_wmean"]),
        ("3. bio proof",
         ["ct_purity_mean", "ct_purity_std", "ct_purity_wmean", "coverage"]),
        ("4. rare cell types -- where the method should earn its keep",
         ["rare_homogeneity_mean", "rare_homogeneity_std",
          "rare_f1_macro_mean", "rare_f1_macro_std",
          "rare_coverage_mean", "rare_coverage_std"]),
        ("housekeeping", ["n_dead", "n_mc", "median_size"]),
    ]
    for title, keys in groups:
        print(f"\n--- {title}")
        sub = df.reindex(keys).dropna(how="all")
        print(sub.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n--- reference points on this graph (K=220, per-batch, self=0)")
    for k, v in REFERENCE.items():
        print(f"    {k:<12} aff {v['coh_aff_pb']:.4f}   gram {v['coh_gram_pb']:.4f}")
    print(f"    SEACells published: ct_purity {SEACELLS_REF['ct_purity']:.4f}  "
          f"batch_entropy {SEACELLS_REF['batch_entropy']:.4f}  "
          f"coverage {SEACELLS_REF['coverage']:.3f}")
    print("\nsaved: files/tables/d41_pancreas/d41_pancreas_scoreboard.csv")
