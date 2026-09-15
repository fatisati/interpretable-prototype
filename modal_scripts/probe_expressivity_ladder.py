"""EXPRESSIVITY PROBE -- is our learned z worse than plain PCA at the SAME dimension?

THE QUESTION
------------
scProto's internal_fraction is 0.2926; SEACells' is 0.4465. Is that because
  (a) our LOSS points the wrong way, or
  (b) the FEATURES our assignment head sees cannot support a partition that good?
These need opposite fixes, so guessing is expensive.

THE METHOD
----------
Hand a linear head SEACells' own 200 labels as a SUPERVISED target and let it copy them.
No loss, no competing terms, fit IN-SAMPLE on purpose -- the question is what a feature
space CAN represent, not how it generalizes, so memorizing is allowed and makes the test
as generous as possible. Then measure the internal_fraction of the partition that head
produces. A feature space that cannot support a good partition even when handed the
answer key will never support one under any loss.

THE CONTROL THAT MATTERS: PCA AT THE SAME DIMENSION
---------------------------------------------------
z is a LEARNED 16-d representation. PCA-16 is an untrained, closed-form 16-d
representation of the same cells. The encoder is supposed to make z BETTER than that.

    linear head on PCA-16  <  linear head on z-16   -> encoder is concentrating signal;
                                                       the features are fine, look at the loss
    linear head on PCA-16  >  linear head on z-16   -> THE ENCODER IS DESTROYING INFORMATION
                                                       relative to a trivial baseline at equal
                                                       dimension. No loss tuning fixes that.

The 8/16/32/50 PCA ladder answers "does raw DIMENSION help?" from the same run, and it
also addresses a real objection to the earlier frozen-z probe: that probe held z fixed
from one trained run, while in reality the encoder is trainable and z can reshape itself.
PCA-16 sidesteps that entirely -- it is a fixed, assumption-free reference point at the
same dimension, so "is 16-d enough?" gets answered without depending on any trained z.

NO BIAS ANYWHERE. Our real head is logits[cell,p] = dot(z[cell], prototype[p]) with no
bias term (confirmed at runtime: head bias present = False), so every probe here uses
fit_intercept=False. This matches the architecture exactly and keeps the comparison
honest. Note the consequence: without a bias the argmax is unchanged by scaling z, so the
assignment depends only on the DIRECTION of a feature vector from the origin. The origin
therefore matters, and features are used AS-IS -- not re-centred -- because re-centring
would move the origin and change what a no-bias head can express.

CAVEAT, stated up front: each probe imitates SEACells' specific LABEL IDENTITIES, not
internal_fraction directly. Some other partition in the same family could score higher
without imitating SEACells. Every number is a LOWER BOUND on what that family can
express -- strong evidence, not proof.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/probe_expressivity_ladder.py
Cost: no training at all. A few minutes on a T4 (GPU only to load the checkpoint for z).
"""
import time
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

REF_RUN = dict(name="lf_L16_g015_uOFF", gamma=0.15, usage=0.0)
PCA_DIMS = [8, 16, 32, 50]      # 16 is the matched-dimension control against our z
C_WEAK = 100.0                  # weak regularization: this is a CAPACITY test, not a
                                # generalization test, so we want near-unpenalized MLE


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def probe_ladder():
    import os
    os.environ["WANDB_MODE"] = "disabled"
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

    import numpy as np, pandas as pd, scipy.sparse as sp, pickle
    import scanpy as sc
    from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.evaluation.collapse_diagnostics import labels_from_trainer

    adata = sc.read_h5ad(DATASETS["s28nsc"]["path"])
    print(f"[data] {adata.shape}   obsm keys: {list(adata.obsm.keys())}")

    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo()
    _m = float(_Atri.sum()); _row, _col, _dat = _Atri.row, _Atri.col, _Atri.data
    print(f"[graph] m = {_m:,.0f}")

    def partition_stats(li):
        li = np.asarray(li, dtype=np.int64)
        same = (li[_row] == li[_col]) & (li[_row] >= 0)
        e = float(_dat[same].sum())
        sizes = np.bincount(li[li >= 0], minlength=max(int(li.max()) + 1, 1)).astype(float)
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        return {"internal_fraction": e / _m,
                "aggregate_density": e / pairs if pairs else float("nan"),
                "n_alive": int(len(alive)),
                "median_size": float(np.median(alive)),
                "max_over_median": float(alive.max() / np.median(alive))}

    def to_ints(series):
        lab = pd.Series(series).reindex(adata.obs_names).to_numpy()
        ok = ~pd.isna(lab)
        li = np.full(len(lab), -1, dtype=np.int64)
        li[ok] = pd.Categorical(pd.Series(lab[ok]).astype(str)).codes
        return li

    # ---------------------------------------------------------------- SEACells target
    mdir = "/vol/models/s28nsc"
    cands = sorted(d for d in os.listdir(mdir)
                   if d.startswith("seacell") and
                   os.path.exists(os.path.join(mdir, d, "seacell_sc.h5ad")))
    pick = next((d for d in cands if "bklam02" in d or "bk02" in d), cands[0])
    print(f"[seacells] dirs={cands}  using {pick}")
    from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import _reconstruct_from_delta
    sad = sc.read_h5ad(os.path.join(mdir, pick, "seacell_sc.h5ad"))
    if sad.uns.get("_seacell_delta"):
        sad = _reconstruct_from_delta(sad, os.path.join(mdir, pick))
    sea_li = to_ints(pd.Series(sad.obs["SEACell"].astype(str).values, index=sad.obs_names))
    sea_stats = partition_stats(sea_li)
    print(f"[seacells] {sea_stats}")

    # ------------------------------------------------------- our z + our trained labels
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
        'nassoc_size_scale': 'full', 'nassoc_cpm_norm': 'gful',
        'nassoc_gamma': REF_RUN["gamma"], 'lambda_nassoc': 1000, 'nassoc_exact': 1,
        'lambda_proto_usage': REF_RUN["usage"], 'latent_dims': 16,
    }
    trainer, *_ = find_metacells(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        label_key="celltypes", niche_key="niches_2D",
        lambda_config=lc, load_pretrain=True, load_umap=True, skip_eval=True,
        trainer_kwargs={"experiment_name": REF_RUN["name"]})
    import torch
    trainer.model.eval()
    with torch.no_grad():
        Z = trainer.encode_adata(adata, trainer.model, z_idx=1).cpu().numpy()
    trained_stats = partition_stats(to_ints(labels_from_trainer(trainer, adata).dropna()))
    print(f"[ref] z={Z.shape}   our head bias present: "
          f"{getattr(trainer.model.prototypes, 'bias', None) is not None}")
    print(f"[ref] trained partition {trained_stats}")

    # ------------------------------------------------------------------------- features
    # PCA is computed ONCE at max dim and truncated, so PCA-8 is exactly the first 8
    # components of PCA-50 -- a clean nested ladder.
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= max(PCA_DIMS):
        PCA_FULL = np.asarray(adata.obsm["X_pca"])[:, :max(PCA_DIMS)]
        print(f"[pca] reusing adata.obsm['X_pca'] -> {PCA_FULL.shape}")
    else:
        print(f"[pca] computing {max(PCA_DIMS)}-comp PCA")
        tmp = adata.copy(); sc.pp.pca(tmp, n_comps=max(PCA_DIMS))
        PCA_FULL = np.asarray(tmp.obsm["X_pca"]); del tmp

    feats = {"z16 (OUR ENCODER)": Z}
    for d in PCA_DIMS:
        feats[f"pca{d}"] = PCA_FULL[:, :d]
    # BANKSY: the embedding SEACells itself partitioned (seacell_X_bklam02_K200). Included
    # because it is the one feature space KNOWN to contain spatial context, which scProto's
    # encoder never receives. Free to test; drop this line if it is not wanted.
    for k in adata.obsm.keys():
        if "bklam02" in k or k == "X_bk02":
            feats[f"BANKSY {k}"] = np.asarray(adata.obsm[k])
            print(f"[banksy] found {k} -> {adata.obsm[k].shape}")
            break
    else:
        print("[banksy] no bklam02 obsm key on this adata -- skipping that row")

    # ------------------------------------------------------------------------- probes
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    ok = sea_li >= 0
    y = sea_li[ok]
    out = {}

    print("\n" + "=" * 84)
    print("PROBE: linear head, NO BIAS (matches our real head), fit to SEACells' 200 labels")
    print("in-sample. Features used AS-IS -- not re-centred, because a no-bias head is")
    print("origin-sensitive and re-centring would change what it can express.")
    print("=" * 84)

    for tag, X in feats.items():
        t0 = time.time()
        try:
            # scale to comparable magnitude WITHOUT moving the origin
            Xs = X / (float(np.abs(X).mean()) + 1e-12)
            clf = LogisticRegression(fit_intercept=False, C=C_WEAK, max_iter=400, n_jobs=-1)
            clf.fit(Xs[ok], y)
            pred = clf.predict(Xs[ok])
            li = np.full(len(sea_li), -1, dtype=np.int64); li[ok] = pred
            st = partition_stats(li)
            st["accuracy"] = float((pred == y).mean())
            st["ari"] = float(adjusted_rand_score(y, pred))
            st["dim"] = int(X.shape[1])
            st["seconds"] = time.time() - t0
            out[tag] = st
            print(f"[{tag:<20}] dim={st['dim']:<3} int_frac={st['internal_fraction']:.4f} "
                  f"acc={st['accuracy']:.4f} ari={st['ari']:.4f} "
                  f"density={st['aggregate_density']:.5f} alive={st['n_alive']:<4} "
                  f"({st['seconds']:.0f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            out[tag] = {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------------- summary
    lo, hi = trained_stats["internal_fraction"], sea_stats["internal_fraction"]
    print("\n\n===== HOW MUCH OF THE internal_fraction GAP EACH FEATURE SPACE REACHES =====")
    print(f"floor  = what our trained loss achieved : {lo:.4f}")
    print(f"target = SEACells                       : {hi:.4f}")
    print(f"gap                                     : {hi - lo:.4f}\n")
    print(f"{'features':<22}{'dim':>5}{'int_frac':>10}{'gap closed':>12}{'acc':>8}{'ari':>8}"
          f"{'density':>10}{'max/med':>9}")
    for tag, st in out.items():
        if "error" in st:
            print(f"{tag:<22}   FAILED"); continue
        gc = (st["internal_fraction"] - lo) / (hi - lo) * 100
        print(f"{tag:<22}{st['dim']:>5}{st['internal_fraction']:>10.4f}{gc:>11.0f}%"
              f"{st['accuracy']:>8.3f}{st['ari']:>8.3f}{st['aggregate_density']:>10.5f}"
              f"{st['max_over_median']:>9.1f}")
    print(f"{'SEACells (target)':<22}{'--':>5}{hi:>10.4f}{100:>11.0f}%{'--':>8}{'--':>8}"
          f"{sea_stats['aggregate_density']:>10.5f}{sea_stats['max_over_median']:>9.1f}")
    print(f"{'our trained run':<22}{16:>5}{lo:>10.4f}{0:>11.0f}%{'--':>8}{'--':>8}"
          f"{trained_stats['aggregate_density']:>10.5f}{trained_stats['max_over_median']:>9.1f}")

    zr = out.get("z16 (OUR ENCODER)", {})
    pr = out.get("pca16", {})
    if "internal_fraction" in zr and "internal_fraction" in pr:
        print("\n>>> THE MATCHED-DIMENSION VERDICT (pca16 vs our z16, both 16-d, both no bias):")
        print(f">>>   our learned z16 : {zr['internal_fraction']:.4f}")
        print(f">>>   plain pca16     : {pr['internal_fraction']:.4f}")
        if pr["internal_fraction"] > zr["internal_fraction"]:
            print(">>>   PCA WINS -> the encoder is DESTROYING information relative to a")
            print(">>>   trivial baseline at equal dimension. Not a loss-tuning problem.")
        else:
            print(">>>   z WINS -> the encoder is concentrating signal as intended; 16-d")
            print(">>>   features are not the bottleneck, so look at the loss.")
    print("\nREMINDER: each probe imitates SEACells' LABEL IDENTITIES, so every number is a")
    print("LOWER BOUND on what that feature space can express -- evidence, not proof.")
    return {"seacells": sea_stats, "trained": trained_stats, "probes": out}


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = probe_ladder.remote()
    print("wall (s):", time.time() - t0)
    import json; print(json.dumps(r, indent=2, default=str))
