"""EXPRESSIVITY LADDER -- what can our assignment head represent AT ALL?

THE QUESTION
------------
scProto's internal_fraction is 0.2926; SEACells' is 0.4465. Is that because
  (a) our LOSS points the wrong way, or
  (b) our MODEL physically cannot produce a partition that good?
These need completely different fixes, so they must be separated before more tuning.

THE METHOD
----------
Hand our assignment head SEACells' own 200 labels as a SUPERVISED target and let it copy
them. Delete the loss, delete the competing terms, fit in-sample. If the head can copy a
good partition, the architecture is fine and the loss is at fault. If it cannot copy it
even when handed the answer key, no loss will ever get there.

Already measured (diag_and_train_correct_gamma.py):
    SEACells (target)                 int_frac 0.4465
    best linear softmax on our z      int_frac 0.3156   <- closes only 15% of the gap
    what our trained loss achieved    int_frac 0.2926

THIS SCRIPT decomposes that ceiling along three axes, to say WHICH restriction binds:

  1. BIAS.  Our head is  logits[cell,p] = dot(z[cell], prototype[p])  with NO bias.
     Without a bias, scaling z by any c>0 leaves the argmax unchanged:
         argmax_p dot(c*z, proto_p) = argmax_p c*dot(z, proto_p) = argmax_p dot(z, proto_p)
     so the assignment depends ONLY ON THE DIRECTION of z, never its magnitude, and each
     metacell is a CONE radiating from the origin. Adding a per-prototype bias b[p] turns
     the decision boundaries from hyperplanes THROUGH THE ORIGIN into arbitrary offset
     hyperplanes -- i.e. general polytopes, magnitude now matters. fit_intercept=True
     measures exactly what that would buy, without writing any model code.

  2. DIMENSION. 16-d z vs 50-d X_pca. 8->16 latent dims was already worth +0.025 int_frac
     in real training, so dimensionality is a live suspect.

  3. LINEARITY. A one-hidden-layer MLP on z -- the boundaries need not be flat at all.
     Upper bound on what any re-parameterized head could reach on this same z.

HOW TO READ THE RESULT
----------------------
  a bias alone recovers most of the gap  -> add a bias to the prototype layer (small change)
  50-d PCA recovers it but 16-d z cannot -> raise latent_dims; the encoder is compressing
                                            away what the partition needs
  only the MLP recovers it               -> the LINEAR head is the binding constraint;
                                            the head itself must change
  nothing recovers it                    -> SEACells' partition is not a function of these
                                            features at all; the graph must enter the
                                            ASSIGNMENT, not just the loss

CAVEAT, stated up front: each probe fits SEACells' specific LABEL IDENTITIES, not
internal_fraction directly. Some other partition in the same family could score higher
without imitating SEACells. So every number here is a LOWER BOUND on what that family can
express, not a proven ceiling. It is strong evidence, not proof.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/probe_expressivity_ladder.py
Cost: no training. A few minutes on a T4 (GPU only to load the checkpoint reliably), cents.
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

    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo()
    _m = float(_Atri.sum()); _row, _col, _dat = _Atri.row, _Atri.col, _Atri.data

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
    print(f"[seacells] using {pick}")
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
    print(f"[ref] z={Z.shape}  trained partition {trained_stats}")

    # ---------------------------------------------------------------------- X_pca (50-d)
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= 50:
        P = np.asarray(adata.obsm["X_pca"])[:, :50]
        print(f"[pca] reusing adata.obsm['X_pca'] -> {P.shape}")
    else:
        print("[pca] computing 50-comp PCA (same n_comps the affinity was built with)")
        tmp = adata.copy(); sc.pp.pca(tmp, n_comps=50)
        P = np.asarray(tmp.obsm["X_pca"]); del tmp
    # scale-match PCA to z so the linear fits are conditioned comparably
    P = P / (P.std(0).mean() + 1e-12) * (Z.std(0).mean() + 1e-12)

    # -------------------------------------------------------------------------- probes
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    ok = sea_li >= 0
    y = sea_li[ok]
    out = {}

    def run_probe(tag, X, make_clf, note):
        t0 = time.time()
        try:
            clf = make_clf()
            clf.fit(X[ok], y)
            pred = clf.predict(X[ok])
            li = np.full(len(sea_li), -1, dtype=np.int64); li[ok] = pred
            st = partition_stats(li)
            st["accuracy"] = float((pred == y).mean())
            st["ari"] = float(adjusted_rand_score(y, pred))
            st["seconds"] = time.time() - t0
            st["note"] = note
            out[tag] = st
            print(f"[{tag}] {note}\n        int_frac={st['internal_fraction']:.4f} "
                  f"acc={st['accuracy']:.4f} ari={st['ari']:.4f} "
                  f"density={st['aggregate_density']:.5f} alive={st['n_alive']} "
                  f"({st['seconds']:.0f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            out[tag] = {"error": f"{type(e).__name__}: {e}", "note": note}

    print("\n" + "=" * 80)
    print("EXPRESSIVITY LADDER -- each row is the BEST that function family can do when")
    print("handed SEACells' labels directly. In-sample on purpose: this measures what the")
    print("family CAN represent, not how it generalizes, so memorizing is allowed.")
    print("=" * 80)

    run_probe("z16_nobias", Z,
              lambda: LogisticRegression(fit_intercept=False, max_iter=300, n_jobs=-1),
              "16-d z, no bias  <- EXACTLY our current head: softmax(dot(z, proto))")
    run_probe("z16_bias", Z,
              lambda: LogisticRegression(fit_intercept=True, max_iter=300, n_jobs=-1),
              "16-d z, WITH bias  <- what adding b[p] to the prototype layer would buy")
    run_probe("pca50_nobias", P,
              lambda: LogisticRegression(fit_intercept=False, max_iter=300, n_jobs=-1),
              "50-d PCA, no bias  <- does more DIMENSION help, holding the head fixed?")
    run_probe("pca50_bias", P,
              lambda: LogisticRegression(fit_intercept=True, max_iter=300, n_jobs=-1),
              "50-d PCA, WITH bias  <- dimension AND bias together")

    try:
        from sklearn.neural_network import MLPClassifier
        run_probe("z16_mlp", Z,
                  lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=60,
                                        early_stopping=False, random_state=0),
                  "16-d z, 1-hidden-layer MLP  <- upper bound for ANY head on this same z")
    except Exception as e:
        print(f"[z16_mlp] skipped: {e}")

    # ------------------------------------------------------------------------- summary
    print("\n\n===== HOW MUCH OF THE internal_fraction GAP EACH FAMILY CAN REACH =====")
    lo, hi = trained_stats["internal_fraction"], sea_stats["internal_fraction"]
    print(f"floor  = what our trained loss achieved : {lo:.4f}")
    print(f"target = SEACells                       : {hi:.4f}")
    print(f"gap                                     : {hi - lo:.4f}\n")
    print(f"{'family':<16}{'int_frac':>10}{'gap closed':>12}{'acc':>8}{'ari':>8}"
          f"{'density':>10}{'max/med':>9}")
    for tag, st in out.items():
        if "error" in st:
            print(f"{tag:<16}   FAILED"); continue
        gc = (st["internal_fraction"] - lo) / (hi - lo) * 100
        print(f"{tag:<16}{st['internal_fraction']:>10.4f}{gc:>11.0f}%{st['accuracy']:>8.3f}"
              f"{st['ari']:>8.3f}{st['aggregate_density']:>10.5f}{st['max_over_median']:>9.1f}")
    print(f"{'SEACells':<16}{hi:>10.4f}{100:>11.0f}%{'--':>8}{'--':>8}"
          f"{sea_stats['aggregate_density']:>10.5f}{sea_stats['max_over_median']:>9.1f}")
    print(f"{'our trained run':<16}{lo:>10.4f}{0:>11.0f}%{'--':>8}{'--':>8}"
          f"{trained_stats['aggregate_density']:>10.5f}{trained_stats['max_over_median']:>9.1f}")
    print("\nREMINDER: each probe imitates SEACells' LABEL IDENTITIES, so every number is a")
    print("LOWER BOUND on what that family can express -- strong evidence, not proof.")
    return {"seacells": sea_stats, "trained": trained_stats, "probes": out}


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = probe_ladder.remote()
    print("wall (s):", time.time() - t0)
    import json; print(json.dumps(r, indent=2, default=str))
