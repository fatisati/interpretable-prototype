"""[D64] NB smoke test: isolate WHICH step of the NB path fails and print the
FULL traceback (the probe's output was truncated to one line).

cvae_epochs=1 so it is fast and cheap.
"""
import modal

app = modal.App("scproto-nb-smoke")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy", "scipy", "pandas", "anndata", "scanpy", "torch",
                 "scarches", "wandb", "tqdm", "statsmodels", "diptest",
                 "palantir", "faiss-cpu", "scikit-learn", "matplotlib",
                 "scib-metrics", "leidenalg", "python-igraph", "ipython")
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{n} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{n}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{n}/batch2cat.json"
        for n in ["lung", "lung_fetal_donor", "lung_fetal_organoid", "brain",
                  "breast", "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas"]])
)
volume = modal.Volume.from_name("scproto-data")
DS, BATCH_KEY, LABEL_KEY, K = "pbmc-immune", "study", "final_annotation", 300
TAG = "arbf_ctb20_seed0"


def _env():
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


@app.function(image=image, gpu="L4", timeout=5400, volumes={"/vol": volume})
def smoke():
    import os, traceback
    import numpy as np
    _env()
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes, LAMBDA_PROTO_UMAP_PRECON)

    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "recon_loss": "nb", "lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
        "lambda_umap": 0.0, "lambda_recon": 1.0, "lambda_gmkl": 1.0,
        "lambda_proto_recon": 1.0, "proto_recon_form": "mix3",
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "proto_recon_stopgrad": True, "proto_recon_dec_sg": 0,
    }
    t = get_trainer(
        experiment_name="nbsmoke", cvae_epochs=1, affinity_type="arbf",
        dataset_id=resolved, l2norm=1, assignment_metric="dotp",
        batch_size=1024, umap_steps_per_epoch=None, umap_similarity="proto",
        freeze_batch_embedding=1, freeze_decoder=0,
        num_prototypes=_resolve_num_prototypes(K, resolved),
        **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")},
    )
    print("STEP 0 ok: trainer built")
    print("   recon_loss      :", getattr(t, "recon_loss", None))
    print("   use_counts      :", getattr(t.train_ds, "use_counts", None))
    X = t.train_ds.adata.X
    xs = X[:200].toarray() if hasattr(X, "toarray") else X[:200]
    print(f"   X[:200] max     : {float(np.max(xs)):.1f}  integer={bool(np.allclose(xs, np.round(xs)))}")
    print("   n_cells, n_genes:", t.train_ds.adata.shape)
    print("   model name      :", t.get_model_name())

    for step, fn in (("pretrain_encoder", lambda: t.pretrain_encoder()),
                     ("init_prototypes", lambda: t.init_prototypes()),
                     ("setup_umap_edges", lambda: t._setup_umap_edges(10)),
                     ("one_umap_epoch", lambda: t.continue_train_umap_edges(
                         epochs=1, early_stop=False))):
        try:
            fn()
            print(f"STEP ok: {step}")
        except Exception:
            print(f"STEP FAILED: {step}\n" + "=" * 70)
            traceback.print_exc()
            print("=" * 70)
            return {"failed_at": step, "tb": traceback.format_exc()}
    return {"failed_at": None}


@app.local_entrypoint()
def main():
    r = smoke.remote()
    print("\n\n######## RESULT ########")
    print("failed at:", r["failed_at"])
    if r.get("tb"):
        print(r["tb"][-4000:])
