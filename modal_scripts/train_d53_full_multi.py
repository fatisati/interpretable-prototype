"""[D53] Full-graph runs of the FIXED loss for lung / pbmc-immune / crcx.

THE FIXED LOSS (user decision 2026-09-25): joint cohesion (gram+aff, spans per
graph, 50/50 authority) + community with DETACHED prototypes (umap_proto_sg=1,
encoder-only gradient), lambda_umap tuned per (dataset, graph) by the
gradient-share strategy (~40% of the encoder gradient, within-run ratios).

Program: transcriptional datasets get NH + FULL runs (NH via scproto-d46; FULL
via this app). Multi-batch spatial (crcx) gets FULL only. s28nsc keeps its
cohesion-only [D39]-E result (the routed arm failed dge_r on the scoreboard).

Spans measured 2026-09-25 with the current V[b] cohesion on each FULL graph:
  lung         span_aff 0.029801  span_gram 0.138232  -> 25.84 / 5.57
  pbmc-immune  span_aff 0.026743  span_gram 0.101852  -> 28.80 / 7.56
  crcx         (filled after its measurement completes)

Deploy + spawn:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy modal_scripts/train_d53_full_multi.py
"""
import time

import modal

app = modal.App("scproto-d53")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
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

AUTHORITY = 1.5403

# ds -> (K, span_aff, span_gram) on the FULL arbf graph, current V[b] cohesion
DATASETS = {
    "lung":        (300, 0.029801, 0.138232),
    "pbmc-immune": (300, 0.026743, 0.101852),
    "crcx":        (670, None, None),   # filled by --span-aff/--span-gram args
}


def build_config(lam_gram, lam_aff, lam_umap=0.0, grad_probe=0,
                 umap_proto_sg=0):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": 0.0,
        "lambda_proto_recon": 0.0,
        "lambda_umap": lam_umap,
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 500.0,
        "proto_reinit_every": 0,
        "grad_probe": grad_probe,
        "umap_proto_sg": umap_proto_sg,
        "latent_dims": 8,
    }


def validate(lc, lam_gram, lam_aff, span_aff, span_gram):
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_recon", "lambda_proto_recon"):
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    auth = lam_gram * span_gram + lam_aff * span_aff
    assert abs(auth - AUTHORITY) < 0.05, f"authority drifted: {auth}"
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["latent_dims"] == 8 and lc["proto_reinit_every"] == 0
    return {"config_tag": "d53full", "authority": round(auth, 4),
            "lambda_cohesion(gram)": lc["lambda_cohesion"],
            "lambda_cohesion_aff": lc["lambda_cohesion_aff"],
            "lambda_umap": lc["lambda_umap"],
            "umap_proto_sg": lc["umap_proto_sg"],
            "grad_probe": lc["grad_probe"]}


def _env():
    import os
    import sys
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
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


@app.function(image=image, timeout=900, volumes={"/vol": volume})
def probe(ds: str, lam_gram: float, lam_aff: float,
          span_aff: float, span_gram: float, lam_umap: float = 0.0,
          grad_probe: int = 0, umap_proto_sg: int = 0):
    _env()
    r = validate(build_config(lam_gram, lam_aff, lam_umap, grad_probe,
                              umap_proto_sg), lam_gram, lam_aff,
                 span_aff, span_gram)
    r["ds"] = ds
    r["K"] = DATASETS[ds][0]
    return r


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run(ds: str, lam_gram: float, lam_aff: float,
        span_aff: float, span_gram: float, lam_umap: float = 0.0,
        epochs: int = 10, suffix: str = "", grad_probe: int = 0,
        umap_proto_sg: int = 0, aff_type: str = "arbf"):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    K = DATASETS[ds][0]
    name = f"d53full{suffix}_{ds}"
    lc = build_config(lam_gram, lam_aff, lam_umap, grad_probe, umap_proto_sg)
    checked = validate(lc, lam_gram, lam_aff, span_aff, span_gram)
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}  ds={ds} K={K}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=epochs, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=aff_type, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics
