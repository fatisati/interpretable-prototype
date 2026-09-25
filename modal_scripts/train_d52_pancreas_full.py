"""[D50] s28nsc: joint cohesion + ROUTED community (encoder-only), lambda measured.

THE DECISION, WRITTEN BEFORE RUNNING
====================================
[D39] found "community kills prototypes" on s28nsc/bk02 -- but at lambda_umap=1.0
UNROUTED, the dose that also killed prototypes on pancreas (89% of the prototype
gradient, measured). With [D48] routing (umap sees DETACHED prototypes; its
gradient reaches the encoder only) and a MEASURED lambda, pancreas showed the
routed term improves purity, held-out cohesion, modularity and generalization
with 0 dead (decB). This asks whether the routed recipe transfers to spatial --
which would reframe [D39] as a dosage artifact, not a dataset split.

Config = the [D39] E-arm exactly (joint cohesion 10.0 gram / 22.3 aff, bk02,
K=200, LD=16 for comparability with the frozen [D39] scoreboard), plus
lambda_umap routed. Two phases:
  1. measure: 1 epoch, lambda_umap=1.0, grad_probe=25 -> in-config gradient
     shares (norms are NOT portable across graphs or code paths -- measured
     twice already: 17.71 vs 1.65 for the same term).
  2. real run: 10 epochs at lambda_umap set for ~40% encoder share (decB's
     validated operating point).

EXPECTED (phase 2): n_dead=0; coh_aff/dge_r >= the [D39] E-arm's
(0.4576 joint_purity / 0.8105 dge_r / 0 dead); any clear regression on those =
the routed term does NOT transfer to spatial, and cohesion-only stays the
s28nsc config. VERDICT: [ ] transfers  [ ] fails  [ ] inconclusive because ___

Deploy + spawn (server-side, survives disconnects):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy modal_scripts/train_d50_s28nsc_routed.py
"""
import time

import modal

app = modal.App("scproto-d52")

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

# [D39] E-arm constants -- s28nsc/bk02, K=200, LD=16, spans from the d39 header
LAM_GRAM, LAM_AFF = 3.67, 11.14
SPAN_GRAM, SPAN_AFF = 0.209982, 0.069124
AUTHORITY = 1.5403


def build_config(lam_umap=0.0, grad_probe=0, umap_proto_sg=0):
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
        "lambda_cohesion": LAM_GRAM,
        "lambda_cohesion_aff": LAM_AFF,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 500.0,
        "proto_reinit_every": 0,
        "coh_diag_every": 25,
        "grad_probe": grad_probe,
        "umap_proto_sg": umap_proto_sg,     # [D48] encoder-only community
        "latent_dims": 8,                   # 9d default -- full-graph pancreas
    }


def validate(lc):
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_recon", "lambda_proto_recon"):
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    auth = LAM_GRAM * SPAN_GRAM + LAM_AFF * SPAN_AFF
    assert abs(auth - AUTHORITY) < 0.05, f"authority drifted: {auth}"
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    return {"config_tag": "d50", "graph": "arbf-full", "K": 220, "latent_dims": 8,
            "authority": round(auth, 4),
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
def probe(lam_umap: float = 0.0, grad_probe: int = 0, umap_proto_sg: int = 0):
    _env()
    return validate(build_config(lam_umap, grad_probe, umap_proto_sg))


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run(lam_umap: float = 0.0, epochs: int = 10, suffix: str = "",
        grad_probe: int = 0, umap_proto_sg: int = 0):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    name = f"d52fg{suffix}"
    lc = build_config(lam_umap, grad_probe, umap_proto_sg)
    checked = validate(lc)
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "pancreas", cvae_epochs=50, train_epochs=epochs, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=220,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics
