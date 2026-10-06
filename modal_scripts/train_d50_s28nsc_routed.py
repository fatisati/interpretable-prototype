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
     validated operating point) -- CORRECTION [D55]: this was never actually
     true. decB's real ratio-to-cohesion (measured from its own grad_probe.jsonl,
     matching [D46]'s convention) is 7.60x, not ~40% share. "rtA" below used
     lumap=0.02, which is itself ~9.1x ratio-to-cohesion, not the intended 40%
     either -- so this phase never ran at a properly calibrated dose. See [D55].

EXPECTED (phase 2): n_dead=0; coh_aff/dge_r >= the [D39] E-arm's
(0.4576 joint_purity / 0.8105 dge_r / 0 dead); any clear regression on those =
the routed term does NOT transfer to spatial, and cohesion-only stays the
s28nsc config.
VERDICT: [x] FAILS -- confirmed at TWO doses. rtA (lumap=0.02, ~9.1x ratio,
uncalibrated) collapsed dge_rbo_avg to 0.137 vs the 0.8105 baseline. The one
PROPERLY calibrated dose ([D46], share=0.25x ratio, a different, more careful
run) still lost on every scoreboard metric (ct_purity -8.5%, dge_r -4.5%) even
without collapsing. Community does not transfer to s28nsc at any dose tested,
calibrated or not. Cohesion-only ([D39]-E) stays s28nsc's config -- confirmed,
not reopened, by the 2026-09-26 `d50rt_cvA` run (cohesion + CVAE-recon, no
community): dge_r=0.796 vs baseline 0.8105, n_dead=3 vs 0 -- recon alone is a
mild regression too, not a fix. Do not re-run this arm.

Deploy + spawn (server-side, survives disconnects):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy modal_scripts/train_d50_s28nsc_routed.py
"""
import time

import modal

app = modal.App("scproto-d50")

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
LAM_GRAM, LAM_AFF = 10.0, 22.3
SPAN_GRAM, SPAN_AFF = 0.076776, 0.034573
AUTHORITY = 1.5403


def build_config(lam_umap=0.0, grad_probe=0, umap_proto_sg=0, lam_recon=0.0,
                 epoch_dump_assign=0):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "epoch_dump_assign": epoch_dump_assign,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": lam_recon,
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
        "latent_dims": 16,                  # [D39] comparability, not the 9d default
    }


def validate(lc):
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_proto_recon"):
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    auth = LAM_GRAM * SPAN_GRAM + LAM_AFF * SPAN_AFF
    assert abs(auth - AUTHORITY) < 0.05, f"authority drifted: {auth}"
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    return {"config_tag": "d50", "graph": "bk02", "K": 200, "latent_dims": 16,
            "authority": round(auth, 4),
            "lambda_cohesion(gram)": lc["lambda_cohesion"],
            "lambda_cohesion_aff": lc["lambda_cohesion_aff"],
            "lambda_umap": lc["lambda_umap"],
            "umap_proto_sg": lc["umap_proto_sg"],
            "lambda_recon": lc["lambda_recon"],
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
def probe(lam_umap: float = 0.0, grad_probe: int = 0, umap_proto_sg: int = 0,
          lam_recon: float = 0.0, epoch_dump_assign: int = 0):
    _env()
    return validate(build_config(lam_umap, grad_probe, umap_proto_sg, lam_recon,
                                 epoch_dump_assign))


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run(lam_umap: float = 0.0, epochs: int = 10, suffix: str = "",
        grad_probe: int = 0, umap_proto_sg: int = 0, lam_recon: float = 0.0,
        freeze_be: bool = False, epoch_dump_assign: int = 0):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    name = f"d50rt{suffix}"
    lc = build_config(lam_umap, grad_probe, umap_proto_sg, lam_recon,
                      epoch_dump_assign)
    checked = validate(lc)
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}  freeze_be={freeze_be}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=epochs, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=freeze_be,
    )
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics
