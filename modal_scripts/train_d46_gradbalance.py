"""[D46] cohesion + proto_recon + community, lambdas balanced on MEASURED gradient norms.

WHY THIS RUN
------------
[D39] turned community OFF because at its inherited lambda (lambda_umap=1,
lambda_proto_recon=0.01) it cost 125/200 prototypes and halved dge_r. But that
was ONE lambda setting, never a calibrated one -- the lambdas came from a config
written for a different loss. "Community is harmful" may really be "community at
that weight is harmful".

This run puts all three terms back and sets their weights so each contributes a
comparable PULL on the shared weights, measured rather than assumed:

    influence of term i  =  lambda_i * || d(term_i) / d(params) ||       ([D45])

not the loss VALUE share, which is a poor proxy -- cohesion's printed value stays
large while its encoder gradient decays as assignments saturate.

TWO STAGES
----------
  stage 1 (this file, MEASURE=True): run a few steps with grad_probe=5 and
     PLACEHOLDER lambdas, to read gnorm_enc for every term off
     <dump>/grad_probe.jsonl. Nothing is concluded from stage 1's metrics.
  stage 2 (MEASURE=False, LAMBDAS filled in): the real 10-epoch run at the
     balanced lambdas.

BALANCING RULE, stated before seeing any number: take cohesion as the anchor
(it is the term the model is built around, [D39]) and set

    lambda_other = SHARE * (lambda_coh * gnorm_coh) / gnorm_other

with SHARE = 0.25, i.e. community and proto_recon each pull at a QUARTER of
cohesion's strength. Rationale: [D39] measured community at full strength
destroying the partition, so the question this run asks is whether a clearly
subordinate community term helps or still hurts -- not whether an equal one does.

WATCH, IN THIS ORDER (the [D39] failure signature):
  1. unused_proto per epoch -- community killed 125/200 by epoch 10 last time,
     first deaths at epoch 2. If deaths appear again at a quarter strength, the
     term is harmful at any weight that matters and we stop.
  2. dge_r / joint_purity at the end -- community halved dge_r (0.8105 -> 0.3603).
  3. cohesion_aff trend -- it FELL under community before (0.0078 -> 0.0051)
     where cohesion-only rose (0.0077 -> 0.0207).

REFERENCE, same graph/K ([D39], s28nsc/bk02/K=200, cohesion-only):
    dge_r 0.8105   joint_purity 0.4576   ct_purity 0.8681   n_cov 16   0 dead
  and with community at the OLD uncalibrated weight:
    dge_r 0.3603   joint_purity 0.3951   ct_purity 0.8235   n_cov 11   125 dead

NOTE this run is K=200 and latent_dims=16 to stay comparable with [D39]. That is
deliberately NOT the CLAUDE.md 9d default (K from the dataset config = 800,
latent_dims 8) -- a calibration experiment has to match the runs it is calibrated
against. Do not copy these two values into a fresh experiment.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_d46_gradbalance.py
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
        "leidenalg", "python-igraph", "ipython",
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

MEASURE = False         # stage 1 done; these are its measured lambdas.
LAM_GRAM, LAM_AFF = 10.0, 22.3      # [D39] measured spans, s28nsc/bk02, K=200
# stage-2 values, computed from stage 1 (leave as None until measured):
# MEASURED stage 1, s28nsc/bk02, step 0 (lam*g_enc w.r.t. encoder params):
#     umap 18.73 | proto_recon 3.78 | cohesion_gram 0.362 + cohesion_aff 0.426
#   -> community pulled 23.8x cohesion, proto_recon 4.8x. THAT is why [D39] saw
#      community destroy the partition: it was never calibrated, just inherited.
#   balanced at SHARE=0.25 of cohesion's total influence (0.788):
LAM_UMAP   = 0.0105      # was 1.0     -> 95x smaller
LAM_PRECON = 0.00052     # was 0.01    -> 19x smaller
# CAVEAT: one probe sample, at step 0 after waypoint init. [D45] warns cohesion's
# encoder gradient DECAYS as assignments saturate, which would make these two too
# generous later. grad_probe=50 stays on in stage 2 so the drift is visible.


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(measure: bool, lam_umap, lam_precon):
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    for k, v in (("HOME_DIR", "/vol/"), ("DATA_DIR", "/vol/data/"),
                 ("MODEL_DIR", "/vol/models/"), ("CODE_DIR", "/vol/code/")):
        os.environ[k] = v
    sys.path.insert(0, "/vol/code"); os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    # stage 1 uses PLACEHOLDER weights purely so every term is active and
    # therefore probe-able; its metrics are meaningless and are not read.
    lu = 1.0 if measure else lam_umap
    lp = 0.01 if measure else lam_precon
    name = "d46_probe" if measure else "d46_balanced"

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_recon": 0.0,
        "lambda_umap": lu,              # community ON, weight under test
        "lambda_proto_recon": lp,       # proto_recon ON, weight under test
        "lambda_cohesion": LAM_GRAM,
        "lambda_cohesion_aff": LAM_AFF,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0,     # CLAUDE.md 9d: ALWAYS 1.0
        "cohesion_global": 2, "coh_mem_init": "full", "coh_mem_hlife": 0.0,
        # hlife 0: the 500 used in [D39] was a measured no-op (0.988 at refresh),
        # so 0 is the honest setting and keeps this comparable to those runs.
        "proto_reinit_every": 0,        # so dead prototypes are VISIBLE, not repaired
        "grad_probe": 5 if measure else 50,
        "latent_dims": 16,              # match [D39]; NOT the 9d default
    }
    assert lc["cohesion_self"] == 1.0
    assert lc["proto_reinit_every"] == 0, "reinit must stay off so deaths are visible"
    if not measure:
        assert lam_umap and lam_precon, "stage 2 needs measured lambdas"

    print(f"\n{'='*78}\n=== {name}\n"
          f"    cohesion {LAM_GRAM}/{LAM_AFF} + umap {lu} + proto_recon {lp}\n"
          f"    grad_probe={lc['grad_probe']}  reinit OFF  K=200  latent 16\n"
          f"    WATCH: unused_proto from epoch 1 -- [D39] saw first deaths at epoch 2\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=1 if measure else 10,
        eval_freq=2, patience=4, batch_size=1024, affinity_type="bk02",
        num_prototypes=200, lambda_config=lc,
        trainer_kwargs={"experiment_name": name}, early_stop=False)
    print(f"\n[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> metrics: {metrics}")
    print(f"[{name}] grad probe -> {trainer.get_dump_path()}/grad_probe.jsonl")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    run.remote(MEASURE, LAM_UMAP, LAM_PRECON)
    print("\nStage 1 done -> read grad_probe.jsonl, compute:")
    print("   lambda_other = 0.25 * (lambda_coh * gnorm_coh) / gnorm_other")
    print("then set MEASURE=False, fill LAM_UMAP/LAM_PRECON, and rerun.")
