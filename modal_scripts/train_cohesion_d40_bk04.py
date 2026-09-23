"""[D40] The cohesion winner on a SECOND BANKSY graph (bk04), vs SEACells on that
same graph.

WHY bk04 AND NOT 0.5 / 0.8 / 1.0
--------------------------------
A head-to-head is only meaningful if BOTH sides are computed on the same graph.
Checked what actually exists (2026-09-23):

    lambda   affinity              SEACells baseline            usable?
    0.2      bk02          yes     seacell_X_bklam02_K200  yes  DONE ([D39])
    0.4      bk04          yes     seacell_X_bklam04_K200  yes  THIS RUN -- free
    0.5      banksy0.5     yes     --                      no   needs a SEACells fit
    0.8      bk08          yes     --                      no   needs a SEACells fit
    1.0      --            no      --                      no   needs both

So bk04 is the one extra graph that costs nothing but a training run. 0.5/0.8
need new SEACells fits before they can be compared to anything, and that is a
separate decision about spending CPU-hours.

Both the bk04 affinity and the bklam04 SEACells assignments were bridged from
the fatemehs-hashemig volume to fatemeh-helmholtz so everything scores in one
place, exactly as was done for bk02.

WHAT THIS ANSWERS
-----------------
[D39] on bk02 established, with the community term OFF throughout:
    cohesion ALONE reaches coh_aff 0.0296 / joint_purity 0.4472 / 0 dead,
    beating SEACells on ct_purity and dge_tpr, behind on joint_purity and dge_r.
The open question is whether that ranking is a property of the METHOD or an
accident of the bk02 graph. If the winner holds on bk04 too, the result is about
the objective. If it flips, it is about the graph, and the lambda sweep becomes
the real experiment.

ARM: whichever of aff-only / gram-only / gram+aff wins on bk02 ([D39] arms
A, C, E). Set ARM_LAMBDAS below to that winner's (gram, aff) pair before
running -- the placeholder is the aff-only winner from [D39] arm A.

Everything else identical to [D39]: cohesion_self=1.0 (plain mean-similarity,
self-loop 1), cohesion_diag=1.0, cohesion_global=2 + [D32] full init + [D31]
hlife 500, [D35] per-step prototype renorm, [D37] fixed normalizer, NO reinit,
NO early stop, 10 fixed epochs.

REFERENCE, bk02 (measured, scratchpad/metric_sanity.py -- all analytic checks
exact). NOTE these are bk02 numbers; the bk04 graph has its own floor/target and
the eval pass recomputes them:
    random-200   coh_aff 0.0039   joint_purity 0.1069     <- floor
    ct x niche   coh_aff 0.0152   joint_purity 1.0000     <- biological target
    SEACells     coh_aff 0.0418   joint_purity 0.5035   dge_r 0.8487  n_cov 19
    d39_A        coh_aff 0.0296   joint_purity 0.4472   dge_r 0.8006  n_cov 16

VERDICT (fill after the eval pass):
  [ ] ranking holds on bk04 -> the result is about the objective, not the graph
  [ ] ranking flips        -> graph-dependent; the lambda sweep is the experiment
  [ ] inconclusive-because-...

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d40_bk04.py
Then: add this run name to eval_all_vs_seacells.py, point it at bk04 + the
bklam04 baseline, and rerun that pass for joint_purity / dge_r / n_covered.
Cost: one T4 arm, 10 epochs, ~15 min.
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

# measured spans at cohesion_self=1 (d30 header), authority 1.5403 held fixed
SPAN_GRAM, SPAN_AFF = 0.076776, 0.034573
AUTHORITY = 20.0 * 0.077017

# (name, lambda_gram, lambda_aff) -- SET THIS to the bk02 winner
ARM_LAMBDAS = ("d40_bk04_aff", 0.0, 44.6)


@app.function(image=image, gpu="T4", timeout=5400, volumes={"/vol": volume})
def run(name: str, lam_gram: float, lam_aff: float):
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

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": 0.0,
        # community OFF -- [D39] measured it costing 125/200 prototypes and
        # halving dge_r on bk02, and decisions.md records the retraction of the
        # "it aligns batches" rationale that had been keeping it.
        "lambda_umap": 0.0,
        "lambda_proto_recon": 0.0,
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
        "coh_diag_every": 50,
        "latent_dims": 16,
    }
    _auth = lam_gram * SPAN_GRAM + lam_aff * SPAN_AFF
    assert abs(_auth - AUTHORITY) < 0.05, f"authority drifted: {_auth}"
    assert lc["lambda_umap"] == 0.0 and lc["lambda_proto_recon"] == 0.0

    print(f"\n{'='*78}\n=== {name}  on bk04 (BANKSY lambda 0.4)\n"
          f"    lambda_gram={lam_gram}  lambda_aff={lam_aff}  (authority {_auth:.4f})\n"
          f"    community OFF, no reinit, no early stop, 10 fixed epochs\n"
          f"    baseline to beat: seacell_X_bklam04_K200 on this SAME graph\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=10, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk04", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False)
    print(f"\n[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    name, lam_gram, lam_aff = ARM_LAMBDAS
    run.remote(name, lam_gram, lam_aff)
    print("\nNext: register this run in eval_all_vs_seacells.py against bk04 +")
    print("seacell_X_bklam04_K200, then rerun it for joint_purity / dge_r / n_covered.")
