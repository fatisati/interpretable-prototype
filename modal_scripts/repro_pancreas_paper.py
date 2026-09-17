"""REPRODUCE THE PAPER'S PANCREAS ROW -- published nassoc mode, nothing else.

WHY
---
The paper's pancreas column (sections/results.tex) is:

                  Modularity      Batch Entropy      Purity
    SEACells      0.67 +- 0.05    0.19 +- 0.30       0.97 +- 0.08
    scProto       0.61 +- 0.08    1.04 +- 0.66       0.98 +- 0.07

The caption defines them as: modularity = mean +- std ACROSS BATCHES; batch entropy and
purity = mean +- std across metacells. Cross-checking against SEACells' stored
metrics.json confirms the key mapping exactly:

    paper 0.67 = mean_modularity_batch           0.6729
    paper 0.19 = mean_batch_entropy              0.1916
    paper 0.97 = weighted_mean_cell_type_purity  0.9653

Applying that same mapping to the stored scProto run proto_umap_ds-panc_NP220_LD20_...
gives modularity 0.6145 (paper 0.61, matches) and purity 0.9762 (paper 0.98, matches)
but batch entropy 0.3895 against the paper's 1.04 -- a 2.7x mismatch. So that stored run
is NOT the one behind the paper's column, and we do not currently know which of the
several proto_umap_ds-panc_NP220_* runs is.

WHAT THIS RUNS
--------------
The PUBLISHED objective, exactly as written in sections/method.tex:

    M[k,j]   = A[k,j] / sqrt(vol[k]*vol[j])       A = S' W S,  vol = S' d
    L_nassoc = (1/K) sum_k (M[k,k]-1)^2 + alpha/(K(K-1)) sum_{k!=j} M[k,j]^2

i.e. nassoc_diag_norm='volume' + nassoc_diag_loss='mse' -- both DEFAULTS. No gamma
exists in this objective at all. lambda_nassoc=1, usage ON at 0.1, affinity 'arbf'
(PCA), K=220. Nothing from today's work is enabled: nassoc_exact, nassoc_ht,
nassoc_cpm_norm and nassoc_gamma_scale all stay at their behaviour-preserving defaults.

TWO ARMS, because the latent dimension is the main unknown: the stored run I read is
LD20, but train_pancreas_nadn_pair.py names a DIFFERENT run with no LD token as
canonical. Running both settles it.

THIS IS ALSO A REGRESSION TEST. Today's commits (427d2e3 [D9], c8e21a1 [D9b],
d3a98ed [D10]) all claim to be behaviour-preserving at their defaults. If this
reproduces the stored numbers, that claim is verified. If it does not, today's edits
broke the default path and everything else from today is suspect.

KNOWN REPRODUCTION RISK, stated up front: train_epochs is set to 20 per CLAUDE.md
section 5, but the published runs predate that rule and may have used more. If the
numbers come out low, epochs is the first suspect, not the loss.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/repro_pancreas_paper.py
Cost: 2 x ~8-10 min on a T4 (16,382 cells). Commits after every arm.
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

# (latent_dims, tag). None = leave at the config default, matching the no-LD run name.
ARMS = [(20, "LD20"), (None, "LDdef")]

PAPER = dict(modularity=0.61, batch_entropy=1.04, purity=0.98)
SEACELLS = dict(modularity=0.67, batch_entropy=0.19, purity=0.97)
STORED_LD20 = dict(modularity=0.6145, batch_entropy=0.3895, purity=0.9762)


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def repro():
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

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    out = {}
    for dim, tag in ARMS:
        name = f"repro_paper_{tag}"
        print("\n" + "=" * 78)
        print(f"=== {name}: PUBLISHED nassoc (volume + MSE-to-1), lambda_nassoc=1, "
              f"usage=0.1, latent={dim or 'default'}")
        print("=" * 78)
        # LAMBDA_PROTO_UMAP_PRECON already carries lambda_nassoc=1, lambda_proto_usage=0.1,
        # proto_usage_mode='ema', nassoc_agg='max'. Everything the paper used is a DEFAULT,
        # so the only thing to set is latent_dims. Nothing from today is enabled.
        lc = dict(LAMBDA_PROTO_UMAP_PRECON)
        if dim is not None:
            lc["latent_dims"] = dim

        t0 = time.time()
        try:
            trainer, metrics, _ = run_mc_task(
                "pancreas",
                cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="arbf", num_prototypes=220,
                lambda_config=lc,
                trainer_kwargs={"experiment_name": name},
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            out[name] = {"error": str(e)}
            volume.commit()
            continue

        el = time.time() - t0
        print(f"[{name}] done in {el:.0f}s  model={trainer.get_model_name()}")
        got = dict(
            modularity=metrics.get("mean_modularity_batch"),
            batch_entropy=metrics.get("mean_batch_entropy"),
            purity=metrics.get("weighted_mean_cell_type_purity"),
        )
        print(f"[{name}] >>> modularity(batch-mean)={got['modularity']}  "
              f"batch_entropy={got['batch_entropy']}  purity={got['purity']}  "
              f"unused={metrics.get('n_unused_protos')}")
        out[name] = {"latent": dim, "elapsed": el, "paper_keys": got, "metrics": metrics}
        volume.commit()

    def f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "--"

    print("\n" + "=" * 92)
    print("REPRODUCTION vs THE PAPER'S PANCREAS ROW")
    print("=" * 92)
    print(f"{'source':<26}{'modularity':>14}{'batch_entropy':>16}{'purity':>10}")
    print(f"{'PAPER scProto':<26}{f(PAPER['modularity']):>14}"
          f"{f(PAPER['batch_entropy']):>16}{f(PAPER['purity']):>10}")
    print(f"{'PAPER SEACells':<26}{f(SEACELLS['modularity']):>14}"
          f"{f(SEACELLS['batch_entropy']):>16}{f(SEACELLS['purity']):>10}")
    print(f"{'stored LD20 run':<26}{f(STORED_LD20['modularity']):>14}"
          f"{f(STORED_LD20['batch_entropy']):>16}{f(STORED_LD20['purity']):>10}")
    for k, v in out.items():
        if "error" in v:
            print(f"{k:<26}  ERROR: {v['error'][:50]}")
            continue
        g = v["paper_keys"]
        print(f"{k:<26}{f(g['modularity']):>14}{f(g['batch_entropy']):>16}"
              f"{f(g['purity']):>10}")
    print("\nPASS = a run matching the PAPER row (esp. batch_entropy ~1.04, the one the")
    print("       stored LD20 run misses at 0.39). That both identifies the paper's")
    print("       config AND verifies today's commits are behaviour-preserving.")
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    print(repro.remote())
