"""[D36] SHORT diagnostic: is cohesion_aff's gradient dead, or just small?

Same config as d35_final (aff sim, cohesion_self=1.0 simple, D29+D31+D32
memory, no reinit, no early stop) but capped at 2 epochs with a TIGHT
diagnostic cadence (coh_diag_every=10, vs the default 50), so we get several
[D36 DEBUG] lines per epoch instead of ~10 per full 20-epoch run.

Each [D36 DEBUG] line prints, at the SAME step:
    cohesion_aff(batch-metric)      -- what's already in the epoch summary
    cohesion_aff(TRUE full-graph)   -- the exact full-N value, computed fresh
    grad_norm(wrt soft_assign)      -- ACTUAL measured gradient norm of
                                        cohesion_aff_loss w.r.t. this batch's
                                        soft assignment (torch.autograd.grad,
                                        retain_graph=True -- doesn't touch the
                                        real backward() pass)

DECISION BEFORE RUNNING -- what each outcome means
---------------------------------------------------
  grad_norm == -1.0 (printed) or "GRAPH DISCONNECTED"
      -> user's hypothesis confirmed: the gradient path is genuinely broken.
         Next step: bisect the graph (check _S_eff / _Sc construction) to
         find exactly where it detaches.
  grad_norm > 0 but tiny (< ~1e-4) and NOT growing across the 2 epochs
      -> alive but the dilution mechanism (live gradient only through the
         batch's rows vs a size_g denominator spanning the whole memory) is
         suppressing it enough to be practically inert. Next step: consider
         whether cohesion_global=2's per-step gradient scale needs a fix
         (this predates D31/D32 -- it's inherent to plain D29 too).
  grad_norm meaningfully larger, OR cohesion_aff(TRUE full-graph) is clearly
  higher than the batch-metric and RISING across steps
      -> it's training; the flat 4-decimal print in d35_final was a rounding
         artefact of the batch-vs-full-N scale gap in the metric's own
         formula, not a training bug. Next step: fix the metric's own
         reporting scale (mirror nassoc's full-scale rescale), not the loss.

VERDICT (fill in after reading the log):
  [ ] dead gradient confirmed -- find the disconnect
  [ ] alive but diluted -- consider a gradient-scale fix
  [ ] alive and fine -- only the printed metric was misleading
  [ ] inconclusive-because-...

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d36_debug.py
Cost: 2 epochs only, a few minutes on a T4, well under $0.10.
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
        "leidenalg", "python-igraph",
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

SPAN_AFF_SELF = 0.034573
AUTHORITY = 20.0 * 0.077017
LAM_AFF = round(AUTHORITY / SPAN_AFF_SELF, 1)   # 44.6


@app.function(image=image, gpu="T4", timeout=1800, volumes={"/vol": volume})
def run():
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
        "lambda_cohesion": 0.0,
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
        "coh_diag_every": 10,       # [D36] tight cadence for this short debug run
        "latent_dims": 16,
    }

    print(f"\n{'='*78}\n=== d36_debug -- 2 EPOCHS ONLY, tight [D36] diagnostic cadence\n"
          f"    same config as d35_final. WATCH the [D36 DEBUG] lines:\n"
          f"    grad_norm == -1 -> dead gradient (disconnected graph)\n"
          f"    grad_norm tiny  -> alive but diluted by the memory's size_g scale\n"
          f"    TRUE full-graph value vs batch-metric -> confirms/refutes the\n"
          f"    ~n_batch/N scale-mismatch hypothesis directly\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=2, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": "d36_debug"},
        early_stop=False)
    print(f"\n[d36_debug] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[d36_debug] >>> metrics: {metrics}")
    volume.commit()
    return metrics


@app.local_entrypoint()
def main():
    run.remote()
    print("\nDone. Read the [D36 DEBUG] lines above to fill the verdict.")
