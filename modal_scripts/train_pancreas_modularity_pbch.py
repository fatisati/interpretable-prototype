"""PANCREAS: batch-aware soft modularity ([D12] modularity_agg='pbch'), usage OFF, reinit ON.

WHY THE PREVIOUS PANCREAS MODULARITY RUN FAILED
-----------------------------------------------
modu_panc_r36_l10_uOFF gave batch entropy 0.2471 against the published loss's 0.4220 --
close to SEACells' 0.19, which is exactly the behaviour the paper criticises. The cause
was NOT the lambda. It was computing modularity GLOBALLY on a graph that
sections/method.tex itself describes as "approximately block-diagonal" by batch. On a
block-diagonal graph, global modularity's optimum IS batch-separated communities,
because putting each technology in its own prototypes maximises within-community edges
against the null.

THE FIX ([D12])
---------------
Per batch b, exactly mirroring the published nassoc's per-batch construction:

    W_b[i,j]       = affinity[i,j] if batch(i)==b or batch(j)==b, else 0
    degree_b[i]    = degree[i] for cells INSIDE b; edges INTO b otherwise
    total_degree_b = sum of degree_b

    affinity_inside_b[k] = (total_degree_b / n_sampled_b)
                         * sum over SAMPLED edges touching b of assign[h,k]*assign[t,k]
    volume_b[k]          = sum over sampled cells of degree_b[i]*assign[i,k]/prob_in_batch[i]
    contribution_b[k]    = (affinity_inside_b[k]
                            - resolution * volume_b[k]*volume_running_b[k] / total_degree_b)
                           / total_degree_b

    modularity = sum over k of ( max over b of contribution_b[k] )

Per-prototype max across batches, then sum -- the direct analogue of nassoc's
element-wise max on its diagonal. A prototype spanning nine technologies is judged nine
times on within-batch coherence and scores if it is coherent in AT LEAST ONE; it is
never charged for cross-batch affinity it does not have.

Each batch keeps its OWN total_degree_b in both the rescaling and the normalizer, and
its OWN volume EMA -- a batch sees only ~1024/9 = 114 sampled edges, so a shared EMA
would reinstate the variance bias measure_modularity_estimator.py measured at 1.9x for
25-50 cell prototypes.

On single-batch data this reduces EXACTLY to the global version (max over one element,
degree_b == degree, total_degree_b == total_degree).

RESOLUTION = 16.58, and why it is SHARED not per-batch
------------------------------------------------------
Leiden bisection for K=220 on each batch's own subgraph gives:

    fluidigmc1 10.65   smarter 14.02   celseq 15.05   inDrop4 15.11   inDrop2 16.58
    inDrop1    17.13   inDrop3 20.24   celseq2 20.60  smartseq2 22.71
    median 16.58                       global (whole graph) 36.52

The global 36.52 is calibrated against the whole graph's total_degree and is ~2.2x too
strong for per-batch contributions. Among per-batch values we take ONE SHARED bar
because a lower resolution_b means a smaller charge and therefore a HIGHER
contribution -- so with max aggregation, per-batch resolutions would make the max
systematically select whichever batch is graded most leniently (fluidigmc1 at 10.65)
rather than the batch where a prototype is genuinely best. A shared bar also keeps
"how much evidence is required to group two cells" one scientific standard rather than
one that varies by sequencing technology.

LAMBDA = 0.2, derived not swept
-------------------------------
The per-batch charge gradient is ~9x the global one because total_degree_b ~
total_degree/9 sits in the denominator:

    charge_gradient = 9 * resolution / (K * B)
    lambda = f * K * S_typical / (9 * resolution * q_pos)
           = f * 220 * 0.5 / (9 * 16.58 * 0.35) = f * 2.11

f is anchored on the s28nsc sweep, where single-batch means global == per-batch, so it
transfers: f = 0.03-0.3 works, f >= 3 hurts. lambda = 0.2 is f = 0.10.

NOTE the earlier pancreas run gives NO evidence about lambda, because its failure is
attributed to the aggregation, not the weight.

Uncertainty: the derivation rests on the degree_b/prob_in_batch cancellation, a flat
factor of 9, and S_typical/q_pos ~ 1.43 read off the previous run. Could be off 2-3x --
check the epoch-1 terms.

REFERENCES TO BEAT
------------------
  published loss (reproduced)  purity 0.9041  batch_entropy 0.4220  modularity 0.6924  coverage 1.00
  global modularity, lambda=10 purity 0.9703  batch_entropy 0.2471  modularity 0.5070  coverage 0.79
  CPM (gamma 0.042, lambda 50) purity 0.9030  batch_entropy 1.0907  modularity 0.4652  coverage 0.93

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_pancreas_modularity_pbch.py
Cost: ~13 min on a T4.
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

RES_PBCH = 16.58        # median of the nine per-batch Leiden bisections
LAMBDA = 0.2            # f = 0.10 via lambda = f * K*S / (9*resolution*q_pos)


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
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

    name = "modupb_panc_r1658_l02_uOFF_reinit"
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0,                 # modularity REPLACES nassoc
        "lambda_modularity": LAMBDA,
        "modularity_agg": "pbch",             # [D12] per batch, max per prototype
        "modularity_res_pbch": RES_PBCH,
        "lambda_proto_usage": 0.0,            # usage OFF
        "proto_reinit_every": 2,              # revive dead prototypes
        "proto_reinit_thresh": 0.01,
    }
    assert lc["lambda_nassoc"] == 0.0 and lc["lambda_proto_usage"] == 0.0
    assert "latent_dims" not in lc            # paper default = 8
    print(f"\n{'='*78}\n=== {name}\n"
          f"    modularity_agg=pbch  resolution={RES_PBCH}  lambda={LAMBDA}\n"
          f"    usage OFF, reinit every 2 epochs, latent default(8), K=220, arbf\n{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "pancreas", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=220,
        lambda_config=lc, trainer_kwargs={"experiment_name": name})
    print(f"\n[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()

    def g(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "--"

    print("\n" + "=" * 92)
    print("PANCREAS -- batch-aware modularity vs the references")
    print("=" * 92)
    print(f"{'run':<34}{'purity':>10}{'batch_ent':>12}{'modularity':>12}{'coverage':>10}")
    print(f"{'modularity pbch (this run)':<34}{g(metrics,'purity'):>10}"
          f"{g(metrics,'batch_entropy'):>12}{g(metrics,'modularity'):>12}"
          f"{g(metrics,'coverage'):>10}")
    print(f"{'--- references ---':<34}")
    print(f"{'published loss (reproduced)':<34}{0.9041:>10.4f}{0.4220:>12.4f}{0.6924:>12.4f}{1.0000:>10.4f}")
    print(f"{'modularity GLOBAL, lambda=10':<34}{0.9703:>10.4f}{0.2471:>12.4f}{0.5070:>12.4f}{0.7857:>10.4f}")
    print(f"{'CPM (gamma 0.042, lambda 50)':<34}{0.9030:>10.4f}{1.0907:>12.4f}{0.4652:>12.4f}{0.9286:>10.4f}")
    print("\nPASS = batch_entropy back near or above 0.4220 (the global version's 0.2471")
    print("       was the batch-separation failure) WITHOUT losing modularity/coverage.")
    return metrics


@app.local_entrypoint()
def main():
    run.remote()
