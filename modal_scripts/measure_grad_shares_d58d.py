"""[D58d] FIX #3 (final): full pass over ALL edges, gradients computed every
step, but optimizer.step() is a no-op -- model never moves. Average gnorm
over the whole pass.

WHY, after d58b and d58c both had real problems
-------------------------------------------------
d58b: full pass WITH real updates -> the measurement pass itself distorted
  the model toward recon/community (their raw gradient is 15-40x cohesion's
  at lambda=1), so the measured gnorm didn't match the real run's own
  fresh-state gnorm. Verified: real run's OWN step-0 share was
  4.98%/6.27%/47.70%/41.05%, nowhere near the intended 25% each.
d58c: single step, no updates -> uncontaminated, but only one minibatch --
  not "a full pass over all edges" (a noisy, unrepresentative sample), and
  this run never actually executed (Modal workspace hit its spend limit
  mid-launch).

THE FIX (user's own diagnosis): load the pretrained checkpoint, run one full
pass over every edge (correct threshold, train_epochs=10 matching the real
train_d59 deployment), compute gradients every step via the existing
grad_probe hook, but monkeypatch `trainer.optimizer.step` to a no-op so the
model NEVER moves during the whole pass. Average gnorm_enc across every
logged step for the final per-term estimate -- representative (sees the
whole graph, not one batch) AND uncontaminated (weights are bit-identical
from the first step to the last).

Note: the cohesion memory buffer (coh_mem, an EMA of soft assignments) still
updates during the pass -- that's `coh_mem_init='full'`'s own documented
job (CLAUDE.md 9d [D32]), not weight drift, and is desired here exactly as
it is during real training's first pass.

Run (only once the Modal workspace spend limit is raised):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_d58d.py
"""
import json
import time

import modal

app = modal.App("scproto-d58d-probe")

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

NH_TAG = "arbf_nodeheldout20_seed0"
REAL_TRAIN_EPOCHS = 10   # matches train_d59's ACTUAL deployment, all 5 datasets

NH_DATASETS = {
    "pancreas": ("tech", "celltype", 220),
    "lung": ("batch", "cell_type", 300),
    "pbmc-immune": ("study", "final_annotation", 300),
}
FULL_DATASETS = {
    "s28nsc": ("bk02", 800),
    "crcx": ("bk05", 670),
}


def build_probe_config():
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_proto_recon": 0.0,
        "lambda_umap": 1.0, "lambda_recon": 1.0,
        "lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0,
        "proto_reinit_every": 0,
        "grad_probe": 1,             # log every step of the pass
        "umap_proto_sg": 1,
    }


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
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


def _average_full_pass(dump_path):
    """Average gnorm_enc across EVERY logged step of the frozen-weight pass."""
    import os
    p = os.path.join(dump_path, "grad_probe.jsonl")
    if not os.path.exists(p):
        return {"error": f"no grad_probe.jsonl at {p}"}
    with open(p) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    sums = {"cohesion_gram": 0.0, "cohesion_aff": 0.0, "umap": 0.0, "recon": 0.0}
    n = 0
    for line in lines:
        row = json.loads(line)
        if "error" in row:
            continue
        terms = {k: v for k, v in row.items() if k != "step" and isinstance(v, dict)}
        if not all(t in terms for t in sums):
            continue
        for t in sums:
            sums[t] += terms[t]["gnorm_enc"]
        n += 1
    if n == 0:
        return {"error": "no usable rows", "rows_seen": len(lines)}
    return {"n_steps_averaged": n, **{t: sums[t] / n for t in sums}}


def _build_trainer(ds_id, batch_key, label_key, K, affinity_type, lc, name):
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)
    import os
    lc = dict(lc)
    lc['affinity_type'] = affinity_type
    resolved_ds = _resolve_ds_id(ds_id, batch_key=batch_key, label_key=label_key,
                                 niche_key=None, num_prototypes=K)
    resolved_K = _resolve_num_prototypes(K, resolved_ds)
    t = get_trainer(
        experiment_name=name, cvae_epochs=50,
        affinity_type=lc.get('affinity_type', 'arbf'),
        dataset_id=resolved_ds, l2norm=1, assignment_metric='dotp',
        batch_size=1024, umap_steps_per_epoch=None,
        umap_similarity=lc.get('umap_similarity', 'proto'),
        freeze_batch_embedding=1, freeze_decoder=0,
        **({} if resolved_K is None else {'num_prototypes': resolved_K}),
        **{k: v for k, v in lc.items() if k not in ('umap_similarity', 'affinity_type')},
    )
    ckpt_path = os.path.join(t.get_pretrain_dump_path(), 'pretrain_checkpoint.pth')
    if os.path.exists(ckpt_path):
        t.load_pretrain_checkpoint()
    else:
        t.pretrain_encoder()
        t.init_prototypes()
        t.save_pretrain_checkpoint()
    if hasattr(t.model, 'freeze_batch_embedding'):
        t.model.freeze_batch_embedding()
    return t


def _measure_frozen(t):
    """Build the correctly-thresholded edge dataset, run ONE full pass with
    optimizer.step() no-op'd so weights never move, return the averaged
    per-term gnorm_enc over the whole pass."""
    t._setup_umap_edges(REAL_TRAIN_EPOCHS)
    t.optimizer.step = lambda *a, **kw: None    # <- the whole fix: freeze weights
    t.continue_train_umap_edges(epochs=1, early_stop=False)
    return _average_full_pass(t.get_dump_path())


@app.function(image=image, gpu="A100", timeout=1800, volumes={"/vol": volume})
def probe_nodeheldout(ds: str):
    _env()
    bk, lk, K = NH_DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad"
    lc = build_probe_config()
    name = f"d58d_{ds}"
    t0 = time.time()
    t = _build_trainer(train_h5, bk, lk, K, "arbf", lc, name)
    shares = _measure_frozen(t)
    print(f"[{ds}] FROZEN full-pass gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.function(image=image, gpu="A100", timeout=1800, volumes={"/vol": volume})
def probe_full(ds: str):
    _env()
    aff, K = FULL_DATASETS[ds]
    lc = build_probe_config()
    name = f"d58d_{ds}"
    t0 = time.time()
    t = _build_trainer(ds, None, None, K, aff, lc, name)
    shares = _measure_frozen(t)
    print(f"[{ds}] FROZEN full-pass gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.local_entrypoint()
def main():
    calls = {}
    for ds in NH_DATASETS:
        calls[ds] = probe_nodeheldout.spawn(ds)
    for ds in FULL_DATASETS:
        calls[ds] = probe_full.spawn(ds)
    print(f"Spawned {len(calls)} parallel FROZEN full-pass probes: {list(calls)}")

    results = {}
    for ds, c in calls.items():
        ds2, r = c.get()
        results[ds2] = r

    print("\n" + "=" * 100)
    print("FROZEN full-pass gnorm_enc (weights never moved) -> lambda_i = 0.25/gnorm_i")
    print("=" * 100)
    SHARE = 0.25
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:<14} FAILED: {r}")
            continue
        print(f"\n{ds}  (averaged over {r['n_steps_averaged']} steps):")
        for term in ("cohesion_gram", "cohesion_aff", "umap", "recon"):
            g = r[term]
            lam = SHARE / g if g > 0 else float("inf")
            print(f"  {term:<14} gnorm_enc={g:.6e}   lambda = {lam:.5f}")
