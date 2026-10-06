"""[D58b] FIX: d58's probe measured gradients on a DEGENERATE 2-edge dataset,
not a full pass -- redo properly.

THE BUG, FOUND BY RE-CHECKING BEFORE TRUSTING d58's NUMBERS
-------------------------------------------------------------
`_setup_umap_edges(epochs)` (trainers/scproto.py:1010) sets
    weight_threshold = weights.max() / epochs
and only edges surviving that threshold enter the EdgeDataset used for
training. `find_metacells`/`run_mc_task` passes `train_epochs` straight
through as this `epochs` value (tasks.py:288). d58's probe used
`train_epochs=1`, so `weight_threshold = weights.max() / 1 = weights.max()`
-- the most restrictive threshold possible. Confirmed in every one of the 5
probe logs: the FIRST (and only trained-on) EdgeDataset was exactly "2 edges"
at the single top weight value in every dataset (pancreas/lung/pbmc-immune/
crcx/s28nsc all show this). A second, much larger EdgeDataset (918K-4.6M
edges) appears LATER in each log, but that is built by an EVAL-time code path
AFTER the (degenerate) GPROBE reading was already taken -- it never
contributed to what was measured.

So every lambda_i computed from d58 came from a near-empty, single-weight-
value edge sample -- not remotely a representative pass over the graph. User
caught this: "wait to get the full pass on the pretrained encoder to find
lambdas."

THE FIX
-------
Build the EdgeDataset at the SAME threshold the real deployment will use
(train_epochs=20, so weight_threshold = max/20, matching train_d59's real
run), but only actually RUN one epoch of measurement over it -- not all 20 --
by calling the trainer's two lower-level methods directly instead of the
`train_umap_edges` convenience wrapper that couples threshold-epochs to
loop-epochs 1:1:

    t._setup_umap_edges(20)                    # correct threshold, real edges
    t.continue_train_umap_edges(epochs=1, ...) # ONE real epoch of measurement

This replicates find_metacells' own setup (tasks.py:192-269: resolve ds_id,
build trainer via get_trainer, load pretrain checkpoint, freeze batch
embedding) up to the point where it would normally call
`train_umap_edges(epochs=train_epochs)`, then substitutes the two-call
version above instead of that single call. Everything else (pretrain
checkpoint reuse, cohesion config, lambda values) is identical to d58 and to
the real train_d59 deployment.

grad_probe=25 (not 1) since a full epoch now has real steps (hundreds to
thousands) worth watching -- the LAST record in grad_probe.jsonl (not the
first) is what gets read, i.e. the gradient state AFTER a full representative
pass over the pretrained encoder, not at step 0 right after waypoint init.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_d58b.py
"""
import json
import time

import modal

app = modal.App("scproto-d58b-probe")

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
REAL_TRAIN_EPOCHS = 20   # matches train_d59 -- the threshold the probe must match

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
        "grad_probe": 25,
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


def _extract_last(dump_path):
    import os
    p = os.path.join(dump_path, "grad_probe.jsonl")
    if not os.path.exists(p):
        return {"error": f"no grad_probe.jsonl at {p}"}
    with open(p) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    best = None
    for line in lines:
        row = json.loads(line)
        if "error" in row:
            continue
        terms = {k: v for k, v in row.items() if k != "step" and isinstance(v, dict)}
        if all(t in terms for t in ("cohesion_gram", "cohesion_aff", "umap", "recon")):
            best = (row.get("step", -1), terms)
    if best is None:
        return {"error": "no row had all 4 terms", "rows_seen": len(lines)}
    step, terms = best
    return {"step": step, **{t: terms[t]["gnorm_enc"] for t in
             ("cohesion_gram", "cohesion_aff", "umap", "recon")}}


def _build_trainer(ds_id, batch_key, label_key, K, affinity_type, lc, name):
    """Replicates find_metacells (tasks.py:192-269) up to the point it would
    call train_umap_edges -- pretrain load + freeze -- then stops there so we
    can drive the edge-threshold / epoch-count split ourselves."""
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)

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

    ckpt_path = __import__('os').path.join(t.get_pretrain_dump_path(), 'pretrain_checkpoint.pth')
    import os
    if os.path.exists(ckpt_path):
        t.load_pretrain_checkpoint()
    else:
        t.pretrain_encoder()
        t.init_prototypes()
        t.save_pretrain_checkpoint()
    if hasattr(t.model, 'freeze_batch_embedding'):
        t.model.freeze_batch_embedding()
    return t


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/vol": volume})
def probe_nodeheldout(ds: str):
    _env()
    bk, lk, K = NH_DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad"
    lc = build_probe_config()
    name = f"d58b_{ds}"
    t0 = time.time()
    t = _build_trainer(train_h5, bk, lk, K, "arbf", lc, name)
    t._setup_umap_edges(REAL_TRAIN_EPOCHS)
    t.continue_train_umap_edges(epochs=1, early_stop=False)
    shares = _extract_last(t.get_dump_path())
    print(f"[{ds}] FULL-PASS gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/vol": volume})
def probe_full(ds: str):
    _env()
    aff, K = FULL_DATASETS[ds]
    lc = build_probe_config()
    name = f"d58b_{ds}"
    t0 = time.time()
    t = _build_trainer(ds, None, None, K, aff, lc, name)
    t._setup_umap_edges(REAL_TRAIN_EPOCHS)
    t.continue_train_umap_edges(epochs=1, early_stop=False)
    shares = _extract_last(t.get_dump_path())
    print(f"[{ds}] FULL-PASS gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.local_entrypoint()
def main():
    calls = {}
    for ds in NH_DATASETS:
        calls[ds] = probe_nodeheldout.spawn(ds)
    for ds in FULL_DATASETS:
        calls[ds] = probe_full.spawn(ds)
    print(f"Spawned {len(calls)} parallel FULL-PASS probes: {list(calls)}")

    results = {}
    for ds, c in calls.items():
        ds2, r = c.get()
        results[ds2] = r

    print("\n" + "=" * 100)
    print("FULL-PASS gnorm_enc (threshold matches train_epochs=20) -> lambda_i = 0.25/gnorm_i")
    print("=" * 100)
    SHARE = 0.25
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:<14} FAILED: {r}")
            continue
        print(f"\n{ds}  (measured at step {r['step']}):")
        for term in ("cohesion_gram", "cohesion_aff", "umap", "recon"):
            g = r[term]
            lam = SHARE / g if g > 0 else float("inf")
            print(f"  {term:<14} gnorm_enc={g:.6e}   lambda = {lam:.5f}")
