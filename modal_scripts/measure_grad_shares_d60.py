"""[D60] Frozen full-pass gradient-norm probe -- the measurement d58d INTENDED
to take but did not.

WHY d58d IS INVALID (found 2026-10-06, before spending any new GPU time)
------------------------------------------------------------------------
d58d:180 froze the weights with

    t.optimizer.step = lambda *a, **kw: None

but `scproto.py` has TWO optimizers and that is the wrong one:

    pretrain optimizer   built scproto.py:813-859  stored  self.optimizer (line 859)
    stage-2 optimizer    built scproto.py:1133     stored  self._umap_state['optimizer']
                                                            (line 1140)

`_run_umap_epoch` reads `optimizer = s['optimizer']` (scproto.py:1294) -- it never
touches `self.optimizer`. So the patch hit an optimizer that is never stepped during
the pass and THE WEIGHTS MOVED FOR THE WHOLE RUN. d58d is a relabelled d58b, i.e.
exactly the failure its own docstring was written to fix.

Proof in d58d's own output (mean of first 10% vs last 10% of the pass, gnorm_enc):

    dataset    gram             aff              community        recon
    pancreas   0.135 -> 0.169   0.084 -> 0.109   1.67 -> 2.06     19.7 -> 20.5
    lung       0.117 -> 0.199   0.066 -> 0.117   2.46 -> 2.32     3.64 -> 6.17
    immune     0.055 -> 0.088   0.033 -> 0.051   2.32 -> 1.72     4.51 -> 8.65
    crcx       0.047 -> 0.028   0.036 -> 0.014   6.45 -> 1.99     5.45 -> 7.05

`recon`'s gnorm cannot move 1.92x with frozen weights -- it never touches the
cohesion memory, so nothing else in the pass could move it. And the drift direction
is the one d58d's docstring describes for d58b: at all-lambdas-1 the model trains
toward recon/community, their gnorms fall, cohesion's rises.

WHAT THIS SCRIPT CHANGES, AND NOTHING ELSE
------------------------------------------
Kept verbatim from d58d (these parts were right):
  - real training path: load_pretrain_checkpoint() (or pretrain+init_prototypes),
    freeze_batch_embedding=1, dataset-default K, batch_size=1024,
    umap_steps_per_epoch=None, umap_similarity='proto', umap_proto_sg=1
  - `_setup_umap_edges(REAL_TRAIN_EPOCHS)` with REAL_TRAIN_EPOCHS=10, so
    weight_threshold = max_weight/10 matches train_d59's ACTUAL deployment.
    (Probing at train_epochs=1 is the d58/d58b bug: measured 9.4x error on
    s28nsc's umap gnorm -- see files/notes/lambda_calibration.md.)
  - all four lambdas = 1, grad_probe=1, exit after ONE pass.

Changed:
  1. Freeze THREE ways so no single mechanism is load-bearing:
       (a) patch `_umap_state['optimizer'].step`  <- the one actually used
       (b) zero every param_group['lr']
       (c) base_lr=0.0 on the trainer, so the optimizer is BUILT with lr=0
  2. ASSERT bit-exact parameter equality before/after the pass. The run FAILS if
     anything moved. This is the check d58d lacked.
  3. ASSERT the logged row count matches len(loader). grad_probe has silently
     stopped after step 0/1 on some configs (d58p wrote 1 row; s28nsc/d46_probe
     wrote 2) -- a mean over 1 row must raise, not return a number.
  4. Report median (what sets the lambda; gnorm is positive and right-skewed) plus
     mean/std/CV, AND the first-50-step median separately so the cohesion memory's
     warm-up is visible rather than silently averaged in.
  5. T4, not A100. The probe is tiny; d58d asked for a GPU it does not need.

NOTE -- the cohesion memory DOES still update during the pass, on purpose.
`coh_mem_init='full'` filling in over the pass is [D32]'s documented job and is what
happens during real training's first pass too. So gram/aff gnorm legitimately rises
as the memory warms; recon's must NOT (see assertion 2 / verification 3).

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_d60.py
"""
import json
import statistics as st
import time

import modal

app = modal.App("scproto-d60-probe")

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
TERMS = ("cohesion_gram", "cohesion_aff", "umap", "recon")
WARM = 50                # first-N-steps window reported separately (cold memory)

NH_DATASETS = {
    "pancreas": ("tech", "celltype", 220),
    "lung": ("batch", "cell_type", 300),
    "pbmc-immune": ("study", "final_annotation", 300),
}
FULL_DATASETS = {
    "s28nsc": ("bk02", 800),
    "crcx": ("bk05", 670),
}

# Target share vectors, both chosen a priori -- NOT tuned on any metric.
SHARES = {
    "A_flat": {"cohesion_gram": 0.25, "cohesion_aff": 0.25,
               "umap": 0.25, "recon": 0.25},
    "B_subsystem": {"cohesion_gram": 1 / 6, "cohesion_aff": 1 / 6,
                    "umap": 1 / 3, "recon": 1 / 3},
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
        "base_lr": 0.0,              # [D60] freeze (c): optimizer BUILT with lr=0
                                     #   (scproto.py:1133 uses self.base_lr).
                                     #   'blr' is in constants.py:84 and numeric, so
                                     #   the probe gets its own run folder and cannot
                                     #   clobber a real run.
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


def _summarize_full_pass(dump_path, expected_steps):
    """Per-term stats over EVERY logged step of the frozen pass.

    Raises if the probe logged far fewer rows than the loader has batches --
    grad_probe has silently stopped after step 0/1 on some configs, and a 'mean'
    over one row is not a measurement.
    """
    import os
    p = os.path.join(dump_path, "grad_probe.jsonl")
    if not os.path.exists(p):
        raise RuntimeError(f"[D60] no grad_probe.jsonl at {p}")

    with open(p) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    series, errors = {t: [] for t in TERMS}, []
    for line in lines:
        row = json.loads(line)
        if "error" in row:
            errors.append(row["error"])
            continue
        terms = {k: v for k, v in row.items() if k != "step" and isinstance(v, dict)}
        if not all(t in terms for t in TERMS):
            continue
        for t in TERMS:
            series[t].append(terms[t]["gnorm_enc"])

    n = len(series[TERMS[0]])
    if n < max(10, int(0.9 * expected_steps)):
        raise RuntimeError(
            f"[D60] probe logged only {n} usable rows but the loader has "
            f"{expected_steps} batches -- grad_probe stopped early, measurement "
            f"is NOT a full pass. errors={errors[:3]}")

    out = {"n_steps": n, "expected_steps": expected_steps,
           "n_probe_errors": len(errors)}
    for t in TERMS:
        g = series[t]
        w = g[:WARM]
        out[t] = {
            "median": st.median(g),
            "mean": st.mean(g),
            "std": st.pstdev(g),
            "cv": st.pstdev(g) / st.mean(g) if st.mean(g) else 0.0,
            "step0": g[0],
            f"median_first{WARM}": st.median(w),
            f"median_last{WARM}": st.median(g[-WARM:]),
        }
    return out


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
    """ONE full pass over every edge with the weights BIT-EXACTLY frozen."""
    import os
    import torch

    # correct edge threshold -- matches deployment, NOT train_epochs=1
    t._setup_umap_edges(REAL_TRAIN_EPOCHS)

    s = t._umap_state
    expected_steps = len(s['loader'])
    opt = s['optimizer']

    # (a) patch the optimizer _run_umap_epoch actually uses (scproto.py:1294)
    opt.step = lambda *a, **kw: None
    # (b) zero every param group's lr, so even a re-created optimizer steps by 0
    for g in opt.param_groups:
        g['lr'] = 0.0
    # base_lr=0.0 in the config is freeze (c) -- it was already built with lr=0.
    print(f"   [D60] freeze: optimizer={type(opt).__name__} "
          f"lrs={[g['lr'] for g in opt.param_groups]}  expected_steps={expected_steps}")

    # don't write a pointless checkpoint of an unchanged model
    t.umap_checkpoint_freq = 0

    # start from a clean probe log so a stale file cannot pad the row count
    probe_path = os.path.join(t.get_dump_path(), 'grad_probe.jsonl')
    if os.path.exists(probe_path):
        os.remove(probe_path)

    # (c) bit-exact fingerprint before/after -- the primary gate
    before = {n: p.detach().clone() for n, p in t.model.named_parameters()}

    t.continue_train_umap_edges(epochs=1, early_stop=False)

    moved = [n for n, p in t.model.named_parameters()
             if not torch.equal(p.detach(), before[n])]
    if moved:
        raise RuntimeError(
            f"[D60] WEIGHTS MOVED during the probe -- measurement INVALID "
            f"({len(moved)} tensors changed). First 5: {moved[:5]}")
    print(f"   [D60] VERIFIED frozen: all {len(before)} parameter tensors bit-identical")

    return _summarize_full_pass(t.get_dump_path(), expected_steps)


def _run(ds, ds_id, bk, lk, K, aff):
    _env()
    lc = build_probe_config()
    t0 = time.time()
    t = _build_trainer(ds_id, bk, lk, K, aff, lc, f"d60_{ds}")
    res = _measure_frozen(t)
    print(f"[{ds}] frozen full-pass done in {time.time()-t0:.0f}s: "
          f"{res['n_steps']}/{res['expected_steps']} steps")
    volume.commit()
    return ds, res


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def probe_nodeheldout(ds: str):
    bk, lk, K = NH_DATASETS[ds]
    return _run(ds, f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad",
                bk, lk, K, "arbf")


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def probe_full(ds: str):
    aff, K = FULL_DATASETS[ds]
    return _run(ds, ds, None, None, K, aff)


def _sig4(x):
    """4 significant digits -- NOT 2 decimal places. CLAUDE.md 9d's rule exists for
    FILENAME LENGTH (lcoh4.03750478377344 blew the 255-byte limit); blind 2dp would
    turn recon=0.0187 into 0.02."""
    if x == 0 or x != x:
        return "0"
    from math import floor, log10
    return f"%.{max(0, 3 - int(floor(log10(abs(x)))))}f" % x


@app.local_entrypoint()
def main():
    calls = {}
    for ds in NH_DATASETS:
        calls[ds] = probe_nodeheldout.spawn(ds)
    for ds in FULL_DATASETS:
        calls[ds] = probe_full.spawn(ds)
    print(f"[D60] spawned {len(calls)} parallel FROZEN full-pass probes: {list(calls)}")

    results, failed = {}, {}
    for ds, c in calls.items():
        try:
            ds2, r = c.get()
            results[ds2] = r
        except Exception as e:
            failed[ds] = f"{type(e).__name__}: {e}"
            print(f"[{ds}] FAILED: {failed[ds]}")

    print("\n" + "=" * 104)
    print("[D60] FROZEN full-pass gnorm_enc -- weights VERIFIED bit-identical over the pass")
    print("=" * 104)

    for ds, r in results.items():
        print(f"\n### {ds}   ({r['n_steps']}/{r['expected_steps']} steps, "
              f"{r['n_probe_errors']} probe errors)")
        print(f"{'term':<15}{'median':>12}{'mean':>12}{'std':>12}{'CV':>7}"
              f"{'step0':>12}{'med_first50':>13}{'med_last50':>13}")
        for t in TERMS:
            d = r[t]
            print(f"{t:<15}{d['median']:>12.5g}{d['mean']:>12.5g}{d['std']:>12.5g}"
                  f"{d['cv']:>7.2f}{d['step0']:>12.5g}"
                  f"{d[f'median_first{WARM}']:>13.5g}{d[f'median_last{WARM}']:>13.5g}")
        # recon must be flat with frozen weights -- it never touches the cohesion memory
        rc = r['recon']
        ratio = (rc[f'median_last{WARM}'] / rc[f'median_first{WARM}']
                 if rc[f'median_first{WARM}'] else float('nan'))
        flag = "OK" if 0.8 <= ratio <= 1.25 else "<-- CHECK: recon should be FLAT when frozen"
        print(f"  recon last50/first50 = {ratio:.3f}  {flag}")

    for label, sh in SHARES.items():
        print(f"\n### lambdas, share vector {label} "
              f"(basis: full-pass MEDIAN; 4 significant digits)\n")
        print(f"{'dataset':<14}{'gram':>12}{'aff':>12}{'community':>12}{'recon':>12}")
        for ds, r in results.items():
            vals = [sh[t] / r[t]['median'] if r[t]['median'] > 0 else float('inf')
                    for t in TERMS]
            print(f"{ds:<14}" + "".join(f"{_sig4(v):>12}" for v in vals))

    out = {"results": results, "failed": failed, "shares": SHARES,
           "real_train_epochs": REAL_TRAIN_EPOCHS}
    with open("d60_gradshares.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[D60] wrote d60_gradshares.json")
    if failed:
        print(f"[D60] {len(failed)} dataset(s) FAILED: {list(failed)}")
