"""[D59] Frozen full-pass gradnorm probe on the (type x batch)-stratified split,
with FIVE terms -- including proto_recon, which has never been measured.

Same protocol as D56 (measure_grad_shares_d60.py), three things different:
  1. the NEW split  pbmc-immune_arbf_ctb20_seed0  (stratified by cell type x batch)
  2. lambda_proto_recon = 1 as well, so grad_probe logs its gnorm
     (scproto.py:3467 already has 'proto_recon' in its _term_map)
  3. proto_recon_stopgrad = 0 -- the configuration we intend to DEPLOY, so the
     measured gradient includes the path through `scores` into the encoder.
     With the default True, `scores.detach()` means proto_recon can only move the
     prototypes and decoder, never the assignment -- which is why D46's
     proto_recon result was a null.

`init_attributes` (trainers/base.py:22-28) setattr's EVERY kwarg, registered in
defaults.py or not, so proto_recon_stopgrad reaches the trainer. It is NOT in
ABBREVIATIONS, so it will not appear in the run-folder name -- hence the explicit
experiment_name.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_ctb.py
"""
import json
import statistics as st

import modal

app = modal.App("scproto-ctb-probe")

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
        for name in ["lung", "lung_fetal_donor", "lung_fetal_organoid", "brain",
                     "breast", "heart", "eye", "gut_fetal", "skin", "COVID",
                     "pancreas"]
    ])
)

volume = modal.Volume.from_name("scproto-data")

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
TAG = "arbf_ctb20_seed0"
REAL_TRAIN_EPOCHS = 10
GRAD_PROBE = 5
WARM = 50
PROTO_DRIFT_TOL = 1e-4
TERMS = ("cohesion_gram", "cohesion_aff", "umap", "recon", "proto_recon")

# share vectors to report lambdas for, both a priori
SHARES = {
    # B: equal share per SUBSYSTEM, CVAE recon (the D58 base, on the new split)
    "B_recon": {"cohesion_gram": 1/6, "cohesion_aff": 1/6,
                "umap": 1/3, "recon": 1/3, "proto_recon": 0.0},
    # B with proto_recon REPLACING CVAE recon, same subsystem shares
    "B_protorecon": {"cohesion_gram": 1/6, "cohesion_aff": 1/6,
                     "umap": 1/3, "recon": 0.0, "proto_recon": 1/3},
}


def build_probe_config():
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
        "lambda_umap": 1.0, "lambda_recon": 1.0,
        "lambda_proto_recon": 1.0,          # <- measured for the first time
        "proto_recon_stopgrad": 0,          # <- deployment config; see docstring
        "proto_recon_hard": False,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "umap_proto_sg": 1, "grad_probe": GRAD_PROBE,
        "base_lr": 0.0,                     # freeze (c): optimizer BUILT with lr=0
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


def _summarize(dump_path, expected_steps):
    import os
    p = os.path.join(dump_path, "grad_probe.jsonl")
    assert os.path.exists(p), f"no grad_probe.jsonl at {p}"
    series = {t: [] for t in TERMS}
    errors = 0
    for line in open(p):
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            errors += 1
            continue
        for t in TERMS:
            if t in row and isinstance(row[t], dict):
                series[t].append(row[t]["gnorm_enc"])
    exp_rows = -(-expected_steps // GRAD_PROBE)
    n = len(series["cohesion_gram"])
    assert n >= max(10, int(0.9 * exp_rows)), (
        f"probe logged {n} rows, expected ~{exp_rows} -- not a full pass "
        f"(errors={errors})")
    out = {"n_rows": n, "expected_rows": exp_rows, "n_probe_errors": errors}
    for t in TERMS:
        g = series[t]
        if len(g) < 10:
            out[t] = None
            continue
        out[t] = {"median": st.median(g), "mean": st.mean(g),
                  "std": st.pstdev(g), "step0": g[0],
                  "median_first": st.median(g[:WARM]),
                  "median_last": st.median(g[-WARM:])}
    return out


@app.function(image=image, gpu="L4", timeout=14400, volumes={"/vol": volume})
def probe():
    import os
    _env()
    import torch
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)

    lc = build_probe_config()
    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    assert os.path.exists(train_h5), f"missing {train_h5}"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    t = get_trainer(
        experiment_name="ctbprobe", cvae_epochs=50, affinity_type="arbf",
        dataset_id=resolved, l2norm=1, assignment_metric="dotp",
        batch_size=1024, umap_steps_per_epoch=None, umap_similarity="proto",
        freeze_batch_embedding=1, freeze_decoder=0,
        num_prototypes=_resolve_num_prototypes(K, resolved),
        **{k: v for k, v in lc.items()
           if k not in ("umap_similarity", "affinity_type")},
    )
    ck = os.path.join(t.get_pretrain_dump_path(), "pretrain_checkpoint.pth")
    if os.path.exists(ck):
        t.load_pretrain_checkpoint()
    else:
        t.pretrain_encoder(); t.init_prototypes(); t.save_pretrain_checkpoint()
    if hasattr(t.model, "freeze_batch_embedding"):
        t.model.freeze_batch_embedding()
    assert getattr(t, "proto_recon_stopgrad", True) == 0, \
        "proto_recon_stopgrad did not reach the trainer"

    t._setup_umap_edges(REAL_TRAIN_EPOCHS)
    s = t._umap_state
    expected = len(s["loader"])
    opt = s["optimizer"]
    opt.step = lambda *a, **kw: None
    for g in opt.param_groups:
        g["lr"] = 0.0
    t.umap_checkpoint_freq = 0
    pp = os.path.join(t.get_dump_path(), "grad_probe.jsonl")
    if os.path.exists(pp):
        os.remove(pp)
    before = {nm: p.detach().clone() for nm, p in t.model.named_parameters()}
    print(f"   [ctb-probe] frozen: lrs={[g['lr'] for g in opt.param_groups]} "
          f"expected_steps={expected}")

    t.continue_train_umap_edges(epochs=1, early_stop=False)

    enc_moved, drift = [], {}
    for nm, p in t.model.named_parameters():
        if torch.equal(p.detach(), before[nm]):
            continue
        if "prototypes" in nm:
            drift[nm] = float((p.detach() - before[nm]).norm()
                              / before[nm].norm().clamp(min=1e-12))
        else:
            enc_moved.append(nm)
    assert not enc_moved, f"ENCODER WEIGHTS MOVED: {enc_moved[:5]}"
    worst = max(drift.values(), default=0.0)
    assert worst <= PROTO_DRIFT_TOL, f"prototypes drifted {worst:.3e}"
    print(f"   [ctb-probe] VERIFIED frozen; prototype drift {worst:.3e}")

    res = _summarize(t.get_dump_path(), expected)
    volume.commit()
    return res


def _sig4(x):
    from math import floor, log10
    if x == 0 or x != x:
        return "0"
    return f"%.{max(0, 3 - int(floor(log10(abs(x)))))}f" % x


@app.local_entrypoint()
def main():
    print(f"[D59] probe on {DS} / {TAG}, FIVE terms at lambda=1, frozen full pass")
    r = probe.remote()
    print(f"\nrows {r['n_rows']}/{r['expected_rows']}  probe errors {r['n_probe_errors']}")
    print(f"\n{'term':<16}{'median':>12}{'mean':>12}{'std':>12}{'CV':>7}"
          f"{'step0':>12}{'first50':>12}{'last50':>12}")
    for t in TERMS:
        d = r.get(t)
        if not d:
            print(f"{t:<16}{'NOT MEASURED':>12}")
            continue
        cv = d["std"] / d["mean"] if d["mean"] else 0
        print(f"{t:<16}{d['median']:>12.5g}{d['mean']:>12.5g}{d['std']:>12.5g}"
              f"{cv:>7.2f}{d['step0']:>12.5g}"
              f"{d['median_first']:>12.5g}{d['median_last']:>12.5g}")

    for name, sh in SHARES.items():
        print(f"\n### lambdas, share vector {name} (basis: full-pass MEDIAN)\n")
        for t in TERMS:
            if sh.get(t, 0.0) == 0.0 or not r.get(t):
                continue
            print(f"    {t:<16} share={sh[t]:.4f}  "
                  f"lambda = {_sig4(sh[t] / r[t]['median'])}")
    with open("files/tables/gradnorm_calibration/ctb_gradshares.json", "w") as f:
        json.dump({"result": r, "shares": SHARES, "tag": TAG}, f, indent=2)
    print("\n[D59] wrote files/tables/gradnorm_calibration/ctb_gradshares.json")
