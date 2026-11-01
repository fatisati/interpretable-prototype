"""[D65] Encoder-path gradient-norm probe for the CROSS-BATCH SIGNATURE terms.

Measures gnorm_enc at a TEMPORARY lambda=1 for every active term, so each
deployed lambda can be solved as  lambda_i = share_i / gnorm_i  (CLAUDE.md
9d-1). Runs for one likelihood at a time: --lik mse | nb.

WHAT IS NEW HERE vs the D63/D64 probes
  lambda_xbatch      = 1.0   diagonal pull  (same prototype, across batches)
  lambda_xbatch_off  = 1.0   off-diagonal push (different prototypes)
both never measured before, so their deployed lambdas cannot be guessed.

WHY lambda=1 AND NOT THE DEPLOYED VALUE
CLAUDE.md 9d-1 records the trap directly: pre-2026-09-26 probe files measured
cohesion at its REAL authority/span lambda rather than at 1, so an old file's
cohesion gnorm_enc has to be DIVIDED by its reported lambda before it is
comparable. Conflating the two is how s28nsc's recon lambda ended up ~11x too
high. This probe forces EVERY term to lambda=1 for a direct reading.

ROUTING ASSERTION (the [D63] lesson turned into a guard)
With xbatch_proto_sg=1 the cross-batch terms must put EXACTLY ZERO gradient on
the prototypes, so there is only ONE budget to calibrate. [D63] measured what
the alternative costs: proto_recon's lambda was set from the encoder gradient,
which is structurally zero for a detached-score term, so its prototype-path
pull ran 5.7-18.6x GM-KL's and collapsed the prototypes (median nearest-
prototype cosine 0.81 -> 0.9997). This probe FAILS if that routing is wrong.

The pass is frozen: optimizer lr=0 and opt.step() no-oped, then every parameter
is compared before/after. proto_recon is OFF here -- [D63] closed it at
lambda_proto_recon = 0 after 5 doses spanning 100x.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run \
       modal_scripts/measure_grad_shares_d65.py --lik mse
"""
import json
import os
import statistics as st

import modal

app = modal.App("scproto-d65-probe")

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
TERMS = ("cohesion_gram", "cohesion_aff", "recon", "gmkl",
         "xbatch", "xbatch_off")

# [D65] the deployed share vectors (user decision 2026-11-01: cross-batch takes
# 1/8, the other four keep their 1:1:2:2 ratio and are renormalized to sum to 1)
SHARES = {
    "D65_diag": {"cohesion_gram": 0.1458, "cohesion_aff": 0.1458,
                 "recon": 0.2917, "gmkl": 0.2917, "xbatch": 0.1250},
    # arm BOTH holds the diagonal CONSTANT and adds the off-diagonal at its own
    # share, so DIAG vs BOTH isolates the off-diagonal's effect. The cost is
    # that BOTH carries more total new force -- exactly the confound [D62]
    # called out -- and that is recorded in the ADR rather than hidden.
    "D65_both": {"cohesion_gram": 0.1296, "cohesion_aff": 0.1296,
                 "recon": 0.2593, "gmkl": 0.2593,
                 "xbatch": 0.1111, "xbatch_off": 0.1111},
}


def build_probe_config(lik):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
        "lambda_recon": 1.0, "lambda_gmkl": 1.0,
        # the two new terms, measured for the first time
        "lambda_xbatch": 1.0, "lambda_xbatch_off": 1.0,
        "xbatch_margin": 0.5,
        "xbatch_mem_hlife": 5.0,     # derived: 33506 cells / ~6700 per step
        "xbatch_mem_mix": 0.5,
        "xbatch_proto_sg": 1,        # encoder-only; ASSERTED below
        "xbatch_min_size": 1.0,
        "xbatch_weight_sg": 1,
        # [D63] CLOSED at 0 after 5 doses spanning 100x, both forms, both
        # routings, recon on and off. Not revived here.
        "lambda_proto_recon": 0.0,
        "lambda_umap": 0.0,          # community off ([D60] d60_leid_noc recipe)
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "prot_init": "leid", "grad_probe": GRAD_PROBE,
        "recon_loss": lik,
        "base_lr": 0.0,              # optimizer BUILT with lr=0
    }


def _stage_data_locally():
    """[D64] h5py does NON-SEQUENTIAL seeks, which a network-backed Modal volume
    serves unreliably for a large file ('wrong B-tree signature' on key X, even
    though the file is byte-identical by md5). Copy to container-local disk and
    read from there -- the same fix CLAUDE.md records for Drive-streamed paths."""
    import shutil
    src, dst = "/vol/data", "/tmp/data"
    need = [
        f"node_heldout/{DS}_{TAG}_train.h5ad",
        f"node_heldout/{DS}_{TAG}_train_label_encoder.pkl",
        f"node_heldout/{DS}_{TAG}_split.npz",
        "scpoli/Immune_ALL_human_hvg.h5ad",
    ]
    for rel in need:
        s_, d_ = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if not os.path.exists(d_) or os.path.getsize(d_) != os.path.getsize(s_):
            shutil.copyfile(s_, d_)
        assert os.path.getsize(d_) == os.path.getsize(s_), f"short copy: {rel}"
    import h5py
    with h5py.File(os.path.join(dst, need[0]), "r") as f:
        _ = f["X"]["data"][:1000]
    print(f"   [D65] staged {len(need)} files to {dst}; h5py data read OK")
    return dst + "/"


def _env():
    import sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = _stage_data_locally()
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
    p = os.path.join(dump_path, "grad_probe.jsonl")
    assert os.path.exists(p), f"no grad_probe.jsonl at {p}"
    series = {t: [] for t in TERMS}
    values = {t: [] for t in TERMS}
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
                values[t].append(row[t]["value"])
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
                  "median_last": st.median(g[-WARM:]),
                  "value_median": st.median(values[t])}
    return out


@app.function(image=image, gpu="T4", timeout=14400, volumes={"/vol": volume})
def probe(lik: str = "mse"):
    _env()
    import torch
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)

    # [D61] STALE-COPY GUARD: assert the new code is actually on this container
    # before anything else. D61's six arms (~$6 of GPU) silently trained the
    # PREVIOUS scproto.py because `modal volume put` with a leading-/ remote
    # path uploaded nowhere, and the whole ADR had to be invalidated.
    import inspect
    from interpretable_ssl.trainers import scproto as _sp
    _src = inspect.getsource(_sp)
    assert "[D65] CROSS-BATCH SIGNATURE ALIGNMENT" in _src, (
        "STALE scproto.py on this container -- the D65 term is not in the code "
        "that is about to run. Re-upload with a RELATIVE remote path and verify "
        "by md5 (CLAUDE.md 11).")

    lc = build_probe_config(lik)
    train_h5 = os.path.join(os.environ["DATA_DIR"],
                            f"node_heldout/{DS}_{TAG}_train.h5ad")
    assert os.path.exists(train_h5), f"missing {train_h5}"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    t = get_trainer(
        experiment_name=f"d65probe_{lik}", cvae_epochs=50, affinity_type="arbf",
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

    # [CLAUDE.md 9/11] the config must reach the RUN, not just the attribute
    for key, want in (("lambda_xbatch", 1.0), ("lambda_xbatch_off", 1.0),
                      ("coh_proto_sg", 1), ("xbatch_proto_sg", 1),
                      ("lambda_gmkl", 1.0)):
        got = getattr(t, key, None)
        assert got == want, f"{key} did not reach trainer: {got!r} != {want!r}"
    assert getattr(t, "recon_loss", "mse") == lik
    _mn = t.get_model_name()
    for tok in ("lxb", "lxbof"):
        assert tok in _mn, f"{tok} missing from model name: {_mn}"
    print(f"   [D65] model folder: {_mn}")

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
    print(f"   [D65 probe] frozen: lrs={[g['lr'] for g in opt.param_groups]} "
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
    print(f"   [D65 probe] VERIFIED frozen; prototype drift {worst:.3e}")

    res = _summarize(t.get_dump_path(), expected)
    res["lik"] = lik

    # ---- ROUTING: with every *_proto_sg=1, ONLY gmkl may touch the prototypes
    gp = {t_: [] for t_ in TERMS}
    with open(os.path.join(t.get_dump_path(), "grad_probe.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            for t_ in TERMS:
                if t_ in row and isinstance(row[t_], dict):
                    gp[t_].append(row[t_]["gnorm_proto"])
    routing = {t_: st.median(v) for t_, v in gp.items() if v}
    print(f"   [D65 ROUTING] median gnorm_proto: "
          f"{ {k: float(f'{v:.3e}') for k, v in routing.items()} }")
    for t_ in ("cohesion_gram", "cohesion_aff", "xbatch", "xbatch_off"):
        assert routing.get(t_, 1.0) == 0.0, (
            f"{t_} leaks gradient into the prototypes (gnorm_proto="
            f"{routing.get(t_)}). With xbatch_proto_sg=1 it must be exactly 0, "
            f"or there are TWO budgets to calibrate and [D63] repeats.")
    assert routing.get("gmkl", 0) > 0.0, "gmkl puts NO gradient on protos!"
    res["routing_gnorm_proto"] = routing
    volume.commit()
    return res


def _sig4(x):
    from math import floor, log10
    if x == 0 or x != x:
        return "0"
    return f"{round(x, -int(floor(log10(abs(x)))) + 3):g}"


@app.local_entrypoint()
def main(lik: str = "mse"):
    assert lik in ("mse", "nb"), lik
    print(f"[D65] encoder-path gradient probe, recon_loss={lik}, "
          f"ALL terms at temporary lambda=1")
    res = probe.remote(lik)

    print(f"\n[D65] {res['n_rows']}/{res['expected_rows']} rows, "
          f"{res['n_probe_errors']} probe errors")
    print(f"\n{'term':<16}{'median gnorm':>14}{'CV':>8}{'first':>12}"
          f"{'last':>12}{'value':>12}")
    for t in TERMS:
        r = res.get(t)
        if not r:
            print(f"{t:<16}{'-- not logged --':>14}")
            continue
        cv = r["std"] / r["median"] if r["median"] else float("nan")
        print(f"{t:<16}{r['median']:>14.6g}{cv:>8.2f}"
              f"{r['median_first']:>12.4g}{r['median_last']:>12.4g}"
              f"{r['value_median']:>12.4g}")

    print("\n[D65] DEPLOYED LAMBDAS  (lambda = share / median gnorm_enc, 9d-1)")
    out = {"lik": lik, "probe": res, "lambdas": {}}
    for name, sh in SHARES.items():
        print(f"\n  {name}")
        lam = {}
        for t_, s_ in sh.items():
            r = res.get(t_)
            if not r or s_ == 0:
                continue
            lam[t_] = round(s_ / r["median"], 8)
            print(f"    {t_:<16} share={s_:.4f}  ->  lambda={lam[t_]:.8g}")
        out["lambdas"][name] = lam
        # realized shares, as a check that the solve is self-consistent
        tot = sum(lam[t_] * res[t_]["median"] for t_ in lam)
        chk = {t_: round(100 * lam[t_] * res[t_]["median"] / tot, 1) for t_ in lam}
        print(f"    realized shares (%): {chk}")

    outdir = os.path.join("files", "tables", "d65_xbatch")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"d65_gnorm_{lik}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[D65] wrote {p}")
    print("[D65] paste these medians into GNORM_ENC in "
          "modal_scripts/train_d65_xbatch.py")
