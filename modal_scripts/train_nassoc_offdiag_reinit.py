"""OPTION 1: published nassoc diagonal + WORKING reinit, with and without a real off-diagonal.

THE TWO THINGS THIS TESTS
-------------------------
A) Does a WORKING reinit rescue the known spatial collapse of the published loss?
   decisions.md 1-3 recorded 15/200 survivors for published nassoc on s28nsc. Every run
   since then that carried `prie` in its folder name had reinit SILENTLY DISABLED ([D13]):
   the reinit path reads _c_k_ema/_n_k_ema, and both were only written inside the
   `lambda_proto_usage > 0` branch -- so with usage OFF they stayed None and the block
   never ran. Fixed in [D13]. This is the first run where reinit can actually fire.

B) Does the off-diagonal, given enough weight, fight the merge bias?
   The diagonal is containment = affinity_inside / volume. Merging two prototypes ALWAYS
   raises containment (the cross-affinity becomes internal), so the diagonal alone prefers
   fewer, larger groups -- that IS the collapse. The off-diagonal penalises exactly that
   cross-affinity, so in principle it is the brake.

WHY alpha = 2000 AND NOT 3 OR 10 -- MEASURED, NOT GUESSED
----------------------------------------------------------
Scored the two terms on a real K=200 partition (Leiden-modularity, res 19.431) on bk02:

    diagonal   (containment - 1)^2  mean over 200 entries    = 0.32164
    offdiag    (M_kj)^2             mean over 39800 entries  = 0.0000845
    ratio                                                    = 3806x

At the default alpha=1 the off-diagonal is 0.026% of the nassoc loss -- structurally
inert, because it averages over all K^2-K prototype pairs and nearly all of them share no
edges, so the mean is dominated by zeros. alpha=3 or 10 would change nothing.
offdiag_norm='min' (bills leakage to the SMALLER prototype, so a rare prototype leaking
into a big one registers) improves it only to 2124x.

    alpha = 2000   ~ parity between the two terms
    alpha =  200   ~ off-diagonal at 10% of the diagonal

A CAVEAT WORTH RECORDING, because it may make arm C worse rather than better
-----------------------------------------------------------------------------
On the SAME measurement the RANDOM partition scores BETTER on the off-diagonal
(2.52e-05) than the good one (8.45e-05). Mean-of-squares penalises CONCENTRATED
between-group affinity, and a correct spatial partition necessarily has a few strongly
coupled adjacent-niche pairs (max M_kj = 0.11 for Leiden vs 0.0073 for random). So a
heavy off-diagonal may push toward spreading boundary affinity rather than separating
better. Arms B and C measure which effect wins.

ARMS
----
    A  alpha=1     (published, unchanged)  offdiag_norm default   reinit ON
    C  alpha=2000  offdiag_norm='min'                             reinit ON

Both: published volume diagonal (the default), lambda_nassoc=1, usage OFF, latent 16,
K=200, bk02 -- matching the modularity sweep so the numbers are comparable.

REFERENCES TO BEAT (s28nsc / bk02, K=200, latent 16)
----------------------------------------------------
    modularity lambda=10   DGE 0.822   modularity 0.3087   dead   0   w_purity 0.8686
    modularity lambda=1    DGE 0.672   modularity 0.3893   dead  42   w_purity 0.8473
    scProto best (CPM)     DGE 0.842   int_frac 0.3338
    Leiden-modularity      DGE 0.856   int_frac 0.5147
    SEACells bk02          DGE 0.849   int_frac 0.4465

WATCH unused_proto AND the `[reinit] revived N` line -- if that line never appears, the
[D13] fix did not take and the run says nothing about reinit.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_nassoc_offdiag_reinit.py
Cost: ~13 min per arm on a T4, run in parallel.
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

ARMS = [
    # (name, nassoc_alpha, offdiag_norm)
    ("nao_A_alpha1_reinit",    1.0,    None),
    ("nao_C_alpha2000_min",    2000.0, "min"),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, alpha: float, offdiag_norm):
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
        "lambda_nassoc": 1.0,          # published weight
        "lambda_modularity": 0.0,      # no modularity term
        "lambda_proto_usage": 0.0,     # usage OFF -- the config where prototypes die
        "proto_reinit_every": 2,       # [D13] this can finally fire
        "proto_reinit_thresh": 0.01,
        "nassoc_alpha": alpha,
        "latent_dims": 16,         # match the modularity sweep (LD16) for comparability
    }
    if offdiag_norm is not None:
        lc["nassoc_offdiag_norm"] = offdiag_norm
    # published diagonal = nassoc_diag_norm 'volume', which is already the default --
    # assert we are NOT silently inheriting a CPM/pair variant from the preset.
    assert lc.get("nassoc_diag_norm", "volume") == "volume", lc.get("nassoc_diag_norm")
    assert lc.get("nassoc_diag_loss", "mse") in ("mse",), lc.get("nassoc_diag_loss")
    assert lc["lambda_proto_usage"] == 0.0 and lc["proto_reinit_every"] == 2

    print(f"\n{'='*78}\n=== {name}\n"
          f"    published volume diagonal, lambda_nassoc=1, alpha={alpha}, "
          f"offdiag_norm={offdiag_norm or 'default'}\n"
          f"    usage OFF, reinit every 2 epochs [D13], latent 16, K=200, bk02\n{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name})
    print(f"\n[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    results = list(run.starmap(ARMS))
    print("\n" + "=" * 96)
    print("OPTION 1 -- published nassoc diagonal + working reinit, s28nsc/bk02 K=200 LD16")
    print("=" * 96)
    print(f"{'arm':<26}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<26}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 96)
    print(f"{'modularity lambda=10':<26}{'--':>10}{'--':>11}{0.3087:>12.4f}{'--':>10}"
          "   dead 0, DGE 0.822")
    print(f"{'modularity lambda=1':<26}{'--':>10}{'--':>11}{0.3893:>12.4f}{'--':>10}"
          "   dead 42, DGE 0.672")
    print("\nCHECK: grep the log for '[reinit] revived' -- if absent, [D13] did not take.")
