"""[D39-gram] Same direction test, GRAM (2-hop) similarity instead of aff.

THE QUESTION
------------
[D36] measured cohesion's gradient as alive (norm 4e-5..2e-4, never
disconnected) yet true full-graph cohesion FELL during training, 0.0076 ->
0.0044, while unused_proto rose 0 -> 151. Two explanations remain and they
need opposite fixes:

  (a) direction RIGHT, strength too low -- cohesion is outvoted by the other
      loss terms. Fix: raise its weight (and [D37] already did, 8.6x, by
      correcting the normalizer).
  (b) direction WRONG -- cohesion's own gradient degrades the partition.
      Then no weight helps and the formulation itself must change.

Arm A settles it: run cohesion ALONE, every other term zeroed. Nothing can
outvote it. If cohesion still falls, the direction is wrong -- (b). If it
rises, the direction was fine all along -- (a).

Arm B then asks the follow-up you actually care about: does adding the
community (UMAP) term back help or fight it?

REFERENCE VALUES, measured on this exact graph (s28nsc/bk02, K=200,
scratchpad/metric_sanity.py, all analytic checks exact):

    one-group        coh_aff 0.0005   joint_purity 0.1002
    random-200       coh_aff 0.0039   joint_purity 0.1069   <- FLOOR
    celltype (K=18)  coh_aff 0.0058   joint_purity 0.4252
    ct x niche (175) coh_aff 0.0152   joint_purity 1.0000   <- TARGET
    singletons       coh_aff 1.0000   joint_purity 1.0000   (degenerate)

    prior run START  coh_aff 0.0076   (halfway to the target)
    prior run END    coh_aff 0.0044   (nearly back to random)

Note coh_aff = K/N + (1/N)*sum_k within[k]/size[k], and at K=200 the first
term is a CONSTANT 0.00342 every partition gets free. So the informative
part is what exceeds it:  random 0.0005, prior-start 0.0042, target 0.0122.

ARMS
----
  d39_A_cohonly   lambda_cohesion_aff=44.6, EVERYTHING else 0
                  (umap 0, proto_recon 0, nassoc 0, usage 0, modularity 0)
  d39_B_cohcomm   the same + community: lambda_umap=1, lambda_proto_recon=0.01

Both: cohesion_self=1.0 (the plain mean-similarity formula, self-loop 1),
cohesion_diag=1.0, cohesion_global=2 with [D32] full init + [D31] hlife 500,
[D35] per-step prototype renorm, [D37] fixed normalizer, NO reinit, NO early
stop, 10 fixed epochs.

WHAT TO WATCH (per epoch, now that [D38] returns them):
    coh_aff      -- must RISE toward 0.0152, must not sink to 0.0039
    joint_purity -- (cell type, niche) purity, the real criterion
    n_dead       -- prototypes no cell chose
  plus the [D36 DEBUG] lines every 25 steps: true full-graph coh_aff and the
  live gradient norm.

VERDICT (fill after reading):
  A: [ ] cohesion rises alone -> direction RIGHT, it was outvoted
     [ ] cohesion still falls alone -> direction WRONG, reformulate
  B: [ ] community helps      [ ] community fights cohesion   [ ] neutral

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d39_direction.py
Cost: 2 arms x 10 epochs on a T4, in parallel, ~15 min each.
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

# GRAM span 0.076776 (d30 header, measured full-graph hard assignments WITH
# sim[c][c]=1) -> 1.5403/0.076776 = 20.1. Same authority as the aff arms' 44.6,
# just a different scale because the gram similarity spans a wider range.
LAM_GRAM = 20.1

# (name, lambda_umap, lambda_proto_recon)
ARMS = [
    ("d39_C_gramonly", 0.0, 0.0),     # gram cohesion ALONE -- direction test
    ("d39_D_gramcomm", 1.0, 0.01),    # gram cohesion + community
]


@app.function(image=image, gpu="T4", timeout=5400, volumes={"/vol": volume})
def run(name: str, lam_umap: float, lam_proto_recon: float):
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
        # --- everything that is not cohesion / community ---
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": 0.0,
        # --- the two knobs that define the arm ---
        "lambda_umap": lam_umap,
        "lambda_proto_recon": lam_proto_recon,
        # --- cohesion: the plain formula ---
        "lambda_cohesion": LAM_GRAM,      # drives the GRAM path
        "lambda_cohesion_aff": 0.0,
        "cohesion_sim": "gram",           # 2-hop: cos of affinity rows
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,       # sim(c,c) = 1
        "cohesion_self": 1.0,       # self counts like any other member
        "cohesion_global": 2,       # [D29] remembered full-graph profiles
        "coh_mem_init": "full",     # [D32]
        "coh_mem_hlife": 500.0,     # [D31]
        "proto_reinit_every": 0,    # NO reinit
        "coh_diag_every": 25,       # [D36] probe cadence
        "latent_dims": 16,
    }
    assert lc["lambda_cohesion"] == 20.1 and lc["lambda_cohesion_aff"] == 0.0
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    if name.startswith("d39_C"):
        assert lc["lambda_umap"] == 0.0 and lc["lambda_proto_recon"] == 0.0, \
            "arm A must have NOTHING but cohesion"

    print(f"\n{'='*78}\n=== {name}\n"
          f"    lambda_cohesion(GRAM)={LAM_GRAM}  lambda_umap={lam_umap}  "
          f"lambda_proto_recon={lam_proto_recon}\n"
          f"    cohesion_self=1.0, diag=1.0, D35 renorm, D37 normalizer fix\n"
          f"    NO reinit, NO early stop, 10 fixed epochs\n"
          f"    TARGETS: coh_gram random 0.0048 | celltype 0.0157 | ct-niche 0.0307\n"
          f"             joint_purity  random 0.1069 | ct-niche 1.0000\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=10, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False)
    print(f"\n[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    results = list(run.starmap(ARMS))
    print("\n" + "=" * 96)
    print("[D39] direction test -- the metrics that matter")
    print("=" * 96)
    print(f"{'arm':<18}{'coh_aff':>10}{'joint_purity':>14}{'n_dead':>9}")
    for name, m in results:
        def g(k, f="{:.4f}"):
            v = m.get(k)
            return f.format(v) if isinstance(v, (int, float)) else "--"
        print(f"{name:<18}{g('coh_aff'):>10}{g('joint_purity'):>14}"
              f"{g('n_dead', '{:.0f}'):>9}")
    print("-" * 96)
    print(f"{'random-200':<16}{0.0039:>10.4f}{0.1069:>14.4f}{'--':>9}  <- FLOOR")
    print(f"{'prior run START':<16}{0.0076:>10.4f}{'--':>14}{'--':>9}")
    print(f"{'ct x niche':<16}{0.0152:>10.4f}{1.0000:>14.4f}{'--':>9}  <- TARGET")
    print("\nArm A above 0.0076 -> direction RIGHT (it was being outvoted).")
    print("Arm A at/below 0.0039 -> direction WRONG, the formulation needs work.")
