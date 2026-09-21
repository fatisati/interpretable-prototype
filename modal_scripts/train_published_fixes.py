"""PUBLISHED NeurIPS loss on s28nsc/bk02, plus the two corrections, on one comparable footing.

WHY pub_A EXISTS -- a verification that should have come first
--------------------------------------------------------------
The "31 of 200 prototypes live" collapse quoted throughout this session came from `nao_A`,
which ran with `lambda_proto_usage = 0.0`. That is the published loss with its
ANTI-COLLAPSE TERM REMOVED, plus reinit on and latent 16. The actual published
configuration has never been run on bk02 here. Before any diagnosis of "the published loss
collapses on spatial graphs" can stand, the real thing has to be measured.

    published = LAMBDA_PROTO_UMAP_PRECON + defaults:
        lambda_umap 1, lambda_proto_recon 0.01, lambda_nassoc 1
        lambda_proto_usage 0.1  (USAGE ON)        prot_init 'waypoint'
        nassoc_diag True, nassoc_diag_norm 'volume'
        nassoc_offdiag_norm 'sqrt', nassoc_offdiag_agg 'pair', nassoc_alpha 1.0
        no reinit

THE TWO CORRECTIONS
-------------------
OFF-DIAGONAL (pub_B). leakage = affinity_between / min(vol_k, vol_j), hinged at the
measured p95 of a reference Leiden partition's leakages (0.04698 under min(vol) -- NOT the
0.0343 measured under sqrt), then averaged over LIVE prototypes only ([D24] 'hnlv').
Measured on bk02: overlap-pathology detection 4.32x and K-transfer 0.93 (K=200 -> 573),
against 1.28x and 0.10 for the published mean-over-all-pairs, whose weight therefore drifts
10x and never transfers. Under min(vol) the dead-slot dilution seen with 'sqrt' (65x)
disappears -- killing prototypes RAISES the term instead. alpha = 30, roughly a third of
the measured parity value of 86.6, because at parity the off-diagonal ranks partition
quality backwards.

DIAGONAL (pub_C). The containment diagonal (affinity_within / volume) is replaced by the
cohesion form with sim[i][j] = aff[i][j], i.e. ratio association. Containment's merge
arithmetic is unconditional: volume[k] ALREADY contains the between-affinity, so merging
moves that weight into the numerator while the denominator merely adds -- the ratio rises
for any data. A cell COUNT contains no affinity, so merging has to earn the cells it adds.
cohesion_form='pcex' [D23] additionally excludes cell c from its own divisor, which is a
0.3% correction at 290 cells and a 50% correction at 2 -- it stops small prototypes, i.e.
the rare states, being systematically under-scored.

B, C, D: usage OFF, reinit ON. latent 16 throughout so all four are directly comparable
(the paper used latent 8; that is a deliberate deviation for comparability with this
session's other runs).

READ K_live AND median size FIRST. A run with K_live 192 and median 15 is worse than one
with K_live 170 and median 260 -- median metacell size, not the live count, is what tracks
DGE.

REFERENCES (s28nsc/bk02, K=200, LD16)
    modularity l=10            DGE 0.822  purity 0.8686  median 288  K_live 200
    modularity l=10 + cohesion DGE 0.803  purity 0.8710  median 292  K_live 200  15 states
    SEACells bk02              DGE 0.849                 median 258  K_live 200  19 states

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_published_fixes.py
Cost: ~13 min per arm on a T4, 4 arms in parallel, about $0.52.
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
        "ipython", "leidenalg", "python-igraph",
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

HINGE_THRESH = 0.04698   # p95 of a reference K=200 Leiden partition's leakages, min(vol)

#          (name,              offdiag_alpha, lambda_cohesion, lambda_usage)
ARMS = [
    ("pub_A_published",           1.0,  0.0, 0.1),
    ("pub_B_offdiag",            30.0,  0.0, 0.0),
    ("pub_C_diagonal",            1.0,  5.0, 0.0),
    ("pub_D_both",               30.0,  5.0, 0.0),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, alpha: float, lam_coh: float, lam_usage: float):
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

    corrected_od = alpha > 1.0
    corrected_dg = lam_coh > 0.0

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 1.0,
        "nassoc_alpha": alpha,
        "nassoc_diag": (not corrected_dg),
        "nassoc_offdiag_norm": ("min" if corrected_od else "sqrt"),
        "nassoc_offdiag_agg": ("hnlv" if corrected_od else "pair"),
        "nassoc_offdiag_thresh": (HINGE_THRESH if corrected_od else 0.0),
        "lambda_cohesion": lam_coh,
        "cohesion_sim": "aff",
        "cohesion_form": "pcex",
        "cohesion_self": 0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": lam_usage,
        "proto_reinit_every": (0 if lam_usage > 0 else 2),
        "proto_reinit_rel": 0.1,
        "latent_dims": 16,
    }
    assert lc["lambda_nassoc"] == 1.0 and lc["lambda_modularity"] == 0.0
    if name.endswith("published"):
        assert lc["nassoc_diag"] and lc["nassoc_offdiag_norm"] == "sqrt"
        assert lc["lambda_proto_usage"] == 0.1 and lc["proto_reinit_every"] == 0

    print(f"\n{'='*78}\n=== {name}\n"
          f"    off-diagonal: alpha={alpha} "
          f"{'CORRECTED min/hnlv thr=' + str(HINGE_THRESH) if corrected_od else 'published sqrt/pair'}\n"
          f"    diagonal:     {'cohesion sim=aff pcex (ratio assoc)' if corrected_dg else 'published containment'}\n"
          f"    usage={lam_usage}  reinit={'OFF' if lam_usage > 0 else 'ON'}  latent 16  K=200  bk02\n"
          f"    WATCH K_live AND median size -- K_live alone hides a sliver collapse\n{'='*78}")

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
    print("PUBLISHED LOSS + CORRECTIONS -- s28nsc/bk02 K=200 LD16")
    print("=" * 96)
    print(f"{'arm':<22}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<22}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 96)
    print(f"{'modularity l=10':<22}{'--':>10}{'--':>11}{0.3087:>12.4f}{'--':>10}"
          "   DGE 0.822, median 288")
    print(f"{'modu + cohesion':<22}{'--':>10}{'--':>11}{0.3063:>12.4f}{'--':>10}"
          "   DGE 0.803, median 293, 15 states")
    print("\nK_live and median size must be read from purity_per_mc.csv / size_per_mc.csv.")
