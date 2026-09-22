"""IS THE COMMUNITY LOSS NEEDED AT ALL?  -- the ablation never run.

FOR keeping it: it is the ONLY term with explicit REPULSION (-log(1-q_neg) on random
pairs). Cohesion is pure attraction; nothing in it separates unrelated cells.

AGAINST: the Leiden warm start showed the community loss ALONE degrading a good partition
(median 312, max/median 2.6) to median 133, max/median 45.2, minimum size 1. It is the
only term measured actively DESTROYING a good partition. It also samples EDGES, so
high-degree cells are over-represented and their prototypes absorb mass -- the
rich-get-richer mechanism behind the size skew.

REDUNDANCY: both act on <assign_i, assign_j>. The community loss applies -log (saturating,
dominated by whichever individual pairs are worst); cohesion applies an affinity-weighted
linear form aggregated per prototype (never saturates). So cohesion largely subsumes the
community loss's POSITIVE half. The genuinely unique part is the negatives.

Both arms: cohesion (sim=aff, pcex, self-excluded) as the ONLY global term. No nassoc, no
modularity, no usage loss, reinit ON. The single difference is lambda_umap.

WHAT EACH OUTCOME MEANS
    B collapses, A does not   -> the repulsion is load-bearing, keep the community loss
    B holds or beats A        -> cohesion subsumes it, and we can drop the one term
                                 measured to destroy good partitions

READ median size, not K_live. A run with K_live 192 and median 15 is worse than one with
K_live 170 and median 260.

REFERENCES (s28nsc/bk02, K=200, LD16)
    modularity l=10 + cohesion   DGE 0.803  median 292  K_live 200
    modularity l=10              DGE 0.822  median 288  K_live 200
    published                    DGE 0.406  median   5  K_live 190
    corrected diagonal (pub_C)   DGE 0.643  median 156  K_live 189
    SEACells                     DGE 0.849  median 258  K_live 200

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_community_ablation.py
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

#          (name,                 lambda_umap, lambda_cohesion)
ARMS = [
    ("abl_A_community_ON",   1.0, 5.0),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, lam_umap: float, lam_coh: float):
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
        "lambda_umap": lam_umap,          # THE ABLATION
        "lambda_nassoc": 0.0,             # no diagonal, no off-diagonal
        "lambda_modularity": 0.0,         # no size control at all
        "lambda_cohesion": lam_coh,       # the only global term
        "cohesion_sim": "aff",
        "cohesion_form": "pcex",
        "cohesion_self": 0,
        "lambda_proto_usage": 0.0,
        "proto_reinit_every": 2,
        "proto_reinit_rel": 0.1,
        "latent_dims": 16,
    }
    assert lc["lambda_nassoc"] == 0.0 and lc["lambda_modularity"] == 0.0
    assert lc["lambda_cohesion"] > 0 and lc["lambda_proto_usage"] == 0.0

    banner = "=" * 78
    print("\n" + banner)
    print(f"=== {name}")
    print(f"    community loss lambda_umap={lam_umap}   cohesion lambda={lam_coh} (sim=aff, pcex)")
    print("    NO nassoc, NO modularity, usage OFF, reinit ON, latent 16, K=200, bk02")
    print("    WATCH median size, not K_live")
    print(banner)

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
    results = [run.remote(*ARMS[0])]
    print("\n" + "=" * 96)
    print("IS THE COMMUNITY LOSS NEEDED?  -- s28nsc/bk02 K=200 LD16, cohesion only")
    print("=" * 96)
    print(f"{'arm':<24}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<24}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 96)
    print(f"{'modularity + cohesion':<24}{'--':>10}{'--':>11}{0.3063:>12.4f}{'--':>10}"
          "   DGE 0.803, median 293")
    print("\nmedian size must be read from size_per_mc.csv -- K_live alone hides slivers.")
