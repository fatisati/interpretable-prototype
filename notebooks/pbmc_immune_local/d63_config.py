"""The D63/D64 recipe, copied from modal_scripts/train_d64_nb.py so the local run
differs from Modal ONLY in the measured gradient norms (re-measured here)."""
import json, os
from _local_env import ROOT

SHARE_ENC = {"kernel": 1 / 8, "recon": 1 / 4, "gmkl": 1 / 4}
SHARE_PROTO = {"gmkl": 0.5, "proto_recon": 0.5}
GRAD_PROBE = 5
GNORM_JSON = os.path.join(ROOT, 'notebooks', 'pbmc_immune_local', 'tables', 'gradshares_local.json')

# Modal-measured values (train_d64_nb.py) -- kept only to compare against the local probe
GNORM_MODAL = {"mse": {"enc": {"cohesion_gram": 0.043067, "cohesion_aff": 0.02464,
                               "recon": 11.834, "gmkl": 11.223},
                       "proto": {"gmkl": 0.055115, "proto_recon": 5.6654}}}


def base_recipe(lik="mse"):
    """Routing + fixed options shared by probe and arms (lambdas filled separately)."""
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "recon_loss": lik,
        "proto_recon_form": "mix3", "proto_recon_dec_sg": 0, "proto_recon_stopgrad": True,
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "grad_probe": GRAD_PROBE, "seed": 31,
    }


def probe_config(lik="mse"):
    return base_recipe(lik) | {"lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
                               "lambda_umap": 1.0, "lambda_recon": 1.0,
                               "lambda_proto_recon": 1.0, "lambda_gmkl": 1.0,
                               "base_lr": 0.0}


def arm_config(prot_init, recon_on, lik="mse", gnorm=None):
    if gnorm is None:
        gnorm = json.load(open(GNORM_JSON))[lik]
    ge, gp = gnorm["enc"], gnorm["proto"]
    lam_gmkl = round(SHARE_ENC["gmkl"] / ge["gmkl"], 6)
    lam_pr = round((SHARE_PROTO["proto_recon"] / SHARE_PROTO["gmkl"])
                   * lam_gmkl * gp["gmkl"] / gp["proto_recon"], 8)
    return base_recipe(lik) | {
        "prot_init": prot_init,
        "lambda_cohesion": round(SHARE_ENC["kernel"] / ge["cohesion_gram"], 6),
        "lambda_cohesion_aff": round(SHARE_ENC["kernel"] / ge["cohesion_aff"], 6),
        "lambda_recon": round(SHARE_ENC["recon"] / ge["recon"], 6) if recon_on else 0.0,
        "lambda_gmkl": lam_gmkl, "lambda_proto_recon": lam_pr, "lambda_umap": 0.0,
    }


def validate(lc, recon_on, lik="mse"):
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_umap"):
        assert float(lc[k]) == 0.0, f"{k} must be 0"
    assert lc["coh_proto_sg"] == 1 and lc["umap_proto_sg"] == 1
    assert lc["proto_recon_form"] == "mix3" and lc["proto_recon_stopgrad"] is True
    assert lc["proto_recon_dec_sg"] == 0 and lc["recon_loss"] == lik
    assert (float(lc["lambda_recon"]) > 0) == recon_on
    assert float(lc["lambda_gmkl"]) > 0 and float(lc["lambda_proto_recon"]) > 0
    assert lc["cohesion_self"] == 1.0 and lc["proto_reinit_every"] == 0
