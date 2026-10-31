"""Local reproduction of one D63/D64 arm (default: d63_mse-rec = MSE, CVAE recon ON, leiden init).
Saves the epoch-0 (post-init, pre-training) labels as the same-init baseline, then trains
TRAIN_EPOCHS epochs with early_stop=False, places held-out cells with the frozen encoder,
and writes clusters_full.npz / latent / prototypes / RECIPE.md like modal_scripts/train_d64_nb.py.

env: ARM=d63_mse-rec|d63_mse-norec   EPOCHS=10   USE_MPS=1 (optional)
     RUN_TAG=<name>  output folder + experiment name (default ARM); use it for a device/seed replicate
                     so it never overwrites another run of the same arm
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
from d63_config import arm_config, validate, GNORM_JSON
import numpy as np, pandas as pd, torch
from interpretable_ssl.utils import get_device
from interpretable_ssl.experiments.tasks import get_trainer, _resolve_ds_id, _resolve_num_prototypes

ARM = os.environ.get("ARM", "d63_mse-rec")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
RUN_TAG = os.environ.get("RUN_TAG", ARM)
LIK = "nb" if "nb" in ARM else "mse"
recon_on = not ARM.endswith("norec")
prot_init = "waypoint" if "wayp" in ARM else "leid"
OUT = os.path.join(ROOT, "notebooks", "pbmc_immune_local", "tables", RUN_TAG)
os.makedirs(OUT, exist_ok=True)

lc = arm_config(prot_init, recon_on, LIK)
validate(lc, recon_on, LIK)
print(f"[{ARM}] device={get_device()} epochs={EPOCHS} lambdas:",
      {k: lc[k] for k in ("lambda_cohesion", "lambda_cohesion_aff", "lambda_recon", "lambda_gmkl", "lambda_proto_recon")})

resolved = _resolve_ds_id(TRAIN_H5, batch_key=BATCH_KEY, label_key=LABEL_KEY, niche_key=None, num_prototypes=K)
t = get_trainer(experiment_name=RUN_TAG + "_local", cvae_epochs=50, affinity_type="arbf", dataset_id=resolved,
                l2norm=1, assignment_metric="dotp", batch_size=1024, umap_steps_per_epoch=None,
                umap_similarity="proto", freeze_batch_embedding=1, freeze_decoder=0,
                num_prototypes=_resolve_num_prototypes(K, resolved),
                **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")})
ck = os.path.join(t.get_pretrain_dump_path(), "pretrain_checkpoint.pth")
assert os.path.exists(ck), f"no pretrain checkpoint at {ck} -- run probe_gradnorms.py first"
t.load_pretrain_checkpoint()
t.model.freeze_batch_embedding()
mn = t.get_model_name(); print(f"[{ARM}] model folder = {mn}")
assert "prfm-mix3" in mn and "frzbe1" in mn

# ---- full data + held-out ids, for placement ----
from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
ad_full = get_preprocessed_adata(DS)
d = np.load(SPLIT_NPZ, allow_pickle=True)
test_ids = [str(x) for x in d["test_ids"]]
ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
assert t.check_conditions_compatible(t.model, ad_test)
test_mask = np.zeros(len(ad_full), bool); test_mask[ad_full.obs_names.get_indexer(test_ids)] = True
tr_pos = ad_full.obs_names.get_indexer(list(t.train_ds.adata.obs_names))


def place_all(tag):
    """train assignments (current prototypes) + frozen-encoder placement of held-out cells."""
    with torch.no_grad():
        t.model.eval()
        z_te = t.encode_adata(ad_test, t.model, z_idx=1)
        z_te = z_te.cpu().numpy() if hasattr(z_te, "cpu") else np.asarray(z_te)
        a_te = t.model.prototypes(torch.as_tensor(z_te, device=t.device)).argmax(dim=1).cpu().numpy()
        z_tr = t.encode_adata(t.train_ds.adata, t.model, z_idx=1)
        z_tr = z_tr.cpu().numpy() if hasattr(z_tr, "cpu") else np.asarray(z_tr)
        a_tr = t.model.prototypes(torch.as_tensor(z_tr, device=t.device)).argmax(dim=1).cpu().numpy()
    t.model.train()
    lab = np.zeros(len(ad_full), int); lab[tr_pos] = a_tr; lab[test_mask] = a_te
    zf = np.zeros((len(ad_full), z_tr.shape[1]), np.float32); zf[tr_pos] = z_tr; zf[test_mask] = z_te
    P = t.model.get_prototypes().detach().cpu().numpy()
    od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, "clusters_full.npz"), labels=lab, test_mask=test_mask)
    np.save(os.path.join(od, "latent_z_full.npy"), zf); np.save(os.path.join(od, "proto_vectors.npy"), P)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12); Cm = Pn @ Pn.T; np.fill_diagonal(Cm, -9)
    nn_cos = Cm.max(1); iu = np.triu_indices_from(Cm, 1)
    col = {"median_nn_cos": float(np.median(nn_cos)), "frac_pairs_gt_0.99": float((Cm[iu] > 0.99).mean()),
           "max_nn_cos": float(nn_cos.max()), "n_dead": int(K - len(np.unique(a_tr)))}
    print(f"[{ARM}] {tag}: n_mc={len(np.unique(lab))} collapse={col}")
    return col


t0 = time.time()
t._setup_umap_edges(EPOCHS)                 # prototype init (leiden/waypoint) + eps calibration
col0 = place_all("epoch0")                  # same-init baseline, as D60/D63 used
t1 = time.time()
t.continue_train_umap_edges(epochs=EPOCHS, early_stop=False)
train_s = time.time() - t1
t.save_umap_checkpoint(); t.save_clusters()
col = place_all("final")
summary = {"arm": ARM, "run_tag": RUN_TAG, "epochs": EPOCHS, "device": get_device(), "lambdas": {k: lc[k] for k in lc if k.startswith("lambda_")},
           "train_seconds": round(train_s, 1), "collapse_epoch0": col0, "collapse_final": col, "model_name": mn,
           "dump_path": t.get_dump_path()}
json.dump(summary, open(os.path.join(OUT, "arm_summary.json"), "w"), indent=2, default=str)
print(f"[{ARM}] trained {EPOCHS} epochs in {train_s:.0f}s ({train_s/EPOCHS:.0f}s/epoch); total {time.time()-t0:.0f}s")
