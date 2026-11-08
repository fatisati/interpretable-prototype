"""Frozen full-pass gradient-norm probe (local copy of modal_scripts/measure_grad_shares_d63.py).
Six terms at lambda=1, lr=0, optimizer step disabled; writes tables/gradshares_local.json
with encoder-path and prototype-path median norms, and prints the derived D63 lambdas.
Also creates/reuses the 50-epoch MSE pretrain checkpoint every arm loads."""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
from d63_config import probe_config, arm_config, GNORM_JSON, GNORM_MODAL, GRAD_PROBE
import torch
from interpretable_ssl.utils import get_device
from interpretable_ssl.experiments.tasks import get_trainer, _resolve_ds_id, _resolve_num_prototypes

TERMS = ("cohesion_gram", "cohesion_aff", "umap", "recon", "gmkl", "proto_recon")
LIK = os.environ.get("LIK", "mse")
t0 = time.time()
print(f"device={get_device()}  likelihood={LIK}")
lc = probe_config(LIK)
assert os.path.exists(TRAIN_H5), f"missing {TRAIN_H5} -- run prepare_pbmc_immune.py first"
resolved = _resolve_ds_id(TRAIN_H5, batch_key=BATCH_KEY, label_key=LABEL_KEY, niche_key=None, num_prototypes=K)
t = get_trainer(experiment_name=f"probe_{LIK}", cvae_epochs=50, affinity_type="arbf", dataset_id=resolved,
                l2norm=1, assignment_metric="dotp", batch_size=1024, umap_steps_per_epoch=None,
                umap_similarity="proto", freeze_batch_embedding=1, freeze_decoder=0,
                num_prototypes=_resolve_num_prototypes(K, resolved),
                **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")})
ck = os.path.join(t.get_pretrain_dump_path(), "pretrain_checkpoint.pth")
if os.path.exists(ck):
    t.load_pretrain_checkpoint()
else:
    t1 = time.time(); t.pretrain_encoder(); t.init_prototypes(); t.save_pretrain_checkpoint()
    print(f"[probe] pretrain 50 epochs took {time.time()-t1:.0f}s")
t.model.freeze_batch_embedding()
for k_, v_ in (("coh_proto_sg", 1), ("lambda_gmkl", 1.0), ("proto_recon_form", "mix3"),
               ("proto_recon_dec_sg", 0), ("proto_recon_stopgrad", True)):
    assert getattr(t, k_, None) == v_, f"{k_} did not reach trainer"
print("[probe] model folder:", t.get_model_name())

t._setup_umap_edges(10)
s = t._umap_state; expected = len(s["loader"]); opt = s["optimizer"]
opt.step = lambda *a, **kw: None
for g in opt.param_groups: g["lr"] = 0.0
t.umap_checkpoint_freq = 0
pp = os.path.join(t.get_dump_path(), "grad_probe.jsonl")
if os.path.exists(pp): os.remove(pp)
before = {nm: p.detach().clone() for nm, p in t.model.named_parameters()}
print(f"[probe] frozen pass: {expected} steps, probing every {GRAD_PROBE}")
t1 = time.time()
t.continue_train_umap_edges(epochs=1, early_stop=False)
print(f"[probe] one frozen pass took {time.time()-t1:.0f}s")
moved = [nm for nm, p in t.model.named_parameters() if not torch.equal(p.detach(), before[nm]) and "prototypes" not in nm]
assert not moved, f"ENCODER WEIGHTS MOVED: {moved[:5]}"

enc = {k: [] for k in TERMS}; pro = {k: [] for k in TERMS}; errors = 0
for line in open(pp):
    if not line.strip(): continue
    row = json.loads(line)
    if "error" in row: errors += 1; continue
    for k in TERMS:
        if isinstance(row.get(k), dict):
            enc[k].append(row[k]["gnorm_enc"]); pro[k].append(row[k]["gnorm_proto"])
n = len(enc["cohesion_gram"]); exp_rows = -(-expected // GRAD_PROBE)
assert n >= max(10, int(0.9 * exp_rows)), f"probe logged {n} rows, expected ~{exp_rows} (errors={errors})"
res = {"enc": {k: st.median(v) for k, v in enc.items() if v},
       "proto": {k: st.median(v) for k, v in pro.items() if v},
       "n_rows": n, "errors": errors, "expected_steps": expected}
for k in ("cohesion_gram", "cohesion_aff", "umap"):
    assert res["proto"][k] == 0.0, f"{k} leaks into prototypes"
assert res["proto"]["gmkl"] > 0 and res["proto"]["proto_recon"] > 0
os.makedirs(os.path.dirname(GNORM_JSON), exist_ok=True)
allres = json.load(open(GNORM_JSON)) if os.path.exists(GNORM_JSON) else {}
allres[LIK] = res; json.dump(allres, open(GNORM_JSON, "w"), indent=2)
print(f"\n[probe] rows={n} errors={errors}")
print(f"{'term':14s} {'enc_local':>10s} {'enc_modal':>10s}   {'proto_local':>11s} {'proto_modal':>11s}")
gm = GNORM_MODAL.get(LIK, {"enc": {}, "proto": {}})
for k in TERMS:
    print(f"{k:14s} {res['enc'].get(k, float('nan')):10.4g} {gm['enc'].get(k, float('nan')):10.4g}   "
          f"{res['proto'].get(k, float('nan')):11.4g} {gm['proto'].get(k, float('nan')):11.4g}")
lc_arm = arm_config("leid", True, LIK, gnorm=res)
print("\n[probe] derived d63-style lambdas (recon ON):",
      {k: lc_arm[k] for k in ("lambda_cohesion", "lambda_cohesion_aff", "lambda_recon", "lambda_gmkl", "lambda_proto_recon")})
print(f"total {time.time()-t0:.0f}s")
