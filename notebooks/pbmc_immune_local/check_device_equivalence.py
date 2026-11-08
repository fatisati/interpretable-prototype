"""Prove that one stage-2 training step computes the SAME numbers on Apple MPS and CPU.

Builds the d63_mse-rec trainer twice (USE_MPS=1 / 0), copies the full state from the MPS
trainer into the CPU one (model weights incl. prototypes, cohesion memory, calibrated
epsilon, EMAs), feeds BOTH the same edge batch, removes the two stochastic parts of the
forward pass (encoder dropout; the CVAE reparameterisation sample -> its mean), zeroes the
learning rate, and compares every per-step metric that _run_umap_epoch returns.

Result on 2026-10-31 (torch 2.15, M3 Ultra): worst relative difference 1.3e-6 over
loss / recon / proto_recon / cohesion gram / cohesion aff / kl / umap / batch_entropy / effk.
Without the noise removal the two devices differ by ~1% simply because their random
streams differ (dropout masks, reparameterisation noise, epsilon calibration).

Run:  USE_MPS=1 python -I notebooks/pbmc_immune_local/check_device_equivalence.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
from d63_config import arm_config
import numpy as np, torch, torch.nn as nn
import torch.distributions as td
from interpretable_ssl.experiments.tasks import get_trainer, _resolve_ds_id, _resolve_num_prototypes

lc = arm_config("leid", True, "mse")
resolved = _resolve_ds_id(TRAIN_H5, batch_key=BATCH_KEY, label_key=LABEL_KEY, niche_key=None, num_prototypes=K)


def build(use_mps):
    os.environ['USE_MPS'] = '1' if use_mps else '0'
    t = get_trainer(experiment_name=f"xdev_{'mps' if use_mps else 'cpu'}", cvae_epochs=50, affinity_type="arbf",
                    dataset_id=resolved, l2norm=1, assignment_metric="dotp", batch_size=1024, umap_steps_per_epoch=1,
                    umap_similarity="proto", freeze_batch_embedding=1, freeze_decoder=0,
                    num_prototypes=_resolve_num_prototypes(K, resolved),
                    **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")})
    t.load_pretrain_checkpoint(); t.model.freeze_batch_embedding(); t._setup_umap_edges(10)
    t._run_umap_epoch()                      # one step so every lazily-created buffer exists
    return t


tm, tc = build(True), build(False)
assert str(tm.device).startswith('mps') and str(tc.device) == 'cpu', (tm.device, tc.device)

# ---- copy state mps -> cpu
tc.model.load_state_dict({k: v.detach().cpu() for k, v in tm.model.state_dict().items()})
for name, v in list(vars(tm).items()):
    if name in ('model', 'train_ds', 'query', 'device', '_umap_state') or name.startswith('_coh_g'):
        continue
    if torch.is_tensor(v):
        setattr(tc, name, v.detach().cpu().clone())
    elif isinstance(v, (int, float, bool)) and getattr(tc, name, None) != v:
        setattr(tc, name, v)                 # calibrated epsilon, step counters
print(f"epsilon synced: {tm.epsilon:.6f}")

# ---- make the forward deterministic on both devices
for t in (tm, tc):
    for g in t._umap_state['optimizer'].param_groups:
        g['lr'] = 0.0
    for m in t.model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
torch.randn_like = lambda x, **k: torch.zeros_like(x)
td.Normal.rsample = lambda self, sample_shape=torch.Size(): self.loc.expand(self._extended_shape(sample_shape)).clone()
td.Normal.sample = td.Normal.rsample

# ---- same batch into both
tc._umap_state['edge_dataset'].rng = np.random.RandomState(7); torch.manual_seed(123)
batch = next(iter(tc._umap_state['loader']))
for t in (tm, tc):
    t._umap_state['loader'] = [batch]
mm, mc = tm._run_umap_epoch(), tc._run_umap_epoch()

keys = ['loss', 'recon', 'proto_recon', 'cohesion_term', 'cohesion_aff_term', 'gmkl', 'kl', 'umap', 'batch_entropy', 'effk']
print(f"{'metric':20s} {'mps':>14s} {'cpu':>14s} {'rel diff':>10s}")
worst = 0.0
for k in keys:
    if k in mm and k in mc and isinstance(mm[k], (int, float)):
        a, b = float(mm[k]), float(mc[k]); rd = abs(a - b) / max(abs(b), 1e-9)
        if abs(b) > 1e-6:
            worst = max(worst, rd)
        print(f"{k:20s} {a:14.6f} {b:14.6f} {rd:10.2e}")
print(f"worst rel diff over non-zero metrics: {worst:.2e}")
assert worst < 1e-4, "MPS and CPU steps disagree beyond float32 noise"
print("PASS: one training step is numerically identical on mps and cpu")
