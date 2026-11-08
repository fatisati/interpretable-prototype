"""Time real d63_mse-rec training steps on cpu vs mps. env: USE_MPS=0|1, STEPS=40.
Same trainer kwargs as train_d63_arm.py, except umap_steps_per_epoch=STEPS to truncate
the epoch, and a separate experiment_name so nothing of the real run is touched."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
from d63_config import arm_config
import torch
from interpretable_ssl.utils import get_device
from interpretable_ssl.experiments.tasks import get_trainer, _resolve_ds_id, _resolve_num_prototypes
STEPS = int(os.environ.get("STEPS", "40"))
dev = get_device()
lc = arm_config("leid", True, "mse")
resolved = _resolve_ds_id(TRAIN_H5, batch_key=BATCH_KEY, label_key=LABEL_KEY, niche_key=None, num_prototypes=K)
t = get_trainer(experiment_name=f"bench_{dev}", cvae_epochs=50, affinity_type="arbf", dataset_id=resolved,
                l2norm=1, assignment_metric="dotp", batch_size=1024, umap_steps_per_epoch=STEPS,
                umap_similarity="proto", freeze_batch_embedding=1, freeze_decoder=0,
                num_prototypes=_resolve_num_prototypes(K, resolved),
                **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")})
t.load_pretrain_checkpoint(); t.model.freeze_batch_embedding()
print(f"[bench] device={t.device} model device={next(t.model.parameters()).device} threads={torch.get_num_threads()}")
t0 = time.time(); t._setup_umap_edges(10); print(f"[bench] setup {time.time()-t0:.1f}s")
t._run_umap_epoch()   # warm-up (MPS kernel compilation, allocator)
for rep in range(2):
    t0 = time.time(); m = t._run_umap_epoch(); dt = time.time() - t0
    print(f"[bench] {dev} rep{rep}: {STEPS} steps in {dt:.1f}s = {dt/STEPS:.3f} s/step | loss={m.get('loss', m) if isinstance(m, dict) else m}")
