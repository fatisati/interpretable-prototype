"""[D65] Batch entropy on the HELD-OUT half, same protocol as the meeting note.

Batch entropy = how evenly the studies are mixed inside each metacell, averaged
over metacells weighted by size. 0 = a metacell is one study only, 1 = perfectly
mixed. It is the check that we did not simply split the data by batch.

This is the TEST-SPLIT number (meeting note reports 0.482 / 0.573 / 0.574), NOT
the full-dataset `batch_entropy` in metrics.json (which is ~0.98 on the same
runs). Mixing the two would compare different quantities.

Reported for all cells and for rare cells separately, since rare populations are
where a batch-split failure would show up first.

Run:  python modal_scripts/view_batch_entropy_d65.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels      # noqa: E402
from modal_scripts.score_nodeheldout import test_mask_for                # noqa: E402

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
ARMS = [
    ("stage-1 only", "d60_epoch0_leid"),
    ("Leiden on CVAE", "d59_leiden_ctb"),
    ("scProto (no xbatch)", "d60_leid_noc"),
    ("D65 diag-mse", "d65_diag-mse"),
    ("D65 both-mse", "d65_both-mse"),
    ("D65 diag-nb", "d65_diag-nb"),
    ("D65 both-nb", "d65_both-nb"),
]

n, h5, bk, lk, nk, K = DATASETS[DS]
h5_path = os.path.join(os.environ["DATA_DIR"], h5)
import anndata                                                            # noqa: E402
ad = anndata.read_h5ad(h5_path)
cts = np.asarray(ad.obs[lk]).astype(str)
bts = np.asarray(ad.obs[bk]).astype(str)
mask = test_mask_for(DS, h5_path, n)

freq = pd.Series(cts).value_counts()
rare_types = set(freq[freq <= np.quantile(freq.values, 0.25)].index)
is_rare = np.array([c in rare_types for c in cts])
print(f"{DS}: {n:,} cells, held-out {int(mask.sum()):,}, "
      f"rare types {len(rare_types)} ({int(is_rare.sum()):,} cells)")


def batch_entropy(labels, sel):
    """size-weighted mean entropy of the batch mix inside each metacell,
    normalized by log(n_batches) so 1.0 = perfectly mixed."""
    ents, sizes = [], []
    nb = len(np.unique(bts))
    for mc in np.unique(labels[sel]):
        sub = bts[sel & (labels == mc)]
        if sub.size == 0:
            continue
        _, cnt = np.unique(sub, return_counts=True)
        p = cnt / cnt.sum()
        e = float(-(p * np.log(p)).sum() / np.log(nb))
        ents.append(e); sizes.append(sub.size)
    if not ents:
        return float("nan"), 0
    w = np.asarray(sizes, float)
    return float((np.asarray(ents) * w).sum() / w.sum()), len(ents)


print(f"\n{'arm':<22}{'entropy test ALL':>19}{'entropy test RARE':>20}{'n_mc':>7}")
print("-" * 68)
for name, d in ARMS:
    lab = load_labels(os.path.join(RUNS, d), n)
    if lab is None:
        print(f"{name:<22}{'labels not found':>19}")
        continue
    lab = np.asarray(lab)
    e_all, n_all = batch_entropy(lab, mask)
    e_rare, _ = batch_entropy(lab, mask & is_rare)
    print(f"{name:<22}{e_all:>19.3f}{e_rare:>20.3f}{n_all:>7}")
print("\n0 = metacell holds one study only; 1 = studies perfectly mixed.")
