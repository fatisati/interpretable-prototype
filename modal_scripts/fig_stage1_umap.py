"""UMAP of the STAGE-1 (pretrained scPoli CVAE, epoch-0) latent for pbmc-immune.

2x2 panels:
  A  cell types (16), labels at cluster medians
  B  rare (bottom-quartile types) highlighted over gray commons
  C  batch
  D  Megakaryocyte progenitors only, colored by batch -- the home/satellite split

Load-if-exists: UMAP coords cached next to the figure. FORCE_RERUN=False.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO)
os.chdir(REPO)

from modal_scripts.score_all_baselines import DATASETS  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs  # noqa: E402

FORCE_RERUN = False
DS = "pbmc-immune"
OUT = os.path.join(REPO, "files", "figures", "d60_ctb")
os.makedirs(OUT, exist_ok=True)
COORDS = os.path.join(OUT, f"{DS}_stage1_umap_coords.npy")

n, h5, bk, lk, nk, K = DATASETS[DS]
h5p = os.path.join(os.environ["DATA_DIR"], h5)
obs = read_obs(h5p, [bk, lk])
cts = np.asarray(obs[lk]).astype(str)
bts = np.asarray(obs[bk]).astype(str)

z = np.load(os.path.join(REPO, "files", "volume_runs", DS,
                         "d59_epoch0", "latent_z_full.npy"))
assert len(z) == n == len(cts)

if os.path.exists(COORDS) and not FORCE_RERUN:
    emb = np.load(COORDS)
    print(f"loaded UMAP coords from {COORDS}")
else:
    import umap
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                    random_state=0).fit_transform(zn)
    np.save(COORDS, emb)
    print(f"computed UMAP, saved coords to {COORDS}")

freq = pd.Series(cts).value_counts()
qcut = np.quantile(freq.values, 0.25)
rare_types = list(freq[freq <= qcut].index)
rare = np.isin(cts, rare_types)
print(f"rare types ({len(rare_types)}): {rare_types}")
print(f"batches: {sorted(set(bts))}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# fixed-order palettes (identity job). 16 types: tab20-derived, warm/cool alternating.
TYPE_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
               "#BBBBBB" if False else "#997700", "#DD7788", "#44AA99", "#DDAA33",
               "#004488", "#99CC66", "#BB5566", "#222255", "#F58231", "#88CCAA"]
BATCH_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
RARE_COLORS = ["#D62728", "#7B3FB3", "#1B7837", "#E66100"]
GRAY = "#D9D9D9"

types_sorted = list(freq.index)                 # fixed order: by abundance
batches_sorted = sorted(set(bts))

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#CCCCCC")

# --- A: cell types -------------------------------------------------------------
ax = axes[0, 0]
for i, t in enumerate(types_sorted):
    m = cts == t
    ax.scatter(emb[m, 0], emb[m, 1], s=1.5, c=TYPE_COLORS[i % 16],
               linewidths=0, rasterized=True, label=t)
for i, t in enumerate(types_sorted):
    m = cts == t
    xm, ym = np.median(emb[m, 0]), np.median(emb[m, 1])
    star = " *" if t in rare_types else ""
    ax.text(xm, ym, t + star, fontsize=6.5, ha="center", va="center",
            color="#222222",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax.set_title("A  Stage-1 latent, cell types (* = rare, bottom-quartile)",
             fontsize=11, loc="left")
ax.legend(fontsize=5.5, markerscale=4, loc="upper left",
          bbox_to_anchor=(0.0, -0.02), ncol=4, frameon=False)

# --- B: rare vs common ----------------------------------------------------------
ax = axes[0, 1]
ax.scatter(emb[~rare, 0], emb[~rare, 1], s=1.5, c=GRAY, linewidths=0,
           rasterized=True, label=f"common ({(~rare).sum():,})")
for i, t in enumerate(rare_types):
    m = cts == t
    ax.scatter(emb[m, 0], emb[m, 1], s=8, c=RARE_COLORS[i % 4],
               linewidths=0, rasterized=True, label=f"{t} ({m.sum():,})")
    xm, ym = np.median(emb[m, 0]), np.median(emb[m, 1])
    ax.text(xm, ym, t, fontsize=7.5, ha="center", va="center", fontweight="bold",
            color="#222222",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
ax.set_title("B  Rare types highlighted (gray = common)", fontsize=11, loc="left")
ax.legend(fontsize=7, markerscale=2, loc="best", frameon=False)

# --- C: batch -------------------------------------------------------------------
ax = axes[1, 0]
for i, b in enumerate(batches_sorted):
    m = bts == b
    ax.scatter(emb[m, 0], emb[m, 1], s=1.5, c=BATCH_COLORS[i % 5],
               linewidths=0, rasterized=True, label=f"{b} ({m.sum():,})")
ax.set_title("C  Batch (study)", fontsize=11, loc="left")
ax.legend(fontsize=7, markerscale=4, loc="best", frameon=False)

# --- D: Megakaryocyte progenitors by batch -- the home/satellite split ----------
ax = axes[1, 1]
mk = [t for t in rare_types if "egakaryo" in t]
mk = mk[0] if mk else rare_types[0]
mmk = cts == mk
ax.scatter(emb[~mmk, 0], emb[~mmk, 1], s=1.5, c=GRAY, linewidths=0,
           rasterized=True)
for i, b in enumerate(batches_sorted):
    m = mmk & (bts == b)
    if m.sum() == 0:
        continue
    ax.scatter(emb[m, 0], emb[m, 1], s=22, c=BATCH_COLORS[i % 5],
               edgecolors="white", linewidths=0.4,
               label=f"{b} (n={m.sum()})")
ax.set_title(f"D  {mk} only, colored by batch", fontsize=11, loc="left")
ax.legend(fontsize=8, markerscale=1.5, loc="best", frameon=False)

fig.suptitle(f"{DS}: stage-1 (pretrained CVAE) latent, UMAP of z "
             f"(cosine, n={n:,})", fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
p = os.path.join(OUT, f"{DS}_stage1_umap.png")
fig.savefig(p, dpi=200, bbox_inches="tight")
print(f"saved {p}")
