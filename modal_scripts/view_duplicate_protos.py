"""Are there BATCH-SPECIFIC prototypes holding the SAME cell type?
(user's duplication question, and a test of the claim that mix3 cannot fix it)

A proto is 'batch-specific' if >= BATCH_FRAC of its cells come from one batch.
A DUPLICATE SET = two or more batch-specific protos with the SAME majority cell
type but DIFFERENT dominant batches.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO)
os.chdir(REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs  # noqa: E402

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
BATCH_FRAC = 0.80
MIN_CELLS = 10

n, h5, bk, lk, nk, K = DATASETS[DS]
obs = read_obs(os.path.join(os.environ["DATA_DIR"], h5), [bk, lk])
cts = np.asarray(obs[lk]).astype(str)
bts = np.asarray(obs[bk]).astype(str)
freq = pd.Series(cts).value_counts()
rare = set(freq[freq <= np.quantile(freq.values, 0.25)].index)

ARMS = ["d60_epoch0_leid", "d60_leid_noc", "d61_pr3_lv", "d61_pr3_lvdec",
        "d61_pr3_lvdec_nr", "d61_pr3_sgdec_nr"]

print(f"batch-specific = >={BATCH_FRAC:.0%} of a proto's cells from ONE batch; "
      f"protos with >={MIN_CELLS} cells\n")
for arm in ARMS:
    lab = load_labels(os.path.join(RUNS, arm), n)
    if lab is None:
        print(f"-- {arm}: no labels"); continue
    bs = []                       # (mc, type, batch, n)
    n_protos = 0
    for mc in np.unique(lab):
        m = lab == mc
        if m.sum() < MIN_CELLS:
            continue
        n_protos += 1
        bc = Counter(bts[m]); b, cnt = bc.most_common(1)[0]
        if cnt / m.sum() >= BATCH_FRAC:
            t = Counter(cts[m]).most_common(1)[0][0]
            bs.append((int(mc), t, b, int(m.sum())))
    by_type = defaultdict(list)
    for mc, t, b, c in bs:
        by_type[t].append((b, mc, c))
    dup_types, dup_protos, rare_dup = 0, 0, 0
    detail = []
    for t, lst in by_type.items():
        if len({b for b, _, _ in lst}) >= 2:      # same type, >=2 distinct batches
            dup_types += 1
            dup_protos += len(lst)
            if t in rare:
                rare_dup += 1
            detail.append((t, lst))
    print(f"=== {arm} ===  protos>={MIN_CELLS} cells: {n_protos}, "
          f"batch-specific: {len(bs)} ({100*len(bs)/n_protos:.0f}%)")
    print(f"    DUPLICATE sets (same type, batch-specific protos in >=2 batches): "
          f"{dup_types} types / {dup_protos} protos; rare types involved: {rare_dup}")
    detail.sort(key=lambda d: -len(d[1]))
    for t, lst in detail[:6]:
        s = ", ".join(f"{b}:mc{mc}({c})" for b, mc, c in sorted(lst)[:8])
        flag = " <RARE>" if t in rare else ""
        print(f"      {t:<34} {len(lst)} protos  {s}{flag}")
    for t, lst in detail:
        if t in rare and (t, lst) not in detail[:6]:
            s = ", ".join(f"{b}:mc{mc}({c})" for b, mc, c in sorted(lst)[:8])
            print(f"      {t:<34} {len(lst)} protos  {s} <RARE>")
    print()
