"""Per-prototype, per-batch majority-label view: is the same prototype holding
DIFFERENT cell types in different batches? (user's wrong-alignment hypothesis)

For every arm: for each prototype with >=3 cells in >=2 batches, collect the
majority type of its members in each batch. Count protos whose batch-majorities
all AGREE vs DISAGREE; slice by whether a rare type is involved; list the worst
disagreeing examples with counts.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO)
os.chdir(REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa: E402
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs  # noqa: E402

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
MIN = 3

n, h5, bk, lk, nk, K = DATASETS[DS]
h5p = os.path.join(os.environ["DATA_DIR"], h5)
obs = read_obs(h5p, [bk, lk])
cts = np.asarray(obs[lk]).astype(str)
bts = np.asarray(obs[bk]).astype(str)

freq = pd.Series(cts).value_counts()
qcut = np.quantile(freq.values, 0.25)
rare_types = set(freq[freq <= qcut].index)

ARMS = ["d60_epoch0_leid", "d59_leiden_ctb", "d60_leid_noc",
        "d61_pr3_lv", "d62_pr16", "d62_pr8"]

for arm in ARMS:
    lab = load_labels(os.path.join(RUNS, arm), n)
    if lab is None:
        print(f"-- {arm}: no labels"); continue
    agree = disagree = single = 0
    dis_rows = []
    for mc in np.unique(lab):
        m = lab == mc
        majs = {}
        for b in np.unique(bts[m]):
            s = m & (bts == b)
            if s.sum() < MIN:
                continue
            c = Counter(cts[s])
            majs[b] = (c.most_common(1)[0][0], int(s.sum()))
        if len(majs) < 2:
            single += 1
            continue
        types = {t for t, _ in majs.values()}
        if len(types) == 1:
            agree += 1
        else:
            disagree += 1
            dis_rows.append((int(mc), int(m.sum()), majs,
                             bool(types & rare_types)))
    multi = agree + disagree
    print(f"\n=== {arm} ===  protos(>=2 batches with >={MIN} cells): {multi}, "
          f"single/small: {single}")
    print(f"  batch-majorities AGREE:    {agree:4d}  ({100*agree/multi:.1f}%)")
    print(f"  batch-majorities DISAGREE: {disagree:4d}  ({100*disagree/multi:.1f}%)")
    nr = sum(1 for r in dis_rows if r[3])
    print(f"  disagreeing protos involving a RARE type: {nr}")
    dis_rows.sort(key=lambda r: -r[1])
    print("  worst 8 disagreeing protos (mc, n_cells, per-batch majority(count)):")
    for mc, nc, majs, has_rare in dis_rows[:8]:
        parts = ", ".join(f"{b}:{t}({c})" for b, (t, c) in sorted(majs.items()))
        flag = "  <RARE>" if has_rare else ""
        print(f"    mc{mc:<4d} n={nc:<5d} {parts}{flag}")
    # rare-focused: every proto where ANY batch-majority is rare
    print("  ALL protos where some batch-majority is a rare type:")
    shown = 0
    for mc in np.unique(lab):
        m = lab == mc
        majs = {}
        for b in np.unique(bts[m]):
            s = m & (bts == b)
            if s.sum() < MIN:
                continue
            c = Counter(cts[s])
            majs[b] = (c.most_common(1)[0][0], int(s.sum()))
        if len(majs) >= 2 and any(t in rare_types for t, _ in majs.values()):
            parts = ", ".join(f"{b}:{t}({c})" for b, (t, c) in sorted(majs.items()))
            ok = "AGREE" if len({t for t, _ in majs.values()}) == 1 else "DISAGREE"
            print(f"    mc{int(mc):<4d} {ok:<8s} {parts}")
            shown += 1
    if shown == 0:
        print("    (none -- no rare type is a batch-majority in any multi-batch proto)")
