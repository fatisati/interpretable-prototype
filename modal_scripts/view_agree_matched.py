"""Is d62_pr16's AGREE% gain real, or a SELECTION artifact?
mix3 kills/merges prototypes, shrinking the denominator. Recompute the
batch-vote agreement at several MATCHED minimum-size thresholds, and report
ABSOLUTE counts (agree / disagree), not just the fraction.
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd
REPO=r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0,REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS)
n,h5,bk,lk,nk,K=DATASETS[DS]
obs=read_obs(os.path.join(os.environ["DATA_DIR"],h5),[bk,lk])
cts=np.asarray(obs[lk]).astype(str); bts=np.asarray(obs[bk]).astype(str)
ARMS=["d60_epoch0_leid","d59_leiden_ctb","d60_leid_noc","d62_pr16","d63_mse-rec","d63_mse-norec"]
def agree(lab, minc):
    a=d=0
    for mc in np.unique(lab):
        m=lab==mc; majs=[]
        for b in np.unique(bts[m]):
            s=m&(bts==b)
            if s.sum()>=minc: majs.append(Counter(cts[s]).most_common(1)[0][0])
        if len(majs)<2: continue
        if len(set(majs))==1: a+=1
        else: d+=1
    return a,d
print("batch-vote agreement at MATCHED thresholds (agree / disagree / %agree)\n")
print(f"{'arm':<18}" + "".join(f"{'>='+str(m)+' cells':>26}" for m in (3,10,20,30)))
for arm in ARMS:
    lab=load_labels(os.path.join(RUNS,arm),n)
    if lab is None: continue
    cells=[]
    for m in (3,10,20,30):
        a,d=agree(lab,m)
        cells.append(f"{a:>4}/{d:<4} {100*a/(a+d) if a+d else 0:>6.1f}%     ")
    print(f"{arm:<18}" + "".join(f"{c:>26}" for c in cells))
