"""Do same-type cells from DIFFERENT batches land on the SAME prototype?
Granularity-free: ratio of cross-batch to within-batch co-assignment probability.

  p_k(t,b) = fraction of type-t cells in batch b assigned to proto k
  within(t) = mean over batches b     of  sum_k p_k(t,b)^2
  cross(t)  = mean over batch PAIRS   of  sum_k p_k(t,b1)*p_k(t,b2)
  ratio(t)  = cross / within     1.0 = batch does not matter at all
                                 0.0 = batches never share a prototype
The ratio cancels granularity: using more protos per type lowers BOTH terms.
"""
import os, sys
import numpy as np, pandas as pd
REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS); MINC=20
n,h5,bk,lk,nk,K = DATASETS[DS]
obs=read_obs(os.path.join(os.environ["DATA_DIR"],h5),[bk,lk])
cts=np.asarray(obs[lk]).astype(str); bts=np.asarray(obs[bk]).astype(str)
freq=pd.Series(cts).value_counts()
rare=set(freq[freq<=np.quantile(freq.values,0.25)].index)
ARMS=["d60_epoch0_leid","d59_leiden_ctb","d60_leid_noc","d61_pr3_lv","d62_pr16","d62_pr8"]
def ratios(lab):
    out={}
    for t in freq.index:
        mt=cts==t
        bs=[b for b in np.unique(bts[mt]) if (mt&(bts==b)).sum()>=MINC]
        if len(bs)<2: continue
        P={}
        for b in bs:
            s=mt&(bts==b); v=np.bincount(lab[s],minlength=K).astype(float)
            P[b]=v/v.sum()
        w=np.mean([ (P[b]**2).sum() for b in bs])
        c=np.mean([ (P[bs[i]]*P[bs[j]]).sum()
                    for i in range(len(bs)) for j in range(i+1,len(bs))])
        out[t]=(c/w if w>0 else np.nan, int(mt.sum()), len(bs))
    return out
res={a:ratios(load_labels(os.path.join(RUNS,a),n)) for a in ARMS
     if load_labels(os.path.join(RUNS,a),n) is not None}
types=[t for t in freq.index if t in res[ARMS[0]]]
print(f"cross/within co-assignment RATIO (1.0 = batch-blind). types with >=2 "
      f"batches having >={MINC} cells\n")
print(f"{'cell type':<32}{'n':>7}{'nb':>4}  " + "".join(f"{a.replace('d61_pr3_','').replace('d60_',''):>12}" for a in res))
for t in types:
    star=" *" if t in rare else "  "
    r0=res[ARMS[0]][t]
    print(f"{t[:30]:<30}{star}{r0[1]:>7}{r0[2]:>4}  " +
          "".join(f"{res[a][t][0]:>12.3f}" for a in res))
print()
for a in res:
    vals=np.array([res[a][t][0] for t in types]); wts=np.array([res[a][t][1] for t in types])
    rv=[res[a][t][0] for t in types if t in rare]
    print(f"{a:<20} mean={vals.mean():.3f}  weighted={np.average(vals,weights=wts):.3f}"
          + (f"  rare-only={np.mean(rv):.3f}" if rv else "  rare-only=n/a"))
