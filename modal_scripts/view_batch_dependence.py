"""Duplication WITHOUT a hard 'batch-specific' threshold (the old >=80%-one-batch
rule missed every partially-mixed duplicate -- user's objection, 2026-10-23).

Question asked properly: for cells of ONE cell type, does WHICH PROTOTYPE a cell
lands on DEPEND on its batch?

  build the (prototype x batch) contingency table for that type's cells
  V = Cramer's V of that table      0 = prototype choice independent of batch
                                    1 = prototype choice fully determined by batch
  V is biased upward with many prototypes and few cells, so we subtract a
  PERMUTATION NULL: reshuffle the batch labels among that type's cells and
  recompute. excess = V_obs - V_null is the real batch dependence.

Also lists concrete duplicate pairs by ENRICHMENT (no 80% rule): a prototype is
'leaning to batch b' for type t if it holds >= MINC of t's cells and batch b is
>= ENR times more frequent there than in that type overall.
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd
REPO=r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0,REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS)
MINB=20; MINC=5; ENR=2.0; NPERM=20
n,h5,bk,lk,nk,K=DATASETS[DS]
obs=read_obs(os.path.join(os.environ["DATA_DIR"],h5),[bk,lk])
cts=np.asarray(obs[lk]).astype(str); bts=np.asarray(obs[bk]).astype(str)
freq=pd.Series(cts).value_counts()
rare=set(freq[freq<=np.quantile(freq.values,0.25)].index)
rng=np.random.RandomState(0)

def cramers_v(lab_t, bat_t):
    tab=pd.crosstab(lab_t, bat_t).values.astype(float)
    if tab.shape[0]<2 or tab.shape[1]<2: return np.nan
    nn=tab.sum()
    exp=np.outer(tab.sum(1), tab.sum(0))/nn
    chi2=((tab-exp)**2/np.maximum(exp,1e-12)).sum()
    return float(np.sqrt(chi2/(nn*(min(tab.shape)-1))))

ARMS=["d60_epoch0_leid","d59_leiden_ctb","d60_leid_noc","d61_pr3_lv","d62_pr16","d62_pr8"]
types=[t for t in freq.index
       if len([b for b in np.unique(bts[cts==t]) if ((cts==t)&(bts==b)).sum()>=MINB])>=2]
res={}
for arm in ARMS:
    lab=load_labels(os.path.join(RUNS,arm),n)
    if lab is None: continue
    row={}
    for t in types:
        m=(cts==t)&np.isin(bts,[b for b in np.unique(bts[cts==t])
                                if ((cts==t)&(bts==b)).sum()>=MINB])
        lt,bt=lab[m],bts[m]
        v=cramers_v(lt,bt)
        nulls=[cramers_v(lt, rng.permutation(bt)) for _ in range(NPERM)]
        row[t]=(v, float(np.mean(nulls)), v-float(np.mean(nulls)), int(m.sum()))
    res[arm]=row
print(f"BATCH DEPENDENCE of prototype choice, per cell type "
      f"(Cramer's V minus permutation null; 0 = none, higher = more duplication)\n")
hdr=f"{'cell type':<32}{'n':>6}  " + "".join(f"{a.replace('d60_','').replace('d59_',''):>16}" for a in res)
print(hdr)
for t in types:
    star=" *" if t in rare else "  "
    print(f"{t[:30]:<30}{star}{res[ARMS[0]][t][3]:>6}  " +
          "".join(f"{res[a][t][2]:>16.3f}" for a in res))
print()
for a in res:
    ex=np.array([res[a][t][2] for t in types]); w=np.array([res[a][t][3] for t in types])
    print(f"{a:<20} mean excess V={ex.mean():.3f}  weighted={np.average(ex,weights=w):.3f}"
          f"  (raw V mean={np.mean([res[a][t][0] for t in types]):.3f},"
          f" null={np.mean([res[a][t][1] for t in types]):.3f})")
print(f"\n\nENRICHMENT-based duplicate pairs (no 80% rule; proto holds >={MINC} of the"
      f" type's cells and a batch is >={ENR}x over-represented there)\n")
for arm in ["d60_epoch0_leid","d60_leid_noc","d62_pr16"]:
    lab=load_labels(os.path.join(RUNS,arm),n)
    print(f"=== {arm} ===")
    tot=0
    for t in types:
        mt=cts==t
        base=pd.Series(bts[mt]).value_counts(normalize=True)
        lean={}
        for mc in np.unique(lab[mt]):
            s=mt&(lab==mc)
            if s.sum()<MINC: continue
            fr=pd.Series(bts[s]).value_counts(normalize=True)
            for b,f in fr.items():
                if base.get(b,0)>0 and f/base[b]>=ENR and f>=0.4:
                    lean.setdefault(b,[]).append((int(mc),int(s.sum()),round(f,2)))
        if len(lean)>=2:
            tot+=sum(len(v) for v in lean.values())
            flag=" <RARE>" if t in rare else ""
            desc="; ".join(f"{b}->" + ",".join(f"mc{mc}(n={c},{p:.0%})" for mc,c,p in v[:3])
                           for b,v in sorted(lean.items()))
            print(f"  {t[:28]:<28} {desc}{flag}")
    print(f"  total protos in batch-leaning duplicate sets: {tot}\n")
