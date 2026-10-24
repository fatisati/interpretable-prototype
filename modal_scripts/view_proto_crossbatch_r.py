"""PROTOTYPE cross-batch coherence measured in GENE SPACE, not by labels.

For prototype k and batch b (>= MINC cells of k in b):
    sig(k,b)[gene] = mean expr in (k,b)  -  mean expr over ALL batch-b cells
                     (within-batch contrast: the batch effect cancels)
    r_cross(k)     = mean over batch PAIRS of  pearson_r( sig(k,b1), sig(k,b2) )

r_cross(k) ~ 0.84 would mean "this prototype means the SAME biology in every
batch it spans" (that is the measured median for genuinely same-type pairs);
r_cross(k) ~ 0 means a chimera. Label-free, so it is immune to annotations
that disagree across studies (e.g. Oetjen's Megakaryocyte progenitors).

NULL for scale: r between sig(k,b1) and sig(j,b2) for RANDOM k != j.
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd, anndata
REPO=r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0,REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS)
MINC=10
n,h5,bk,lk,nk,K=DATASETS[DS]
ad=anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"],h5))
X=ad.X.toarray() if hasattr(ad.X,"toarray") else np.asarray(ad.X)
X=np.asarray(X,dtype=np.float32)
cts=np.asarray(ad.obs[lk]).astype(str); bts=np.asarray(ad.obs[bk]).astype(str)
freq=pd.Series(cts).value_counts()
rare=set(freq[freq<=np.quantile(freq.values,0.25)].index)
batches=np.unique(bts)
bg={b: X[bts==b].mean(0) for b in batches}
# held-out split, so we can also report train vs test
import anndata as _ad
names=np.asarray(ad.obs_names).astype(str)
d=np.load(os.path.join(REPO,"node_heldout_data",f"{DS}_arbf_ctb20_seed0_split.npz"),
          allow_pickle=True)
tid=set(str(x) for x in d["test_ids"]); test=np.array([x in tid for x in names])

def r(a,b):
    a=a-a.mean(); b=b-b.mean()
    d_=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d_) if d_>0 else np.nan

ARMS=["d60_epoch0_leid","d59_leiden_ctb","d60_leid_noc","d61_pr3_lv","d62_pr16","d62_pr8"]
rng=np.random.RandomState(0)
print(f"prototype CROSS-BATCH signature coherence; (proto,batch) groups need "
      f">={MINC} cells\n")
print(f"{'arm':<20}{'protos':>8}{'r_cross':>10}{'wtd':>9}{'rare':>9}"
      f"{'common':>9}{'NULL':>9}{'frac<0.3':>10}")
out={}
for arm in ARMS:
    lab=load_labels(os.path.join(RUNS,arm),n)
    if lab is None: continue
    sigs={}; maj={}; size={}
    for k in np.unique(lab):
        mk=lab==k
        if mk.sum()<MINC: continue
        maj[k]=Counter(cts[mk]).most_common(1)[0][0]
        for b in batches:
            s=mk&(bts==b)
            if s.sum()>=MINC:
                sigs[(k,b)]=X[s].mean(0)-bg[b]
                size[(k,b)]=int(s.sum())
    per={}
    for k in maj:
        bs=[b for b in batches if (k,b) in sigs]
        if len(bs)<2: continue
        vals=[r(sigs[(k,bs[i])],sigs[(k,bs[j])])
              for i in range(len(bs)) for j in range(i+1,len(bs))]
        per[k]=(float(np.mean(vals)), sum(size[(k,b)] for b in bs))
    keys=list(sigs)
    nulls=[]
    for _ in range(400):
        (k1,b1),(k2,b2)=keys[rng.randint(len(keys))],keys[rng.randint(len(keys))]
        if k1!=k2 and b1!=b2: nulls.append(r(sigs[(k1,b1)],sigs[(k2,b2)]))
    v=np.array([p[0] for p in per.values()]); w=np.array([p[1] for p in per.values()])
    rv=[per[k][0] for k in per if maj[k] in rare]
    cv=[per[k][0] for k in per if maj[k] not in rare]
    out[arm]=dict(per=per,maj=maj,mean=v.mean(),wtd=float(np.average(v,weights=w)),
                  rare=float(np.mean(rv)) if rv else np.nan, n_rare=len(rv),
                  common=float(np.mean(cv)), null=float(np.median(nulls)),
                  frac_low=float((v<0.3).mean()), n=len(per))
    gs=[size[kb] for kb in size]
    out[arm]['grp']=float(np.median(gs))
    # size-matched control: restrict to (proto,batch) groups with >=40 cells
    per40={}
    for k in maj:
        bs=[b for b in batches if (k,b) in sigs and size[(k,b)]>=40]
        if len(bs)<2: continue
        per40[k]=float(np.mean([r(sigs[(k,bs[i])],sigs[(k,bs[j])])
                     for i in range(len(bs)) for j in range(i+1,len(bs))]))
    out[arm]['r40']=float(np.mean(list(per40.values()))) if per40 else float('nan')
    out[arm]['n40']=len(per40)
    o=out[arm]
    print(f"{arm:<20}{o['n']:>8}{o['mean']:>10.3f}{o['wtd']:>9.3f}"
          f"{o['rare']:>9.3f}{o['common']:>9.3f}{o['null']:>9.3f}{o['frac_low']:>10.1%}"
          f"  | med_grp={o['grp']:.0f}  r(groups>=40c)={o['r40']:.3f} (n={o['n40']})")
print(f"\n(rare = prototypes whose majority cell type is a rare type; "
      f"n_rare per arm: " + ", ".join(f"{a}={out[a]['n_rare']}" for a in out) + ")")
print("\nDELTA vs its own epoch-0 baseline (d60_epoch0_leid):")
base=out["d60_epoch0_leid"]
for a in out:
    if a=="d60_epoch0_leid": continue
    print(f"  {a:<20} r_cross {out[a]['mean']-base['mean']:+.3f}   "
          f"rare {out[a]['rare']-base['rare']:+.3f}   "
          f"common {out[a]['common']-base['common']:+.3f}   "
          f"frac<0.3 {out[a]['frac_low']-base['frac_low']:+.1%}")
