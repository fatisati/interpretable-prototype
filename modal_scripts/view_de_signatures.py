"""CIDER-style within-batch DE signatures: is Oetjen's Megakaryocyte-progenitor
population the SAME biological state as the 10X/Freytag/Sun satellites?

  signature(type t, batch b) = mean expr of (t,b) cells  -  mean expr of ALL
                               batch-b cells        <- within-batch contrast,
                                                       batch effect cancels
  similarity = Pearson r between two groups' signatures (CIDER's IDER idea)
No model, no prototypes, no latent -- raw expression only.
"""
import os, sys
import numpy as np, pandas as pd, anndata
REPO=r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0,REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS
DS="pbmc-immune"; MINC=10
n,h5,bk,lk,nk,K=DATASETS[DS]
ad=anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"],h5))
X=ad.X.toarray() if hasattr(ad.X,"toarray") else np.asarray(ad.X)
cts=np.asarray(ad.obs[lk]).astype(str); bts=np.asarray(ad.obs[bk]).astype(str)
print(f"X {X.shape}, range [{X.min():.2f},{X.max():.2f}]")
sig={}; sizes={}
for b in np.unique(bts):
    mb=bts==b; bg=X[mb].mean(0)
    for t in np.unique(cts[mb]):
        s=mb&(cts==t)
        if s.sum()<MINC: continue
        sig[(t,b)]=X[s].mean(0)-bg; sizes[(t,b)]=int(s.sum())
keys=list(sig)
def r(a,b):
    x,y=sig[a],sig[b]
    return float(np.corrcoef(x,y)[0,1])
TARGET="Megakaryocyte progenitors"
mk=[k for k in keys if k[0]==TARGET]
print(f"\n=== {TARGET} groups: " + ", ".join(f"{b}(n={sizes[(t,b)]})" for t,b in mk))
print("\n--- cross-batch similarity BETWEEN Megakaryocyte-progenitor groups ---")
for i in range(len(mk)):
    for j in range(i+1,len(mk)):
        print(f"    {mk[i][1]:<9} vs {mk[j][1]:<9}  r = {r(mk[i],mk[j]):+.3f}")
print("\n--- for each Mk-prog group: its TOP cross-batch matches (any type) ---")
for k in mk:
    cand=[(r(k,o),o) for o in keys if o[1]!=k[1]]
    cand.sort(reverse=True)
    print(f"  {k[1]} (n={sizes[k]}):")
    for v,o in cand[:4]:
        flag="  <-- SAME TYPE" if o[0]==TARGET else ""
        print(f"      r={v:+.3f}  {o[0]} / {o[1]}{flag}")
    same=[(v,o) for v,o in cand if o[0]==TARGET]
    if same and same[0][0] < cand[0][0]:
        rank=[i for i,(v,o) in enumerate(cand) if o[0]==TARGET]
        print(f"      (best same-type match ranks #{rank[0]+1} of {len(cand)})")
print("\n--- CONTROL: distribution of cross-batch r ---")
same_t=[r(a,b) for i,a in enumerate(keys) for b in keys[i+1:]
        if a[0]==b[0] and a[1]!=b[1]]
diff_t=[r(a,b) for i,a in enumerate(keys) for b in keys[i+1:]
        if a[0]!=b[0] and a[1]!=b[1]]
print(f"    SAME type, different batch : n={len(same_t)}  "
      f"median r={np.median(same_t):+.3f}  10th pct={np.percentile(same_t,10):+.3f}")
print(f"    DIFF type, different batch : n={len(diff_t)}  "
      f"median r={np.median(diff_t):+.3f}  90th pct={np.percentile(diff_t,90):+.3f}")
print(f"    => a tau separating them sits near "
      f"{(np.percentile(same_t,10)+np.percentile(diff_t,90))/2:+.3f}")
