"""Two tests:
(1) WHY proto_recon merges prototypes: is its TARGET (each prototype's member
    mean expression) nearly identical across prototypes? Compare the pairwise
    correlation of RAW member means vs of background-subtracted signatures.
(2) Is the user's sparsity argument right: are RARE cells sparser (lower counts,
    fewer detected genes) so their reconstruction is intrinsically harder?
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd, anndata
REPO=r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0,REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS)
n,h5,bk,lk,nk,K=DATASETS[DS]
ad=anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"],h5))
X=ad.X.toarray() if hasattr(ad.X,"toarray") else np.asarray(ad.X); X=X.astype(np.float32)
C=ad.layers["counts"]; C=C.toarray() if hasattr(C,"toarray") else np.asarray(C)
cts=np.asarray(ad.obs[lk]).astype(str); bts=np.asarray(ad.obs[bk]).astype(str)
freq=pd.Series(cts).value_counts()
rare=set(freq[freq<=np.quantile(freq.values,0.25)].index)

print("TEST 1 -- how DISCRIMINATIVE is proto_recon's target vs the r-penalty's?\n")
print("   (pairwise correlation between DIFFERENT prototypes' targets;")
print("    high = the two prototypes are being pulled to nearly the same place)\n")
print(f"{'arm':<16}{'RAW member mean':>18}{'bg-subtracted sig':>20}")
for arm in ["d60_leid_noc","d62_pr16"]:
    lab=load_labels(os.path.join(RUNS,arm),n)
    ks=[k for k in np.unique(lab) if (lab==k).sum()>=30]
    raw=np.stack([X[lab==k].mean(0) for k in ks])
    gm=X.mean(0)
    sig=raw-gm                                   # global background subtracted
    def pw(M):
        Mc=M-M.mean(1,keepdims=True)
        Mn=Mc/np.linalg.norm(Mc,axis=1,keepdims=True)
        R=Mn@Mn.T; iu=np.triu_indices_from(R,1)
        return float(np.median(R[iu]))
    print(f"{arm:<16}{pw(raw):>18.4f}{pw(sig):>20.4f}")

print("\n\nTEST 2 -- are RARE cells sparser? (the user's argument)\n")
rows=[]
for t in freq.index:
    m=cts==t
    rows.append((t, int(m.sum()), float(C[m].sum(1).mean()),
                 float((C[m]>0).sum(1).mean()), t in rare))
df=pd.DataFrame(rows, columns=["type","n","mean_total_counts","mean_genes_detected","rare"])
print(df.sort_values("n",ascending=False).to_string(index=False,
      float_format=lambda v: f"{v:,.0f}"))
print(f"\n  RARE   types: mean total counts = "
      f"{df[df.rare].mean_total_counts.mean():,.0f}, "
      f"genes detected = {df[df.rare].mean_genes_detected.mean():,.0f}")
print(f"  COMMON types: mean total counts = "
      f"{df[~df.rare].mean_total_counts.mean():,.0f}, "
      f"genes detected = {df[~df.rare].mean_genes_detected.mean():,.0f}")
