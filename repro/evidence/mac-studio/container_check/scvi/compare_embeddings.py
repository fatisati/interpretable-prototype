"""Compare baseline run folders pairwise: coordinates, neighbours, clusters."""
import sys, hashlib, itertools, json
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari
def nn(x, k=15):
    i = NearestNeighbors(n_neighbors=k+1, algorithm="brute").fit(x).kneighbors(x, return_distance=False)
    return np.sort(i[:, 1:], axis=1)
runs = {}
for d in sys.argv[1:]:
    df = pd.read_csv(f"{d}/embedding.csv"); c = json.load(open(f"{d}/config.json"))
    lat = [c_ for c_ in df.columns if c_.startswith("latent_")]
    runs[d.rstrip("/").split("/")[-1]] = (df, df[lat].to_numpy(np.float64), hashlib.sha256(open(f"{d}/embedding.csv","rb").read()).hexdigest()[:12], c)
for n,(df,x,h,c) in runs.items():
    print(f"{n}: sha {h}, epochs {c.get('epochs_trained')}, clusters train/full {df.cluster_train_run.nunique()}/{df.cluster_full_run.nunique()}")
for a,b in itertools.combinations(runs,2):
    (da,xa,ha,_),(db,xb,hb,_) = runs[a],runs[b]
    assert (da.cell_id==db.cell_id).all()
    d = np.abs(xa-xb); share = (nn(xa)==nn(xb)).all(axis=1).mean()
    print(f"\n{a} vs {b}: file identical {ha==hb}")
    print(f"   coordinates: max diff {d.max():.3g}, median {np.median(d):.3g} (latent sd {xa.std():.2f})")
    print(f"   same 15 neighbours: {share:.2%} of cells")
    for col in ("cluster_train_run","cluster_full_run"):
        k = (da[col].notna() & db[col].notna()).to_numpy()
        print(f"   {col}: ARI {ari(da.loc[k,col].astype(str), db.loc[k,col].astype(str)):.4f}")
