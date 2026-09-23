import h5py, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
from anndata._io.specs.registry import _REGISTRY, IOSpec
@_REGISTRY.register_read(h5py.Dataset, IOSpec('null','0.1.0'))
def _read_null(e,_reader): return None
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

a=ad.read_h5ad('data/NSCLC_3D_section_28.h5ad')
CT=a.obs['celltypes'].astype(str).values; NI=a.obs['niches_2D'].astype(str).values
z=np.load('ck/z_f2.npy'); P=np.load('ck/proto_vectors.npy')
eps=torch.load('ck/f2.pth',map_location='cpu',weights_only=False).get('epsilon',0.05)
S=torch.softmax(torch.from_numpy(z@P.T).float()/float(eps),dim=1).numpy()   # soft assignment, 200-d
hard=pd.read_csv('ck/cell_assignments.csv',dtype={'cell_id':str}).set_index('cell_id')['metacell_id'].reindex(a.obs_names).values
H=pd.get_dummies(pd.Series(hard)).values.astype(np.float32)                 # hard label, one-hot
print(f"epsilon(temperature)={eps}   soft-assign entropy mean={float(-(S*np.log(S+1e-9)).sum(1).mean()):.3f} nats (max {np.log(200):.2f})")

reps={'z_8d':z,'soft_assign_200d':S,'hard_metacell_onehot':H}
rows=[]
for ct in pd.unique(CT):
    m=CT==ct; y=NI[m]; vc=pd.Series(y).value_counts(); vc=vc[vc>=30]
    if len(vc)<2: continue
    keep=np.isin(y,vc.index); y=y[keep]
    if len(y)<150: continue
    r={'cell_type':ct,'chance':1.0/len(vc)}
    for rn,R in reps.items():
        Xr=R[m][keep]; Xr=Xr[:,np.asarray(Xr).std(0)>1e-9]
        X=np.nan_to_num(StandardScaler().fit_transform(Xr)); pred=np.empty_like(y)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
            c=LogisticRegression(max_iter=2000,class_weight='balanced'); c.fit(X[tr],y[tr]); pred[te]=c.predict(X[te])
        r[rn]=balanced_accuracy_score(y,pred)
    rows.append(r)
d=pd.DataFrame(rows)
print("\n====== WHERE DOES THE NICHE SIGNAL DIE? (mean balanced acc, 5-fold) ======")
base=d['chance'].mean(); print(f"  chance                        {base:.3f}")
for c in reps:
    print(f"  {c:28s}  {d[c].mean():.3f}   lift {d[c].mean()-base:+.3f}")
print(f"\n  z -> soft assignment : retains {(d['soft_assign_200d'].mean()-base)/(d['z_8d'].mean()-base)*100:.0f}% of z's niche signal")
print(f"  z -> hard metacell   : retains {(d['hard_metacell_onehot'].mean()-base)/(d['z_8d'].mean()-base)*100:.0f}% of z's niche signal")
d.to_csv('niche_sep_pipeline.csv',index=False); print("\nsaved -> niche_sep_pipeline.csv")
