import h5py, numpy as np, pandas as pd, warnings
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
reps={'expression_PCA50': a.obsm['X_pca'][:,:50],
      'scProto_z_8d':     np.load('ck/z_f2.npy'),
      'expression_PCA8':  a.obsm['X_pca'][:,:8]}   # dimension-matched control

rows=[]
for ct in pd.unique(CT):
    m=CT==ct; y=NI[m]
    vc=pd.Series(y).value_counts(); vc=vc[vc>=30]
    if len(vc)<2: continue
    keep=np.isin(y,vc.index); y=y[keep]
    if len(y)<150: continue
    r={'cell_type':ct,'n_cells':int(m.sum()),'n_niches':len(vc),'chance':1.0/len(vc)}
    for rn,R in reps.items():
        X=StandardScaler().fit_transform(R[m][keep]); pred=np.empty_like(y)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
            c=LogisticRegression(max_iter=2000,class_weight='balanced'); c.fit(X[tr],y[tr]); pred[te]=c.predict(X[te])
        r[rn]=balanced_accuracy_score(y,pred)
    rows.append(r)
d=pd.DataFrame(rows).sort_values('n_cells',ascending=False)
for c in reps: d[c+'_lift']=d[c]-d['chance']
pd.set_option('display.width',250)
print(d[['cell_type','n_cells','n_niches','chance','expression_PCA50','expression_PCA8','scProto_z_8d']].round(3).to_string(index=False))
print("\n=============== MEAN BALANCED ACCURACY (5-fold CV) ===============")
print(f"  chance                       : {d['chance'].mean():.3f}")
for c in reps: print(f"  {c:28s} : {d[c].mean():.3f}   lift {d[c+'_lift'].mean():+.3f}")
r50,r8,rz=d['expression_PCA50_lift'].mean(),d['expression_PCA8_lift'].mean(),d['scProto_z_8d_lift'].mean()
print(f"\n  z retains {rz/r50*100:.0f}% of the niche signal in PCA50")
print(f"  z retains {rz/r8*100:.0f}% of the niche signal in a DIMENSION-MATCHED PCA8")
d.to_csv('niche_sep_with_z.csv',index=False); print("\nsaved -> niche_sep_with_z.csv")
