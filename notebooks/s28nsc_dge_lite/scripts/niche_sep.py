import h5py, numpy as np, pandas as pd
from anndata._io.specs.registry import _REGISTRY, IOSpec
@_REGISTRY.register_read(h5py.Dataset, IOSpec('null','0.1.0'))
def _read_null(elem,_reader): return None
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

a = ad.read_h5ad('data/NSCLC_3D_section_28.h5ad')
CT, NI = a.obs['celltypes'].astype(str).values, a.obs['niches_2D'].astype(str).values
reps = {'expression_PCA': a.obsm['X_pca'][:, :50], 'spatial_xy': a.obsm['spatial']}

rows=[]
for ct in pd.unique(CT):
    m = CT==ct
    y = NI[m]
    vc = pd.Series(y).value_counts()
    vc = vc[vc>=30]                     # niches with enough cells in this cell type
    if len(vc)<2: continue
    keep = np.isin(y, vc.index); y=y[keep]
    if len(y)<150: continue
    r={'cell_type':ct,'n_cells':int(m.sum()),'n_niches':len(vc),'chance':1.0/len(vc)}
    for rn, R in reps.items():
        X = StandardScaler().fit_transform(R[m][keep])
        sk = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = np.empty_like(y)
        for tr,te in sk.split(X,y):
            clf = LogisticRegression(max_iter=1500, class_weight='balanced', C=1.0)
            clf.fit(X[tr],y[tr]); pred[te]=clf.predict(X[te])
        r[rn]=balanced_accuracy_score(y,pred)
    rows.append(r)

d=pd.DataFrame(rows).sort_values('n_cells',ascending=False)
d['expr_lift']=d['expression_PCA']-d['chance']
d['spat_lift']=d['spatial_xy']-d['chance']
pd.set_option('display.width',200)
print(d.round(3).to_string(index=False))
print("\n================ SUMMARY (balanced accuracy, 5-fold CV) ================")
print(f"cell types tested                 : {len(d)}")
print(f"mean chance level                 : {d['chance'].mean():.3f}")
print(f"mean accuracy from EXPRESSION PCA : {d['expression_PCA'].mean():.3f}   lift {d['expr_lift'].mean():+.3f}")
print(f"mean accuracy from SPATIAL xy     : {d['spatial_xy'].mean():.3f}   lift {d['spat_lift'].mean():+.3f}")
d.to_csv('niche_separability.csv',index=False)
print("\nsaved -> niche_separability.csv")
