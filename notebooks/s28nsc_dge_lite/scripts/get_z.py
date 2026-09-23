import h5py, numpy as np, pandas as pd, torch, torch.nn.functional as F
from anndata._io.specs.registry import _REGISTRY, IOSpec
@_REGISTRY.register_read(h5py.Dataset, IOSpec('null','0.1.0'))
def _read_null(e,_reader): return None
import anndata as ad

sd = torch.load('ck/f2.pth', map_location='cpu', weights_only=False)['model_state_dict']
W_e, b_e = sd['scpoli_cvae.encoder.FC.L0.expr_L.weight'], sd['scpoli_cvae.encoder.FC.L0.expr_L.bias']
W_c      = sd['scpoli_cvae.encoder.FC.L0.cond_L.weight']
emb      = sd['scpoli_cvae.embeddings.0.weight']          # (1,10) -> single condition
W_m, b_m = sd['scpoli_cvae.encoder.mean_encoder.weight'], sd['scpoli_cvae.encoder.mean_encoder.bias']
P        = sd['prototypes.weight']                         # (200,8)
cond_vec = (emb @ W_c.T).squeeze(0)                        # constant across cells

a = ad.read_h5ad('data/NSCLC_3D_section_28.h5ad')
saved = pd.read_csv('ck/cell_assignments.csv', dtype={'cell_id':str}).set_index('cell_id')
saved = saved['metacell_id'].reindex(a.obs_names).values

Xs = {'adata.X(log1p-norm)': np.asarray(a.X.todense() if hasattr(a.X,'todense') else a.X, dtype=np.float32)}
cnt = a.layers['counts']
cnt = np.asarray(cnt.todense() if hasattr(cnt,'todense') else cnt, dtype=np.float32)
Xs['log1p(counts)'] = np.log1p(cnt)
Xs['counts_raw']    = cnt

def encode(X, act):
    z=[]
    for i in range(0, X.shape[0], 8192):
        h = torch.from_numpy(X[i:i+8192]) @ W_e.T + b_e + cond_vec
        if act=='relu': h = F.relu(h)
        z.append(h @ W_m.T + b_m)
    return torch.cat(z).numpy()

best=None
for xn, X in Xs.items():
    for act in ('relu','identity'):
        z = encode(X, act)
        for metric in ('dotp','cos'):
            zz, PP = (z, P.numpy())
            if metric=='cos':
                zz = zz/ (np.linalg.norm(zz,axis=1,keepdims=True)+1e-9)
                PP = PP/ (np.linalg.norm(PP,axis=1,keepdims=True)+1e-9)
            pred = (zz @ PP.T).argmax(1)
            # assignments may be relabelled: score by best 1-1 agreement via mode matching
            agree = pd.crosstab(pred, saved).max(1).sum()/len(saved)
            print(f"  X={xn:22s} act={act:8s} metric={metric:5s} -> assignment agreement {agree:.4f}")
            if best is None or agree>best[0]: best=(agree,xn,act,metric,z)
print(f"\nBEST: agreement {best[0]:.4f}  X={best[1]} act={best[2]} metric={best[3]}")
np.save('ck/z_f2.npy', best[4]); print("saved ck/z_f2.npy", best[4].shape)
