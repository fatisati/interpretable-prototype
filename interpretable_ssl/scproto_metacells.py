from collections import Counter
import numpy as np

def extract_proto_labels(adata, sample_proto_sim, label_keys, k=5):
    proto_to_label = {key: [] for key in label_keys}
    n_protos = sample_proto_sim.shape[1]

    for proto in range(n_protos):
        # take similarities for this prototype across all cells
        cell_sims = sample_proto_sim[:, proto]
        # get top-k cells for this prototype
        idx = np.argsort(cell_sims)[::-1][:k]

        for label_key in label_keys:
            labels = adata.obs[label_key].iloc[idx]
            majority = Counter(labels).most_common(1)[0][0]
            proto_to_label[label_key].append(majority)
    return proto_to_label


def generate_metacell_adata(metacells, proto_labels):
    import torch
    import pandas as pd
    import anndata as ad

    X = metacells.detach().cpu().numpy() if torch.is_tensor(metacells) else np.asarray(metacells)
        
    obs_dict = {}

    for key, vals in proto_labels.items():
        obs_dict[key] = pd.Categorical(vals) if not isinstance(vals, pd.Categorical) else vals
    obs = pd.DataFrame(obs_dict, index=[f"proto_{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=X, obs=obs)
    return adata
