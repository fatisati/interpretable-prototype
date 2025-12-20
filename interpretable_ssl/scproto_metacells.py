from collections import Counter
import numpy as np
from scipy.special import softmax

def extract_proto_labels(
    adata, sample_proto_sim, label_keys, tau=0.05, similarity = 'normal'
):
    if similarity == 'normal':
        proto_to_label = {k: [] for k in label_keys}
        labels = {k: adata.obs[k].values for k in label_keys}

        assign = sample_proto_sim.argmax(axis=1)

        for p in range(sample_proto_sim.shape[1]):
            idx = np.where(assign == p)[0]
            for k in label_keys:
                if len(idx) == 0:
                    proto_to_label[k].append(None)
                else:
                    vals, cnts = np.unique(labels[k][idx], return_counts=True)
                    proto_to_label[k].append(vals[cnts.argmax()])

        return proto_to_label
        
    proto_to_label = {key: [] for key in label_keys}
    labels_arr = {k: adata.obs[k].values for k in label_keys}

    for p in range(sample_proto_sim.shape[1]):
        # w = softmax(sample_proto_sim[:, p] / tau)
        w = sample_proto_sim[:, p]
        for key in label_keys:
            score = {}
            for lab, s in zip(labels_arr[key], w):
                score[lab] = score.get(lab, 0.0) + s
            proto_to_label[key].append(max(score, key=score.get))

    return proto_to_label

def extract_proto_labels_v1(adata, sample_proto_sim, label_keys, k=5):
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
