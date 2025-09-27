import numpy as np
import faiss
from sklearn.decomposition import PCA


def get_knn(X, k):
    _, d = X.shape
    X = X.astype("float32")  # FAISS needs float32

    # --- FAISS exact L2 index ---
    index = faiss.IndexFlatL2(d)
    index.add(X)

    # Search k+1 because first neighbor = self
    D, I = index.search(X, k + 1)  # shapes: (N, k+1)
    return D, I


def homogeneity(X, I):
    N, d = X.shape
    k = I.shape[1]
    scores = np.zeros(N)

    for i in range(N):
        neigh = X[I[i, 1:]]  # exclude self
        pca = PCA(n_components=min(d, k - 1))
        pca.fit(neigh)
        eigvals = pca.explained_variance_ratio_
        entropy = -(eigvals * np.log(eigvals + 1e-12)).sum()
        scores[i] = 1.0 / (1.0 + entropy)

    return scores

def calc_sigma(D, l):
    knn_dists = np.sqrt(D)

    # take distance to l-th neighbor (skip self at position 0)
    lnn = knn_dists[:, l]

    # back to torch
    return np.clip(lnn, 1e-8, None) # avoid zeros

def compute_row_marginals(sigma, heterogeneity, alpha=1.0, beta=1.0, clip=(0.25, 4.0)):
    sigma_w = (sigma / sigma.mean()) ** alpha
    hetero_w = (heterogeneity / heterogeneity.mean()) ** beta
    w = sigma_w * hetero_w
    w = np.clip(w, clip[0], clip[1])
    r = w / w.sum()  # row marginals
    return r
