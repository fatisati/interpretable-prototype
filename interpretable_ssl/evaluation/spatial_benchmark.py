from sklearn.neighbors import NearestNeighbors
import numpy as np
import scvi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.cluster import KMeans


def get_scproto_embeddings(trainer, model, adata):
    adata.obsm["X_scproto"] = trainer.encode_adata(adata, model).cpu().numpy()
    print(f"scproto emb shape: {adata.obsm['X_scproto'].shape}")


def add_metacell_labels(adata, model):
    X_mapped = model.prototypes(torch.tensor(adata.obsm["X_scproto"], device="cuda"))
    prototype_assignments = torch.argmax(X_mapped, dim=1)
    adata.obs["proto_labels"] = prototype_assignments.cpu().numpy()


def tissue_continuity_score(adata, label_key="metacell", k=6):
    continuity_scores = []

    for embryo in adata.obs["batch"].unique():
        sub = adata[adata.obs["batch"] == embryo].copy()
        coords = sub.obsm["spatial"]
        labels = sub.obs[label_key].values

        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        _, neighbors = nn.kneighbors(coords)

        scores = [
            np.mean(labels[i] == labels[neighbors[i][1:]])  # skip self
            for i in range(len(labels))
        ]
        continuity_scores.extend(scores)

    return np.mean(continuity_scores)


def set_random_metacells(n, adata):
    # Set a random seed for reproducibility (optional)
    np.random.seed(42)

    # Assign random integer labels between 0 and N (e.g., 10 classes)
    n_labels = n
    adata.obs["random_metacells"] = np.random.randint(0, n_labels, size=adata.n_obs)


def baseline_continuity(adata, ref_adata, latent_dim, num_proto):
    adata.obs["pca_kmeans_500"] = (
        KMeans(n_clusters=num_proto, random_state=0)
        .fit_predict(adata.obsm["X_pca"][:, :latent_dim])
        .astype(str)
    )
    pca_score = tissue_continuity_score(adata, label_key="pca_kmeans_500", k=10)
    if "X_scvi" not in adata.obsm:
        add_scvi_embeddings(ref_adata, adata, latent_dim)
    adata.obs["scvi_kmeans_500"] = (
        KMeans(n_clusters=num_proto, random_state=0)
        .fit_predict(adata.obsm["X_scvi"][:, :latent_dim])
        .astype(str)
    )
    scvi_score = tissue_continuity_score(adata, label_key="scvi_kmeans_500", k=10)
    return pca_score, scvi_score


def add_scvi_embeddings(ref_adata, adata, latent_dim):
    # Set up the reference AnnData
    scvi.model.SCVI.setup_anndata(ref_adata)

    # Train SCVI on the reference data
    model = scvi.model.SCVI(ref_adata, n_latent=latent_dim)
    model.train(max_epochs=100)
    # Encode all data (ref_adata + others) into latent space
    scvi.model.SCVI.prepare_query_anndata(adata, model)
    adata.obsm[f"X_scvi_{latent_dim}"] = model.get_latent_representation(adata)


def evaluate_embedding_f1(
    ref_adata, query_adata, latent_key="X_emb", label_key="cell_type", k=10
):
    # Get latent embeddings and labels
    X_ref = ref_adata.obsm[latent_key]
    y_ref = ref_adata.obs[label_key].values
    X_query = query_adata.obsm[latent_key]
    y_query = query_adata.obs[label_key].values

    # Train k-NN classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_ref, y_ref)

    # Predict and compute F1 scores
    y_pred = clf.predict(X_query)
    return {
        "f1_micro": f1_score(y_query, y_pred, average="micro"),
        "f1_macro": f1_score(y_query, y_pred, average="macro"),
        "f1_weighted": f1_score(y_query, y_pred, average="weighted"),
    }


def assign_embedding_by_index(full_adata, target_adata, obsm_key="X_emb"):
    """
    Assigns embeddings from full_adata.obsm[obsm_key] to target_adata.obsm[obsm_key]
    by matching obs_names.
    """
    emb_dict = {
        idx: vec for idx, vec in zip(full_adata.obs_names, full_adata.obsm[obsm_key])
    }
    target_adata.obsm[obsm_key] = np.array(
        [emb_dict[idx] for idx in target_adata.obs_names]
    )


def evaluate_multiple_embeddings(
    ref_adata, query_adata, embeddings, label_key="cell_type", k=50
):
    """
    embeddings: list of (name, latent_key) tuples, e.g.
                [("scvi", "X_scvi"), ("scproto", "X_scproto")]
    Returns: pd.DataFrame with rows for each embedding, columns: f1_micro, f1_macro, f1_weighted
    """
    results = []
    for name, latent_key in embeddings:
        scores = evaluate_embedding_f1(
            ref_adata, query_adata, latent_key=latent_key, label_key=label_key, k=k
        )
        scores["model"] = name
        results.append(scores)

    df = pd.DataFrame(results)
    return df.set_index("model")


def celltype_purity(adata, label_key="metacell", celltype_key="cell_type"):
    groups = adata.obs.groupby(label_key)[celltype_key]
    purities = [v.value_counts().max() / len(v) for _, v in groups]
    return np.mean(purities)


def merge_labels_embeddings(adata, ref, query, keys):
    combined = ref.concatenate(query, index_unique=None)

    # Reorder to match original adata index
    combined = combined[adata.obs_names]

    for key in keys:
        if key in combined.obsm:
            adata.obsm[key] = combined.obsm[key]
        elif key in combined.obs:
            adata.obs[key] = combined.obs[key]


def evaluate_trainer(trainer):
    model = trainer.load_model()
    adata = trainer.dataset.adata
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
    if "X_scproto" not in adata.obsm:
        get_scproto_embeddings(trainer, model, adata)
    add_metacell_labels(adata, model)
    c_purity = celltype_purity(adata, "proto_labels")
    b_purity = celltype_purity(adata, "proto_labels", "batch")
    tc_score = tissue_continuity_score(adata, label_key="proto_labels", k=10)

    print(f"c, b purity: {c_purity}, {b_purity}, tc_score: {tc_score}")
    r, q = trainer.ref.adata, trainer.query.adata
    assign_embedding_by_index(adata, r, "X_scproto")
    assign_embedding_by_index(adata, q, "X_scproto")

    embeddings = [
        ("scproto", "X_scproto"),
    ]

    df = evaluate_multiple_embeddings(r, q, embeddings)
    return df
