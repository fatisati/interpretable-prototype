import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import torch
import numpy as np
import pandas as pd
from collections import Counter
import anndata


def plot_marker_gene_expressions(
    adata,
    cell_types_to_filter=["CD4+ T cells", "CD8+ T cells"],
    x_gene="CD4",
    y_gene="CD8A",
    cell_type_column="cell_type",
    raw_adata=None,
):
    """
    Plot CD8A vs TYROBP expression colored by density and cell type.

    Args:
        adata (AnnData): The input AnnData object containing gene expression data.
        cell_types_to_filter (list): List of cell types to include in the plot.
        x_gene (str): The name of the gene to plot on the x-axis.
        y_gene (str): The name of the gene to plot on the y-axis.
        cell_type_column (str): Column name in adata.obs specifying cell types.

    Returns:
        None
    """
    # Step 1: Filter cells based on cell type
    adata_filtered = adata[adata.obs[cell_type_column].isin(cell_types_to_filter)]
    print(len(adata_filtered))
    print(len(adata_filtered.var))
    # Convert sparse matrix to dense if necessary, then flatten
    x = adata_filtered[:, x_gene].X
    y = adata_filtered[:, y_gene].X

    # Check if x and y are sparse, convert to dense if needed
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(y, "toarray"):
        y = y.toarray()

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()

    # Step 3: Get cell type information
    cell_types = adata_filtered.obs[cell_type_column]

    # Step 4: Compute density for density-colored scatter plot
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Step 5: Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # Plot 1: Density-colored scatter plot
    axes[0].scatter(x, y, c=density, cmap="viridis", s=5, alpha=0.8)
    axes[0].set_title(f"{y_gene} vs {x_gene} (Density Colored)")
    axes[0].set_xlabel(f"{x_gene} Expression")
    axes[0].set_ylabel(f"{y_gene} Expression")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis"), ax=axes[0], label="Density"
    )

    # Plot 2: Cell type-colored scatter plot
    unique_cell_types = cell_types_to_filter
    palette = sns.color_palette("hsv", len(unique_cell_types))
    color_dict = dict(zip(unique_cell_types, palette))
    for cell_type in unique_cell_types:
        idx = cell_types == cell_type
        axes[1].scatter(
            x[idx],
            y[idx],
            c=[color_dict[cell_type]] * sum(idx),
            label=cell_type,
            s=5,
            alpha=0.8,
        )

    if raw_adata is not None:
        for cell_type in unique_cell_types:

            cell_adata = raw_adata[raw_adata.obs[cell_type_column] == cell_type]
            x = cell_adata[:, x_gene].X
            y = cell_adata[:, y_gene].X
            xmean = np.mean(x)
            ymean = np.mean(y)
            axes[0].scatter(
                xmean,
                ymean,
                facecolor=color_dict[cell_type],
                edgecolor="black",
                s=20,
                label=f"{cell_type} Mean",
                alpha=0.5,
                linewidth=1,
            )
            axes[1].scatter(
                xmean,
                ymean,
                facecolor=color_dict[cell_type],
                edgecolor="black",
                s=20,
                label=f"{cell_type} Mean",
                alpha=0.5,
                linewidth=1,
            )

    axes[1].set_title(f"{y_gene} vs {x_gene} (Cell Type Colored)")
    axes[1].set_xlabel(f"{x_gene} Expression")
    axes[1].set_ylabel(f"{y_gene} Expression")
    axes[1].legend(title="Cell Type", loc="best", markerscale=3)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return plt


def get_proto_sample_ind(scores, proto_id, k=10, use_knn=True):
    if use_knn:
        prototype_scores = scores[:, proto_id]

        # Find the indices of the k-nearest samples
        return np.argsort(-prototype_scores)[:k]
    else:
        # score is np arr
        # for each cell the id of assigned proto
        sample_proto_id = scores.argmax(1)
        # return sample ids which assigned to proto_id
        return np.where(sample_proto_id == proto_id)[0]


def assign_prototype_labels(
    adata, scores, prot_cnts, k=10, cell_type_column="cell_type", use_knn=True
):
    # # Ensure similarity_tensor is a numpy array
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # # Step 1: Assign each sample to the prototype with the highest similarity
    # prototype_assignments = np.argmax(similarity_tensor, axis=1)

    # Step 3: Calculate majority cell type and confidence for each prototype
    prototype_labels = []
    prototype_confidences = []

    for prototype in range(prot_cnts):
        proto_sample_ind = get_proto_sample_ind(scores, prototype, use_knn=use_knn)

        # Get the cell types of the k-nearest samples
        proto_sample_labels = adata.obs.iloc[proto_sample_ind][cell_type_column]

        if len(proto_sample_labels) > 0:
            # Count the frequency of each cell type
            label_counts = Counter(proto_sample_labels)

            # Determine the majority cell type and its confidence
            majority_label, majority_count = label_counts.most_common(1)[0]
            confidence = majority_count / k
        else:
            # Handle prototypes with no k-nearest samples
            majority_label = "Unknown"
            confidence = 0.0

        prototype_labels.append(majority_label)
        prototype_confidences.append(confidence)

    # Step 4: Create a DataFrame for prototype labels and confidences
    prototype_df = pd.DataFrame(
        {
            "prototype": range(prot_cnts),
            "prototype_label": prototype_labels,
            "prototype_confidence": prototype_confidences,
        }
    )

    return prototype_df


def generate_proto_adata(x, cell_types, gene_panel=None, confidence=None):
    """
    Generate an AnnData object for prototypes from input data.

    Args:
        x (np.ndarray or torch.Tensor): A 2D array or tensor of shape (n_prototypes, n_features),
                                        where each row represents a prototype's data.
        cell_types (list): A list of cell type labels for each prototype.
        gene_panel (list): A list of gene names corresponding to the features (columns) in x.

    Returns:
        AnnData: A new AnnData object containing prototype data, cell types, and gene panel.
    """
    # Convert x to numpy if it's a PyTorch tensor
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Validate input dimensions
    assert (
        len(cell_types) == x.shape[0]
    ), "Number of cell types must match the number of prototypes."

    # Create an AnnData object
    proto_adata = anndata.AnnData(X=x)

    # Add cell types to proto_adata.obs
    proto_adata.obs["cell_type"] = pd.Categorical(cell_types)

    if confidence is not None:
        proto_adata.obs["confidence"] = confidence
    # Add gene panel to proto_adata.var
    if gene_panel is not None:
        proto_adata.var.index = pd.Index(gene_panel)

    return proto_adata
