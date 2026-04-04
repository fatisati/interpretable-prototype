import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scanpy as sc
from scipy.stats import gaussian_kde
from collections import Counter
from scipy.spatial.distance import cdist
random_state = 42


def calc_umap_v2(z, proto=None, labels=None, k=10, metric="euclidean", random_state=0):
    if torch.is_tensor(z): 
        z = z.detach().cpu().numpy()
    data = z if proto is None else np.vstack(
        (z, proto.detach().cpu().numpy() if torch.is_tensor(proto) else proto)
    )
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, use_rep="X", metric=metric, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    umap = adata.obsm["X_umap"]

    if proto is None:
        return umap, None, None

    z_umap, proto_umap = umap[:len(z)], umap[len(z):]
    sim = adata.obsp["connectivities"].toarray()[len(z):, :len(z)]

    if k is not None:
        topk = np.argsort(-sim, axis=1)[:, :k]
        proto_labels = [Counter(labels[idx]).most_common(1)[0][0] for idx in topk]
    else:
        dist = cdist(z, proto if not torch.is_tensor(proto) else proto.detach().cpu().numpy())
        assign = np.argmin(dist, axis=1)
        proto_labels = [
            Counter(labels[np.where(assign == p)[0]]).most_common(1)[0][0]
            for p in range(len(proto))
        ]


    return z_umap, proto_umap, proto_labels

def calculate_umap(embeddings, prototypes=None, metric="euclidean"):
    num_cells = embeddings.shape[0]

    # Convert embeddings to numpy arrays if they are tensors
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    if prototypes is not None:
        num_prototypes = len(prototypes)

        # Convert prototypes to numpy arrays if they are tensors
        if torch.is_tensor(prototypes):
            prototypes = prototypes.detach().cpu().numpy()

        # Combine embeddings and prototypes for UMAP
        combined_data = np.vstack((embeddings, prototypes))
        combined_adata = sc.AnnData(combined_data)
        combined_adata.obs["type"] = ["cell"] * num_cells + [
            "prototype"
        ] * num_prototypes

        # Perform UMAP on the combined data with specified metric
        sc.pp.neighbors(
            combined_adata, use_rep="X", metric=metric, random_state=random_state
        )
        # sc.pp.neighbors(combined_adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(combined_adata, random_state=random_state)

        # Extract UMAP embeddings
        umap_embedding = combined_adata.obsm["X_umap"]
        cell_umap = umap_embedding[:num_cells]
        prototype_umap = umap_embedding[num_cells:]
    else:
        # Perform UMAP on the embeddings only with specified metric
        adata = sc.AnnData(embeddings)
        sc.pp.neighbors(adata, use_rep="X", metric=metric, random_state=random_state)
        sc.tl.umap(adata, random_state=random_state)

        # Extract UMAP embeddings
        umap_embedding = adata.obsm["X_umap"]
        cell_umap = umap_embedding
        prototype_umap = None

    return cell_umap, prototype_umap


def plot_umap(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    augmentation_labels=None,
    save_plot=True,
    save_path=None,
):
    # Determine the number of subplots based on whether augmentation labels are provided
    n_plots = 3 if augmentation_labels is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 8))

    # Optionally plot cell embeddings colored by augmentation
    if augmentation_labels is not None:
        unique_augmentations = np.unique(augmentation_labels)
        unique_colors_augmentations = plt.cm.get_cmap(
            "tab20", len(unique_augmentations)
        )
        colors_augmentations = ListedColormap(
            unique_colors_augmentations(np.linspace(0, 1, len(unique_augmentations)))
        )

        for i, aug in enumerate(unique_augmentations):
            indices = np.where(augmentation_labels == aug)[0]
            axes[0].scatter(
                cell_umap[indices, 0],
                cell_umap[indices, 1],
                label=aug,
                color=colors_augmentations(i),
                alpha=0.6,
                s=20,
            )

        if prototype_umap is not None:
            axes[0].scatter(
                prototype_umap[:, 0],
                prototype_umap[:, 1],
                color="white",
                edgecolor="black",
                s=100,
                marker="o",
                label="Prototypes",
            )

        axes[0].set_title("UMAP of Cell Embeddings with Augmentations Highlighted")
        axes[0].set_xlabel("UMAP Dimension 1")
        axes[0].set_ylabel("UMAP Dimension 2")

    # Plot cell embeddings colored by cell type
    unique_cell_types = np.unique(cell_types)
    unique_colors = plt.cm.get_cmap("tab20", len(unique_cell_types))
    colors = ListedColormap(unique_colors(np.linspace(0, 1, len(unique_cell_types))))

    for i, cell_type in enumerate(unique_cell_types):
        indices = np.where(cell_types == cell_type)[0]
        axes[1 if augmentation_labels is not None else 0].scatter(
            cell_umap[indices, 0],
            cell_umap[indices, 1],
            label=cell_type,
            color=colors(i),
            alpha=0.6,
            s=20,
        )

    if prototype_umap is not None:
        axes[1 if augmentation_labels is not None else 0].scatter(
            prototype_umap[:, 0],
            prototype_umap[:, 1],
            color="white",
            edgecolor="black",
            s=100,
            marker="o",
            label="Prototypes",
        )

    axes[1 if augmentation_labels is not None else 0].set_title(
        "UMAP of Cell Embeddings with Cell Types Highlighted"
    )
    axes[1 if augmentation_labels is not None else 0].set_xlabel("UMAP Dimension 1")
    axes[1 if augmentation_labels is not None else 0].set_ylabel("UMAP Dimension 2")

    # Plot cell embeddings colored by study
    unique_studies = np.unique(study_labels)
    unique_colors_studies = plt.cm.get_cmap("tab20", len(unique_studies))
    colors_studies = ListedColormap(
        unique_colors_studies(np.linspace(0, 1, len(unique_studies)))
    )

    for i, study in enumerate(unique_studies):
        indices = np.where(study_labels == study)[0]
        axes[2 if augmentation_labels is not None else 1].scatter(
            cell_umap[indices, 0],
            cell_umap[indices, 1],
            label=study,
            color=colors_studies(i),
            alpha=0.6,
            s=20,
        )

    if prototype_umap is not None:
        axes[2 if augmentation_labels is not None else 1].scatter(
            prototype_umap[:, 0],
            prototype_umap[:, 1],
            color="white",
            edgecolor="black",
            s=100,
            marker="o",
            label="Prototypes",
        )

    axes[2 if augmentation_labels is not None else 1].set_title(
        "UMAP of Cell Embeddings with Studies Highlighted"
    )
    axes[2 if augmentation_labels is not None else 1].set_xlabel("UMAP Dimension 1")
    axes[2 if augmentation_labels is not None else 1].set_ylabel("UMAP Dimension 2")

    # Adjust layout to make space for the legends
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    # Adding legends below the plots
    if augmentation_labels is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.20, -0.1),
            ncol=3,
            title="Augmentations",
        )

    handles, labels = axes[
        1 if augmentation_labels is not None else 0
    ].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.50 if n_plots == 3 else 0.25, -0.1),
        ncol=3,
        title="Cell Types",
    )

    handles, labels = axes[
        2 if augmentation_labels is not None else 1
    ].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.80 if n_plots == 3 else 0.75, -0.1),
        ncol=3,
        title="Studies",
    )

    if save_plot:
        plt.savefig(f"{save_path}/ref-umap.png", bbox_inches="tight")
    else:
        plt.show()


from scipy.stats import gaussian_kde

import matplotlib.colors as mcolors


def plot_3umaps(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    prototype_labels=None,
    save_plot=True,
    save_path_list=None,
    w = None,
    w_label = None
):

    def plot_scatter(
        ax,
        data_umap,
        labels=None,
        label_title=None,
        prototypes=None,
        prototype_labels=None,
        exclude_prototypes=False,
        color_by_density=False,
        w=None,
    ):
        if color_by_density:
            if w is None:
                # Calculate point density
                xy = np.vstack([data_umap[:, 0], data_umap[:, 1]])
                density = gaussian_kde(xy)(xy)
                c = density
            else:
                c = w
            scatter = ax.scatter(
                data_umap[:, 0],
                data_umap[:, 1],
                c=c,
                cmap="viridis",
                s=20,
                alpha=0.6,
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            
            cbar.set_label(w_label if w_label is not None else 'Density')
        else:
            unique_labels = np.unique(labels)
            unique_colors = plt.cm.get_cmap("tab20", len(unique_labels))

            colors = ListedColormap(
                unique_colors(np.linspace(0, 1, len(unique_labels)))
            )

            for i, label in enumerate(unique_labels):
                indices = np.where(labels == label)[0]
                ax.scatter(
                    data_umap[indices, 0],
                    data_umap[indices, 1],
                    label=label,
                    color=colors(i),
                    alpha=0.6,
                    s=20,
                )

                if prototypes is not None and not exclude_prototypes:
                    prototype_indices = [
                        idx for idx, lbl in enumerate(prototype_labels) if lbl == label
                    ]
                    if prototype_indices:

                        ax.scatter(
                            prototypes[prototype_indices, 0],
                            prototypes[prototype_indices, 1],
                            color=colors(i),
                            edgecolor="black",
                            linewidth=0.5,
                            s=70,
                            zorder=2,
                            alpha=0.8,
                            # label="Prototypes",
                        )

            # Plot prototypes with labels not in cell types
            if prototypes is not None:
                extra_indices = [
                    idx
                    for idx, lbl in enumerate(prototype_labels)
                    if lbl not in unique_labels
                ]
                if extra_indices:
                    ax.scatter(
                        prototypes[extra_indices, 0],
                        prototypes[extra_indices, 1],
                        color="white",  # Use a distinct color for extra prototypes
                        edgecolor="black",
                        linewidth=0.5,
                        s=70,
                        zorder=3,
                        alpha=0.9,
                        label="Extra Prototypes",  # Add a legend for extra prototypes
                    )
        if label_title:
            ax.set_title(f"UMAP of Cell Embeddings with {label_title} Highlighted")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def add_legend(fig, ax, title, bbox_anchor):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=bbox_anchor,
            ncol=3,
            title=title,
        )

    # Determine the number of subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot cell embeddings colored by density (without prototypes)
    plot_scatter(axes[0], cell_umap, color_by_density=True, w=w)
    axes[0].set_title(f"UMAP of Cell Embeddings Colored by {w_label if w_label is not None else 'Density'}")

    # Plot cell embeddings colored by cell types (with prototypes)
    plot_scatter(
        axes[1], cell_umap, cell_types, "Cell Types", prototype_umap, prototype_labels
    )
    if np.unique(cell_types).shape[0] < 50:
        add_legend(fig, axes[1], "Cell Types", (0.50, -0.1))

    # Plot cell embeddings colored by studies (without prototypes)
    plot_scatter(axes[2], cell_umap, study_labels, "Studies", exclude_prototypes=True)
    add_legend(fig, axes[2], "Studies", (0.85, -0.1))

    plt.tight_layout(rect=[0, 0.2, 1, 1])

    if (save_path_list is not None) and save_plot:
        for save_path in save_path_list:
            fig.savefig(
                save_path, bbox_inches="tight", pad_inches=0.5
            )  # Increase pad_inches as needed

    return fig


def plot_3umaps2(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    proto_labels,
    proto_confidence=None,
    save_plot=True,
    save_path_list=None,
    proto_size_base=100,
):

    def plot_scatter(
        ax,
        data_umap,
        labels=None,
        label_title=None,
        prototypes=None,
        proto_labels=None,
        proto_confidence=None,
        exclude_prototypes=False,
        color_by_density=False,
        label_to_color_idx=None,
        shared_colors=None,
    ):
        if color_by_density:
            # Calculate point density
            xy = np.vstack([data_umap[:, 0], data_umap[:, 1]])
            density = gaussian_kde(xy)(xy)
            scatter = ax.scatter(
                data_umap[:, 0],
                data_umap[:, 1],
                c=density,
                cmap="viridis",
                s=20,
                alpha=0.6,
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label("Density")
        else:
            for label, idx in label_to_color_idx.items():
                indices = np.where(np.array(labels) == label)[0]
                ax.scatter(
                    data_umap[indices, 0],
                    data_umap[indices, 1],
                    label=label,
                    color=shared_colors(idx),
                    alpha=0.6,
                    s=20,
                )

        # Visualize prototypes
        if prototypes is not None and not exclude_prototypes:
            visualize_prototypes(
                ax,
                prototypes,
                proto_labels,
                proto_confidence,
                proto_size_base,
                label_to_color_idx,
                shared_colors,
            )

        if label_title:
            ax.set_title(f"UMAP of Cell Embeddings with {label_title} Highlighted")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def visualize_prototypes(
        ax,
        prototypes,
        proto_labels,
        proto_confidence,
        proto_size_base,
        label_to_color_idx,
        shared_colors,
    ):
        """
        Visualizes prototypes on the UMAP plot.
        - Prototypes are colored by their labels and sized by confidence.
        """
        if proto_confidence is None:
            proto_confidence = np.ones(prototypes.shape[0])  # Default confidence

        for label, idx in label_to_color_idx.items():
            indices = np.where(np.array(proto_labels) == label)[0]
            ax.scatter(
                prototypes[indices, 0],
                prototypes[indices, 1],
                color=shared_colors(idx),
                edgecolor="k",
                linewidth=0.5,
                s=proto_size_base
                * proto_confidence[indices],  # Scale size by confidence
                alpha=0.8,
                label=f"Prototype: {label}",
            )

    def add_legend(fig, ax, title, bbox_anchor):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=bbox_anchor,
            ncol=3,
            title=title,
        )

    # Combine cell types and prototype labels for a shared colormap
    all_labels = np.unique(np.concatenate([cell_types, proto_labels]))
    label_to_color_idx = {label: idx for idx, label in enumerate(all_labels)}
    shared_colors = plt.cm.get_cmap("tab20", len(all_labels))

    # Determine the number of subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot cell embeddings colored by density (without prototypes)
    plot_scatter(axes[0], cell_umap, color_by_density=True)
    axes[0].set_title("UMAP of Cell Embeddings Colored by Density")

    # Plot cell embeddings colored by cell types (with prototypes)
    plot_scatter(
        axes[1],
        cell_umap,
        cell_types,
        "Cell Types",
        prototypes=prototype_umap,
        proto_labels=proto_labels,
        proto_confidence=proto_confidence,
        label_to_color_idx=label_to_color_idx,
        shared_colors=shared_colors,
    )
    add_legend(fig, axes[1], "Cell Types", (0.50, -0.1))

    # Plot cell embeddings colored by studies (without prototypes)
    plot_scatter(
        axes[2],
        cell_umap,
        study_labels,
        "Studies",
        exclude_prototypes=True,
        label_to_color_idx=label_to_color_idx,
        shared_colors=shared_colors,
    )
    add_legend(fig, axes[2], "Studies", (0.85, -0.1))

    plt.tight_layout(rect=[0, 0.2, 1, 1])

    if save_plot and save_path_list is not None:
        for save_path in save_path_list:
            fig.savefig(
                save_path, bbox_inches="tight", pad_inches=0.5
            )  # Increase pad_inches as needed

    return fig


import matplotlib.pyplot as plt

def plot_f1_scores_per_class(f1_scores_dict, class_names):
    """
    Plots a grouped bar plot of F1 scores per class for multiple splits, with values displayed on top of the bars.

    Parameters:
    - f1_scores_dict: Dictionary where keys are split names (e.g., 'Train', 'Validation', 'Test')
      and values are lists of per-class F1 scores.
      Example: {
          'Train': [0.8, 0.7, 0.9],
          'Validation': [0.75, 0.65, 0.85],
          'Test': [0.78, 0.68, 0.88]
      }
    - class_names: List of class names corresponding to the F1 scores.
    """
    # Number of classes and splits
    num_classes = len(class_names)
    num_splits = len(f1_scores_dict)
    bar_width = 0.25
    indices = range(num_classes)

    # Create the plot
    plt.figure(figsize=(12, 6))
    for i, (split_name, f1_scores) in enumerate(f1_scores_dict.items()):
        bar_positions = [x + i * bar_width for x in indices]
        bars = plt.bar(
            bar_positions,
            f1_scores,
            width=bar_width,
            label=split_name,
        )

        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Formatting
    plt.xticks(
        [x + bar_width * (num_splits / 2 - 0.5) for x in indices],
        class_names,
        rotation=45,
        ha="right",
    )
    plt.xlabel("Class Names")
    plt.ylabel("F1 Score")
    plt.title("F1 Score comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_latent_umap_with_protos(t, color_key='niches_2D', max_cells=50000,
                                  use_model_assignments=True, joint_umap=True):
    """
    Plot UMAP of latent space with prototypes labeled by majority vote.

    Args:
        t: SCProtoTrainer object (after setup())
        color_key: obs column to color by (default 'niches_2D')
        max_cells: subsample if more cells than this
        use_model_assignments: if True, use model's soft scores for assignment
                               if False, use Euclidean distance
        joint_umap: if True, compute UMAP on cells+protos together (protos on manifold)
                    if False, place protos at centroid of assigned cells

    Returns:
        fig, z_umap, proto_umap, proto_labels
    """
    ad = t.train_ds.adata
    model = t.model

    if ad.n_obs > max_cells:
        idx = np.random.choice(ad.n_obs, max_cells, replace=False)
        ad = ad[idx].copy()

    z = t.encode_adata(ad, model, z_idx=1).detach().cpu().numpy()
    proto = model.get_prototypes().detach().cpu().numpy()
    n_protos = proto.shape[0]

    # Get model's actual assignments (scores -> argmax)
    if use_model_assignments:
        scores = t.encode_adata(ad, model, z_idx=2).detach().cpu().numpy()
        assignments = scores.argmax(axis=1)
    else:
        dist = cdist(z, proto, metric='euclidean')
        assignments = np.argmin(dist, axis=1)

    if joint_umap:
        # Joint UMAP: protos and cells together (protos live on same manifold)
        combined = np.vstack([z, proto])
        adata = sc.AnnData(combined)
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(adata)
        z_umap = adata.obsm['X_umap'][:len(z)]
        proto_umap = adata.obsm['X_umap'][len(z):]
    else:
        # Separate UMAP: place protos at centroid of assigned cells
        adata = sc.AnnData(z)
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(adata)
        z_umap = adata.obsm['X_umap']

        proto_umap = np.zeros((n_protos, 2))
        for p in range(n_protos):
            mask = assignments == p
            if mask.sum() > 0:
                proto_umap[p] = z_umap[mask].mean(axis=0)
            else:
                dists = np.linalg.norm(z - proto[p], axis=1)
                nearest = np.argmin(dists)
                proto_umap[p] = z_umap[nearest]

    # Label each prototype by majority vote of assigned cells
    labels = ad.obs[color_key].values
    proto_labels = []
    proto_sizes = []
    for p in range(n_protos):
        assigned_mask = assignments == p
        n_assigned = assigned_mask.sum()
        proto_sizes.append(n_assigned)
        if n_assigned == 0:
            proto_labels.append(None)
        else:
            assigned_labels = labels[assigned_mask]
            majority_label = Counter(assigned_labels).most_common(1)[0][0]
            proto_labels.append(majority_label)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10')
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(z_umap[mask, 0], z_umap[mask, 1],
                   c=[label_to_color[lbl]], label=lbl, alpha=0.5, s=10)

    # Plot prototypes - size proportional to n_cells, white if unused
    for i, plbl in enumerate(proto_labels):
        if plbl is None:
            c = 'white'
            size = 50
        else:
            c = label_to_color.get(plbl, 'white')
            size = max(50, min(300, proto_sizes[i] // 2))
        ax.scatter(proto_umap[i, 0], proto_umap[i, 1],
                   c=[c], edgecolor='black', s=size, linewidth=1, zorder=10)
        ax.annotate(str(i), (proto_umap[i, 0], proto_umap[i, 1]),
                    fontsize=7, ha='center', va='center')

    # Add stats to title
    n_used = sum(1 for s in proto_sizes if s > 0)
    umap_type = "joint" if joint_umap else "centroid"
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_title(f'Latent UMAP ({umap_type}) with Prototypes ({n_used}/{n_protos} used)')
    plt.tight_layout()
    return fig, z_umap, proto_umap, proto_labels


def _plot_umap_with_protos(ax, z_umap, proto_umap, labels, proto_labels, title, cmap_dict, s_cell=3, s_proto=50):
    """Plot UMAP with cells colored by label and prototypes marked."""
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(z_umap[mask, 0], z_umap[mask, 1], c=[cmap_dict[lbl]], label=lbl, alpha=0.5, s=s_cell)
    for i, plbl in enumerate(proto_labels):
        if plbl is not None:
            c = cmap_dict.get(plbl, 'white')
            ax.scatter(proto_umap[i, 0], proto_umap[i, 1], c=[c], edgecolor='black', s=s_proto, linewidth=1, zorder=10)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def _get_proto_labels_majority(assignments, labels, n_protos):
    """Assign each proto a label by majority voting."""
    proto_labels = []
    for p in range(n_protos):
        mask = assignments == p
        if mask.sum() > 0:
            proto_labels.append(Counter(labels[mask]).most_common(1)[0][0])
        else:
            proto_labels.append(None)
    return proto_labels


def compare_kmeans_scproto(t, n_clusters=None, figsize=(10, 4), color_key='niches_2D'):
    """Side-by-side UMAP: KMeans on PCA vs SCProto."""
    from sklearn.cluster import KMeans

    ad = t.train_ds.adata
    X_pca = ad.obsm['X_pca']
    labels = np.array(ad.obs[color_key].values)
    n_clusters = n_clusters or t.nmb_prototypes

    # Distinct colormap
    unique_labels = np.unique(labels)
    n = len(unique_labels)
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]
    cmap_dict = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    # === KMeans on PCA ===
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_pca)
    km_assignments = km.labels_
    km_centers = km.cluster_centers_

    # Joint UMAP for KMeans
    combined_km = np.vstack([X_pca, km_centers])
    tmp_km = sc.AnnData(combined_km)
    sc.pp.neighbors(tmp_km, use_rep='X', n_neighbors=15)
    sc.tl.umap(tmp_km)
    z_umap_km = tmp_km.obsm['X_umap'][:len(X_pca)]
    proto_umap_km = tmp_km.obsm['X_umap'][len(X_pca):]
    km_proto_labels = _get_proto_labels_majority(km_assignments, labels, n_clusters)

    # === SCProto ===
    with torch.no_grad():
        z = t.encode_adata(ad, t.model, z_idx=1).detach().cpu().numpy()
        scores = t.encode_adata(ad, t.model, z_idx=2)
    sc_assignments = scores.argmax(1).cpu().numpy()
    proto = t.model.get_prototypes().detach().cpu().numpy()
    n_protos = proto.shape[0]

    # Joint UMAP for SCProto
    combined_sc = np.vstack([z, proto])
    tmp_sc = sc.AnnData(combined_sc)
    sc.pp.neighbors(tmp_sc, use_rep='X', n_neighbors=15)
    sc.tl.umap(tmp_sc)
    z_umap_sc = tmp_sc.obsm['X_umap'][:len(z)]
    proto_umap_sc = tmp_sc.obsm['X_umap'][len(z):]
    sc_proto_labels = _get_proto_labels_majority(sc_assignments, labels, n_protos)

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _plot_umap_with_protos(axes[0], z_umap_km, proto_umap_km, labels, km_proto_labels,
                           f'KMeans on PCA ({n_clusters} clusters)', cmap_dict)
    _plot_umap_with_protos(axes[1], z_umap_sc, proto_umap_sc, labels, sc_proto_labels,
                           f'SCProto ({n_protos} prototypes)', cmap_dict)

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, markerscale=0.8)
    plt.tight_layout()
    return fig


def build_spatial_context(adata, radius, pca_key='X_pca', spatial_key='spatial'):
    """Compute spatial context as mean PCA of neighbors within radius.

    Args:
        adata: AnnData with spatial coordinates and PCA
        radius: spatial radius for neighbor search
        pca_key: key in obsm for PCA embeddings
        spatial_key: key in obsm for spatial coordinates

    Returns:
        context: (n_cells, n_pcs) array of mean neighbor embeddings
        stats: dict with neighbor statistics
    """
    from sklearn.neighbors import NearestNeighbors

    Xsp = adata.obsm[spatial_key]
    Xpca = adata.obsm[pca_key]

    nn = NearestNeighbors(radius=radius).fit(Xsp)
    neigh = nn.radius_neighbors(Xsp, return_distance=False)
    neigh = [idx[idx != i] for i, idx in enumerate(neigh)]  # exclude self

    ctx = np.stack([Xpca[idx].mean(0) if len(idx) else Xpca[i] for i, idx in enumerate(neigh)])

    n_neighbors = [len(idx) for idx in neigh]
    stats = {
        'n_empty': sum(n == 0 for n in n_neighbors),
        'mean_neighbors': np.mean(n_neighbors),
        'median_neighbors': np.median(n_neighbors),
    }

    return ctx, stats


def build_seacells_affinity(adata, n_waypoint_eigs=10, build_on='X_pca', cache_path=None):
    """Build affinity using SEACells library.

    Args:
        adata: AnnData object with PCA computed
        n_waypoint_eigs: number of eigenvectors for diffusion
        build_on: obsm key to build kernel on
        cache_path: path to save/load affinity. If exists, loads from cache.

    Returns:
        affinity: kernel matrix from SEACells
        model: SEACells model object (None if loaded from cache)
    """
    import scipy.sparse as sp

    # Check cache
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading SEACells affinity from cache: {cache_path}")
        aff = sp.load_npz(cache_path)
        return aff, None

    import SEACells.core

    ad = adata.copy()
    n_seacells = max(10, int(np.sqrt(ad.n_obs)))  # dummy, just need kernel

    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_on,
        n_SEACells=n_seacells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=False,
        verbose=True,
    )

    # Build kernel (this is what we want)
    model.construct_kernel_matrix()
    aff = model.kernel_matrix

    # Save cache
    if cache_path is not None:
        sp.save_npz(cache_path, aff)
        print(f"Saved SEACells affinity to: {cache_path}")

    return aff, model


def plot_spatial_context_comparison(
    ad,
    aff,
    color=None,
    context_key='context',
    figsize=(15, 4),
    **plot_kwargs,
):
    """Compare 3 UMAPs: PCA, Spatial Context, SEACells Affinity.

    Args:
        ad: AnnData with X_pca and context in obsm
        aff: precomputed SEACells affinity matrix
        color: obs column(s) for coloring
        context_key: obsm key for spatial context
        figsize: figure size
        **plot_kwargs: passed to sc.pl.umap

    Returns:
        dict with ad_pca, ad_ctx, ad_aff (each with X_umap)
    """
    # --- 1. PCA UMAP ---
    ad_pca = ad.copy()
    sc.pp.neighbors(ad_pca, use_rep='X_pca')
    sc.tl.umap(ad_pca)

    # --- 2. Context UMAP ---
    ad_ctx = ad.copy()
    sc.pp.neighbors(ad_ctx, use_rep=context_key)
    sc.tl.umap(ad_ctx)

    # --- 3. SEACells Affinity UMAP ---
    ad_aff = ad.copy()
    ad_aff.obsp['connectivities'] = aff.tocsr()
    ad_aff.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': None,
        'params': {'method': 'precomputed'},
    }
    sc.tl.umap(ad_aff)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    titles = ['PCA UMAP', 'Context UMAP', 'SEACells Affinity UMAP']
    ads = [ad_pca, ad_ctx, ad_aff]

    for ax, a, title in zip(axes, ads, titles):
        sc.pl.umap(a, color=color, ax=ax, show=False, title=title, legend_loc='none', **plot_kwargs)

    # Single shared legend using scanpy's colors
    if color is not None:
        color_key = f'{color}_colors'
        if color_key in ads[0].uns:
            colors_list = ads[0].uns[color_key]
            categories = ads[0].obs[color].cat.categories
        else:
            categories = sorted(ads[0].obs[color].unique())
            cmap = plt.cm.tab10 if len(categories) <= 10 else plt.cm.tab20
            colors_list = [cmap(i % cmap.N) for i in range(len(categories))]

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markersize=8, label=l)
                   for l, c in zip(categories, colors_list)]
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=9)

    plt.tight_layout()
    plt.show()

    return {'ad_pca': ad_pca, 'ad_ctx': ad_ctx, 'ad_aff': ad_aff, 'fig': fig}


def plot_aff(ad, aff, celltype=None, niches=None, celltype_key='cell_type',
             niche_key='niches_2D', color=None, k=None, random_state=0, **plot_kwargs):
    """Plot UMAP from precomputed affinity using scanpy.

    Args:
        ad: AnnData object
        aff: sparse affinity matrix
        celltype: filter to this celltype (None = all)
        niches: filter to these niches - single value or list (None = all)
        celltype_key: obs column for celltype
        niche_key: obs column for niche
        color: obs column(s) for coloring. Default: [celltype_key, niche_key]
        k: subsample to k cells (None = all)
        random_state: random seed for subsampling
        **plot_kwargs: passed to sc.pl.umap

    Returns:
        ad: AnnData with X_umap computed
    """
    ad = ad.copy()

    # Subsample
    if k is not None and k < ad.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(ad.n_obs, size=k, replace=False)
        ad = ad[idx].copy()
        aff = aff[idx][:, idx]

    # Set connectivities
    ad.obsp['connectivities'] = aff.tocsr()
    ad.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': None,
        'params': {'method': 'precomputed'},
    }
    if celltype is not None:
        ad = ad[ad.obs[celltype_key] == celltype].copy()
    if niches is not None:
        ad = ad[ad.obs[niche_key].isin(niches)].copy()

    # UMAP
    sc.tl.umap(ad)

    # Plot
    if color is None:
        color = [niche_key]
    sc.pl.umap(ad, color=color, **plot_kwargs)

    return ad


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _rbo(a, b, p=0.9):
    k = min(len(a), len(b))
    if k == 0:
        return 0.0
    overlap = 0.0
    for d in range(1, k + 1):
        overlap += len(set(a[:d]) & set(b[:d])) / d * (p ** (d - 1))
    return overlap * (1 - p)


def compare_dge_metrics(base_dir):
    """Compare DGE metrics across all methods (standalone, no trainer needed).

    Args:
        base_dir: directory containing model folders with *_dge.csv and gt/singlecell_dge.csv

    Returns:
        DataFrame with jaccard, rbo, logfc_corr, logfc_std (index = model names)
    """
    import pandas as pd

    # Load ground truth
    sc_path = os.path.join(base_dir, 'gt', 'singlecell_dge.csv')
    if not os.path.exists(sc_path):
        print(f"Ground truth not found: {sc_path}")
        return None
    sc_dge = pd.read_csv(sc_path)

    # Find all methods
    methods = {}
    for entry in os.listdir(base_dir):
        entry_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_dir):
            continue

        proto_path = os.path.join(entry_dir, 'prototype_dge.csv')
        baseline_path = os.path.join(entry_dir, 'baseline_dge.csv')

        if os.path.exists(proto_path):
            methods[entry] = pd.read_csv(proto_path)
        elif os.path.exists(baseline_path):
            methods[entry] = pd.read_csv(baseline_path)

    if not methods:
        print("No DGE files found")
        return None

    # Compute scores
    results = []
    for name, dge in methods.items():
        scores = []
        for (ct, niche), m_grp in dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
            if len(s_grp) < 10 or len(m_grp) < 3:
                continue

            m_genes = m_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            merged = m_grp.merge(s_grp, on='gene', suffixes=('_m', '_s'))
            corr = merged['logfoldchange_m'].corr(merged['logfoldchange_s']) if len(merged) > 5 else np.nan

            scores.append({
                'jaccard': _jaccard(m_genes, s_genes),
                'rbo': _rbo(m_genes, s_genes),
                'logfc_corr': corr,
            })

        if scores:
            sdf = pd.DataFrame(scores)
            results.append({
                'name': name,
                'jaccard': sdf['jaccard'].mean(),
                'rbo': sdf['rbo'].mean(),
                'logfc_corr': sdf['logfc_corr'].mean(),
                'logfc_std': dge['logfoldchange'].std(),
            })

    return pd.DataFrame(results).set_index('name').sort_values('jaccard', ascending=False)


def load_dge_results(base_dir):
    """Load all DGE results from model directory.

    Args:
        base_dir: directory containing model folders with *_dge.csv

    Returns:
        dict: {model_name: DataFrame} including 'singlecell' from gt/
    """
    import pandas as pd

    results = {}

    # Load ground truth
    sc_path = os.path.join(base_dir, 'gt', 'singlecell_dge.csv')
    if os.path.exists(sc_path):
        results['singlecell'] = pd.read_csv(sc_path)

    # Load all models
    for entry in os.listdir(base_dir):
        entry_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_dir):
            continue

        proto_path = os.path.join(entry_dir, 'prototype_dge.csv')
        baseline_path = os.path.join(entry_dir, 'baseline_dge.csv')

        if os.path.exists(proto_path):
            results[entry] = pd.read_csv(proto_path)
        elif os.path.exists(baseline_path):
            results[entry] = pd.read_csv(baseline_path)

    print(f"Loaded {len(results)} DGE results: {list(results.keys())}")
    return results


def plot_volcano_grid(dge_dict, models, niches, celltype=None, rename=None,
                      pval_thr=0.05, logfc_thr=0.5, figsize=None, top_k=5):
    """Plot volcano grid: rows=niches, cols=models.

    Args:
        dge_dict: dict from load_dge_results()
        models: list of model names to show (columns)
        niches: list of niche names to show (rows)
        celltype: filter to this celltype (None = all)
        rename: dict to rename model names for display
        pval_thr: p-value threshold for significance
        logfc_thr: logFC threshold for significance
        figsize: figure size (auto if None)
        top_k: number of top significant genes to label per subplot (by neg_log_p)

    Returns:
        Figure
    """
    n_rows = len(niches)
    n_cols = len(models)
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, niche in enumerate(niches):
        for j, model in enumerate(models):
            ax = axes[i, j]

            if model not in dge_dict:
                ax.set_title(f'{model}: not found')
                ax.axis('off')
                continue

            dge = dge_dict[model].copy()

            # Filter
            if celltype is not None:
                dge = dge[dge['celltype'] == celltype]
            dge = dge[dge['niche'] == niche]

            if len(dge) == 0:
                ax.set_title(f'{rename.get(model, model) if rename else model}\n{niche[:20]}: no data')
                ax.axis('off')
                continue

            # Volcano
            dge['neg_log_p'] = -np.log10(dge['pval'].clip(lower=1e-300))
            ax.scatter(dge['logfoldchange'], dge['neg_log_p'], s=2, alpha=0.4, c='gray')

            # Significant genes
            sig = dge[(dge['pval_adj'] < pval_thr) & (abs(dge['logfoldchange']) > logfc_thr)]
            ax.scatter(sig['logfoldchange'], sig['neg_log_p'], s=6, c='red', alpha=0.7)

            # Label top_k significant genes
            if top_k and len(sig) > 0:
                top = sig.nlargest(min(top_k, len(sig)), 'neg_log_p')
                for _, row in top.iterrows():
                    ax.annotate(row['gene'], (row['logfoldchange'], row['neg_log_p']),
                                fontsize=6, ha='center', va='bottom',
                                xytext=(0, 4), textcoords='offset points')

            # Lines
            ax.axhline(-np.log10(0.05), ls='--', c='blue', alpha=0.3)
            ax.axvline(0, ls='--', c='black', alpha=0.3)

            # Title
            display_name = rename.get(model, model) if rename else model
            ax.set_title(f'{display_name}\n{niche[:25]} ({len(sig)} sig)', fontsize=10)

            if i == n_rows - 1:
                ax.set_xlabel('LogFC')
            if j == 0:
                ax.set_ylabel('-log10(p)')

    plt.tight_layout()
    plt.show()
    return fig


def load_expression_data(base_dir, adata, models, celltype_key='celltype',
                         niche_key='niches_2D'):
    """Load metacell/single-cell expression into a dict of AnnDatas from saved files.

    For each model in base_dir:
      - Loads decoded_prototypes.npy (scproto) or avg_expression.npy (baselines)
        → shape (K, n_genes), gene order matches adata.var_names
      - Loads prototype_labels.csv (scproto) or derives labels from clusters.npz
        → niche + celltype per metacell
      - Builds an AnnData with expression in .X, niche/celltype in .obs

    For 'singlecell': wraps the original adata directly (no aggregation).

    Args:
        base_dir: directory containing model folders (same as load_dge_results)
        adata: original single-cell AnnData (for gene names + singlecell data)
        models: list of model names to load
        celltype_key: obs column for celltype in adata
        niche_key: obs column for niche labels in adata

    Returns:
        dict {model_name: AnnData} ready for plot_expression_grid
    """
    import pandas as pd
    import anndata

    gene_names = list(adata.var_names)
    result = {}

    for model in models:
        if model == 'singlecell':
            result['singlecell'] = adata
            continue

        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            print(f"  {model}: directory not found, skipping")
            continue

        # Load expression matrix
        decoded_path = os.path.join(model_dir, 'decoded_prototypes.npy')
        avg_path = os.path.join(model_dir, 'avg_expression.npy')

        if os.path.exists(decoded_path):
            expr = np.load(decoded_path)
        elif os.path.exists(avg_path):
            expr = np.load(avg_path)
        else:
            print(f"  {model}: no expression file found, skipping")
            continue

        # Load labels
        labels_path = os.path.join(model_dir, 'prototype_labels.csv')
        clusters_path = os.path.join(model_dir, 'clusters.npz')

        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            obs_df = pd.DataFrame({
                niche_key: labels_df['niche'].values,
                celltype_key: labels_df['celltype'].values,
            })
        elif os.path.exists(clusters_path):
            # Derive labels from assignments via majority vote
            data = np.load(clusters_path, allow_pickle=True)
            assignments = data['assignments']
            cluster_ids = np.unique(assignments)

            niche_vals = adata.obs[niche_key].values
            ct_vals = adata.obs[celltype_key].values

            mc_niche, mc_ct = [], []
            for c in cluster_ids:
                mask = assignments == c
                vals, counts = np.unique(niche_vals[mask], return_counts=True)
                mc_niche.append(vals[counts.argmax()])
                vals, counts = np.unique(ct_vals[mask], return_counts=True)
                mc_ct.append(vals[counts.argmax()])

            obs_df = pd.DataFrame({niche_key: mc_niche, celltype_key: mc_ct})
        else:
            print(f"  {model}: no labels found, skipping")
            continue

        ad = anndata.AnnData(X=expr, obs=obs_df)
        ad.var_names = gene_names
        result[model] = ad
        print(f"  {model}: {ad.n_obs} metacells x {ad.n_vars} genes")

    print(f"Loaded expression for {len(result)} models: {list(result.keys())}")
    return result


def diagnostic_gene_sets(dge_dict, niche, celltype, sc_key='singlecell',
                         proto_key=None, baseline_key=None,
                         top_k=10, pval_thr=0.05, logfc_thr=0.25):
    """Find 3 diagnostic gene sets for evaluating metacell methods.

    Group 1 — SC ground truth: top significant genes in single-cell.
              Shows whether metacells preserve known signals.
    Group 2 — Proto discoveries: significant in proto, NOT in group 1.
              Shows genes proto amplifies beyond single-cell.
    Group 3 — Baseline discoveries: significant in baseline, NOT in groups 1/2.
              Shows genes baseline finds that others don't.

    All groups ranked by |logfoldchange| descending.
    For groups 2/3, includes the single-cell logFC and pval for validation
    (weak SC trend = plausible discovery, zero SC trend = possible artifact).

    Args:
        dge_dict: dict from load_dge_results()
        niche: niche name to analyze
        celltype: celltype to filter to
        sc_key: key for single-cell in dge_dict
        proto_key: key for prototype method (e.g. 'scproto_...')
        baseline_key: key for baseline method (e.g. 'seacells_...')
        top_k: number of genes per group
        pval_thr: adjusted p-value threshold
        logfc_thr: minimum |logFC| to count as significant

    Returns:
        dict with keys 'sc_top', 'proto_discovered', 'baseline_discovered',
        each a DataFrame with gene, logfoldchange, pval_adj (+ sc_logfc, sc_pval for groups 2/3)
    """
    import pandas as pd

    def _get_sig(key):
        if key is None or key not in dge_dict:
            return pd.DataFrame()
        df = dge_dict[key].copy()
        df = df[(df['celltype'] == celltype) & (df['niche'] == niche)]
        sig = df[(df['pval_adj'] < pval_thr) & (df['logfoldchange'].abs() > logfc_thr)]
        return sig.sort_values('logfoldchange', key=abs, ascending=False)

    def _get_all(key):
        if key is None or key not in dge_dict:
            return pd.DataFrame()
        df = dge_dict[key].copy()
        return df[(df['celltype'] == celltype) & (df['niche'] == niche)]

    sc_sig = _get_sig(sc_key)
    proto_sig = _get_sig(proto_key)
    baseline_sig = _get_sig(baseline_key)
    sc_all = _get_all(sc_key)

    # Group 1: top SC genes
    sc_top = sc_sig.head(top_k)[['gene', 'logfoldchange', 'pval_adj']].reset_index(drop=True)
    sc_gene_set = set(sc_top['gene'])

    # Group 2: proto-only (not in SC top)
    proto_only = proto_sig[~proto_sig['gene'].isin(sc_gene_set)]
    proto_disc = proto_only.head(top_k)[['gene', 'logfoldchange', 'pval_adj']].reset_index(drop=True)
    # Add SC context
    if len(proto_disc) > 0 and len(sc_all) > 0:
        sc_lookup = sc_all.set_index('gene')[['logfoldchange', 'pval_adj']]
        proto_disc['sc_logfc'] = proto_disc['gene'].map(sc_lookup['logfoldchange']).values
        proto_disc['sc_pval'] = proto_disc['gene'].map(sc_lookup['pval_adj']).values
    proto_gene_set = set(proto_disc['gene'])

    # Group 3: baseline-only (not in SC top or proto discoveries)
    used = sc_gene_set | proto_gene_set
    baseline_only = baseline_sig[~baseline_sig['gene'].isin(used)]
    baseline_disc = baseline_only.head(top_k)[['gene', 'logfoldchange', 'pval_adj']].reset_index(drop=True)
    if len(baseline_disc) > 0 and len(sc_all) > 0:
        baseline_disc['sc_logfc'] = baseline_disc['gene'].map(sc_lookup['logfoldchange']).values
        baseline_disc['sc_pval'] = baseline_disc['gene'].map(sc_lookup['pval_adj']).values

    result = {
        'sc_top': sc_top,
        'proto_discovered': proto_disc,
        'baseline_discovered': baseline_disc,
    }

    for name, df in result.items():
        print(f"{name}: {len(df)} genes")
        if len(df) > 0:
            print(df.to_string(index=False))
        print()

    return result


def plot_gene_expression(adata_dict, genes, niche, models, celltype=None,
                         celltype_key='celltype', niche_key='niches_2D',
                         rename=None, figsize=None, ncols=4):
    """Box plots of gene expression: niche vs rest, across models.

    One subplot per gene. Within each subplot, paired boxes (niche | rest)
    for each model side by side.

    Args:
        adata_dict: dict {model_name: AnnData} from load_expression_data()
        genes: list of gene names (one subplot each)
        niche: niche name (cells in niche vs rest)
        models: list of model names to compare
        celltype: filter to this celltype (None = all)
        celltype_key: obs column for celltype
        niche_key: obs column for niche labels
        rename: dict to rename model names for display
        figsize: figure size (auto if None)
        ncols: max columns in subplot grid

    Returns:
        Figure
    """
    if isinstance(genes, str):
        genes = [genes]

    n = len(genes)
    nc = min(n, ncols)
    nr = int(np.ceil(n / nc))
    if figsize is None:
        figsize = (2.5 * len(models) * nc, 3.5 * nr)

    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)

    niche_color = '#e74c3c'
    rest_color = '#bdc3c7'

    # Pre-extract per model: X, gene_names, ct_mask, niche masks
    model_cache = {}
    for model in models:
        if model not in adata_dict:
            continue
        ad = adata_dict[model]
        X = ad.X.toarray() if hasattr(ad.X, 'toarray') else np.array(ad.X)
        gnames = list(ad.var_names)
        if celltype is not None:
            ct_mask = np.array(ad.obs[celltype_key].values == celltype)
        else:
            ct_mask = np.ones(ad.n_obs, dtype=bool)
        niche_labels = np.array(ad.obs[niche_key].values)
        model_cache[model] = (X, gnames, ct_mask, niche_labels)

    from matplotlib.patches import Patch

    for idx, gene in enumerate(genes):
        row, col = divmod(idx, nc)
        ax = axes[row, col]

        box_data = []
        box_colors = []
        positions = []
        tick_positions = []
        tick_labels = []

        for j, model in enumerate(models):
            if model not in model_cache:
                continue

            X, gnames, ct_mask, niche_labels = model_cache[model]
            if gene not in gnames:
                continue

            expr = X[:, gnames.index(gene)]
            in_niche = ct_mask & (niche_labels == niche)
            rest = ct_mask & (niche_labels != niche)

            pos_base = j * 3
            if in_niche.sum() > 0:
                box_data.append(expr[in_niche])
                box_colors.append(niche_color)
                positions.append(pos_base)
            if rest.sum() > 0:
                box_data.append(expr[rest])
                box_colors.append(rest_color)
                positions.append(pos_base + 1)

            display_name = rename.get(model, model) if rename else model
            tick_positions.append(pos_base + 0.5)
            tick_labels.append(display_name)

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                            patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(gene, fontsize=10)
        if col == 0:
            ax.set_ylabel('Expression', fontsize=9)

    # Hide unused axes
    for idx in range(n, nr * nc):
        row, col = divmod(idx, nc)
        axes[row, col].axis('off')

    # Shared legend on first axis
    axes[0, 0].legend(
        handles=[Patch(facecolor=niche_color, alpha=0.7, label=niche),
                 Patch(facecolor=rest_color, alpha=0.7, label='Rest')],
        fontsize=8, loc='upper right')

    suptitle = niche
    if celltype:
        suptitle += f' — {celltype}'
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.show()
    return fig


def plot_signal_amplification(dge_dict, models, niches, celltype=None, rename=None, figsize=None):
    """Grid of scatter plots: rows=models, cols=niches. Shows signal amplification.

    Args:
        dge_dict: dict from load_dge_results()
        models: list of model names (rows)
        niches: list of niche names (cols)
        celltype: filter to this celltype (None = all)
        rename: dict to rename model names for display
        figsize: figure size (auto if None)

    Returns:
        Figure
    """
    if 'singlecell' not in dge_dict:
        print("Need 'singlecell' in dge_dict")
        return None

    sc = dge_dict['singlecell'].copy()
    if celltype:
        sc = sc[sc['celltype'] == celltype]

    n_rows = len(models)
    n_cols = len(niches)
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, model in enumerate(models):
        if model not in dge_dict:
            for j in range(n_cols):
                axes[i, j].set_title(f'{model}: not found')
                axes[i, j].axis('off')
            continue

        proto = dge_dict[model].copy()
        if celltype:
            proto = proto[proto['celltype'] == celltype]

        display_name = rename.get(model, model) if rename else model

        for j, niche in enumerate(niches):
            ax = axes[i, j]

            sc_niche = sc[sc['niche'] == niche]
            proto_niche = proto[proto['niche'] == niche]

            merged = sc_niche.merge(proto_niche, on=['gene', 'niche', 'celltype'], suffixes=('_sc', '_proto'))

            if len(merged) == 0:
                ax.set_title(f'{display_name}\n{niche[:20]}: no data')
                ax.axis('off')
                continue

            # Scatter
            ax.scatter(merged['logfoldchange_sc'], merged['logfoldchange_proto'],
                      s=10, alpha=0.5, c='steelblue')

            # Diagonal
            lim = max(abs(merged['logfoldchange_sc']).max(), abs(merged['logfoldchange_proto']).max()) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4)
            ax.axhline(0, c='gray', lw=0.5)
            ax.axvline(0, c='gray', lw=0.5)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            # Count amplified
            same_dir = (merged['logfoldchange_sc'] * merged['logfoldchange_proto']) > 0
            amplified = abs(merged['logfoldchange_proto']) > abs(merged['logfoldchange_sc'])
            n_amp = (same_dir & amplified).sum()
            pct = 100 * n_amp / len(merged) if len(merged) > 0 else 0

            ax.set_title(f'{display_name}\n{niche[:20]}\n{n_amp}/{len(merged)} ({pct:.0f}%) amplified', fontsize=9)

            if i == n_rows - 1:
                ax.set_xlabel('SC LogFC')
            if j == 0:
                ax.set_ylabel('Metacell LogFC')

            ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    return fig


def compare_runs(base_dir):
    """Compare metrics from all model runs in a directory (standalone, no trainer needed).

    Args:
        base_dir: directory containing model folders with metrics.json

    Returns:
        DataFrame with metrics (index = model names)
    """
    import json
    import pandas as pd

    all_metrics = {}
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        json_path = os.path.join(run_dir, 'metrics.json')
        if os.path.isdir(run_dir) and os.path.exists(json_path):
            with open(json_path) as f:
                all_metrics[entry] = json.load(f)

    rows = []
    for name, m in all_metrics.items():
        row = {'name': name}
        row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
        rows.append(row)

    return pd.DataFrame(rows).set_index('name')


def clean_metrics_df(df, keep=None, remove=None, rename=None):
    """Clean metrics DataFrame for presentation.

    Args:
        df: DataFrame from compare_runs (index = model names)
        keep: list of model names to keep (None = keep all)
        remove: list of model names to remove
        rename: dict mapping old names to new names

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    if keep is not None:
        df = df.loc[[n for n in df.index if n in keep]]
    if remove is not None:
        df = df.loc[[n for n in df.index if n not in remove]]
    if rename is not None:
        df = df.rename(index=rename)

    return df


def plot_metrics_comparison(df, metrics=None, highlight='Ours', baseline='Spectral', figsize=(12, 4)):
    """Bar plot comparing methods, highlighting your method vs baseline.

    Args:
        df: DataFrame from compare_runs/clean_metrics_df (index = model names)
        metrics: list of metric columns to plot. Default: ['modularity', 'ncut', 'metacell_f1']
        highlight: your method name (shown in green)
        baseline: baseline method name (shown in blue)
        figsize: figure size

    Returns:
        Figure
    """
    if metrics is None:
        metrics = ['modularity', 'ncut', 'metacell_f1']
    metrics = [m for m in metrics if m in df.columns]

    # Sort: highlight first, baseline second, others by first metric
    order = []
    if highlight in df.index:
        order.append(highlight)
    if baseline in df.index:
        order.append(baseline)
    others = [n for n in df.index if n not in order]
    if metrics and others:
        others = df.loc[others].sort_values(metrics[0], ascending=False).index.tolist()
    order.extend(others)
    df = df.loc[order]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        colors = []
        for n in df.index:
            if n == highlight:
                colors.append('#2ecc71')  # green
            elif n == baseline:
                colors.append('#3498db')  # blue
            else:
                colors.append('#bdc3c7')  # gray

        bars = ax.bar(range(len(df)), df[m], color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.set_title(m)
        ax.set_ylim(0, df[m].max() * 1.15)

        # Value labels
        for i, v in enumerate(df[m]):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig


def plot_metrics_grouped(df, metrics, groups, group_colors=None, highlight_method=None,
                         figsize=(14, 4), title=None):
    """Bar plot with method groups separated visually.

    Args:
        df: DataFrame (index = model names)
        metrics: dict like {'modularity': 'higher', 'ncut': 'lower', 'f1': 'higher'}
                 values are 'higher' (higher is better) or 'lower' (lower is better)
        groups: dict like {'Graph-based': ['Spectral'], 'Encoder-based': ['Ours', 'VAE']}
        group_colors: dict mapping group names to colors. Default auto-assigns.
        highlight_method: method name to highlight with thick dark border (e.g., 'SCProto')
        figsize: figure size
        title: optional suptitle

    Returns:
        Figure
    """
    metric_names = [m for m in metrics.keys() if m in df.columns]

    # Auto-assign colors if not provided
    default_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    if group_colors is None:
        group_colors = {g: default_colors[i % len(default_colors)] for i, g in enumerate(groups.keys())}

    # Build order and colors with gaps, sorted within each group
    first_metric = list(metrics.keys())[0]
    first_direction = metrics[first_metric]

    order = []
    colors = []
    group_labels = []
    for g, methods in groups.items():
        # Filter to valid methods
        valid = [m for m in methods if m in df.index]
        # Sort within group: best first
        if valid and first_metric in df.columns:
            ascending = (first_direction == 'lower')
            valid = df.loc[valid].sort_values(first_metric, ascending=ascending).index.tolist()
        for m in valid:
            order.append(m)
            colors.append(group_colors.get(g, '#bdc3c7'))
            group_labels.append(g)
        order.append(None)
        colors.append(None)
        group_labels.append(None)
    order = order[:-1]
    colors = colors[:-1]
    group_labels = group_labels[:-1]

    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names):
        direction = metrics[metric]
        x = 0
        xs, vals, cs, names, glabels = [], [], [], [], []
        for name, c, g in zip(order, colors, group_labels):
            if name is None:
                x += 0.5
            else:
                xs.append(x)
                vals.append(df.loc[name, metric])
                cs.append(c)
                names.append(name)
                glabels.append(g)
                x += 1

        bars = ax.bar(xs, vals, color=cs, edgecolor='black', linewidth=0.5)

        # Highlight specific method with thick dark border
        for i, name in enumerate(names):
            if name == highlight_method:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(3)

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

        # Title with arrow
        arrow = '↑' if direction == 'higher' else '↓'
        ax.set_title(f'{metric} ({arrow})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2)

        # Find best in each group
        best_in_group = {}
        for g in groups.keys():
            group_vals = [(i, v) for i, (v, gl) in enumerate(zip(vals, glabels)) if gl == g]
            if group_vals:
                if direction == 'higher':
                    best_idx = max(group_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(group_vals, key=lambda x: x[1])[0]
                best_in_group[best_idx] = True

        # Add star for best in each group, bold for highlighted method
        for i, v in enumerate(vals):
            txt = f'{v:.2f}'
            if i in best_in_group:
                txt = f'★{v:.2f}'
            weight = 'bold' if names[i] == highlight_method else 'normal'
            ax.text(xs[i], v + max(vals)*0.02, txt, ha='center', fontsize=9, fontweight=weight)

        # Add horizontal line for second group average (comparison baseline)
        group_names = list(groups.keys())
        baseline_group = group_names[1] if len(group_names) > 1 else None
        baseline_vals = [v for v, g in zip(vals, glabels) if g == baseline_group]
        if baseline_vals:
            avg = np.mean(baseline_vals)
            ax.axhline(avg, color=group_colors.get(baseline_group, '#3498db'),
                      linestyle='--', alpha=0.5, linewidth=1.5)

    # Legend outside
    handles = [plt.Rectangle((0,0),1,1, color=group_colors[g], label=g)
               for g in groups.keys() if g in group_colors]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(handles), fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.12)

    plt.tight_layout()
    plt.show()
    return fig


def find_discovered_genes_standalone(base_dir, method_name, pval_thr=0.05, logfc_thr=0.5, save=True):
    """Find genes significant in metacell/baseline but not in single-cell.

    Standalone version that works for any method (SCProto, SEACells, Spectral, etc.)

    Args:
        base_dir: model directory containing method subfolders and gt/singlecell_dge.csv
        method_name: folder name (e.g., 'seacells_native_K50', 'spectral_K50', or model folder)
        pval_thr: adjusted p-value threshold for significance
        logfc_thr: absolute log fold change threshold
        save: whether to save discovered_genes.csv to method folder

    Returns:
        DataFrame with discovered genes
    """
    import pandas as pd

    # Load single-cell ground truth
    sc_path = os.path.join(base_dir, 'gt', 'singlecell_dge.csv')
    if not os.path.exists(sc_path):
        print(f"Single-cell DGE not found: {sc_path}")
        return None
    sc_dge = pd.read_csv(sc_path)

    # Load method DGE (try prototype_dge.csv then baseline_dge.csv)
    method_dir = os.path.join(base_dir, method_name)
    proto_path = os.path.join(method_dir, 'prototype_dge.csv')
    baseline_path = os.path.join(method_dir, 'baseline_dge.csv')

    if os.path.exists(proto_path):
        method_dge = pd.read_csv(proto_path)
    elif os.path.exists(baseline_path):
        method_dge = pd.read_csv(baseline_path)
    else:
        print(f"No DGE found for {method_name}")
        print(f"  Tried: {proto_path}")
        print(f"  Tried: {baseline_path}")
        return None

    rows = []
    for (ct, niche), m_grp in method_dge.groupby(['celltype', 'niche']):
        s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
        if len(s_grp) == 0:
            continue

        # Significant in method
        m_sig = m_grp[(m_grp['pval_adj'] < pval_thr) & (abs(m_grp['logfoldchange']) > logfc_thr)]

        # Significant genes in single-cell
        s_sig_genes = set(s_grp[(s_grp['pval_adj'] < pval_thr) & (abs(s_grp['logfoldchange']) > logfc_thr)]['gene'])

        # Method-only genes (discovered)
        for _, gene_row in m_sig.iterrows():
            if gene_row['gene'] not in s_sig_genes:
                sc_match = s_grp[s_grp['gene'] == gene_row['gene']]
                rows.append({
                    'celltype': ct,
                    'niche': niche,
                    'gene': gene_row['gene'],
                    'method_logfc': gene_row['logfoldchange'],
                    'method_pval': gene_row['pval_adj'],
                    'sc_logfc': sc_match['logfoldchange'].values[0] if len(sc_match) > 0 else np.nan,
                    'sc_pval': sc_match['pval_adj'].values[0] if len(sc_match) > 0 else np.nan,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print(f"{method_name}: No discovered genes found")
        return df

    # Sort by logfc
    df = df.sort_values('method_logfc', ascending=False).reset_index(drop=True)

    # Save
    if save:
        save_path = os.path.join(method_dir, 'discovered_genes.csv')
        df.to_csv(save_path, index=False)
        print(f"{method_name}: {len(df)} discovered genes → {save_path}")
    else:
        print(f"{method_name}: {len(df)} discovered genes")

    return df


def find_all_discovered_genes(base_dir, pval_thr=0.05, logfc_thr=0.5):
    """Find discovered genes for all methods in base_dir.

    Args:
        base_dir: model directory (e.g., MODEL_DIR/s28f)
        pval_thr: adjusted p-value threshold
        logfc_thr: log fold change threshold

    Returns:
        dict: {method_name: DataFrame}
    """
    results = {}

    for name in os.listdir(base_dir):
        method_dir = os.path.join(base_dir, name)
        if not os.path.isdir(method_dir):
            continue
        if name == 'gt':
            continue

        # Check if has DGE
        has_dge = (os.path.exists(os.path.join(method_dir, 'prototype_dge.csv')) or
                   os.path.exists(os.path.join(method_dir, 'baseline_dge.csv')))
        if not has_dge:
            continue

        df = find_discovered_genes_standalone(base_dir, name, pval_thr, logfc_thr, save=True)
        if df is not None and len(df) > 0:
            results[name] = df

    return results


def plot_spatial(adata, color_keys=['cell_type', 'niches_2D'], spatial_key='spatial',
                 figsize=None, point_size=3, ncols=2, title=None, save_path=None):
    """Plot cells at spatial locations colored by different annotations.

    Args:
        adata: AnnData with spatial coordinates
        color_keys: list of obs columns to color by (e.g., ['cell_type', 'niches_2D'])
        spatial_key: key in obsm for spatial coordinates
        figsize: figure size, auto-computed if None
        point_size: scatter point size
        ncols: number of columns in subplot grid
        title: overall figure title
        save_path: path to save figure

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt

    coords = adata.obsm[spatial_key]
    n_plots = len(color_keys)
    nrows = (n_plots + ncols - 1) // ncols

    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, key in enumerate(color_keys):
        ax = axes[i]

        if key not in adata.obs.columns:
            ax.set_title(f'{key} not found')
            ax.axis('off')
            continue

        categories = adata.obs[key].astype('category')
        codes = categories.cat.codes
        unique_cats = categories.cat.categories

        # Get or create colors
        color_key = f'{key}_colors'
        if color_key in adata.uns:
            colors = adata.uns[color_key]
        else:
            cmap = plt.cm.get_cmap('tab20', len(unique_cats))
            colors = [cmap(i) for i in range(len(unique_cats))]

        # Plot each category
        for j, cat in enumerate(unique_cats):
            mask = categories == cat
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[colors[j]], s=point_size, label=cat, alpha=0.7)

        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

        # Legend outside (smaller font for many categories)
        fontsize = 8 if len(unique_cats) <= 15 else 6
        ncol = 1 if len(unique_cats) <= 15 else 2
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=fontsize,
                  markerscale=2, ncol=ncol)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig, axes


def compute_metacell_purity(assignments, adata, celltype_key='cell_type', niche_key='niches_2D'):
    """Compute celltype and niche purity for each metacell.

    Args:
        assignments: array of metacell IDs per cell
        adata: AnnData with obs containing celltype and niche columns
        celltype_key: column name for celltype
        niche_key: column name for niche

    Returns:
        DataFrame with columns: metacell_id, celltype_purity, niche_purity,
                                majority_celltype, majority_niche, size
    """
    import pandas as pd
    from collections import Counter

    celltypes = adata.obs[celltype_key].values
    niches = adata.obs[niche_key].values

    rows = []
    for mc_id in np.unique(assignments):
        mask = assignments == mc_id
        size = mask.sum()

        ct_counts = Counter(celltypes[mask])
        niche_counts = Counter(niches[mask])

        majority_ct, ct_max = ct_counts.most_common(1)[0]
        majority_niche, niche_max = niche_counts.most_common(1)[0]

        rows.append({
            'metacell_id': mc_id,
            'celltype_purity': ct_max / size,
            'niche_purity': niche_max / size,
            'majority_celltype': majority_ct,
            'majority_niche': majority_niche,
            'size': size,
        })

    return pd.DataFrame(rows)


def plot_purity_scatter(purity_df, adata=None, celltype_key='cell_type', niche_key='niches_2D',
                        figsize=(12, 5), point_size=30, title=None):
    """Scatter plot of celltype purity vs niche purity.

    Args:
        purity_df: DataFrame from compute_metacell_purity
        adata: AnnData (optional, for color palette)
        celltype_key: column name for celltype colors
        niche_key: column name for niche colors
        figsize: figure size
        point_size: scatter point size
        title: figure title

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get colors
    ct_categories = purity_df['majority_celltype'].unique()
    niche_categories = purity_df['majority_niche'].unique()

    if adata is not None and f'{celltype_key}_colors' in adata.uns:
        ct_palette = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[f'{celltype_key}_colors']))
    else:
        cmap = plt.cm.get_cmap('tab20', len(ct_categories))
        ct_palette = {ct: cmap(i) for i, ct in enumerate(ct_categories)}

    if adata is not None and f'{niche_key}_colors' in adata.uns:
        niche_palette = dict(zip(adata.obs[niche_key].cat.categories, adata.uns[f'{niche_key}_colors']))
    else:
        cmap = plt.cm.get_cmap('tab10', len(niche_categories))
        niche_palette = {n: cmap(i) for i, n in enumerate(niche_categories)}

    x = purity_df['celltype_purity']
    y = purity_df['niche_purity']

    # Subplot 1: color by celltype
    ax = axes[0]
    for ct in ct_categories:
        mask = purity_df['majority_celltype'] == ct
        ax.scatter(x[mask], y[mask], c=[ct_palette[ct]], s=point_size, label=ct, alpha=0.7)
    ax.set_xlabel('Celltype Purity', fontsize=11)
    ax.set_ylabel('Niche Purity', fontsize=11)
    ax.set_title('Color by Celltype', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
    # Legend below
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=7,
              ncol=3, frameon=False, markerscale=1.5)

    # Subplot 2: color by niche
    ax = axes[1]
    for niche in niche_categories:
        mask = purity_df['majority_niche'] == niche
        ax.scatter(x[mask], y[mask], c=[niche_palette[niche]], s=point_size, label=niche, alpha=0.7)
    ax.set_xlabel('Celltype Purity', fontsize=11)
    ax.set_ylabel('Niche Purity', fontsize=11)
    ax.set_title('Color by Niche', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
    # Legend below
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=7,
              ncol=3, frameon=False, markerscale=1.5)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    plt.show()
    return fig, axes


def plot_purity_joint(purity_df, figsize=(6, 6), title=None):
    """Joint distribution plot of celltype vs niche purity.

    Shows 2D KDE with marginal histograms.

    Args:
        purity_df: DataFrame from compute_metacell_purity
        figsize: figure size
        title: plot title

    Returns:
        fig
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    x = purity_df['celltype_purity'].values
    y = purity_df['niche_purity'].values

    fig = plt.figure(figsize=figsize)

    # Main scatter + KDE
    gs = fig.add_gridspec(4, 4)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 2D KDE
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[0:1:100j, 0:1:100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        ax_main.contourf(xi, yi, zi, levels=15, cmap='Blues', alpha=0.7)
        ax_main.contour(xi, yi, zi, levels=5, colors='steelblue', linewidths=0.5)
    except:
        pass

    # Scatter on top
    ax_main.scatter(x, y, s=15, alpha=0.5, c='darkblue', edgecolors='none')
    ax_main.set_xlabel('Celltype Purity', fontsize=11)
    ax_main.set_ylabel('Niche Purity', fontsize=11)
    ax_main.set_xlim(0, 1.02)
    ax_main.set_ylim(0, 1.02)
    ax_main.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax_main.axvline(0.8, color='gray', linestyle='--', alpha=0.5)

    # Marginal histograms
    ax_top.hist(x, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax_top.set_xlim(0, 1.02)
    ax_top.axis('off')

    ax_right.hist(y, bins=30, orientation='horizontal', color='steelblue', alpha=0.7, edgecolor='white')
    ax_right.set_ylim(0, 1.02)
    ax_right.axis('off')

    # Stats annotation
    ct_high = (x > 0.8).mean() * 100
    niche_high = (y > 0.8).mean() * 100
    both_high = ((x > 0.8) & (y > 0.8)).mean() * 100
    ax_main.text(0.05, 0.95, f'CT purity >0.8: {ct_high:.0f}%\nNiche purity >0.8: {niche_high:.0f}%\nBoth >0.8: {both_high:.0f}%',
                 transform=ax_main.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()
    return fig


def plot_purity_comparison(purity_dict, figsize=(12, 5)):
    """Compare purity distributions across multiple models.

    Args:
        purity_dict: dict of {model_name: purity_df}
        figsize: figure size

    Returns:
        fig
    """
    import matplotlib.pyplot as plt

    n_models = len(purity_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, pdf) in zip(axes, purity_dict.items()):
        x = pdf['celltype_purity'].values
        y = pdf['niche_purity'].values

        # 2D KDE
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xi, yi = np.mgrid[0:1:80j, 0:1:80j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=12, cmap='Blues', alpha=0.6)
        except:
            pass

        ax.scatter(x, y, s=12, alpha=0.5, c='darkblue', edgecolors='none')
        ax.set_xlabel('Celltype Purity', fontsize=10)
        ax.set_ylabel('Niche Purity', fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(name, fontsize=12, fontweight='bold')

        # Stats
        ct_high = (x > 0.8).mean() * 100
        niche_high = (y > 0.8).mean() * 100
        both_high = ((x > 0.8) & (y > 0.8)).mean() * 100
        ax.text(0.05, 0.95, f'CT>0.8: {ct_high:.0f}%\nNiche>0.8: {niche_high:.0f}%\nBoth: {both_high:.0f}%',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    return fig


def load_affinity(dataset_id, adata, n_components=50, k_neighbors=50, affinity_type='uctx',
                  spatial=False, graph_mode=None, fold=None, graphs_dir='./graphs'):
    """Load precomputed affinity matrix for a dataset.

    Args:
        dataset_id: dataset name (e.g., 's28f')
        adata: AnnData object (used to get n_cells)
        n_components: number of PCA components used
        k_neighbors: number of neighbors
        affinity_type: type of affinity ('uctx', 'inverse_dist', etc.)
        spatial: whether spatial info was used
        graph_mode: graph mode if any (e.g., 'knn')
        fold: fold number if any
        graphs_dir: directory where affinity is saved (default: './graphs')

    Returns:
        sparse affinity matrix, or None if not found
    """
    import pickle

    n_cells = len(adata)

    # Build filename (matches adata_augmenter.set_graph_name)
    graph_name = f"affinity_{dataset_id}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}"
    if spatial:
        graph_name += "_spatial"
    if fold is not None:
        graph_name += f"_fold{fold}"
    if graph_mode:
        graph_name += f"_{graph_mode}"
    graph_name += ".pkl"

    save_path = os.path.join(graphs_dir, graph_name)

    if not os.path.exists(save_path):
        print(f"Affinity not found: {save_path}")
        return None

    with open(save_path, 'rb') as f:
        aff = pickle.load(f)

    print(f"Loaded affinity: {aff.shape}, nnz={aff.nnz}")
    return aff


def compute_metaclusters(assignments, avg_expression, k=8, n_pcs=20, random_state=42,
                         cell_labels=None, label_strategy='metacells'):
    """Cluster metacells and assign metacluster IDs to cells.

    Args:
        assignments: cell -> metacell ID array (n_cells,)
        avg_expression: metacell expression matrix (n_metacells, n_genes)
        k: number of metaclusters
        n_pcs: number of PCA components
        random_state: random seed
        cell_labels: optional label array per cell (e.g. niche labels). If given,
                     each cluster gets a niche label via majority voting.
        label_strategy: how to assign a label to each cluster (only used if cell_labels given):
            'metacells' - 2-step: cell -> metacell label, then metacell -> cluster label.
                          Each metacell counts as one vote regardless of size.
            'cells'     - 1-step: majority vote of all cells directly in each cluster.
                          Each cell has equal weight.

    Returns:
        dict with:
            - cell_metacluster: metacluster ID for each cell (n_cells,)
            - metacell_metacluster: metacluster ID for each metacell (n_metacells,)
            - pca: PCA coordinates of metacells
            - cell_cluster_label: (only if cell_labels given) niche label per cell
            - metacell_label: (only if label_strategy='metacells') label per metacell
            - cluster_label: majority-voted label per cluster
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # PCA on metacells
    n_pcs = min(n_pcs, avg_expression.shape[0] - 1, avg_expression.shape[1])
    pca = PCA(n_components=n_pcs, random_state=random_state)
    metacell_pca = pca.fit_transform(avg_expression)

    # KMeans on PCA
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    metacell_metacluster = kmeans.fit_predict(metacell_pca)

    # Map to cells
    cell_metacluster = metacell_metacluster[assignments]

    print(f"PCA: {avg_expression.shape} -> {metacell_pca.shape}")
    print(f"KMeans: {k} metaclusters")
    print(f"Cell metacluster distribution: {np.bincount(cell_metacluster)}")

    result = {
        'cell_metacluster': cell_metacluster,
        'metacell_metacluster': metacell_metacluster,
        'pca': metacell_pca,
    }

    if cell_labels is not None:
        cell_labels = np.asarray(cell_labels)

        if label_strategy == 'metacells':
            # Step 1: majority vote per metacell
            n_metacells = avg_expression.shape[0]
            metacell_label = []
            for mc_id in range(n_metacells):
                mask = assignments == mc_id
                if mask.sum() == 0:
                    metacell_label.append('')
                else:
                    counts = pd.Series(cell_labels[mask]).value_counts()
                    metacell_label.append(counts.index[0])
            metacell_label = np.array(metacell_label)
            result['metacell_label'] = metacell_label

            # Step 2: majority vote per cluster from metacell labels
            cluster_label = []
            for c in range(k):
                mc_mask = metacell_metacluster == c
                if mc_mask.sum() == 0:
                    cluster_label.append('')
                else:
                    counts = pd.Series(metacell_label[mc_mask]).value_counts()
                    cluster_label.append(counts.index[0])

        else:  # 'cells'
            # Direct majority vote per cluster from all cell labels
            cluster_label = []
            for c in range(k):
                cell_mask = cell_metacluster == c
                if cell_mask.sum() == 0:
                    cluster_label.append('')
                else:
                    counts = pd.Series(cell_labels[cell_mask]).value_counts()
                    cluster_label.append(counts.index[0])

        cluster_label = np.array(cluster_label)
        cell_cluster_label = cluster_label[cell_metacluster]

        print(f"Cluster labels ({label_strategy}): {dict(enumerate(cluster_label))}")

        result['cluster_label'] = cluster_label
        result['cell_cluster_label'] = cell_cluster_label

    return result


def list_saved_clusters(base_dir):
    """List all saved clusters and metacell expressions in model directory.

    Args:
        base_dir: model directory (e.g., MODEL_DIR/s28f)

    Returns:
        dict: {method_name: {'assignments': array, 'expression': array or None}}
    """
    results = {}

    for name in os.listdir(base_dir):
        method_dir = os.path.join(base_dir, name)
        if not os.path.isdir(method_dir):
            continue

        clusters_path = os.path.join(method_dir, 'clusters.npz')
        avg_expr_path = os.path.join(method_dir, 'avg_expression.npy')  # baselines
        decoded_path = os.path.join(method_dir, 'decoded_prototypes.npy')  # scproto

        if os.path.exists(clusters_path):
            data = np.load(clusters_path, allow_pickle=True)
            assignments = data['assignments']

            # Load expression (try decoded first for scproto, then avg for baselines)
            expr = None
            expr_source = None
            if os.path.exists(decoded_path):
                expr = np.load(decoded_path)
                expr_source = 'decoded_prototypes.npy'
            elif os.path.exists(avg_expr_path):
                expr = np.load(avg_expr_path)
                expr_source = 'avg_expression.npy'

            results[name] = {
                'assignments': assignments,
                'expression': expr,
            }

            # Print summary
            print(f"{name}:")
            print(f"  assignments: {assignments.shape}, {len(np.unique(assignments))} unique clusters")
            if expr is not None:
                print(f"  expression: {expr.shape} (from {expr_source})")
            else:
                print(f"  expression: NOT FOUND")
            print()

    return results


def plot_spatial_with_metaclusters(adata, metaclusters_dict, color_keys=['cell_type', 'niches_2D'],
                                    spatial_key='spatial', figsize=None, point_size=3,
                                    title=None, save_path=None, mask_to_assigned=None,
                                    label_colors=None):
    """Plot spatial: cell_type, niches, and one or more metacluster arrays.

    Args:
        adata: AnnData with spatial coordinates
        metaclusters_dict: dict of {name: array} for metacluster IDs per cell (use -1 for unassigned)
                          e.g., {'SEACells': arr1, 'SCProto': arr2}
                          Can also pass single array for backwards compatibility
        color_keys: list of obs columns for first subplots.
                    Use 'niches_2D*' (with *) to only color assigned cells.
        spatial_key: key in obsm for spatial coordinates
        figsize: figure size
        point_size: scatter point size
        title: overall figure title
        save_path: path to save figure
        mask_to_assigned: list of color_keys that should only show assigned cells (rest gray)
                         e.g., ['niches_2D'] to gray out non-fibroblasts in niches plot

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt

    # Handle single array input (backwards compatibility)
    if isinstance(metaclusters_dict, np.ndarray):
        metaclusters_dict = {'Metacluster': metaclusters_dict}

    if mask_to_assigned is None:
        mask_to_assigned = []

    coords = adata.obsm[spatial_key]
    n_mc = len(metaclusters_dict)
    n_plots = len(color_keys) + n_mc
    ncols = n_plots

    if figsize is None:
        figsize = (5 * ncols, 5)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Get assigned mask from first metacluster array
    first_mc = np.asarray(list(metaclusters_dict.values())[0])
    if first_mc.dtype.kind in ('U', 'O', 'S'):
        assigned_mask = first_mc != ''
    else:
        assigned_mask = first_mc >= 0

    # Plot color_keys (cell_type, niches)
    for i, key in enumerate(color_keys):
        ax = axes[i]

        if key not in adata.obs.columns:
            ax.set_title(f'{key} not found')
            ax.axis('off')
            continue

        categories = adata.obs[key].astype('category')
        unique_cats = categories.cat.categories

        # Get or create colors
        color_key_name = f'{key}_colors'
        if color_key_name in adata.uns:
            colors = adata.uns[color_key_name]
        else:
            cmap = plt.cm.get_cmap('tab20', len(unique_cats))
            colors = [cmap(j) for j in range(len(unique_cats))]

        # Check if should mask to assigned only
        only_assigned = key in mask_to_assigned

        if only_assigned:
            # Plot unassigned cells in gray first
            ax.scatter(coords[~assigned_mask, 0], coords[~assigned_mask, 1],
                      c='lightgray', s=point_size, alpha=0.3)

        # Plot each category
        for j, cat in enumerate(unique_cats):
            cat_mask = categories == cat
            if only_assigned:
                cat_mask = cat_mask & assigned_mask
            if cat_mask.sum() == 0:
                continue
            ax.scatter(coords[cat_mask, 0], coords[cat_mask, 1],
                      c=[colors[j]], s=point_size, label=cat, alpha=0.7)

        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Legend below subplot - 3 columns, larger font
        ax.legend(bbox_to_anchor=(0.5, -0.02), loc='upper center', fontsize=8,
                  markerscale=1.5, ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.5)

    # Build niche color map once (shared across all metacluster subplots)
    if label_colors is not None:
        niche_color_map = label_colors
    else:
        niche_color_map = {}
        for ck in color_keys:
            if ck in adata.obs.columns and f'{ck}_colors' in adata.uns:
                cats = adata.obs[ck].astype('category').cat.categories
                niche_color_map = dict(zip(cats, adata.uns[f'{ck}_colors']))
                break

    mc_cmap = plt.cm.get_cmap('tab10', 10)
    for idx, (mc_name, cell_metacluster) in enumerate(metaclusters_dict.items()):
        ax = axes[len(color_keys) + idx]
        cell_metacluster = np.asarray(cell_metacluster)
        is_labeled = cell_metacluster.dtype.kind in ('U', 'O', 'S')

        if is_labeled:
            # String niche labels — use same colors as the niches subplot
            mask_unassigned = cell_metacluster == ''
            if mask_unassigned.any():
                ax.scatter(coords[mask_unassigned, 0], coords[mask_unassigned, 1],
                          c='lightgray', s=point_size, alpha=0.3)

            unique_labels = [l for l in np.unique(cell_metacluster) if l != '']
            for label in unique_labels:
                mask = cell_metacluster == label
                color = niche_color_map.get(label, 'gray')
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          color=color, s=point_size, alpha=0.7, label=label)

            ax.legend(bbox_to_anchor=(0.5, -0.02), loc='upper center', fontsize=8,
                      markerscale=1.5, ncol=3, frameon=False,
                      handletextpad=0.1, columnspacing=0.5)
        else:
            # Integer cluster IDs — existing behavior
            mask_unassigned = cell_metacluster < 0
            if mask_unassigned.any():
                ax.scatter(coords[mask_unassigned, 0], coords[mask_unassigned, 1],
                          c='lightgray', s=point_size, alpha=0.3, label='Other')

            unique_mc = np.unique(cell_metacluster[~mask_unassigned])
            for j, mc in enumerate(unique_mc):
                mc = int(mc)
                mask = cell_metacluster == mc
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          color=mc_cmap(mc % 10), s=point_size, alpha=0.7, label=f'MC {mc}')

            ax.legend(bbox_to_anchor=(0.5, -0.02), loc='upper center', fontsize=8,
                      markerscale=1.5, ncol=3, frameon=False,
                      handletextpad=0.1, columnspacing=0.5)

        ax.set_title(mc_name, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.subplots_adjust(wspace=0.05, bottom=0.25)  # less space between, room for legend below

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig, axes


def plot_niche_metaclusters(ad, fib_ad, base_dir, methods, k=4,
                            niche_key='niches_2D', color_keys=None,
                            color_by='niche', label_strategy='metacells',
                            point_size=1, figsize=None, save_path=None):
    """Load clusters, assign niche labels via majority voting, and plot spatial.

    Args:
        ad: full AnnData (all cells, with spatial coords)
        fib_ad: filtered AnnData (only the cells that have metacell assignments)
        base_dir: model directory containing method subfolders
        methods: dict of {label: folder_name} e.g. {'SEACells': 'seacells_native_K50'}
        k: number of metaclusters
        niche_key: obs column used for majority-vote labels and color palette
        color_keys: left subplots to show before metacluster plots (default: ['celltypes', niche_key])
        color_by: 'niche' to color metacluster subplots by majority-voted niche label
                  (same palette as niche subplot), or 'cluster' to color by integer cluster ID
        point_size: scatter point size
        figsize: figure size (auto if None)
        save_path: optional path to save figure

    Returns:
        fig, axes
    """
    if color_keys is None:
        color_keys = ['celltypes', niche_key]

    # Build niche color map from fib_ad first (most likely to have colors), then ad
    label_colors = {}
    for source in [fib_ad, ad]:
        color_uns_key = f'{niche_key}_colors'
        if color_uns_key in source.uns and niche_key in source.obs.columns:
            cats = source.obs[niche_key].astype('category').cat.categories
            label_colors = dict(zip(cats, source.uns[color_uns_key]))
            break

    clusters = list_saved_clusters(base_dir)
    niche_labels = fib_ad.obs[niche_key].values

    metaclusters_dict = {}
    for label, folder in methods.items():
        if folder not in clusters:
            print(f"WARNING: {folder} not found in {base_dir}")
            continue
        result = compute_metaclusters(
            clusters[folder]['assignments'],
            clusters[folder]['expression'],
            k=k, cell_labels=niche_labels, label_strategy=label_strategy)

        if color_by == 'niche':
            cell_to_val = dict(zip(fib_ad.obs.index, result['cell_cluster_label']))
            metaclusters_dict[label] = np.array([cell_to_val.get(cid, '') for cid in ad.obs.index])
        else:
            cell_to_val = dict(zip(fib_ad.obs.index, result['cell_metacluster']))
            metaclusters_dict[label] = np.array([cell_to_val.get(cid, -1) for cid in ad.obs.index])

    return plot_spatial_with_metaclusters(
        ad,
        metaclusters_dict=metaclusters_dict,
        color_keys=color_keys,
        mask_to_assigned=[niche_key],
        point_size=point_size,
        figsize=figsize,
        save_path=save_path,
        label_colors=label_colors if color_by == 'niche' else None,
    )


def plot_gsea_comparison(gsea, niche, top_n=10, highlight_method=None,
                         cap=15, pval_thr=0.05, figsize=None, title=None, save_path=None):
    """Bubble plot: pathways × methods for Enrichr GSEA results.

    - Dot size   = -log10(adj p-value), capped at `cap` (fixes EMT domination)
    - Dot color  = overlap ratio (genes found / gene set size)
    - Grey dot   = not significant (adj p > pval_thr)
    - Delta col  = highlight_method -log10(p) minus max of others (shows advantage)

    Args:
        gsea: dict {method: {niche: enrichr_df}} from niche_dge_gsea
        niche: which niche to plot
        top_n: top pathways per method to include (union)
        highlight_method: method for delta column (default: last in dict)
        cap: max -log10(p) for dot size (prevents one pathway dominating scale)
        pval_thr: significance threshold — grey dot if adj p above this
        figsize: figure size (auto if None)
        title: plot title
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    methods = list(gsea.keys())
    if highlight_method is None:
        highlight_method = methods[-1]

    # Union of top_n significant pathways per method, sorted by highlight_method
    all_terms = []
    for method in methods:
        res = (gsea[method] or {}).get(niche)
        if res is not None:
            sig = res[res['Adjusted P-value'] < pval_thr]
            all_terms.extend(sig['Term'].head(top_n).tolist())
    terms = list(dict.fromkeys(all_terms))

    # Sort by highlight_method significance
    res_hi = (gsea[highlight_method] or {}).get(niche)
    if res_hi is not None:
        hi_pval = res_hi.set_index('Term')['Adjusted P-value']
        terms = sorted(terms, key=lambda t: hi_pval.get(t, 1.0))

    def _overlap_ratio(s):
        try:
            a, b = str(s).split('/')
            return int(a) / int(b)
        except Exception:
            return 0.0

    # Gather score and overlap per method per term
    scores   = {m: {} for m in methods}
    overlaps = {m: {} for m in methods}
    for method in methods:
        res = (gsea[method] or {}).get(niche)
        if res is None:
            continue
        r = res.set_index('Term')
        for term in terms:
            if term in r.index:
                pval = r.loc[term, 'Adjusted P-value']
                scores[method][term]   = min(-np.log10(pval + 1e-300), cap)
                overlaps[method][term] = _overlap_ratio(r.loc[term, 'Overlap'])

    sig_thr_score = -np.log10(pval_thr)

    # Delta: highlight_method score - max of others (only among significant)
    others = [m for m in methods if m != highlight_method]
    delta = {}
    for term in terms:
        hi = scores[highlight_method].get(term, 0.0)
        hi = hi if hi >= sig_thr_score else 0.0
        other_max = max(
            (scores[m].get(term, 0.0) if scores[m].get(term, 0.0) >= sig_thr_score else 0.0
             for m in others),
            default=0.0
        )
        delta[term] = hi - other_max

    n_cols = len(methods) + 1
    if figsize is None:
        figsize = (n_cols * 2.0 + 2, len(terms) * 0.45 + 2)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize,
                              gridspec_kw={'width_ratios': [1] * len(methods) + [1.2]})

    cmap_overlap = plt.cm.get_cmap('YlOrRd')

    for xi, method in enumerate(methods):
        ax = axes[xi]
        for yi, term in enumerate(terms):
            s = scores[method].get(term, 0.0)
            o = overlaps[method].get(term, 0.0)
            if s < sig_thr_score:
                ax.scatter(0, yi, s=25, color='lightgray', zorder=2)
            else:
                size  = (s / cap) * 300 + 25
                color = cmap_overlap(o)
                ax.scatter(0, yi, s=size, color=color, zorder=2,
                           edgecolors='gray', linewidths=0.3)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, len(terms) - 0.5)
        ax.set_xticks([])
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms if xi == 0 else [], fontsize=8)
        ax.set_title(method, fontsize=10, fontweight='bold', pad=8)
        ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.15)

    # Delta bar chart
    ax_d = axes[-1]
    delta_vals = [delta[t] for t in terms]
    d_max = max(abs(v) for v in delta_vals) or 1.0
    cmap_delta = plt.cm.get_cmap('RdBu_r')
    for yi, dv in enumerate(delta_vals):
        color = cmap_delta((dv + d_max) / (2 * d_max))
        ax_d.barh(yi, dv, color=color, edgecolor='none', height=0.7)
    ax_d.axvline(0, color='black', linewidth=0.8)
    ax_d.set_yticks(range(len(terms)))
    ax_d.set_yticklabels([])
    ax_d.set_title(f'Δ {highlight_method}\nvs others', fontsize=8,
                   fontweight='bold', pad=4)
    ax_d.invert_yaxis()
    ax_d.grid(axis='y', alpha=0.15)

    # Colorbar for overlap
    sm = plt.cm.ScalarMappable(cmap=cmap_overlap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:-1], label='overlap ratio', shrink=0.35,
                 pad=0.01, location='bottom')

    # Size legend
    legend_ax = axes[0]
    for label, frac in [('low', 0.2), ('mid', 0.6), ('high', 1.0)]:
        legend_ax.scatter([], [], s=frac * 300 + 25, color='gray',
                          alpha=0.5, label=label)
    legend_ax.legend(title=f'-log10(p)\n(capped at {cap})', loc='lower left',
                     fontsize=7, frameon=False)

    fig.suptitle(title or f'GSEA — {niche}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, axes


def plot_niche_tables(dge, gsea, niche, genes=None, pathways=None, top_n=10,
                      highlight_method=None, pval_thr=0.05,
                      figsize=None, save_path=None):
    """Two side-by-side tables for a niche:

    Left  — Gene table:    rows=genes,    cols=methods, values=logFC
    Right — Program table: rows=pathways, cols=methods, values=overlap fraction
                           grey=not significant, red label=unique to highlight_method

    Args:
        dge: dict {method: {niche: dge_df}}
        gsea: dict {method: {niche: enrichr_df}}
        niche: which niche to show
        genes: list of gene names for the gene table
        pathways: list of pathway names for the program table
        highlight_method: method to mark uniqueness (default: last)
        pval_thr: significance threshold for GSEA
        figsize: auto if None
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    methods = list(dge.keys())
    if highlight_method is None:
        highlight_method = methods[-1]

    # Auto-select genes: union of top_n per method, order preserved
    if genes is None:
        seen = {}
        for method in methods:
            res = dge[method].get(niche)
            if res is not None:
                for g in res['names'].head(top_n).tolist():
                    seen[g] = None
        genes = list(seen.keys())

    # Auto-select pathways: union of top_n per method, order preserved
    if pathways is None:
        seen = {}
        for method in methods:
            res = (gsea[method] or {}).get(niche)
            if res is not None:
                for p in res['Term'].head(top_n).tolist():
                    seen[p] = None
        pathways = list(seen.keys())

    # ── Build gene logFC table ────────────────────────────────────────────
    gene_mat = pd.DataFrame('—', index=genes, columns=methods)
    gene_val = pd.DataFrame(0.0, index=genes, columns=methods)
    for method in methods:
        res = dge[method].get(niche)
        if res is None:
            continue
        r = res.set_index('names')
        for gene in genes:
            if gene in r.index:
                lfc = float(r.loc[gene, 'logfoldchanges'])
                gene_mat.loc[gene, method] = f'{lfc:.2f}'
                gene_val.loc[gene, method] = lfc

    # ── Build pathway overlap table ───────────────────────────────────────
    def _overlap_str(s):
        try:
            a, b = str(s).split('/')
            return f'{int(a)/int(b)*100:.0f}%\n({s})'
        except Exception:
            return '—'

    pw_mat  = pd.DataFrame('—',    index=pathways, columns=methods)
    pw_sig  = pd.DataFrame(False,  index=pathways, columns=methods)
    pw_pval = pd.DataFrame(1.0,    index=pathways, columns=methods)
    for method in methods:
        res = (gsea[method] or {}).get(niche)
        if res is None:
            continue
        r = res.set_index('Term')
        for pw in pathways:
            if pw in r.index:
                pval = r.loc[pw, 'Adjusted P-value']
                pw_mat.loc[pw, method]  = _overlap_str(r.loc[pw, 'Overlap'])
                pw_sig.loc[pw, method]  = pval < pval_thr
                pw_pval.loc[pw, method] = pval

    others = [m for m in methods if m != highlight_method]
    unique_pws = [
        pw for pw in pathways
        if pw_sig.loc[pw, highlight_method] and not any(pw_sig.loc[pw, m] for m in others)
    ]

    # ── Layout ────────────────────────────────────────────────────────────
    if figsize is None:
        n_rows = max(len(genes), len(pathways))
        figsize = (len(methods) * 1.5 * 2 + 2, n_rows * 0.32 + 1.5)

    fig, (ax_g, ax_p) = plt.subplots(1, 2, figsize=figsize)

    def _draw_table(ax, row_labels, col_labels, cell_texts,
                    cell_colors, row_label_colors=None, title=''):
        ax.axis('off')
        n_rows, n_cols = len(row_labels), len(col_labels)

        label_w = 0.52           # fraction of axes width for row label column
        cell_w  = (1.0 - label_w) / n_cols   # width per data cell
        row_h   = 1.0 / (n_rows + 1.2)

        # Header row (only over data cells)
        for xi, col in enumerate(col_labels):
            cx = label_w + (xi + 0.5) * cell_w
            ax.text(cx, 1.0 - 0.4 * row_h, col,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', transform=ax.transAxes)

        # Divider under header
        yh = 1.0 - row_h * 0.85
        ax.plot([0, 1], [yh, yh], color='black', linewidth=0.8,
                transform=ax.transAxes, clip_on=False)

        for yi, row in enumerate(row_labels):
            y = 1.0 - (yi + 1.5) * row_h
            # Row label — left-aligned with a small indent
            rl_color = row_label_colors[yi] if row_label_colors else 'black'
            fw = 'bold' if rl_color != 'black' else 'normal'
            ax.text(0.01, y, row, ha='left', va='center',
                    fontsize=10, color=rl_color, fontweight=fw,
                    transform=ax.transAxes)
            # Data cells
            for xi in range(n_cols):
                cx = label_w + (xi + 0.5) * cell_w
                bg = cell_colors[yi][xi]
                rect = plt.Rectangle(
                    [cx - cell_w * 0.47, y - row_h * 0.45],
                    cell_w * 0.94, row_h * 0.9,
                    transform=ax.transAxes, color=bg,
                    zorder=2, clip_on=False)
                ax.add_patch(rect)
                ax.text(cx, y, cell_texts[yi][xi],
                        ha='center', va='center', fontsize=10,
                        zorder=3, transform=ax.transAxes)

            # Light row separator
            ys = y - row_h * 0.5
            ax.plot([0, 1], [ys, ys], color='#cccccc', linewidth=0.4,
                    transform=ax.transAxes, clip_on=False)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # ── Gene table colors: white→orange by logFC ──────────────────────────
    lfc_max = gene_val.values.max() or 1.0
    cmap_lfc = plt.cm.get_cmap('YlOrRd')
    gene_cell_texts  = []
    gene_cell_colors = []
    for gene in genes:
        row_t, row_c = [], []
        for method in methods:
            txt = gene_mat.loc[gene, method]
            val = gene_val.loc[gene, method]
            row_t.append(txt)
            row_c.append(cmap_lfc(val / lfc_max) if txt != '—' else '#f5f5f5')
        gene_cell_texts.append(row_t)
        gene_cell_colors.append(row_c)

    _draw_table(ax_g, genes, methods,
                gene_cell_texts, gene_cell_colors,
                title=f'Gene logFC — {niche}')

    # ── Pathway table colors: grey=ns, yellow-red=sig ─────────────────────
    pw_cell_texts  = []
    pw_cell_colors = []
    pw_row_colors  = []
    cmap_pw = plt.cm.get_cmap('YlOrRd')

    for pw in pathways:
        row_t, row_c = [], []
        for method in methods:
            is_sig = pw_sig.loc[pw, method]
            txt = pw_mat.loc[pw, method] if is_sig else '—'
            pval = pw_pval.loc[pw, method]
            score = min(-np.log10(pval + 1e-300), 20) / 20 if is_sig else 0
            row_t.append(txt)
            row_c.append(cmap_pw(score) if is_sig else '#e8e8e8')
        pw_cell_texts.append(row_t)
        pw_cell_colors.append(row_c)
        pw_row_colors.append('#c0392b' if pw in unique_pws else 'black')

    pw_labels = [f'★ {pw}' if pw in unique_pws else pw for pw in pathways]
    _draw_table(ax_p, pw_labels, methods,
                pw_cell_texts, pw_cell_colors,
                row_label_colors=pw_row_colors,
                title=f'Pathway overlap — {niche}')

    # Legend — only include items that actually appear in the table
    patches = []
    has_nonsig = not pw_sig.values.all()
    if has_nonsig:
        patches.append(mpatches.Patch(color='#e8e8e8', label='not significant'))
    if unique_pws:
        patches.append(mpatches.Patch(color='#c0392b', label=f'★ unique to {highlight_method}'))
    if patches:
        fig.legend(handles=patches, fontsize=8, frameon=False,
                   loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, (ax_g, ax_p)


def plot_niche_summary(dge, gsea, niche, genes, pathways,
                       highlight_method=None, pval_thr=0.05, cap=20,
                       method_colors=None, figsize=None, save_path=None):
    """Combined two-panel figure: gene logFC (top) + GSEA table (bottom).

    Panel A — Grouped bar chart: key genes × methods, height = logFC.
               Shows selective amplification per method.
    Panel B — Heatmap table: methods × pathways, color = -log10(adj p).
               Highlights pathways unique to highlight_method in red.

    Args:
        dge: dict {method: {niche: dge_df}} from niche_dge_gsea
        gsea: dict {method: {niche: enrichr_df}} from niche_dge_gsea
        niche: which niche to show
        genes: list of gene names for Panel A (e.g. ['IGFBP5', 'MMP1', 'FN1'])
        pathways: list of pathway names for Panel B
        highlight_method: method to highlight uniqueness (default: last)
        pval_thr: significance threshold for GSEA
        cap: max -log10(p) for color scale
        method_colors: dict {method: color} — auto if None
        figsize: figure size (auto if None)
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    methods = list(dge.keys())
    if highlight_method is None:
        highlight_method = methods[-1]

    if method_colors is None:
        palette = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    if figsize is None:
        figsize = (max(len(genes) * 1.4, len(pathways) * 1.2) + 2,
                   len(methods) * 0.9 + 5.5)

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.8, 1], hspace=0.55)
    ax_genes = fig.add_subplot(gs[0])
    ax_gsea  = fig.add_subplot(gs[1])

    # ── Panel A: grouped bar chart ────────────────────────────────────────
    n_methods = len(methods)
    bar_w = 0.8 / n_methods
    for mi, method in enumerate(methods):
        res = dge[method].get(niche)
        offsets = np.arange(len(genes)) + (mi - n_methods / 2 + 0.5) * bar_w
        for gi, gene in enumerate(genes):
            lfc = 0.0
            if res is not None:
                row = res[res['names'] == gene]
                if not row.empty:
                    lfc = float(row['logfoldchanges'].iloc[0])
            ax_genes.bar(offsets[gi], lfc, width=bar_w * 0.9,
                         color=method_colors[method],
                         label=method if gi == 0 else None,
                         edgecolor='white', linewidth=0.4)

    ax_genes.set_xticks(np.arange(len(genes)))
    ax_genes.set_xticklabels(genes, fontsize=10, fontweight='bold')
    ax_genes.set_ylabel('log fold change', fontsize=9)
    ax_genes.set_title(f'Gene-level signals — {niche}', fontsize=10,
                        fontweight='bold')
    ax_genes.axhline(0, color='black', linewidth=0.6)
    ax_genes.legend(fontsize=8, frameon=False, loc='upper left')
    ax_genes.spines[['top', 'right']].set_visible(False)

    # ── Panel B: GSEA heatmap ─────────────────────────────────────────────
    mat = pd.DataFrame(0.0, index=methods, columns=pathways)
    sig = pd.DataFrame(False, index=methods, columns=pathways)
    for method in methods:
        res = (gsea[method] or {}).get(niche)
        if res is None:
            continue
        r = res.set_index('Term')
        for pw in pathways:
            if pw in r.index:
                pval = r.loc[pw, 'Adjusted P-value']
                mat.loc[method, pw] = min(-np.log10(pval + 1e-300), cap)
                sig.loc[method, pw] = pval < pval_thr

    others = [m for m in methods if m != highlight_method]
    unique_cols = [
        pw for pw in pathways
        if sig.loc[highlight_method, pw] and not any(sig.loc[m, pw] for m in others)
    ]

    cmap = plt.cm.get_cmap('YlOrRd')
    for yi, method in enumerate(methods):
        for xi, pw in enumerate(pathways):
            s      = mat.loc[method, pw]
            is_sig = sig.loc[method, pw]
            color  = '#e0e0e0' if not is_sig else cmap(s / cap)
            ax_gsea.add_patch(plt.Rectangle(
                [xi - 0.45, yi - 0.45], 0.9, 0.9, color=color, zorder=2))
            if is_sig:
                ax_gsea.text(xi, yi, f'{s:.1f}', ha='center', va='center',
                             fontsize=8, zorder=3,
                             color='white' if s / cap > 0.65 else 'black')

    ax_gsea.set_xticks(range(len(pathways)))
    xlabels = [f'★ {pw}' if pw in unique_cols else pw for pw in pathways]
    ax_gsea.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
    for xi, pw in enumerate(pathways):
        tick = ax_gsea.get_xticklabels()[xi]
        if pw in unique_cols:
            tick.set_color('#c0392b')
            tick.set_fontweight('bold')

    ax_gsea.set_yticks(range(len(methods)))
    ax_gsea.set_yticklabels(methods, fontsize=9, fontweight='bold')
    ax_gsea.set_xlim(-0.5, len(pathways) - 0.5)
    ax_gsea.set_ylim(-0.5, len(methods) - 0.5)
    ax_gsea.invert_yaxis()
    ax_gsea.set_title(f'Pathway enrichment — {niche}', fontsize=10,
                       fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, cap))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_gsea, label=f'-log10(adj p)', shrink=0.6,
                 pad=0.02, location='right')

    patches = [
        mpatches.Patch(color='#e0e0e0', label='not significant'),
        mpatches.Patch(color='#c0392b', label=f'★ unique to {highlight_method}'),
    ]
    ax_gsea.legend(handles=patches, fontsize=7, frameon=False,
                   bbox_to_anchor=(1.25, -0.35), loc='lower right')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, (ax_genes, ax_gsea)


def plot_gsea_table(gsea, niche, pathways, highlight_method=None,
                    pval_thr=0.05, cap=20, figsize=None, save_path=None):
    """Simple heatmap table for selected pathways across methods.

    Rows = methods, columns = pathways (rotated labels for readability).
    Color = -log10(adj p-value), grey = not significant.
    Columns unique to highlight_method get a red bold label + star.

    Args:
        gsea: dict {method: {niche: enrichr_df}} from niche_dge_gsea
        niche: which niche to show
        pathways: list of pathway names to show (in order)
        highlight_method: method to check for uniqueness (default: last)
        pval_thr: significance threshold
        cap: max -log10(p) for color scale
        figsize: auto if None
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    methods = list(gsea.keys())
    if highlight_method is None:
        highlight_method = methods[-1]

    # Build score matrix (methods × pathways)
    mat = pd.DataFrame(0.0, index=methods, columns=pathways)
    sig = pd.DataFrame(False, index=methods, columns=pathways)
    for method in methods:
        res = (gsea[method] or {}).get(niche)
        if res is None:
            continue
        r = res.set_index('Term')
        for pw in pathways:
            if pw in r.index:
                pval = r.loc[pw, 'Adjusted P-value']
                mat.loc[method, pw] = min(-np.log10(pval + 1e-300), cap)
                sig.loc[method, pw] = pval < pval_thr

    # Identify columns (pathways) unique to highlight_method
    others = [m for m in methods if m != highlight_method]
    unique_cols = [
        pw for pw in pathways
        if sig.loc[highlight_method, pw] and not any(sig.loc[m, pw] for m in others)
    ]

    if figsize is None:
        figsize = (len(pathways) * 1.4 + 1.5, len(methods) * 1.0 + 2.0)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap('YlOrRd')

    for yi, method in enumerate(methods):
        for xi, pw in enumerate(pathways):
            s = mat.loc[method, pw]
            is_sig = sig.loc[method, pw]
            color = '#e0e0e0' if not is_sig else cmap(s / cap)
            rect = plt.Rectangle([xi - 0.45, yi - 0.45], 0.9, 0.9,
                                  color=color, zorder=2)
            ax.add_patch(rect)
            if is_sig:
                ax.text(xi, yi, f'{s:.1f}', ha='center', va='center',
                        fontsize=9, zorder=3,
                        color='white' if s / cap > 0.65 else 'black')

    # X-axis — pathway labels rotated, unique ones in red bold
    ax.set_xticks(range(len(pathways)))
    xlabels = [f'★ {pw}' if pw in unique_cols else pw for pw in pathways]
    ax.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=9)
    for xi, pw in enumerate(pathways):
        tick = ax.get_xticklabels()[xi]
        if pw in unique_cols:
            tick.set_color('#c0392b')
            tick.set_fontweight('bold')

    # Y-axis — method labels
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10, fontweight='bold')

    ax.set_xlim(-0.5, len(pathways) - 0.5)
    ax.set_ylim(-0.5, len(methods) - 0.5)
    ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, cap))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f'-log10(adj p), capped at {cap}',
                 shrink=0.5, pad=0.02)

    # Legend
    patches = [
        mpatches.Patch(color='#e0e0e0', label='not significant'),
        mpatches.Patch(color='#c0392b', label=f'★ unique to {highlight_method}'),
    ]
    ax.legend(handles=patches, fontsize=8, frameon=False,
              bbox_to_anchor=(1.0, -0.25), loc='lower right')

    ax.set_title(f'GSEA — {niche}', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_dge_dotplot(dge, niche, top_n=10, figsize=None, title=None, save_path=None):
    """Dot plot of top DGE genes × methods for one niche.

    Dot size = normalized score, color = logFC.
    Genes are the union of top_n per method.

    Args:
        dge: dict {method: {niche: dge_df}} from niche_dge_gsea
        niche: which niche to plot
        top_n: top genes per method to include in union
        figsize: figure size (auto if None)
        title: plot title
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    methods = list(dge.keys())

    # Union of top_n genes across methods (preserve order)
    all_genes = []
    for method in methods:
        res = dge[method].get(niche)
        if res is not None:
            all_genes.extend(res['names'].head(top_n).tolist())
    genes = list(dict.fromkeys(all_genes))

    if figsize is None:
        figsize = (len(methods) * 2 + 1.5, len(genes) * 0.4 + 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    all_scores, all_lfc = [], []
    for method in methods:
        res = dge[method].get(niche)
        if res is not None:
            all_scores.extend(res['scores'].tolist())
            all_lfc.extend(res['logfoldchanges'].tolist())

    max_score = max(all_scores) if all_scores else 1.0
    lfc_min = min(all_lfc) if all_lfc else 0.0
    lfc_max = max(all_lfc) if all_lfc else 1.0

    cmap = plt.cm.get_cmap('RdYlBu_r')

    for xi, method in enumerate(methods):
        res = dge[method].get(niche)
        if res is None:
            continue
        res_indexed = res.set_index('names')
        for yi, gene in enumerate(genes):
            if gene not in res_indexed.index:
                ax.scatter(xi, yi, s=10, color='lightgray', zorder=2)
                continue
            score = res_indexed.loc[gene, 'scores']
            lfc   = res_indexed.loc[gene, 'logfoldchanges']
            size  = (score / max_score) * 300 + 20
            color = cmap((lfc - lfc_min) / (lfc_max - lfc_min + 1e-9))
            ax.scatter(xi, yi, s=size, color=color, zorder=2, edgecolors='gray', linewidths=0.3)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.grid(True, alpha=0.2, zorder=1)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(-0.5, len(genes) - 0.5)
    ax.invert_yaxis()

    # Colorbar for logFC
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=lfc_min, vmax=lfc_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='log fold change', shrink=0.5)

    # Size legend
    for s_label, s_val in [('low', 0.25), ('mid', 0.6), ('high', 1.0)]:
        ax.scatter([], [], s=s_val * 300 + 20, color='gray', alpha=0.6, label=s_label)
    ax.legend(title='score', loc='lower right', frameon=False, fontsize=8)

    ax.set_title(title or f'Top DGE genes — {niche}', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax


def build_metacell_adata(assignments, avg_expression, fib_ad, niche_key='niches_2D'):
    """Build an AnnData for metacells with majority-voted niche labels.

    Args:
        assignments: cell -> metacell ID array (n_cells,), aligned to fib_ad
        avg_expression: (n_metacells, n_genes) expression matrix
        fib_ad: filtered AnnData whose obs contains niche_key
        niche_key: obs column to use for majority voting

    Returns:
        AnnData with shape (n_metacells, n_genes), obs[niche_key] = majority label
    """
    import pandas as pd
    import anndata as ann

    cell_labels = np.asarray(fib_ad.obs[niche_key])
    n_metacells = avg_expression.shape[0]

    metacell_niche = []
    for mc_id in range(n_metacells):
        mask = assignments == mc_id
        if mask.sum() == 0:
            metacell_niche.append('Unknown')
        else:
            counts = pd.Series(cell_labels[mask]).value_counts()
            metacell_niche.append(counts.index[0])

    obs = pd.DataFrame(
        {niche_key: pd.Categorical(metacell_niche)},
        index=[f'mc_{i}' for i in range(n_metacells)]
    )
    return ann.AnnData(X=avg_expression.copy(), obs=obs, var=fib_ad.var.copy())


def run_niche_dge(adata, niche_key='niches_2D', method='wilcoxon', top_k=200):
    """Run niche DGE (each niche vs rest) on single-cell or metacell AnnData.

    Args:
        adata: AnnData with niche labels in obs[niche_key] and lognorm in X
        niche_key: obs column for niches
        method: scanpy rank_genes_groups method ('wilcoxon' or 't-test')
        top_k: number of top genes to keep per niche

    Returns:
        dict: {niche_name: DataFrame with [names, scores, logfoldchanges, pvals_adj]}
    """
    import scanpy as sc

    adata = adata.copy()
    valid = adata.obs[niche_key].value_counts()
    valid = valid[valid > 1].index.tolist()
    adata = adata[adata.obs[niche_key].isin(valid)].copy()

    sc.tl.rank_genes_groups(
        adata, groupby=niche_key, groups=valid,
        reference='rest', method=method, use_raw=False
    )

    results = {}
    for niche in valid:
        df = sc.get.rank_genes_groups_df(adata, group=niche)
        df = df.sort_values('scores', ascending=False).head(top_k)
        results[niche] = df
    return results


def run_gsea(dge_results, gene_sets='MSigDB_Hallmark_2020', top_k=100, organism='Human'):
    """Run Enrichr GSEA on top genes per niche from DGE results.

    Requires gseapy: pip install gseapy

    Args:
        dge_results: dict {niche: DataFrame} from run_niche_dge
        gene_sets: Enrichr library name or list of names
                   e.g. 'MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2023'
        top_k: how many top-scoring genes per niche to submit
        organism: 'Human' or 'Mouse'

    Returns:
        dict: {niche: enrichr results DataFrame sorted by Adjusted P-value}
    """
    import gseapy as gp

    enrichment = {}
    for niche, df in dge_results.items():
        genes = df['names'].head(top_k).tolist()
        print(f"  {niche}: {len(genes)} genes -> Enrichr")
        try:
            enr = gp.enrichr(
                gene_list=genes, gene_sets=gene_sets,
                organism=organism, outdir=None, verbose=False
            )
            enrichment[niche] = enr.results.sort_values('Adjusted P-value')
        except Exception as e:
            print(f"  WARNING: GSEA failed for {niche}: {e}")
            enrichment[niche] = None
    return enrichment


def niche_dge_gsea(fib_ad, base_dir, methods, niche_key='niches_2D',
                   dge_top_k=200, gsea_top_k=100,
                   gene_sets='MSigDB_Hallmark_2020', organism='Human',
                   dge_method='wilcoxon', run_gsea_flag=True):
    """Run niche DGE + Enrichr GSEA for single cells and metacell methods.

    Args:
        fib_ad: filtered AnnData (e.g. fibroblasts), X must be lognorm
        base_dir: model directory with method subfolders
        methods: dict {display_label: folder_name}
                 e.g. {'SEACells': 'seacells_native_K50', 'SCProto': 'scproto_...'}
        niche_key: obs column for niches
        dge_top_k: top genes to keep per niche from DGE
        gsea_top_k: top genes to submit to Enrichr
        gene_sets: Enrichr library name(s)
        organism: 'Human' or 'Mouse'
        dge_method: 'wilcoxon' or 't-test'
        run_gsea_flag: set False to skip GSEA and return only DGE

    Returns:
        dge:  dict {method_label: {niche: dge_df}}
        gsea: dict {method_label: {niche: enrichr_df}}  (empty if run_gsea_flag=False)
    """
    clusters = list_saved_clusters(base_dir)

    dge, gsea = {}, {}

    # --- Single cells ---
    print("=== Single cells ===")
    dge['singlecell'] = run_niche_dge(fib_ad, niche_key=niche_key,
                                       method=dge_method, top_k=dge_top_k)
    if run_gsea_flag:
        print("  Running GSEA...")
        gsea['singlecell'] = run_gsea(dge['singlecell'], gene_sets, gsea_top_k, organism)

    # --- Metacell methods ---
    for label, folder in methods.items():
        print(f"=== {label} ===")
        if folder not in clusters:
            print(f"  WARNING: {folder} not found, skipping")
            continue
        c = clusters[folder]
        if c['expression'] is None:
            print(f"  WARNING: no expression matrix for {folder}, skipping")
            continue

        mc_ad = build_metacell_adata(c['assignments'], c['expression'], fib_ad, niche_key)
        dge[label] = run_niche_dge(mc_ad, niche_key=niche_key,
                                    method=dge_method, top_k=dge_top_k)
        if run_gsea_flag:
            print("  Running GSEA...")
            gsea[label] = run_gsea(dge[label], gene_sets, gsea_top_k, organism)

    return dge, gsea


def plot_spatial_highlight(adata, highlight_key, highlight_values, spatial_key='spatial',
                           figsize=(8, 7), point_size=3, title=None):
    """Plot spatial with specific categories highlighted, rest in gray.

    Args:
        adata: AnnData with spatial coordinates
        highlight_key: obs column to filter (e.g., 'cell_type')
        highlight_values: list of values to highlight (e.g., ['Fibroblasts'])
        spatial_key: key in obsm for spatial coordinates
        figsize: figure size
        point_size: scatter point size
        title: plot title

    Returns:
        fig, ax
    """
    import matplotlib.pyplot as plt

    coords = adata.obsm[spatial_key]
    labels = adata.obs[highlight_key]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot non-highlighted cells in gray
    mask_other = ~labels.isin(highlight_values)
    ax.scatter(coords[mask_other, 0], coords[mask_other, 1],
              c='lightgray', s=point_size, alpha=0.3, label='Other')

    # Plot highlighted cells
    cmap = plt.cm.get_cmap('tab10', len(highlight_values))
    for i, val in enumerate(highlight_values):
        mask = labels == val
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[cmap(i)], s=point_size, alpha=0.8, label=val)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
    return fig, ax

