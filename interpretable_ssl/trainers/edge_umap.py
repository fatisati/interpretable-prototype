"""
Edge-centric Parametric UMAP training.

This module implements UMAP training the way the official implementation does:
- Sample edges (i, j) with probability proportional to their weight p_ij
- For each positive edge, sample K negative edges from non-neighbors
- Loss = attractive (positives) + repulsive (negatives)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, coo_matrix
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings


def find_ab_params(spread: float, min_dist: float) -> Tuple[float, float]:
    """
    Compute a, b parameters for UMAP's low-dimensional similarity kernel.

    This fits the curve q(d) = 1 / (1 + a * d^(2b)) to match the target:
        - 1.0 for d < min_dist
        - exp(-(d - min_dist) / spread) for d >= min_dist

    Args:
        spread: Effective scale of embedded points (default 1.0)
        min_dist: Minimum distance between embedded points (scanpy default 0.5)

    Returns:
        (a, b): Kernel parameters
    """
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)

    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


class EdgeDataset(Dataset):
    """
    Dataset that samples edges from an affinity graph using weighted sampling.

    Each edge is stored once. A WeightedRandomSampler picks edges proportional
    to their affinity weight, so strong edges are visited more often per epoch.
    Negative samples are drawn on-the-fly within the head node's dataset.
    """

    def __init__(
        self,
        affinity: csr_matrix,
        n_epochs: int = 200,
        negative_sample_rate: int = 5,
        seed: int = 42,
        cell_ds: Optional[np.ndarray] = None,
        neg_ds_id: Optional[int] = None,
    ):
        self.n_cells = affinity.shape[0]
        self.negative_sample_rate = negative_sample_rate
        self.rng = np.random.RandomState(seed)

        # Build per-ds cell index arrays for restricted negative sampling
        if cell_ds is not None:
            self.cell_ds = cell_ds
            ds_ids = np.unique(cell_ds)
            self.ds_cells = {ds: np.where(cell_ds == ds)[0] for ds in ds_ids}
        else:
            self.cell_ds = None
            self.ds_cells = None
        self.neg_ds_id = neg_ds_id

        # Convert to COO for easy iteration
        aff_coo = affinity.tocoo()

        # Get edges and weights
        self.head = aff_coo.row.astype(np.int64)
        self.tail = aff_coo.col.astype(np.int64)
        self.weights = aff_coo.data.astype(np.float32)

        # Filter out very weak edges (like official UMAP)
        weight_threshold = self.weights.max() / float(n_epochs)
        mask = self.weights >= weight_threshold
        self.head = self.head[mask]
        self.tail = self.tail[mask]
        self.weights = self.weights[mask]

        # Build adjacency sets for negative sampling (to exclude actual neighbors)
        self.adj_sets = self._build_adjacency_sets(affinity)

        print(f"📊 EdgeDataset: {len(self.head)} edges")
        print(f"   Weight range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")

    def _build_adjacency_sets(self, affinity: csr_matrix) -> dict:
        """Build set of neighbors for each node (for negative sampling exclusion)."""
        adj_sets = {i: set() for i in range(self.n_cells)}
        aff_coo = affinity.tocoo()
        for i, j, w in zip(aff_coo.row, aff_coo.col, aff_coo.data):
            if w > 0:
                adj_sets[i].add(j)
        return adj_sets

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        head = self.head[idx]
        tail = self.tail[idx]
        weight = self.weights[idx]

        # Sample negatives on-the-fly (non-neighbors of head, within head's dataset)
        neighbors = self.adj_sets[head]
        if self.neg_ds_id is not None:
            candidates = self.ds_cells[self.neg_ds_id]
        elif self.cell_ds is not None:
            candidates = self.ds_cells[self.cell_ds[head]]
        else:
            candidates = None
        neg_samples = []
        attempts = 0
        while len(neg_samples) < self.negative_sample_rate and attempts < 100:
            if candidates is not None:
                neg = candidates[self.rng.randint(0, len(candidates))]
            else:
                neg = self.rng.randint(0, self.n_cells)
            if neg != head and neg not in neighbors:
                neg_samples.append(neg)
            attempts += 1

        while len(neg_samples) < self.negative_sample_rate:
            if candidates is not None:
                neg_samples.append(candidates[self.rng.randint(0, len(candidates))])
            else:
                neg_samples.append(self.rng.randint(0, self.n_cells))

        return {
            'head': head,
            'tail': tail,
            'weight': weight,
            'neg_samples': np.array(neg_samples, dtype=np.int64),
        }


def edge_collate_fn(batch):
    """Collate edge samples into batched tensors."""
    heads = torch.tensor([b['head'] for b in batch], dtype=torch.long)
    tails = torch.tensor([b['tail'] for b in batch], dtype=torch.long)
    weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)
    neg_samples = torch.tensor(np.stack([b['neg_samples'] for b in batch]), dtype=torch.long)

    return {
        'head': heads,        # [B]
        'tail': tails,        # [B]
        'weight': weights,    # [B]
        'neg_samples': neg_samples,  # [B, K]
    }


class ParametricUMAPLoss(nn.Module):
    """
    Parametric UMAP loss function.

    For positive edges (i, j) with weight w_ij:
        L_pos = -w_ij * log(q_ij)

    For negative samples (i, k):
        L_neg = -log(1 - q_ik)

    Where q = 1 / (1 + a * d^(2b)) is the low-dim similarity.
    """

    def __init__(
        self,
        min_dist: float = 0.5,
        spread: float = 1.0,
        negative_sample_rate: int = 5,
    ):
        super().__init__()
        self.a, self.b = find_ab_params(spread, min_dist)
        self.negative_sample_rate = negative_sample_rate
        self.eps = 1e-4
        print(f"📐 UMAP kernel: min_dist={min_dist}, spread={spread} -> a={self.a:.4f}, b={self.b:.4f}")

    def compute_q(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute low-dim similarity: q = 1 / (1 + a * d^(2b))

        Args:
            z_i: [B, d] or [B, K, d] embeddings
            z_j: [B, d] or [B, K, d] embeddings

        Returns:
            q: [B] or [B, K] similarities
        """
        d_sq = ((z_i - z_j) ** 2).sum(dim=-1)  # squared distance
        q = 1.0 / (1.0 + self.a * d_sq.pow(self.b))
        return q.clamp(self.eps, 1.0 - self.eps)

    def forward(
        self,
        z_head: torch.Tensor,      # [B, d]
        z_tail: torch.Tensor,      # [B, d]
        z_neg: torch.Tensor,       # [B, K, d]
        weights: torch.Tensor,     # [B]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute UMAP loss.

        Args:
            z_head: Embeddings of source nodes [B, d]
            z_tail: Embeddings of target nodes (positives) [B, d]
            z_neg: Embeddings of negative samples [B, K, d]
            weights: Edge weights (p_ij) [B]

        Returns:
            loss: Scalar loss
            metrics: Dict with q_pos, q_neg, margin, etc.
        """
        # Positive loss: -w * log(q)
        q_pos = self.compute_q(z_head, z_tail)  # [B]
        loss_pos = -(weights * torch.log(q_pos)).mean()

        # Negative loss: -log(1 - q) for each negative
        z_head_exp = z_head.unsqueeze(1)  # [B, 1, d]
        q_neg = self.compute_q(z_head_exp, z_neg)  # [B, K]
        loss_neg = -torch.log(1.0 - q_neg).sum(dim=1).mean()  # sum over K, mean over B

        loss = loss_pos + loss_neg

        metrics = {
            'q_pos': q_pos.mean().item(),
            'q_neg': q_neg.mean().item(),
            'margin': (q_pos.mean() - q_neg.mean()).item(),
            'loss_pos': loss_pos.item(),
            'loss_neg': loss_neg.item(),
            'w_mean': weights.mean().item(),
        }

        return loss, metrics


class EdgeUMAPTrainer:
    """
    Trainer for edge-centric parametric UMAP.

    This trains an encoder network using the UMAP loss on edges sampled
    from an affinity graph.
    """

    def __init__(
        self,
        encoder: nn.Module,
        affinity: csr_matrix,
        X: torch.Tensor,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_epochs: int = 200,
        batch_size: int = 1024,
        negative_sample_rate: int = 5,
        lr: float = 1e-3,
        device: str = 'cuda',
    ):
        """
        Args:
            encoder: Neural network that encodes X[i] -> z[i]
            affinity: Sparse affinity matrix
            X: Data tensor [n_cells, n_features]
            min_dist: UMAP min_dist parameter
            spread: UMAP spread parameter
            n_epochs: Number of training epochs
            batch_size: Batch size (number of edges per step)
            negative_sample_rate: Negatives per positive
            lr: Learning rate
            device: Device to train on
        """
        self.encoder = encoder.to(device)
        self.X = X.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Create edge dataset
        self.edge_dataset = EdgeDataset(
            affinity,
            n_epochs=n_epochs,
            negative_sample_rate=negative_sample_rate,
        )

        # Loss function
        self.loss_fn = ParametricUMAPLoss(
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

        # Data loader
        self.loader = DataLoader(
            self.edge_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=edge_collate_fn,
            num_workers=0,
            drop_last=True,
        )

    def train_epoch(self) -> dict:
        """Train for one epoch, returns average metrics."""
        self.encoder.train()

        total_metrics = {
            'loss': 0, 'q_pos': 0, 'q_neg': 0, 'margin': 0,
            'loss_pos': 0, 'loss_neg': 0, 'w_mean': 0,
        }
        n_batches = 0

        for batch in self.loader:
            head = batch['head'].to(self.device)      # [B]
            tail = batch['tail'].to(self.device)      # [B]
            weights = batch['weight'].to(self.device) # [B]
            neg_samples = batch['neg_samples'].to(self.device)  # [B, K]

            # Get unique node indices for this batch
            all_idx = torch.cat([head, tail, neg_samples.flatten()])
            unique_idx = torch.unique(all_idx)

            # Encode only unique nodes (efficiency)
            X_batch = self.X[unique_idx]
            Z_unique = self.encoder(X_batch)

            # Map back to original indices
            idx_map = {int(idx): i for i, idx in enumerate(unique_idx.cpu().numpy())}

            def gather_embeddings(indices):
                mapped = torch.tensor([idx_map[int(i)] for i in indices.cpu().numpy()],
                                      device=self.device, dtype=torch.long)
                return Z_unique[mapped]

            z_head = gather_embeddings(head)  # [B, d]
            z_tail = gather_embeddings(tail)  # [B, d]

            # For negatives, need to reshape
            B, K = neg_samples.shape
            neg_flat = neg_samples.flatten()
            z_neg_flat = gather_embeddings(neg_flat)  # [B*K, d]
            z_neg = z_neg_flat.view(B, K, -1)  # [B, K, d]

            # Compute loss
            loss, metrics = self.loss_fn(z_head, z_tail, z_neg, weights)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_metrics['loss'] += loss.item()
            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1

        # Average
        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        return total_metrics

    def train(self, epochs: int = None, verbose: bool = True):
        """Train for specified epochs."""
        epochs = epochs or self.n_epochs

        for epoch in range(epochs):
            metrics = self.train_epoch()

            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"loss={metrics['loss']:.4f} | "
                      f"q+={metrics['q_pos']:.3f} | "
                      f"q-={metrics['q_neg']:.3f} | "
                      f"margin={metrics['margin']:.3f}")

            # Reshuffle dataset for next epoch
            self.edge_dataset.reshuffle()

        return metrics

    def encode(self, X: torch.Tensor = None) -> torch.Tensor:
        """Encode data with trained encoder."""
        self.encoder.eval()
        X = X if X is not None else self.X
        with torch.no_grad():
            return self.encoder(X.to(self.device))
