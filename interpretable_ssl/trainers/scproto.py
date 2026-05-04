import math
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import shutil
import time
from logging import getLogger
from tqdm import tqdm
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

from interpretable_ssl.swav_utils import initialize_exp, fix_random_seeds

from interpretable_ssl.configs.larc import LARC
from interpretable_ssl.configs.defaults import *
from interpretable_ssl.utils import *
from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer
from interpretable_ssl.trainers.scproto_utils import *
from interpretable_ssl.trainers.affinity import *
from interpretable_ssl.trainers.scpoli_helpers import *
from interpretable_ssl.trainers.edge_umap import EdgeDataset, ParametricUMAPLoss, edge_collate_fn, find_ab_params
from interpretable_ssl.augmenters.adata_augmenter import *
from interpretable_ssl.models.swav import *
from interpretable_ssl.evaluation.visualization import *
from interpretable_ssl.evaluation.prototype_metrics import *
from interpretable_ssl.evaluation.cd4_marker import *
from interpretable_ssl.evaluation.mc_metric_utils import *

from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils
import wandb

logger = getLogger()


def get_gpu_type_torch():
    if not torch.cuda.is_available():
        return "No GPU available"
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def affinity_report(A):
    import scipy.sparse as sp
    import numpy as np

    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()

    # --- existing ---
    nnz = np.diff(A.indptr)
    mean_deg = nnz.mean()
    med_deg = np.median(nnz)

    # --- NEW: weighted degree ---
    row_sum = np.array(A.sum(axis=1)).flatten()
    deg_min = row_sum.min()
    deg_max = row_sum.max()
    deg_mean = row_sum.mean()

    ent = np.zeros(A.shape[0])
    effk = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i+1]
        if s == e:
            continue
        p = A.data[s:e]
        p = p / (p.sum() + 1e-12)
        ent[i] = -(p * np.log(p + 1e-12)).sum()
        effk[i] = np.exp(ent[i])

    B = A.astype(bool)
    mutual = (B.multiply(B.T)).sum()
    total = B.sum()
    mutual_ratio = float(mutual / (total + 1e-12))

    return {
        "mean_deg": float(mean_deg),
        "med_deg": float(med_deg),

        # NEW
        "deg_min": float(deg_min),
        "deg_max": float(deg_max),
        "deg_mean": float(deg_mean),

        "effk_mean": float(effk.mean()),
        "effk_med": float(np.median(effk)),
        "mutual_ratio": mutual_ratio,
        "frac_empty_rows": float((nnz == 0).mean()),
    }
    
@torch.no_grad()
def _per_batch_sinkhorn(scaled_logits, batch_ids, n_iters=3):
    """Per-batch Sinkhorn normalization.

    Each batch's submatrix is independently normalized to be doubly stochastic:
      - each cell's row sums to 1 (valid assignment distribution)
      - each proto's column sums to N_b/K within batch b (uniform usage per batch)

    Args:
        scaled_logits: (N, K) tensor, already divided by epsilon
        batch_ids:     (N,) integer tensor of batch id per cell (on same device)
        n_iters:       number of Sinkhorn iterations

    Returns:
        Q: (N, K) float tensor, no gradient
    """
    K = scaled_logits.shape[1]
    Q = scaled_logits.exp().clone()
    unique_batches = batch_ids.unique()
    for _ in range(n_iters):
        # row norm: each cell sums to 1
        Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # per-batch column norm: each proto gets N_b/K total weight within batch b
        for b in unique_batches:
            mask = batch_ids == b
            N_b = mask.sum().float()
            col_sums = Q[mask].sum(dim=0).clamp(min=1e-8)
            Q[mask] = Q[mask] / col_sums * (N_b / K)
    # final row norm so rows are valid distributions
    Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return Q


def _med_effk_from_logits(logits, temp):
    p = torch.softmax(logits / temp, dim=1)
    effk = 1.0 / (p * p).sum(dim=1)
    return torch.median(effk).item()


def _solve_temp_for_target_effk(logits, target_effk, iters=40, lo=1e-3, hi=50.0):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        med = _med_effk_from_logits(logits, mid)
        if med < target_effk:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_eps_tau(model, z, effk_aff, student_factor=2.0):
    """Calibrate epsilon and tau based on target effective neighbors."""
    with torch.no_grad():
        logits = model.proto_soft_assignments(z)
    eps = _solve_temp_for_target_effk(logits, target_effk=effk_aff)
    tau = _solve_temp_for_target_effk(logits, target_effk=student_factor * effk_aff)
    return eps, tau


class SCProtoTrainer(AdoptiveTrainer):

    # @log_time('swav')
    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs):
        logger.info(get_gpu_type_torch())

        if "experiment_name" not in kwargs:
            kwargs["experiment_name"] = "scproto"
        super().__init__(dataset, ref_query, parser, **kwargs)
        self.nmb_prototypes = self.num_prototypes
        self.use_projector_out = False

        self.train_augmentation = self.augmentation_type
        self.queue = {}
        if self.condition_key is not None:
            ds_cnt = self.train_.adata.obs[self.condition_key].nunique()
        else:
            ds_cnt = 1
        self.ds_ids = range(ds_cnt)
        self.loss_keys = [
            "swav",
            "recon",
            "kl",
            "proto",
            "commit",
            "aff",
            "proto_recon",
            "r1r2",
        ]
        self.log_hist = {}
        self._metrics_log = {}

    def setup(self):
        fix_random_seeds(self.seed)
        self.dump_path = self.get_dump_path()
        if self.wandb_sweep == 0 and self.debug == 0:
            logger, self.training_stats = initialize_exp(
                self, "epoch", "loss", dump_params=self.wandb_sweep == 0
            )
        self.build_model()
        self.build_data()
        self.build_optimizer()
        self._temperature0 = float(self.temperature)
        self._epsilon0 = float(self.epsilon)
        # self.load_checkpoint()

    def calibrate_temperatures(self, effk_target=None, student_factor=2.0):
        """Calibrate epsilon and tau from data based on affinity effective neighbors."""
        if effk_target is None:
            if hasattr(self, 'aff_stats'):
                effk_target = self.aff_stats['effk_med']
            else:
                logger.warning("No affinity stats available, using default effk=10")
                effk_target = 10.0

        z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
        eps, tau = calibrate_eps_tau(self.model, z, effk_target, student_factor)

        old_eps, old_tau = self.epsilon, self.temperature
        self.epsilon = eps
        self.temperature = tau
        # Update initial values for scheduler
        self._epsilon0 = float(eps)
        self._temperature0 = float(tau)

        logger.info(f"Calibrated: eps {old_eps:.4f} -> {eps:.4f}, tau {old_tau:.4f} -> {tau:.4f} (target effk={effk_target:.1f})")
        print(f"🎯 Calibrated: eps={eps:.4f}, tau={tau:.4f} (from effk={effk_target:.1f})")
        return eps, tau

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val

        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        scpoli_encoder = self.model.scpoli_cvae
        common_dataset_kwargs = dict(
            n_augmentations=self.nmb_views,
            affinity_type=self.affinity_type,
            k_neighbors=self.k_neighbors,
            n_components=self.n_components,
            use_bknn=self.use_bknn,
            condition_keys=[self.condition_key],
            save_dir="./graphs",
            mask_probability=self.mask_probability,
            default_dispersion=self.default_dispersion,
            spatial=self.spatial,
            n_clusters=self.num_prototypes,
            use_counts=(self.use_counts == 1),
            n_proto=self.nmb_prototypes,
            use_manifold_weights=(self.cell_w_mode != "uniform"),
            mc_size=self.mc_size,
            adoptive_eps=(self.adoptive_eps == 1),
            p=self.p,
            k_pos=self.k_pos,
            softm=(self.softm == 1),
            graph_mode=self.graph_mode,
            condition_encoders=scpoli_encoder.condition_encoders,
            conditions_combined_encoder=scpoli_encoder.conditions_combined_encoder,
            # cell_type_keys=[self.cell_type_key],
            # cell_type_encoder=model.cell_type_encoder,
        )
        self.train_ds = MultiCropsDataset(self.train_, **common_dataset_kwargs)
        self.val_.adata = add_condition_combined(
            self.val_.adata, [self.train_.batch_key]
        )
        self.test_ds = MultiCropsDataset(self.val_, **common_dataset_kwargs)

        # Print affinity report if available (use raw affinity for accurate weight stats)
        _aff_for_report = (self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw')
                           else self.train_ds.aff)
        if _aff_for_report is not None:
            self.aff_stats = affinity_report(_aff_for_report)
            logger.info(f"Affinity stats: {self.aff_stats}")
            print(
                f"📊 Affinity: "
                f"wdeg[min/mean/max]={self.aff_stats['deg_min']:.3f}/"
                f"{self.aff_stats['deg_mean']:.3f}/"
                f"{self.aff_stats['deg_max']:.3f}, "
                f"effk_med={self.aff_stats['effk_med']:.1f}, "
                f"mutual={self.aff_stats['mutual_ratio']:.2%}"
            )
        self.train_loader = self.get_data_laoder(self.train_ds)
        self.test_loader = self.get_data_laoder(self.test_ds, drop_last=False)

        self.original_train_loader = self.train_loader

    def get_data_laoder(self, ds, drop_last=True):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            # drop_last=drop_last,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=True,
        )

    def get_model_path(self):
        return os.path.join(self.get_dump_path(), self.get_checkpoint_file())

    def load_model(self):
        model = self.get_model()
        if self.ft_epochs > 0 and self.full_dataset_mode == 0:
            model = self.adapt_model(model, self.query.adata)
        checkpoint_path = self.get_model_path()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # model, _ = apex.amp.initialize(model, self.optimizer, opt_level="O1")

        # model = apex.amp.initialize(model, opt_level="O1")

        # self.model = model
        return model

    def init_prototypes(self):
        if self.prot_init == "kmeans":
            logger.info("initalizing prototypes using kmeans")
            embeddings = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            self.model.init_prototypes_kmeans(embeddings, self.nmb_prototypes)
        elif self.prot_init == "waypoint":
            logger.info("initializing prototypes using waypoint (topology-aware MaxMin)")
            self._init_prototypes_waypoint()

    def _init_prototypes_waypoint(self, n_eigs=10):
        """Topology-aware prototype init mirroring SEACells' initialize_archetypes().

        Steps:
          1. Symmetrise aff_raw and compute its normalised diffusion map
             (top n_eigs eigenvectors of D^{-1/2} A D^{-1/2}).
          2. Greedy MaxMin in diffusion space: iteratively pick the cell
             that is farthest from all already-chosen cells.
          3. Encode the K chosen cells and set them as prototype vectors.

        This guarantees coverage of every topological region of the affinity
        graph, including rare populations — unlike k-means which over-represents
        dense clusters.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        from tqdm import tqdm

        K   = self.nmb_prototypes
        aff = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        N   = aff.shape[0]
        print(f"[waypoint init] N={N} cells, K={K} prototypes, n_eigs={n_eigs}")

        A  = sp.csr_matrix(aff)
        A  = (A + A.T) / 2

        # --- normalised affinity (same normalisation as SEACells) ---
        print("[waypoint init] computing diffusion map ...")
        d          = np.array(A.sum(axis=1)).ravel()
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-8)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L_sym      = D_inv_sqrt @ A @ D_inv_sqrt

        n_eigs = min(n_eigs, N - 2)
        _, vecs = eigsh(L_sym, k=n_eigs, which='LM')   # [N, n_eigs]
        print(f"[waypoint init] diffusion map done — {n_eigs} eigenvectors")

        # --- greedy MaxMin in diffusion space ---
        chosen    = [np.random.randint(0, N)]
        min_dists = np.full(N, np.inf)

        for _ in tqdm(range(K - 1), desc="waypoint MaxMin", unit="proto"):
            last      = chosen[-1]
            d_last    = ((vecs - vecs[last]) ** 2).sum(axis=1)
            min_dists = np.minimum(min_dists, d_last)
            chosen.append(int(min_dists.argmax()))

        chosen = np.array(chosen)
        print(f"[waypoint init] selected {K} seed cells")

        # --- encode chosen cells and set as prototypes ---
        adata_chosen = self.train_ds.adata[chosen]
        embeddings   = self.encode_adata(adata_chosen, self.model, z_idx=1)  # [K, d]

        centers = torch.nn.functional.normalize(embeddings.float(), dim=1)
        self.model.set_prototypes(centers)

    @staticmethod
    def _proto_sim(s_i, s_j, metric='dotp'):
        """Pairwise proto similarity between softmax assignment vectors.

        Args:
            s_i, s_j: (B, K) softmax assignment vectors
            metric: 'dotp' | 'cosine' | 'bhattacharyya' | 'jsd' | 'bhatt_dist'

        Note: epsilon calibration always uses dotp regardless of metric.

        'bhatt_dist' returns Bhattacharyya *distance* = -log(BC), range [0, ∞).
        Caller converts to q via (1 + a * d^(2b))^-1.
        All other metrics return similarity directly used as q.
        """
        if metric == 'cosine':
            return (F.normalize(s_i, dim=-1) * F.normalize(s_j, dim=-1)).sum(dim=-1)
        elif metric == 'bhattacharyya':
            return ((s_i + 1e-8).sqrt() * (s_j + 1e-8).sqrt()).sum(dim=-1)
        elif metric == 'jsd':
            m = 0.5 * (s_i + s_j)
            return (0.5 * (s_i * (s_i / m.clamp(min=1e-8)).log()).sum(dim=-1)
                  + 0.5 * (s_j * (s_j / m.clamp(min=1e-8)).log()).sum(dim=-1))
        elif metric == 'bhatt_dist':
            BC = (s_i.clamp(min=1e-8).sqrt() * s_j.clamp(min=1e-8).sqrt()).sum(dim=-1).clamp(max=1.0 - 1e-6)
            return -BC.log()
        elif metric == 'hellinger':
            BC = (s_i.clamp(min=1e-8).sqrt() * s_j.clamp(min=1e-8).sqrt()).sum(dim=-1).clamp(max=1.0)
            return (1.0 - BC).clamp(min=0.0).sqrt()
        elif metric == 'idot':
            return 1.0 - (s_i * s_j).sum(dim=-1)
        else:  # dotp
            return (s_i * s_j).sum(dim=-1)

    @staticmethod
    def _proto_sim_neg(s_i, s_neg, metric='dotp'):
        """Similarity between s_i (B,K) and negatives s_neg (B,N,K) → (B,N)."""
        if metric == 'cosine':
            return (F.normalize(s_i, dim=-1).unsqueeze(1) * F.normalize(s_neg, dim=-1)).sum(dim=-1)
        elif metric == 'bhattacharyya':
            return ((s_i + 1e-8).sqrt().unsqueeze(1) * (s_neg + 1e-8).sqrt()).sum(dim=-1)
        elif metric == 'jsd':
            s_i_ = s_i.unsqueeze(1)
            m = 0.5 * (s_i_ + s_neg)
            return (0.5 * (s_i_ * (s_i_ / m.clamp(min=1e-8)).log()).sum(dim=-1)
                  + 0.5 * (s_neg * (s_neg / m.clamp(min=1e-8)).log()).sum(dim=-1))
        elif metric == 'bhatt_dist':
            s_i_ = s_i.unsqueeze(1)                                              # (B, 1, K)
            BC = (s_i_.clamp(min=1e-8).sqrt() * s_neg.clamp(min=1e-8).sqrt()).sum(dim=-1).clamp(max=1.0 - 1e-6)
            return -BC.log()                                                      # (B, N)
        elif metric == 'hellinger':
            s_i_ = s_i.unsqueeze(1)                                              # (B, 1, K)
            BC = (s_i_.clamp(min=1e-8).sqrt() * s_neg.clamp(min=1e-8).sqrt()).sum(dim=-1).clamp(max=1.0)
            return (1.0 - BC).clamp(min=0.0).sqrt()                              # (B, N)
        elif metric == 'idot':
            return 1.0 - (s_i.unsqueeze(1) * s_neg).sum(dim=-1)                 # (B, N)
        else:  # dotp
            return (s_i.unsqueeze(1) * s_neg).sum(dim=-1)

    def calibrate_epsilon(self, n_samples=5000):
        """Calibrate epsilon for proto mode.

        calibrate_eps=1: p/q matching (dotp only) — binary-searches epsilon so E[q_pos] = E[p_pos].
          Falls back to effk if target is unreachable OR if the resulting effk_mean < 3.
          Non-dotp metrics always use effk (p/q matching is meaningless for distance kernels).
        calibrate_eps=2: effk alignment — always uses calibrate_effk regardless of metric or p/q result.
        Sets self.epsilon in-place and returns the found value.
        """
        proto_metric = getattr(self, 'umap_proto_metric', 'dotp')
        effk_target = getattr(self, 'umap_proto_effk', 5.0)

        # calibrate_eps=2: always use effk, skip p/q entirely
        if getattr(self, 'calibrate_eps', 1) == 2:
            print(f"[eps calibration] calibrate_eps=2, using effk target={effk_target:.1f}")
            return self.calibrate_effk(effk_target)

        if proto_metric != 'dotp':
            print(f"[eps calibration] metric={proto_metric}, using effk target={effk_target:.1f}")
            return self.calibrate_effk(effk_target)

        import scipy.sparse as sp

        aff = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        coo = sp.coo_matrix(aff)

        # Sample positive pairs
        n = min(n_samples, len(coo.data))
        idx = np.random.choice(len(coo.data), n, replace=False)
        heads = torch.tensor(coo.row[idx], dtype=torch.long, device=self.device)
        tails = torch.tensor(coo.col[idx], dtype=torch.long, device=self.device)
        p_pos = float(coo.data[idx].mean())

        # Encode all cells once
        self.model.eval()
        with torch.no_grad():
            z_all = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            z_all = z_all.to(self.device)
            logits_all = self.model.prototypes(z_all)   # n_cells × K

        def mean_q_pos(eps):
            with torch.no_grad():
                soft = torch.softmax(logits_all / eps, dim=-1)
                q = self._proto_sim(soft[heads], soft[tails], metric='dotp').mean().item()
            return q

        def effk_stats(eps):
            with torch.no_grad():
                soft = torch.softmax(logits_all / eps, dim=-1)
                ek = 1.0 / (soft ** 2).sum(dim=-1)
                return ek.mean().item(), ek.median().item()

        # Check if target is achievable (max q_pos at sharpest epsilon)
        q_at_lo = mean_q_pos(1e-4)
        if q_at_lo < p_pos:
            print(f"[eps calibration] E[p_pos]={p_pos:.4f} unreachable (max E[q_pos]={q_at_lo:.4f}), "
                  f"falling back to effk={effk_target:.1f}")
            return self.calibrate_effk(effk_target)

        lo, hi = 1e-4, 10.0
        for _ in range(60):
            eps = (lo + hi) / 2
            q = mean_q_pos(eps)
            if q < p_pos:
                hi = eps
            else:
                lo = eps

        ek_mean, ek_med = effk_stats(eps)
        if ek_mean < 3.0:
            print(f"[eps calibration] E[p_pos]={p_pos:.4f} → epsilon={eps:.4f} "
                  f"(effk_mean={ek_mean:.2f} < 3.0), falling back to effk={effk_target:.1f}")
            return self.calibrate_effk(effk_target)

        self.epsilon = eps
        print(f"[eps calibration] E[p_pos]={p_pos:.4f} → epsilon={eps:.4f} "
              f"(E[q_pos]={mean_q_pos(eps):.4f}, effk_mean={ek_mean:.2f}, effk_med={ek_med:.2f})")
        return eps

    def calibrate_dist_kernel(self, metric, n_samples=10000):
        """Auto-calibrate dist_min_dist, dist_spread, and (a, b) from positive-edge distances.

        Encodes all cells, computes pairwise distances on positive edges using `metric`,
        then derives:
          dist_min_dist  ← 5th percentile of distances
          dist_spread    ← (90th − 5th percentile) / ln(10)  [so q drops to ~0.1 at 90th pct]
          _dist_a, _dist_b ← find_ab_params(dist_spread, dist_min_dist)

        Sets self.dist_min_dist, self.dist_spread, self._dist_a, self._dist_b in-place.
        """
        import scipy.sparse as sp

        aff = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        coo = sp.coo_matrix(aff)

        n = min(n_samples, len(coo.data))
        idx = np.random.choice(len(coo.data), n, replace=False)
        heads = torch.tensor(coo.row[idx], dtype=torch.long, device=self.device)
        tails = torch.tensor(coo.col[idx], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            z_all = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            z_all = z_all.to(self.device)
            logits_all = self.model.prototypes(z_all)
            soft_all = torch.softmax(logits_all / self.epsilon, dim=-1)
            dists = self._proto_sim(soft_all[heads], soft_all[tails], metric=metric).cpu().numpy()

        p5  = float(np.percentile(dists, 5))
        p90 = float(np.percentile(dists, 90))
        dist_min_dist = max(p5, 1e-4)
        dist_spread   = max((p90 - p5) / np.log(10), 1e-4)

        print(f"[dist kernel calibration] metric={metric}, n={n}")
        print(f"   dist range: p5={p5:.4f}, p50={np.percentile(dists,50):.4f}, p90={p90:.4f}, p99={np.percentile(dists,99):.4f}")
        print(f"   → dist_min_dist={dist_min_dist:.4f}, dist_spread={dist_spread:.4f}")

        try:
            a, b = find_ab_params(dist_spread, dist_min_dist)
            if a <= 0 or b <= 0:
                raise ValueError(f"curve_fit returned non-positive params: a={a:.4f}, b={b:.4f}")
            self._dist_a, self._dist_b = a, b
            print(f"   → a={a:.4f}, b={b:.4f}")
        except Exception as e:
            # Fallback: use manual values from config if curve_fit fails
            _dmd = getattr(self, 'dist_min_dist', 0.1)
            _dsp = getattr(self, 'dist_spread', 0.3)
            print(f"   ⚠ find_ab_params failed ({e}), falling back to manual dist_min_dist={_dmd}, dist_spread={_dsp}")
            self._dist_a, self._dist_b = find_ab_params(_dsp, _dmd)
            print(f"   → a={self._dist_a:.4f}, b={self._dist_b:.4f}")
            return

        self.dist_min_dist = dist_min_dist
        self.dist_spread   = dist_spread

    def calibrate_effk(self, target_effk=5.0, agg=None):
        """Find epsilon so that mean/median effective-k over all cells = target_effk.

        effk_i = 1 / sum_k(s_ik^2) — effective number of protos per cell.
        Higher epsilon → softer assignments → higher effk.
        Sets self.epsilon in-place and returns the found value.
        agg: 'mean' or 'median' (default from umap_proto_effk_agg config, fallback 'mean').
        """
        if agg is None:
            agg = getattr(self, 'umap_proto_effk_agg', 'mean')
        if agg not in ('mean', 'median'):
            raise ValueError(f"effk agg must be 'mean' or 'median', got {agg!r}")

        self.model.eval()
        with torch.no_grad():
            z_all = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            z_all = z_all.to(self.device)
            logits_all = self.model.prototypes(z_all)

        def effk_stat(eps):
            with torch.no_grad():
                soft = torch.softmax(logits_all / eps, dim=-1)
                ek = 1.0 / (soft ** 2).sum(dim=-1)
                return ek.mean().item() if agg == 'mean' else ek.median().item()

        lo, hi = 1e-6, 100.0
        for _ in range(60):
            eps = (lo + hi) / 2
            ek = effk_stat(eps)
            if ek < target_effk:
                lo = eps   # effk too low → softer → larger eps
            else:
                hi = eps   # effk too high → sharper → smaller eps

        self.epsilon = eps
        print(f"[effk calibration] target_effk={target_effk:.1f} → epsilon={eps:.4f} ({agg}_effk={effk_stat(eps):.2f})")
        return eps

    @torch.no_grad()
    def _init_proto_usage_nk_batch(self):
        """Compute per-batch proto usage from full dataset. Called at the start of each epoch.

        Encodes all cells, computes n_k_b = sum of soft assignments for each batch b.
        Stored in self._proto_usage_nk_batch as the EMA starting point for the epoch.
        """
        self.model.eval()
        z_all = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
        z_all = z_all.to(self.device)
        s_all = F.softmax(self.model.prototypes(z_all) / self.epsilon, dim=1)  # (N, K)

        cell_ds_all = self._umap_state['cell_ds']  # (N,) on CPU
        self._proto_usage_nk_batch = {}
        for b in cell_ds_all.unique():
            b_int = b.item()
            mask = (cell_ds_all == b).to(self.device)
            self._proto_usage_nk_batch[b_int] = s_all[mask].sum(dim=0).clamp(min=1e-6).cpu()
        self.model.train()

    def build_model(self):
        self.model = self.get_model()
        self.model = self.model.cuda()
        # logger.info(self.model)
        logger.info(
            f"=======>Building model done. max value for adata fed to scpoli_wrapper: {self.model.scpoli_wrapper.adata.X.max()}"
        )

    def get_model(self):
        # if self.model_version == 1:
        # return model
        # else:
        kwargs = {
            "latent_dim": self.latent_dims,
            "nmb_prototypes": self.num_prototypes,
            "adata": self.train_.adata,
            "multi_layer_proto": self.multi_layer_protos,
            "np2": self.num_prototypes,
            "recon_loss": self.recon_loss,
            "batch_key": self.train_.batch_key,
            "l2norm": self.l2norm,
            "assignment_metric": self.assignment_metric,
            "recon_v": self.recon_v,
        }

        if self.model_type == "gm" or self.model_type == "normal":
            return scProtoGMVAE(
                temperature=self.epsilon,
                beta=self.beta,
                recon_version=self.version,
                kl_type=self.model_type,
                **kwargs,
            )
        if self.model_type == "vqvae":
            return scProtoVQVAE(
                temperature=self.temperature,
                beta=self.beta,
                recon_update_target=self.recon_update_target,
                **kwargs,
            )
        if self.model_type == "hybrid":
            return scProtoHybrid(temperature=self.temperature, beta=self.beta, **kwargs)
        else:
            return SwAVModel(**kwargs)

    def build_optimizer(self):
        opt_type = self.opt
        print(opt_type)
        # ---- Choose optimizer ----
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.wd,
            )
            # LARC only makes sense for SGD
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        elif opt_type == "adam":
            # optimizer = torch.optim.Adam(
            #     self.model.parameters(),
            #     lr=self.base_lr,
            #     weight_decay=self.wd,
            # )

            # ----- optimizer -----
            params_embedding = []
            params = []
            for name, p in self.model.scpoli_cvae.named_parameters():
                if p.requires_grad:
                    if "embedding" in name:
                        params_embedding.append(p)
                    else:
                        params.append(p)

            optimizer = torch.optim.Adam(
                [
                    {"params": params_embedding, "weight_decay": 0},
                    {"params": params},
                ],
                lr=self.base_lr,  # 1e-3,
                eps=0.01,
                weight_decay=self.wd,  # 0.04,
            )

        elif opt_type == "wadam":
            optimizer = torch.optim.AdamW(  # usually better for weighted Adam
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.wd,
            )

        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        self.optimizer = optimizer

        # ---- LR schedule (warmup -> cosine) ----
        total_iters = len(self.train_loader)

        warmup_iters = total_iters * self.warmup_epochs
        warmup_lr = np.linspace(self.start_warmup, self.base_lr, warmup_iters)

        cosine_iters = total_iters * (self.pretraining_epochs - self.warmup_epochs)
        t = np.arange(cosine_iters)

        cosine_lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
            1 + np.cos(np.pi * t / cosine_iters)
        )

        self.lr_schedule = np.concatenate((warmup_lr, cosine_lr))

        logger.info(f"Optimizer '{opt_type}' built successfully.")

    def get_checkpoint_file(self):
        if self.finetuning:
            checkpoint_file = "finetuned-checkpoint.pth.tar"
        elif self.training_type == "semi_supervised":
            checkpoint_file = "semi-pretrain-checkpoint.pth.tar"
        else:
            checkpoint_file = "checkpoint.pth.tar"
        return checkpoint_file

    def save_checkpoint(self, epoch):
        if self.wandb_sweep == 1:
            print("not saving checkpoint", self.debug, self.wandb_sweep)
            return
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # Note: fp16/amp state saving removed (was using apex.amp which is not available)
        # If you need mixed precision, use torch.cuda.amp.GradScaler instead

        checkpoint_file = self.get_checkpoint_file()
        torch.save(save_dict, os.path.join(self.dump_path, checkpoint_file))
        # Skip epoch-specific copies in debug mode to save disk space
        if not self.debug and (epoch % self.checkpoint_freq == 0 or epoch == self.pretraining_epochs - 1):
            shutil.copyfile(
                os.path.join(self.dump_path, checkpoint_file),
                os.path.join(self.dump_path, f"ckp-{epoch}.pth"),
            )

    def log_wandb_loss(self, scores, epoch):
        log_dict = scores
        log_dict["epoch"] = epoch
        if not self.debug:
            wandb.log(log_dict)
        # else:
        #     for k in log_dict:
        #         if k not in self.log_hist:
        #             self.log_hist[k] = []
        #         self.log_hist[k].append(log_dict[k])
        #     print(log_dict)

    # -- UMAP edge training: setup / epoch / public API ------------------

    def _setup_umap_edges(self, epochs: int = None, init_prototypes: bool = True, skip_calibration: bool = False):
        """Build and cache all objects needed for edge-centric UMAP training."""
        epochs = epochs or getattr(self, 'umap_edge_epochs', 200)

        affinity = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff

        adata = self.train_ds.adata
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)

        min_dist = getattr(self, 'umap_min_dist', 0.5)
        spread = getattr(self, 'umap_spread', 1.0)
        neg_rate = getattr(self, 'umap_neg_rate', 5)
        umap_similarity = getattr(self, 'umap_similarity', 'embedding')

        if init_prototypes:
            self.init_prototypes()

        if self.calibrate_eps and umap_similarity == 'proto' and not skip_calibration:
            self.calibrate_epsilon()

        # Build cell -> ds_id mapping
        if self.condition_key is not None:
            batch_labels = self.train_ds.adata.obs[self.condition_key]
            unique_batches = list(batch_labels.unique())
            batch_to_id = {b: i for i, b in enumerate(unique_batches)}
            cell_ds = np.array([batch_to_id[b] for b in batch_labels], dtype=np.int64)
        else:
            cell_ds = np.zeros(len(self.train_ds.adata), dtype=np.int64)

        # Single EdgeDataset over all edges; neg sampling stays within head's dataset.
        # WeightedRandomSampler picks edges proportional to affinity weight each epoch.
        from torch.utils.data import WeightedRandomSampler
        edge_dataset = EdgeDataset(affinity, n_epochs=epochs, negative_sample_rate=neg_rate, cell_ds=cell_ds)
        steps_per_epoch = getattr(self, 'umap_steps_per_epoch', None)
        n_samples = min(len(edge_dataset), steps_per_epoch * self.batch_size) if steps_per_epoch else len(edge_dataset)
        if steps_per_epoch:
            print(f"   umap_steps_per_epoch={steps_per_epoch} → {n_samples} edges/epoch (of {len(edge_dataset)} total)")
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(edge_dataset.weights).float(),
            num_samples=n_samples,
            replacement=True,
        )
        loader = DataLoader(
            edge_dataset, batch_size=self.batch_size, sampler=sampler,
            collate_fn=edge_collate_fn, num_workers=0, drop_last=False,
        )

        _dist_metric = getattr(self, 'umap_proto_metric', 'cosine')
        if umap_similarity != 'proto' or _dist_metric not in ('jsd', 'bhatt_dist', 'hellinger', 'idot'):
            loss_fn = ParametricUMAPLoss(min_dist=min_dist, spread=spread, negative_sample_rate=neg_rate)
        else:
            loss_fn = None
        if _dist_metric in ('jsd', 'bhatt_dist', 'hellinger', 'idot'):
            if self.calibrate_eps and umap_similarity == 'proto' and not skip_calibration:
                self.calibrate_dist_kernel(_dist_metric)
            elif skip_calibration and hasattr(self, '_dist_a'):
                print(f"   {_dist_metric} kernel: reusing calibrated a={self._dist_a:.4f}, b={self._dist_b:.4f}")
            else:
                _dmd = getattr(self, 'dist_min_dist', 0.1)
                _dsp = getattr(self, 'dist_spread', 0.3)
                self._dist_a, self._dist_b = find_ab_params(_dsp, _dmd)
                print(f"   {_dist_metric} kernel: dist_min_dist={_dmd}, dist_spread={_dsp} -> a={self._dist_a:.4f}, b={self._dist_b:.4f}")

        # Precompute per-cell degrees and 2m for degree-weighted positive loss.
        # w_ij = A_ij / (k_i * k_j / 2m) — upweights pairs more connected than expected.
        _A = sp.csr_matrix(affinity)
        _deg = np.array(_A.sum(axis=1)).ravel()          # k_i for each cell
        _two_m = float(_deg.sum())                        # 2m = total edge weight
        self._cell_degree = torch.tensor(_deg, dtype=torch.float32)   # (N,)
        self._two_m = _two_m

        proto_decoupled = getattr(self, 'proto_decoupled', False) and umap_similarity == 'proto'
        if umap_similarity == 'proto' and not proto_decoupled:
            params = list(self.model.scpoli_cvae.parameters()) + list(self.model.prototypes.parameters())
        else:
            params = list(self.model.scpoli_cvae.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.base_lr)

        self._umap_state = {
            'X': X.to(self.device),
            'edge_dataset': edge_dataset,
            'loss_fn': loss_fn,
            'loader': loader,
            'optimizer': optimizer,
            'epoch': 0,
            'cell_ds': torch.tensor(cell_ds, dtype=torch.long),
        }
        self._c_k_ema = None  # reset EMA for proto_usage_mode='max'
        self._gmm_n_total_epochs = epochs
        if proto_decoupled:
            self._init_decoupled_proto_state()
            print(f"   proto_decoupled=True: prototypes updated via EMA of cluster means (excluded from optimizer)")

        print(f"Starting edge-centric UMAP training (similarity={umap_similarity})")
        print(f"   min_dist={min_dist}, spread={spread}, neg_rate={neg_rate}")
        print(f"   lambda_umap={getattr(self, 'lambda_umap', 1.0)}, "
              f"lambda_recon={self.lambda_recon}, lambda_kl={self.lambda_kl}, "
              f"lambda_proto_recon={self.lambda_proto_recon}, lambda_r1r2={self.lambda_r1r2}")

    # ------------------------------------------------------------------
    # Decoupled prototype learning (online GMM EM)
    # ------------------------------------------------------------------

    def _init_decoupled_proto_state(self):
        """Seed running accumulators from current prototype positions.
        Proto weights are NOT changed — this just gives the EMA updater a warm start."""
        K = self.nmb_prototypes
        protos = self.model.get_prototypes().detach().cpu().float()  # (K, D)
        self._proto_running_count  = torch.ones(K)           # S_pi: effective cell count per proto
        self._proto_running_sum    = protos.clone()           # S_mu: weighted sum of embeddings
        self._proto_running_sq_sum = torch.ones_like(protos) # S_var: weighted sum of z² (for resurrect noise)

    def _get_proto_update_eta(self):
        """Return eta for the current epoch (linear schedule: gmm_eta → gmm_eta_end)."""
        epoch   = self._umap_state.get('epoch', 0)
        n_total = getattr(self, '_gmm_n_total_epochs', max(epoch, 1))
        frac    = epoch / max(n_total - 1, 1)
        eta_start = getattr(self, 'gmm_eta', 0.1)
        eta_end   = getattr(self, 'gmm_eta_end', 0.5)
        return eta_start + (eta_end - eta_start) * frac

    @torch.no_grad()
    def _update_protos_ema(self, z_det, soft_assign_det, eta):
        """Update prototype positions as EMA of weighted cluster means.

        Uses the same soft_assign already computed in the forward pass.
        Prototypes are set directly — no gradient involved.

        Args:
            z_det:           (N, D) detached embeddings from this batch
            soft_assign_det: (N, K) detached soft assignments from this batch
            eta:             float, base forgetting factor (per-proto rate adapts based on usage)
        """
        z_cpu  = z_det.cpu().float()
        s_cpu  = soft_assign_det.cpu().float()
        N      = z_cpu.shape[0]

        # --- batch statistics ---
        count_b  = s_cpu.sum(0)               # (K,) effective cells assigned per proto
        sum_b    = s_cpu.T @ z_cpu            # (K, D) weighted sum of embeddings
        sq_sum_b = s_cpu.T @ (z_cpu ** 2)    # (K, D) weighted sum of z² (for variance)

        # --- per-proto update rate: eta_k = eta ^ (usage_k / N) ---
        # barely used proto → eta_k ≈ 1 → running stats frozen (no forgetting)
        # heavily used proto → eta_k ≈ eta → normal EMA update
        usage_frac = (count_b / N).clamp(0.0, 1.0)   # (K,)
        eta_k      = eta ** usage_frac                 # (K,)

        # --- update running accumulators ---
        self._proto_running_count  = eta_k          * self._proto_running_count  + (1 - eta_k)          * count_b
        self._proto_running_sum    = eta_k.unsqueeze(1) * self._proto_running_sum    + (1 - eta_k).unsqueeze(1) * sum_b
        self._proto_running_sq_sum = eta_k.unsqueeze(1) * self._proto_running_sq_sum + (1 - eta_k).unsqueeze(1) * sq_sum_b

        # --- new prototype position = weighted mean of assigned embeddings ---
        count_safe = self._proto_running_count.clamp(min=1e-8)
        new_mu     = self._proto_running_sum / count_safe.unsqueeze(1)   # (K, D)

        # variance per proto (needed only for resurrect noise scale)
        new_var = (self._proto_running_sq_sum / count_safe.unsqueeze(1) - new_mu ** 2).clamp(min=1e-6)

        new_mu_out = new_mu.clone()

        # --- resurrect (opt-in): split dominant proto into the most unused proto ---
        if getattr(self, 'gmm_resurrect', False):
            proto_weights = self._proto_running_count / self._proto_running_count.sum().clamp(min=1e-8)
            thresh        = getattr(self, 'gmm_resurrect_thresh', 3.0) / self.nmb_prototypes

            for k in (proto_weights > thresh).nonzero(as_tuple=True)[0].tolist():
                j = proto_weights.argmin().item()
                if j == k:
                    continue
                # rescale dominant proto mean slightly
                norm_k = new_mu_out[k].norm().clamp(min=1e-8)
                new_mu_out[k] = new_mu_out[k] / norm_k.sqrt()
                # place dead proto near dominant one with small noise
                noise_scale = new_var[k].mean().sqrt().item() * 0.1
                new_mu_out[j] = new_mu_out[k] + torch.randn_like(new_mu_out[k]) * noise_scale
                # split the running count and sums equally between k and j
                half = self._proto_running_count[k].item() / 2.0
                self._proto_running_count[k]    = half
                self._proto_running_count[j]    = half
                self._proto_running_sum[k]      = new_mu_out[k] * half
                self._proto_running_sum[j]      = new_mu_out[j] * half
                self._proto_running_sq_sum[k]   = new_var[k] * half
                self._proto_running_sq_sum[j]   = new_var[k] * half
                # recompute weights after split
                proto_weights = self._proto_running_count / self._proto_running_count.sum().clamp(min=1e-8)

        # --- write new positions into model weights (no gradient) ---
        self.model.set_prototypes(new_mu_out.to(self.device))
        if self.l2norm == 1:
            self.model.normalize_prototypes()

    def _run_umap_epoch(self):
        """Run a single UMAP edge training epoch. Returns metrics dict."""
        s = self._umap_state
        X = s['X']
        loader = s['loader']
        loss_fn = s['loss_fn']
        optimizer = s['optimizer']

        lambda_umap = getattr(self, 'lambda_umap', 1.0)
        lambda_recon = self.lambda_recon
        lambda_kl = self.lambda_kl
        lambda_proto_recon = self.lambda_proto_recon
        lambda_r1r2 = self.lambda_r1r2
        lambda_proto_attract = getattr(self, 'lambda_proto_attract', 0.0)
        lambda_proto_usage = getattr(self, 'lambda_proto_usage', 0.0)
        proto_usage_mode   = getattr(self, 'proto_usage_mode', 'nk')
        lambda_nassoc = getattr(self, 'lambda_nassoc', 0.0)
        nassoc_alpha = getattr(self, 'nassoc_alpha', 1.0)
        nassoc_agg = getattr(self, 'nassoc_agg', 'mean')
        nassoc_diag_loss = getattr(self, 'nassoc_diag_loss', 'mse')
        nassoc_diag = getattr(self, 'nassoc_diag', True)
        use_proto_sim = getattr(self, 'umap_similarity', 'embedding') == 'proto'
        proto_metric = getattr(self, 'umap_proto_metric', 'dotp')
        proto_decoupled = getattr(self, 'proto_decoupled', False) and use_proto_sim
        _proto_eta = self._get_proto_update_eta() if proto_decoupled else 0.1

        self.model.train()
        total_metrics = {
            'loss': 0, 'umap': 0, 'q_pos': 0, 'q_neg': 0, 'margin': 0,
            'loss_pos': 0, 'loss_neg': 0, 'recon': 0, 'kl': 0,
            'proto_recon': 0, 'r1r2': 0, 'proto_attract': 0, 'nassoc': 0, 'proto_usage': 0, 'n_unused_protos': 0,
        }
        # snapshot proto positions at epoch start to measure movement
        if proto_decoupled:
            _proto_mu_epoch_start = self.model.get_prototypes().detach().cpu().clone()
        mode5_c_min = 1.0   # worst-case min coverage seen this epoch (before correction)
        mode5_corr_max = 0.0  # worst-case max boost applied this epoch
        n_batches = 0
        _batch_nk_accum = {}  # batch_id -> [K] cumulative soft assignment sum for batch entropy
        used_proto_ids = set()

        from tqdm import tqdm
        for batch in tqdm(loader, desc='edges'):
            if self.l2norm == 1:
                with torch.no_grad():
                    self.model.normalize_prototypes()

            optimizer.zero_grad()
            proto_recon_loss = torch.tensor(0.0, device=self.device)
            r1r2_loss = torch.tensor(0.0, device=self.device)

            head        = batch['head'].to(self.device)
            tail        = batch['tail'].to(self.device)
            weights     = batch['weight'].to(self.device)
            neg_samples = batch['neg_samples'].to(self.device)
            B, neg_K = neg_samples.shape

            # Single forward pass over unique nodes in this batch
            all_idx = torch.cat([head, tail, neg_samples.flatten()])
            unique_idx = torch.unique(all_idx)
            X_batch = X[unique_idx]
            n_samples = len(X_batch)

            if hasattr(self.train_ds, 'conditions'):
                batch_cond = self.train_ds.conditions[unique_idx.cpu()].to(self.device)
            else:
                n_conds = len(self.model.scpoli_cvae.n_conditions)
                batch_cond = torch.zeros(n_samples, n_conds, dtype=torch.long, device=self.device)

            z_unique, recon_loss, kl_loss = self.model.encoder_out({'x': X_batch, 'batch': batch_cond})

            # torch.unique returns sorted output, so searchsorted gives O(log n) vectorized lookup
            def _gather(indices):
                return torch.searchsorted(unique_idx, indices)

            if use_proto_sim:
                if proto_decoupled:
                    # detached dot-product logits — no gradient to prototypes
                    logits = F.linear(z_unique, self.model.get_prototypes().detach())
                    # same soft_assign as non-decoupled, just with detached protos
                    soft_assign = F.softmax(logits / self.epsilon, dim=1)
                    soft_assign_orig = soft_assign
                else:
                    logits = self.model.prototypes(z_unique)

                # Modes 2, 5, 8: logit correction before softmax (only in non-decoupled mode)
                _usage_mode_pre = 0 if proto_decoupled else getattr(self, 'usage_norm_sim', 0)
                if _usage_mode_pre == 2:
                    cell_ds_all = self._umap_state['cell_ds']
                    batch_ids = cell_ds_all[unique_idx.cpu()]              # (n_unique,) on CPU
                    log_correction = torch.zeros_like(logits)              # (n_unique, K)
                    for b in batch_ids.unique():
                        b_int = b.item()
                        mask = (batch_ids == b).to(self.device)
                        if hasattr(self, '_proto_usage_nk_batch') and b_int in self._proto_usage_nk_batch:
                            n_k_b = self._proto_usage_nk_batch[b_int].to(self.device).clamp(min=1e-6)
                        else:
                            n_k_b = torch.ones(logits.shape[1], device=self.device)
                        log_correction[mask] = torch.log(n_k_b / n_k_b.mean())
                    soft_assign = F.softmax((logits - self.epsilon * log_correction) / self.epsilon, dim=1)
                    soft_assign_orig = F.softmax(logits / self.epsilon, dim=1).detach()  # uncorrected, for EMA
                elif _usage_mode_pre == 5:
                    # coverage-based before-softmax: log-prior correction applied directly to normalized logits
                    # no ε scaling → correction strength is independent of temperature
                    # sa_prelim reflects current step's coverage; no EMA (EMA hides dying protos)
                    with torch.no_grad():
                        sa_prelim = F.softmax(logits / self.epsilon, dim=1)
                        c_k = sa_prelim.max(dim=0).values.clamp(min=1e-8)
                        corr_clamp = getattr(self, 'usage_norm_corr_clamp', 10.0)
                        log_corr = torch.log(c_k / c_k.mean()).clamp(min=-corr_clamp, max=corr_clamp)
                        mode5_c_min = min(mode5_c_min, c_k.min().item())
                        mode5_corr_max = max(mode5_corr_max, (-log_corr).max().item())
                    soft_assign = F.softmax(logits / self.epsilon - log_corr, dim=1)
                    soft_assign_orig = soft_assign
                elif _usage_mode_pre == 8:
                    # pre-softmax double normalization:
                    # 1) shift each column so min=0, divide by max → each proto has at least one cell with value 1
                    # 2) divide by column sum (soft usage) → penalizes protos attracting many cells
                    # handles negative logits safely via the shift
                    col_min = logits.min(dim=0).values
                    L_shifted = logits - col_min
                    L_maxnorm = L_shifted / L_shifted.max(dim=0).values.clamp(min=1e-8)
                    usage_k = L_maxnorm.sum(dim=0).clamp(min=1e-8)
                    L_normed = L_maxnorm / usage_k
                    soft_assign = F.softmax(L_normed / self.epsilon, dim=1)
                    soft_assign_orig = soft_assign
                else:
                    soft_assign = F.softmax(logits / self.epsilon, dim=1)
                    soft_assign_orig = soft_assign

                with torch.no_grad():
                    sa_effk = (1.0 / (soft_assign * soft_assign).sum(dim=1))
                    total_metrics['effk'] = total_metrics.get('effk', 0) + sa_effk.median().item()
                    used_proto_ids.update(soft_assign.argmax(dim=1).unique().cpu().tolist())
                    # accumulate per-batch soft usage for batch entropy
                    _bids = self._umap_state['cell_ds'][unique_idx.cpu()]
                    for _b in _bids.unique():
                        _b_int = _b.item()
                        _nk_b = soft_assign[(_bids == _b).to(self.device)].sum(dim=0).cpu()
                        if _b_int in _batch_nk_accum:
                            _batch_nk_accum[_b_int] += _nk_b
                        else:
                            _batch_nk_accum[_b_int] = _nk_b

                s_head = soft_assign[_gather(head)]
                s_tail = soft_assign[_gather(tail)]
                s_neg = soft_assign[_gather(neg_samples.flatten())].view(B, neg_K, -1)
                _eps = 1e-4

                usage_mode = getattr(self, 'usage_norm_sim', 0)
                t_head = t_tail = t_neg = None  # Sinkhorn targets; set in mode 9

                def _reweight(s_h, s_t, s_n, w, renorm=False):
                    h = s_h / w
                    t = s_t / w
                    n = s_n / w
                    if renorm:
                        h = h / h.sum(dim=1, keepdim=True).detach()
                        t = t / t.sum(dim=1, keepdim=True).detach()
                        n = n / n.sum(dim=2, keepdim=True).detach()
                    return h, t, n

                if usage_mode == 1:
                    # post-softmax, mini-batch global: w = n_k / mean(n_k)
                    n_k = soft_assign.sum(dim=0).clamp(min=1e-6)
                    w = n_k / n_k.mean()
                    s_head_n, s_tail_n, s_neg_n = _reweight(s_head, s_tail, s_neg, w)

                elif usage_mode == 3:
                    # post-softmax, batch-balanced: average n_k across source batches equally
                    cell_ds_all = self._umap_state['cell_ds']
                    batch_ids = cell_ds_all[unique_idx.cpu()]
                    n_k_per_batch = [
                        soft_assign[(batch_ids == b).to(self.device)].sum(dim=0)
                        for b in batch_ids.unique()
                    ]
                    n_k = torch.stack(n_k_per_batch).mean(dim=0).clamp(min=1e-6)
                    w = n_k / n_k.mean()
                    s_head_n, s_tail_n, s_neg_n = _reweight(s_head, s_tail, s_neg, w)

                elif usage_mode == 4:
                    # post-softmax, coverage-based: w = c_k / mean(c_k), no renorm, gradient flows through w
                    c_k = soft_assign.max(dim=0).values
                    c_k = c_k.clamp(min=0.1 * c_k.mean().clamp(min=1e-6))
                    w = c_k / c_k.mean()
                    s_head_n, s_tail_n, s_neg_n = _reweight(s_head, s_tail, s_neg, w)

                elif usage_mode == 6:
                    # post-softmax, coverage-based: w = c_k / mean(c_k), renorm with detached sum, gradient flows through w
                    c_k = soft_assign.max(dim=0).values
                    c_k = c_k.clamp(min=0.1 * c_k.mean().clamp(min=1e-6))
                    w = c_k / c_k.mean()
                    s_head_n, s_tail_n, s_neg_n = _reweight(s_head, s_tail, s_neg, w, renorm=True)

                elif usage_mode == 7:
                    # post-softmax, robust coverage: mean of top-50% assignments per proto
                    # replaces max in mode 6 with mean of above-median cells — parameter-free
                    med_k = soft_assign.median(dim=0).values                       # (K,)
                    mask = soft_assign > med_k.unsqueeze(0)                        # (N, K)
                    c_k = (soft_assign * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                    c_k = c_k.clamp(min=0.1 * c_k.mean().clamp(min=1e-6))
                    w = c_k / c_k.mean()
                    s_head_n, s_tail_n, s_neg_n = _reweight(s_head, s_tail, s_neg, w, renorm=True)

                elif usage_mode == 9:
                    # per-batch Sinkhorn: t is balanced target (no grad), s carries gradients
                    # q = 0.5*(dot(s_i, t_j) + dot(s_j, t_i)) — asymmetric, grad only through s
                    cell_ds_all = self._umap_state['cell_ds']
                    batch_ids = cell_ds_all[unique_idx.cpu()].to(self.device)
                    n_iters = getattr(self, 'sinkhorn_iters', 3)
                    t_all = _per_batch_sinkhorn(logits / self.epsilon, batch_ids, n_iters)
                    t_head = t_all[_gather(head)]
                    t_tail = t_all[_gather(tail)]
                    t_neg  = t_all[_gather(neg_samples.flatten())].view(B, neg_K, -1)
                    s_head_n, s_tail_n, s_neg_n = s_head, s_tail, s_neg

                else:
                    # mode 0: no normalization; modes 2/5: correction already applied pre-softmax
                    s_head_n, s_tail_n, s_neg_n = s_head, s_tail, s_neg

                if t_head is not None:
                    # mode 9: asymmetric Sinkhorn — grad through s, not t
                    q_pos = 0.5 * (
                        (s_head_n * t_tail).sum(dim=-1) +
                        (s_tail_n * t_head).sum(dim=-1)
                    ).clamp(_eps, 1.0 - _eps)
                    q_neg = (s_head_n.unsqueeze(1) * t_neg).sum(dim=-1).clamp(_eps, 1.0 - _eps)
                elif proto_metric in ('jsd', 'bhatt_dist', 'hellinger', 'idot'):
                    d_pos = self._proto_sim(s_head_n, s_tail_n, proto_metric)
                    d_neg = self._proto_sim_neg(s_head_n, s_neg_n, proto_metric)
                    q_pos = (1.0 + self._dist_a * d_pos.pow(self._dist_b)).reciprocal().clamp(_eps, 1.0 - _eps)
                    q_neg = (1.0 + self._dist_a * d_neg.pow(self._dist_b)).reciprocal().clamp(_eps, 1.0 - _eps)
                else:
                    q_pos = self._proto_sim(s_head_n, s_tail_n, proto_metric).clamp(_eps, 1.0 - _eps)
                    q_neg = self._proto_sim_neg(s_head_n, s_neg_n, proto_metric).clamp(_eps, 1.0 - _eps)
                if getattr(self, 'degree_norm_loss', 0):
                    # Degree-normalized positive loss: downweight high-degree edges by 1/sqrt(d_i*d_j)
                    # so each cell contributes equally regardless of neighborhood size.
                    d = self._cell_degree.to(self.device)
                    d_norm = d / d.median()
                    deg_norm = 1.0 / torch.sqrt((d_norm[head] * d_norm[tail]).clamp(min=1e-8))
                    loss_pos = -(deg_norm * torch.log(q_pos)).mean()
                elif getattr(self, 'lambda_degree_weight', 0):
                    # Degree-weighted positive loss: w_ij = A_ij / (k_i * k_j / 2m)
                    # Upweights pairs more connected than expected by chance,
                    # directly aligning the loss with what modularity measures.
                    k = self._cell_degree.to(self.device)
                    k_head = k[head]
                    k_tail = k[tail]
                    null = (k_head * k_tail) / self._two_m
                    deg_w = weights / null.clamp(min=1e-8)   # A_ij / (k_i*k_j/2m)
                    deg_w = deg_w / (deg_w.mean() + 1e-8)   # normalise to mean=1
                    loss_pos = -(deg_w * torch.log(q_pos)).mean()
                else:
                    loss_pos = -torch.log(q_pos).mean()
                loss_neg = -torch.log(1.0 - q_neg).sum(dim=1).mean()
                umap_loss = loss_pos + loss_neg
                metrics = {
                    'q_pos': q_pos.mean().item(), 'q_neg': q_neg.mean().item(),
                    'margin': (q_pos.mean() - q_neg.mean()).item(),
                    'loss_pos': loss_pos.item(), 'loss_neg': loss_neg.item(),
                }
            else:
                z_head = z_unique[_gather(head)]
                z_tail = z_unique[_gather(tail)]
                z_neg = z_unique[_gather(neg_samples.flatten())].view(B, neg_K, -1)
                umap_loss, metrics = loss_fn(z_head, z_tail, z_neg, weights)

            # Auxiliary losses computed once on the shared batch (not per-ds)
            if lambda_proto_recon > 0:
                if proto_decoupled:
                    # scores carry gradient to encoder; protos are detached (updated by GMM)
                    scores = soft_assign
                    protos = self.model.get_prototypes().detach()
                else:
                    scores = soft_assign if use_proto_sim else F.softmax(self.model.prototypes(z_unique) / self.epsilon, dim=1)
                    protos = self.model.get_prototypes()
                K = protos.size(0)
                n_genes = X_batch.size(1)
                unique_conds, inverse_idx = torch.unique(batch_cond, dim=0, return_inverse=True)
                n_unique_conds = unique_conds.size(0)
                proto_expanded = protos.unsqueeze(0).expand(n_unique_conds, -1, -1).reshape(-1, protos.size(1))
                cond_expanded = unique_conds.unsqueeze(1).expand(-1, K, -1).reshape(-1, unique_conds.size(1))
                decoded_all = self.model.decode(proto_expanded, cond_expanded).view(n_unique_conds, K, n_genes)
                recon_x_agg = torch.zeros(n_samples, n_genes, device=self.device)
                for c in range(n_unique_conds):
                    cmask = inverse_idx == c
                    if cmask.any():
                        if proto_decoupled:
                            recon_x_agg[cmask] = scores[cmask] @ decoded_all[c]
                        else:
                            recon_x_agg[cmask] = scores[cmask].detach() @ decoded_all[c]
                proto_recon_loss = F.mse_loss(recon_x_agg, X_batch, reduction="none").sum(dim=-1).mean()

            if lambda_r1r2 > 0:
                r1r2_loss = self.calc_r1r2_loss(z_unique)

            proto_usage_loss = torch.tensor(0.0, device=self.device)
            if lambda_proto_usage > 0 and use_proto_sim:
                if proto_usage_mode == 'max':
                    c_k = soft_assign.max(dim=0).values                   # (K,)
                    proto_usage_loss = -torch.log(c_k.clamp(min=1e-8)).mean()
                elif proto_usage_mode == 'ema':
                    c_k = soft_assign.max(dim=0).values                   # (K,)
                    ema_alpha = getattr(self, 'usage_nk_alpha', 0.9)
                    if self._c_k_ema is None:
                        self._c_k_ema = c_k.detach().cpu()
                        c_k_ema = c_k
                    else:
                        c_k_ema = ema_alpha * self._c_k_ema.to(self.device) + (1 - ema_alpha) * c_k
                        self._c_k_ema = c_k_ema.detach().cpu()
                    proto_usage_loss = -torch.log(c_k_ema.clamp(min=1e-8)).mean() / (1 - ema_alpha)
                else:                                                      # 'nk' (default)
                    n_k = soft_assign.sum(dim=0)                          # (K,)
                    proto_usage_loss = torch.log(1.0 + 1.0 / n_k.clamp(min=1e-8)).mean()

            loss = lambda_umap * umap_loss
            if lambda_recon > 0:
                loss = loss + lambda_recon * recon_loss
            if lambda_kl > 0:
                loss = loss + lambda_kl * kl_loss
            if lambda_proto_recon > 0:
                loss = loss + lambda_proto_recon * proto_recon_loss
            if lambda_r1r2 > 0:
                loss = loss + lambda_r1r2 * r1r2_loss
            if lambda_proto_usage > 0:
                loss = loss + lambda_proto_usage * proto_usage_loss

            nassoc_loss = torch.tensor(0.0, device=self.device)
            if lambda_nassoc > 0 and use_proto_sim:
                # Per-batch nassoc: compute M separately within each batch, then
                # average equally across batches. This avoids the bias where
                # batch-specific protos get free zero off-diagonals (because
                # cross-batch edges are absent in aff_raw). Within each batch
                # the graph is cell-type-structured, so nassoc correctly pushes
                # toward cell-type-pure protos without encoding batch identity.
                S = soft_assign                                          # [n_unique, K]
                K_na = S.shape[1]
                head_local = _gather(head)
                tail_local = _gather(tail)
                eps_na = 1e-8
                off_mask = ~torch.eye(K_na, dtype=torch.bool, device=self.device)

                cell_ds_all = self._umap_state['cell_ds']               # (N_total,) CPU
                # batch id for each unique cell in this mini-batch
                cell_batch = cell_ds_all[unique_idx.cpu()].to(self.device)  # (n_unique,)
                unique_batches = cell_batch.unique()

                M_list = []
                for b in unique_batches:
                    # include all edges whose head is from batch b (tail may be cross-batch)
                    b_head = cell_batch[head_local]
                    edge_mask = (b_head == b)
                    if edge_mask.sum() == 0:
                        continue

                    h_b = head_local[edge_mask]
                    t_b = tail_local[edge_mask]
                    w_b = weights[edge_mask]

                    S_h = S[h_b]                                        # [E_b, K]
                    S_t = S[t_b]                                        # [E_b, K]
                    A_part = S_h.T @ (w_b.unsqueeze(1) * S_t)          # [K, K]
                    A_na = A_part + A_part.T

                    d_na = torch.zeros(n_samples, device=self.device)
                    d_na.scatter_add_(0, h_b, w_b)
                    d_na.scatter_add_(0, t_b, w_b)
                    vol = S.T @ d_na                                    # [K]

                    norm = torch.sqrt((vol[:, None] + eps_na) * (vol[None, :] + eps_na))
                    M_list.append(A_na / norm)                          # [K, K]

                if M_list:
                    if nassoc_agg == 'pbch':
                        # compute nassoc loss independently per batch then average:
                        # nassoc is NOT responsible for cross-batch consistency (that's CVAE+UMAP's job)
                        # within each batch: push diagonal toward 1 (purity) and off-diagonal toward 0 (no redundancy)
                        # use mse for diagonal to avoid blowup when a proto is absent in some batches
                        def _diag_loss(d):
                            if nassoc_diag_loss == 'nll':
                                return -torch.log(d.clamp(min=1e-8)).mean()
                            elif nassoc_diag_loss == 'nll2':
                                return -torch.log((1 - (d - 1) ** 2).clamp(min=1e-8)).mean()
                            else:
                                return ((d - 1) ** 2).mean()
                        batch_losses = [
                            ((_diag_loss(M_b.diag()) if nassoc_diag else 0.0)
                             + nassoc_alpha * (M_b[off_mask] ** 2).mean())
                            for M_b in M_list
                        ]
                        nassoc_loss = torch.stack(batch_losses).mean()
                    else:
                        if nassoc_agg == 'max':
                            # element-wise max: diagonal uses best-case batch (weakest req),
                            # off-diagonal uses worst-case batch (strictest req: no overlap in any batch)
                            M_agg = torch.stack(M_list, dim=0).max(dim=0).values  # [K, K]
                        else:
                            M_agg = torch.stack(M_list, dim=0).mean(dim=0)        # [K, K]
                        diag_M = torch.diag(M_agg)
                        if nassoc_diag_loss == 'nll':
                            diag_term = -torch.log(diag_M.clamp(min=1e-8)).mean()
                        elif nassoc_diag_loss == 'nll2':
                            # per-batch: f(d)=1-(d-1)^2 converts MSE-distance to similarity score in [0,1]
                            # avg across batches then -log: rewards protos used moderately across batches
                            # over single-batch perfect usage; no blowup for individual dead batches
                            diag_per_batch = torch.stack([M.diag() for M in M_list])  # [B, K]
                            avg_f = (1 - (diag_per_batch - 1) ** 2).mean(dim=0)      # [K]
                            diag_term = -torch.log(avg_f.clamp(min=1e-8)).mean()
                        else:
                            diag_term = ((diag_M - 1) ** 2).mean()
                        offdiag_term = nassoc_alpha * (M_agg[off_mask] ** 2).mean()
                        nassoc_loss = (diag_term if nassoc_diag else 0.0) + offdiag_term
                    loss = loss + lambda_nassoc * nassoc_loss

            proto_attract_loss = torch.tensor(0.0, device=self.device)
            if lambda_proto_attract > 0 and use_proto_sim:
                with torch.no_grad():
                    r_i = soft_assign.max(dim=1).values          # (N,) cell rep quality
                    lost_weight = (1.0 - r_i)                    # (N,) high = poorly represented
                    n_k = soft_assign.sum(dim=0).clamp(min=1e-6) # (K,)
                    dead_weight = 1.0 / n_k
                    dead_weight = dead_weight / dead_weight.mean()
                attraction = soft_assign * lost_weight.unsqueeze(1)  # (N, K)
                proto_attract_loss = -(dead_weight * attraction.max(dim=0).values).mean()
                loss = loss + lambda_proto_attract * proto_attract_loss

            loss.backward()
            optimizer.step()

            # Decoupled proto update: online GMM M-step using same soft_assign from forward pass
            if proto_decoupled:
                self._update_protos_ema(z_unique.detach(), soft_assign.detach(), _proto_eta)

            # EMA update for usage_norm_sim=2: update after weights change
            if getattr(self, 'usage_norm_sim', 0) == 2 and use_proto_sim:
                ema_alpha = getattr(self, 'usage_nk_alpha', 0.999)
                with torch.no_grad():
                    for b in batch_ids.unique():
                        b_int = b.item()
                        mask = (batch_ids == b).to(self.device)
                        n_k_b_new = soft_assign_orig[mask].sum(dim=0).clamp(min=1e-6).cpu()
                        if not hasattr(self, '_proto_usage_nk_batch'):
                            self._proto_usage_nk_batch = {}
                        if b_int in self._proto_usage_nk_batch:
                            self._proto_usage_nk_batch[b_int] = ema_alpha * self._proto_usage_nk_batch[b_int] + (1 - ema_alpha) * n_k_b_new
                        else:
                            self._proto_usage_nk_batch[b_int] = n_k_b_new

            total_metrics['loss'] += loss.item()
            total_metrics['umap'] += umap_loss.item()
            total_metrics['recon'] += recon_loss.item()
            total_metrics['kl'] += kl_loss.item()
            total_metrics['r1r2'] += r1r2_loss.item()
            total_metrics['proto_recon'] += proto_recon_loss.item()
            total_metrics['proto_attract'] += proto_attract_loss.item()
            total_metrics['nassoc'] += nassoc_loss.item()
            total_metrics['proto_usage'] += proto_usage_loss.item()
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            n_batches += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        if use_proto_sim:
            total_metrics['n_unused_protos'] = self.nmb_prototypes - len(used_proto_ids)
            if len(_batch_nk_accum) > 1:
                nk_mat = torch.stack(list(_batch_nk_accum.values()), dim=0)       # [B, K]
                nk_norm = nk_mat / nk_mat.sum(dim=0, keepdim=True).clamp(min=1e-8)
                bentropy = -(nk_norm * torch.log(nk_norm.clamp(min=1e-8))).sum(dim=0).mean()
                total_metrics['batch_entropy'] = bentropy.item()

        if getattr(self, 'usage_norm_sim', 0) == 5:
            total_metrics['mode5_c_min'] = mode5_c_min
            total_metrics['mode5_corr_max'] = mode5_corr_max

        if proto_decoupled:
            with torch.no_grad():
                proto_mu_now = self.model.get_prototypes().detach().cpu()
                # mean L2 distance each proto moved this epoch
                proto_move = (proto_mu_now - _proto_mu_epoch_start).norm(dim=1).mean().item()
                # usage spread: coefficient of variation of running counts (high = uneven = risk of collapse)
                counts = self._proto_running_count
                usage_cv = (counts.std() / counts.mean().clamp(min=1e-8)).item()
                # dead protos: weight < 10% of uniform share
                dead = (counts / counts.sum().clamp(min=1e-8) < 0.1 / self.nmb_prototypes).sum().item()
                total_metrics['proto_move']  = proto_move
                total_metrics['proto_usage_cv'] = usage_cv
                total_metrics['proto_dead']  = dead

        s['epoch'] += 1
        return total_metrics

    def _print_umap_epoch(self, epoch, total_epochs, metrics):
        """Print one line of UMAP training progress."""
        extra = ""
        if self.lambda_recon > 0:
            extra += f" | recon={metrics['recon']:.4f}"
        if self.lambda_kl > 0:
            extra += f" | kl={metrics['kl']:.4f}"
        if self.lambda_proto_recon > 0:
            extra += f" | proto_recon={metrics['proto_recon']:.4f}"
        if self.lambda_r1r2 > 0:
            extra += f" | r1r2={metrics['r1r2']:.4f}"
        if getattr(self, 'lambda_proto_attract', 0) > 0:
            extra += f" | proto_attract={metrics['proto_attract']:.4f}"
        if getattr(self, 'lambda_nassoc', 0) > 0:
            extra += f" | nassoc={metrics['nassoc']:.4f}"
        if getattr(self, 'lambda_proto_usage', 0) > 0:
            extra += f" | proto_usage={metrics['proto_usage']:.4f}"
        if getattr(self, 'usage_norm_sim', 0) == 5 and 'mode5_c_min' in metrics:
            extra += f" | c_min={metrics['mode5_c_min']:.3f} corr={metrics['mode5_corr_max']:.2f}"
        if 'proto_move' in metrics:
            extra += (f" | pmove={metrics['proto_move']:.4f}"
                      f" cv={metrics['proto_usage_cv']:.2f}"
                      f" dead={int(metrics['proto_dead'])}")
        effk_str = f" | effk={metrics['effk']:.1f}" if 'effk' in metrics else ""
        unused_str = f" | unused_proto={metrics['n_unused_protos']:.0f}" if 'n_unused_protos' in metrics else ""
        bentropy_str = f" | bentropy={metrics['batch_entropy']:.3f}" if 'batch_entropy' in metrics else ""

        knn_str = ""
        # if epoch % 5 == 0 or epoch == total_epochs:
        #     pca_acc, z_acc = self._niche_knn_acc()
        #     if pca_acc is not None:
        #         knn_str = f" | KNN: {z_acc:.1%} (pca:{pca_acc:.1%})"

        print(f">>> Epoch {epoch}/{total_epochs} | "
              f"loss={metrics['loss']:.4f} | "
              f"q+={metrics['q_pos']:.3f} | "
              f"q-={metrics['q_neg']:.3f} | "
              f"margin={metrics['margin']:.3f}{effk_str}{unused_str}{bentropy_str}{extra}{knn_str}")

    def train_umap_edges(self, epochs: int = None, verbose: bool = True,
                         early_stop: bool = False, eval_freq: int = 10,
                         patience: int = 50, max_epochs: int = None,
                         early_stop_metric: str = 'homophily',
                         min_delta: float = 0.0):
        """Train encoder using edge-centric parametric UMAP (fresh start).

        Args:
            epochs: fixed number of epochs (used when early_stop=False)
            early_stop: if True, stop based on early_stop_metric instead of fixed epochs
            eval_freq: evaluate metric every N epochs (used when early_stop=True)
            patience: stop if no improvement for this many epochs (used when early_stop=True)
            max_epochs: hard cap on epochs regardless of early stopping
            early_stop_metric: 'homophily' or 'modularity' (used when early_stop=True)
            min_delta: minimum improvement in metric to count as progress (default 0.0)
        """
        epochs = epochs or getattr(self, 'pretraining_epochs', 200)
        self._setup_umap_edges(epochs)
        metrics = self.continue_train_umap_edges(
            epochs, verbose,
            early_stop=early_stop, eval_freq=eval_freq,
            patience=patience, max_epochs=max_epochs,
            early_stop_metric=early_stop_metric,
            min_delta=min_delta,
        )
        self.save_clusters()
        return metrics

    def continue_train_umap_edges(self, epochs: int = 50, verbose: bool = True,
                                  early_stop: bool = False, eval_freq: int = 10,
                                  patience: int = 50, max_epochs: int = None,
                                  early_stop_metric: str = 'homophily',
                                  min_delta: float = 0.0):
        """Continue UMAP edge training from current model state for more epochs.

        Rebuilds optimizer/loader/edge_dataset if needed (e.g. after code reload),
        but never touches model weights or prototypes.

        Args:
            epochs: fixed number of epochs (used when early_stop=False)
            early_stop: if True, ignore epochs and stop based on early_stop_metric
            eval_freq: evaluate metric every N epochs (used when early_stop=True)
            patience: stop if no improvement for this many epochs
            max_epochs: hard cap on epochs (used with early_stop=True)
            early_stop_metric: 'homophily' or 'modularity' (used when early_stop=True)
        """
        if not hasattr(self, '_umap_state'):
            self._setup_umap_edges(epochs, init_prototypes=False)

        start = self._umap_state['epoch']

        use_nk = getattr(self, 'usage_norm_sim', 0) == 2

        if not early_stop:
            # --- fixed-epoch mode (original behaviour) ---
            freq = getattr(self, 'umap_checkpoint_freq', 20)
            last_epoch = start + epochs
            for i in range(epochs):
                if use_nk:
                    self._init_proto_usage_nk_batch()
                metrics = self._run_umap_epoch()
                current_epoch = start + i + 1
                if verbose:
                    self._print_umap_epoch(current_epoch, last_epoch, metrics)
                if freq > 0 and (current_epoch % freq == 0 or current_epoch == last_epoch):
                    self.save_umap_checkpoint()
            return metrics

        # --- early-stopping mode ---
        best_score = -1.0
        no_improve_epochs = 0
        epoch_offset = 0

        print(f"Early stopping mode: metric={early_stop_metric}, eval every {eval_freq} epochs, patience={patience}" +
              (f", max_epochs={max_epochs}" if max_epochs else ""))

        # Print initial modularity and coverage before any training and save as baseline checkpoint
        if early_stop_metric == 'modularity':
            init_result = self.modularity()
            best_score = init_result['modularity']
            lk = self.dataset.label_key
            proto_labels = self.label_prototypes(lk)['labels']
            n_covered = proto_labels.nunique()
            n_total = self.train_ds.adata.obs[lk].nunique()
            init_coverage = n_covered / n_total
            print(f"[Epoch 0] initial modularity={best_score:.4f}, coverage={init_coverage:.4f} ({n_covered}/{n_total} cell types) → saving as baseline checkpoint")
            self.save_umap_checkpoint()

        while True:
            if use_nk:
                self._init_proto_usage_nk_batch()
            metrics = self._run_umap_epoch()
            epoch_offset += 1
            current_epoch = start + epoch_offset

            if verbose:
                end_str = f"~{max_epochs}" if max_epochs else "?"
                self._print_umap_epoch(current_epoch, end_str, metrics)

            # Hard cap
            if max_epochs and epoch_offset >= max_epochs:
                print(f"[Early stop] Reached max_epochs={max_epochs}. Saving checkpoint.")
                self.save_umap_checkpoint()
                break

            # Eval block
            if epoch_offset % eval_freq == 0:
                if early_stop_metric == 'modularity':
                    # In recon-only mode prototypes aren't updated during training,
                    # so re-init them from current encoder before scoring.
                    recon_only = (getattr(self, 'lambda_recon', 0) > 0 and
                                  getattr(self, 'lambda_umap', 1) == 0 and
                                  getattr(self, 'lambda_proto', 0) == 0 and
                                  getattr(self, 'lambda_swav', 0) == 0 and
                                  getattr(self, 'lambda_proto_recon', 0) == 0)
                    embedding_mode = getattr(self, 'umap_similarity', 'embedding') == 'embedding'
                    if recon_only or embedding_mode:
                        self.init_prototypes()
                    result = self.modularity()
                    score = result['modularity']
                    proto_labels = self.label_prototypes(lk)['labels']
                    n_covered = proto_labels.nunique()
                    coverage_str = f", coverage={n_covered/n_total:.4f} ({n_covered}/{n_total})"
                else:
                    result = self.edge_homophily()
                    score = result['homophily']
                    coverage_str = ""

                improvement = score - best_score
                if improvement > min_delta:
                    best_score = score
                    no_improve_epochs = 0
                    print(f"  [Early stop] {early_stop_metric} improved to {score:.4f} (+{improvement:.4f}){coverage_str} → saving checkpoint")
                    self.save_umap_checkpoint()
                else:
                    no_improve_epochs += eval_freq
                    print(f"  [Early stop] No improvement ({score:.4f} vs best {best_score:.4f}, min_delta={min_delta}){coverage_str}, "
                          f"no-improve streak: {no_improve_epochs}/{patience}")
                    if no_improve_epochs >= patience:
                        print(f"[Early stop] Patience exhausted. Stopping at epoch {current_epoch}.")
                        break

        return metrics

    def save_umap_checkpoint(self, path=None):
        """Save model + optimizer state so training can resume after Colab restart."""
        if path is None:
            path = os.path.join(self.get_dump_path(), 'umap_checkpoint.pth')
        state = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self._umap_state['epoch'] if hasattr(self, '_umap_state') else 0,
            'optimizer_state_dict': self._umap_state['optimizer'].state_dict() if hasattr(self, '_umap_state') else None,
        }
        torch.save(state, path)
        print(f"Saved UMAP checkpoint to {path} (epoch {state['epoch']})")
        self.save_metacells()
        return path

    def load_umap_checkpoint(self, path=None):
        """Load model weights and rebuild UMAP training state for continue_train_umap_edges.

        Usage in Colab after restart:
            t = SCProtoTrainer(dataset=ds, ...)
            t.setup()
            t.load_umap_checkpoint()
            t.continue_train_umap_edges(epochs=50)
        """
        if path is None:
            path = os.path.join(self.get_dump_path(), 'umap_checkpoint.pth')
        checkpoint = torch.load(path, map_location=self.device)

        # Step 1: restore pretrain weights + init prototypes — matches the state at which
        # epsilon was originally calibrated (before UMAP training began).
        self.load_pretrain_checkpoint()
        self.init_prototypes()

        # Step 2: rebuild training objects and calibrate epsilon on pretrain+proto state.
        self._setup_umap_edges(init_prototypes=False, skip_calibration=False)

        # Step 3: overwrite model weights with the fully trained UMAP checkpoint.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self._umap_state['epoch'] = checkpoint['epoch']

        if checkpoint.get('optimizer_state_dict') is not None:
            self._umap_state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded UMAP checkpoint from {path} (epoch {checkpoint['epoch']})")

    def save_modularity(self):
        import json
        import numpy as np
        result = self.modularity()
        path = os.path.join(self.get_dump_path(), 'modularity.json')
        def _convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            raise TypeError(f'Not serializable: {type(o)}')
        with open(path, 'w') as f:
            json.dump(result, f, indent=2, default=_convert)
        print(f"Saved modularity={result['modularity']:.4f} to {path}")
        return result

    def eval_metacell_quality(self, assignments=None, label='proto', soft_metrics=False):
        """Compute and save cell-type purity, batch entropy, and modularity per metacell.

        Purity and batch entropy are computed per metacell (per-mc Series), then
        saved to CSV. Modularity is computed graph-level and saved to JSON.
        All scalar summaries are also stored in the metrics log.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: assignment method name.

        Returns:
            dict with keys 'purity', 'batch_entropy', 'modularity'.
        """
        import json
        import pandas as pd
        import torch.nn.functional as F

        # Encode once; derive hard assignments AND (optionally) soft matrix S.
        proto_trained = getattr(self, 'umap_similarity', 'embedding') == 'proto'
        protos = self.model.get_prototypes()
        S = None  # (N, K) soft assignment matrix — set below when soft_metrics=True

        with torch.no_grad():
            z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            if proto_trained and protos is not None and protos.shape[0] > 0:
                scores = self.model.prototypes(z)          # (N, K)
                if soft_metrics:
                    print(f"[soft metrics] using epsilon={self.epsilon:.6f} for soft assignments")
                    S = F.softmax(scores / self.epsilon, dim=1).cpu().numpy()
                if assignments is None:
                    assignments = scores.argmax(dim=1).cpu().numpy()
                    label = label or 'proto'
            elif assignments is None:
                from sklearn.cluster import KMeans
                z_np = z.cpu().numpy()
                km = KMeans(n_clusters=self.nmb_prototypes, n_init=3, random_state=42).fit(z_np)
                assignments = km.labels_
                label = label or 'kmeans'

        label = label or 'custom'

        obs = self.train_ds.adata.obs.copy()
        obs['_mc'] = assignments

        dump = self.get_dump_path()

        # Re-save clusters.npz here so it always matches the assignments used for metrics
        # (train_umap_edges saves it from the last epoch; this overwrites with best-checkpoint assignments)
        self.save_clusters(assignments=assignments, label=label)

        # --- Unused prototypes ---
        n_unused = int(self.nmb_prototypes - len(np.unique(assignments)))
        unused_ratio = n_unused / self.nmb_prototypes
        print(f"[{label}] unused protos: {n_unused}/{self.nmb_prototypes} ({unused_ratio:.2%})")

        # --- Metacell sizes (save once, reuse for weighted stats) ---
        mc_sizes = obs['_mc'].astype(str).value_counts().rename('size')
        mc_sizes.index.name = 'metacell'
        mc_sizes.to_csv(os.path.join(dump, 'size_per_mc.csv'))

        # --- Cell-type purity per metacell ---
        lk = self.dataset.label_key
        purity_per_mc = calc_purity(obs, label_key=lk, mc_key='_mc', return_per_mc=True)
        if purity_per_mc is not None:
            purity_per_mc.index.name = 'metacell'
            purity_per_mc.to_csv(os.path.join(dump, 'purity_per_mc.csv'))
            mean_purity = float(purity_per_mc.mean())
            weights_p = mc_sizes.reindex(purity_per_mc.index).fillna(0)
            w_sum_p = weights_p.sum()
            weighted_mean_purity = float((purity_per_mc * weights_p).sum() / w_sum_p)
            weighted_std_purity = float(np.sqrt(((purity_per_mc - weighted_mean_purity) ** 2 * weights_p).sum() / w_sum_p))
            print(f"[{label}] mean cell-type purity: {mean_purity:.4f}  (size-weighted: {weighted_mean_purity:.4f} ± {weighted_std_purity:.4f})")
        else:
            mean_purity = None
            weighted_mean_purity = None
            weighted_std_purity = None
            print(f"[{label}] cell-type purity: label key '{lk}' not in obs, skipped")

        # --- Niche purity per metacell ---
        nk = getattr(self.dataset, 'niche_key', None)
        niche_purity_per_mc = calc_purity(obs, label_key=nk, mc_key='_mc', return_per_mc=True) if nk else None
        if niche_purity_per_mc is not None:
            niche_purity_per_mc.index.name = 'metacell'
            niche_purity_per_mc.to_csv(os.path.join(dump, 'niche_purity_per_mc.csv'))
            mean_niche_purity = float(niche_purity_per_mc.mean())
            weights_n = mc_sizes.reindex(niche_purity_per_mc.index).fillna(0)
            w_sum_n = weights_n.sum()
            weighted_mean_niche_purity = float((niche_purity_per_mc * weights_n).sum() / w_sum_n)
            weighted_std_niche_purity = float(np.sqrt(((niche_purity_per_mc - weighted_mean_niche_purity) ** 2 * weights_n).sum() / w_sum_n))
            print(f"[{label}] mean niche purity: {mean_niche_purity:.4f}  (size-weighted: {weighted_mean_niche_purity:.4f} ± {weighted_std_niche_purity:.4f})")
            self._log_metric('mean_niche_purity', mean_niche_purity)
            self._log_metric('weighted_mean_niche_purity', weighted_mean_niche_purity)
            self._log_metric('weighted_std_niche_purity', weighted_std_niche_purity)
        else:
            mean_niche_purity = None
            weighted_mean_niche_purity = None
            weighted_std_niche_purity = None
            if nk:
                print(f"[{label}] niche purity: niche key '{nk}' not in obs, skipped")

        # --- Batch entropy per metacell ---
        bk = getattr(self.dataset, 'batch_key', None) or getattr(self, 'condition_key', None)
        entropy_per_mc = calc_batch_entropy(obs, batch_key=bk, mc_key='_mc', return_per_mc=True) if bk else None
        if entropy_per_mc is not None:
            entropy_per_mc.index.name = 'metacell'
            entropy_per_mc.to_csv(os.path.join(dump, 'batch_entropy_per_mc.csv'))
            mean_entropy = float(entropy_per_mc.mean())
            weights = mc_sizes.reindex(entropy_per_mc.index).fillna(0)
            w_sum = weights.sum()
            weighted_mean_entropy = float((entropy_per_mc * weights).sum() / w_sum)
            weighted_std_entropy = float(np.sqrt(((entropy_per_mc - weighted_mean_entropy) ** 2 * weights).sum() / w_sum))
            print(f"[{label}] mean batch entropy: {mean_entropy:.4f}  (size-weighted: {weighted_mean_entropy:.4f} ± {weighted_std_entropy:.4f})")
        else:
            mean_entropy = None
            weighted_mean_entropy = None
            weighted_std_entropy = None
            print(f"[{label}] batch entropy: batch key not found, skipped")

        # --- Soft assignment metrics (proto mode only) ---
        soft_purity_per_mc = None
        soft_niche_purity_per_mc = None
        soft_entropy_per_mc = None
        if S is not None:
            K_soft = S.shape[1]
            effective_sizes = S.sum(axis=0)  # (K,) — soft cell count per proto

            pd.Series(effective_sizes, index=[str(k) for k in range(K_soft)], name='effective_size') \
              .rename_axis('metacell').to_csv(os.path.join(dump, 'effective_size_per_mc.csv'))

            if lk in obs.columns:
                lbl_cat = pd.Categorical(obs[lk].values)
                onehot = np.eye(len(lbl_cat.categories))[lbl_cat.codes]  # (N, L)
                lbl_weights = S.T @ onehot                                 # (K, L)
                soft_purity_k = lbl_weights.max(axis=1) / np.maximum(effective_sizes, 1e-10)

                soft_purity_per_mc = pd.Series(
                    soft_purity_k, index=[str(k) for k in range(K_soft)], name='soft_purity'
                ).rename_axis('metacell')
                soft_purity_per_mc.to_csv(os.path.join(dump, 'soft_purity_per_mc.csv'))

                soft_mean_pur   = float(soft_purity_k.mean())
                soft_wmean_pur  = float((soft_purity_k * effective_sizes).sum() / effective_sizes.sum())
                soft_wstd_pur   = float(np.sqrt(((soft_purity_k - soft_wmean_pur) ** 2 * effective_sizes).sum() / effective_sizes.sum()))
                print(f"[{label}] soft mean cell-type purity: {soft_mean_pur:.4f}  (effective-size-weighted: {soft_wmean_pur:.4f} ± {soft_wstd_pur:.4f})")
                self._log_metric('soft_mean_cell_type_purity',          soft_mean_pur)
                self._log_metric('soft_weighted_mean_cell_type_purity', soft_wmean_pur)
                self._log_metric('soft_weighted_std_cell_type_purity',  soft_wstd_pur)

            if bk and bk in obs.columns:
                bat_cat    = pd.Categorical(obs[bk].values)
                bat_onehot = np.eye(len(bat_cat.categories))[bat_cat.codes]  # (N, B)
                bat_weights = S.T @ bat_onehot                                 # (K, B)
                bat_dist    = bat_weights / np.maximum(effective_sizes[:, None], 1e-10)
                soft_entropy_k = -np.sum(bat_dist * np.log(bat_dist + 1e-10), axis=1)

                soft_entropy_per_mc = pd.Series(
                    soft_entropy_k, index=[str(k) for k in range(K_soft)], name='soft_batch_entropy'
                ).rename_axis('metacell')
                soft_entropy_per_mc.to_csv(os.path.join(dump, 'soft_batch_entropy_per_mc.csv'))

                soft_mean_ent  = float(soft_entropy_k.mean())
                soft_wmean_ent = float((soft_entropy_k * effective_sizes).sum() / effective_sizes.sum())
                soft_wstd_ent  = float(np.sqrt(((soft_entropy_k - soft_wmean_ent) ** 2 * effective_sizes).sum() / effective_sizes.sum()))
                print(f"[{label}] soft mean batch entropy: {soft_mean_ent:.4f}  (effective-size-weighted: {soft_wmean_ent:.4f} ± {soft_wstd_ent:.4f})")
                self._log_metric('soft_mean_batch_entropy',          soft_mean_ent)
                self._log_metric('soft_weighted_mean_batch_entropy', soft_wmean_ent)
                self._log_metric('soft_weighted_std_batch_entropy',  soft_wstd_ent)

        # --- Modularity ---
        mod_result = self.modularity(assignments=assignments, label=label)
        mod_path = os.path.join(dump, 'modularity.json')
        def _convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            raise TypeError(f'Not serializable: {type(o)}')
        with open(mod_path, 'w') as f:
            json.dump(mod_result, f, indent=2, default=_convert)

        # --- Per-batch modularity ---
        mean_batch_mod = std_batch_mod = None
        if bk and bk in obs.columns:
            from interpretable_ssl.evaluation.mc_metric_utils import calc_modularity_per_batch
            A_full = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
            batch_mod_s = calc_modularity_per_batch(A_full, assignments, obs[bk].values)
            batch_mod_s.to_csv(os.path.join(dump, 'modularity_per_batch.csv'))
            mean_batch_mod = float(batch_mod_s.mean())
            std_batch_mod = float(batch_mod_s.std())
            print(f"[{label}] per-batch modularity: mean={mean_batch_mod:.4f}, std={std_batch_mod:.4f}")

        # --- Log scalar summaries ---
        if mean_purity is not None:
            self._log_metric('mean_cell_type_purity', mean_purity)
            self._log_metric('weighted_mean_cell_type_purity', weighted_mean_purity)
            self._log_metric('weighted_std_cell_type_purity', weighted_std_purity)
        if mean_entropy is not None:
            self._log_metric('mean_batch_entropy', mean_entropy)
            self._log_metric('weighted_mean_batch_entropy', weighted_mean_entropy)
            self._log_metric('weighted_std_batch_entropy', weighted_std_entropy)
        self._log_metric('modularity', mod_result['modularity'])
        self._log_metric('n_unused_protos', n_unused)
        self._log_metric('unused_proto_ratio', unused_ratio)
        if mean_batch_mod is not None:
            self._log_metric('mean_modularity_batch', mean_batch_mod)
            self._log_metric('std_modularity_batch', std_batch_mod)

        return {
            'purity':        purity_per_mc,
            'soft_purity':   soft_purity_per_mc,
            'niche_purity':  niche_purity_per_mc,
            'batch_entropy': entropy_per_mc,
            'soft_entropy':  soft_entropy_per_mc,
            'modularity':    mod_result,
            'n_unused_protos':  n_unused,
            'unused_proto_ratio': unused_ratio,
        }

    def eval_task2_metrics(self, mc_ad=None, soft_metrics=False):
        """Compute and save task 2 metacell representation metrics.

        Metrics: coverage, DGE consistency (RBO/Kendall/Jaccard), scGraph.
        All scalar summaries are also stored in the metrics log.

        Args:
            mc_ad: metacell AnnData, or None to load from the saved checkpoint path.

        Returns:
            dict with keys 'coverage', 'dge_jaccard_avg', 'scgraph_corr_avg'.
        """
        import scanpy as sc
        from interpretable_ssl.evaluation.mc_metric_utils import calc_task2_metrics

        ad  = self.train_ds.adata
        lk  = self.dataset.label_key
        bk  = getattr(self.dataset, 'batch_key', None) or getattr(self, 'condition_key', None)
        name = self.get_model_name()
        dump = self.get_dump_path()

        # Load metacells if not provided; generate if not yet saved
        if mc_ad is None:
            import anndata
            mc_path = os.path.join(dump, 'metacells.h5ad')
            mc_ad_all = anndata.read_h5ad(mc_path) if os.path.exists(mc_path) else self.save_metacells()
        else:
            mc_ad_all = mc_ad.copy()

        obsm_key = f"{name}_mc_pca"

        # --- Hard labels: majority vote, drop unused protos ---
        proto_labels = self.label_prototypes(lk)['labels']  # Series: proto_id -> label
        mc_ad_hard = mc_ad_all.copy()
        mc_ad_hard.obs[lk] = mc_ad_hard.obs['prototype_id'].map(proto_labels)
        mc_ad_hard = mc_ad_hard[mc_ad_hard.obs[lk].notna()].copy()
        sc.tl.pca(mc_ad_hard)
        mc_ad_hard.obsm[obsm_key] = mc_ad_hard.obsm["X_pca"]

        scalars = calc_task2_metrics(ad, mc_ad_hard, lk, bk, [obsm_key], name, dump)
        for metric_name, value in scalars.items():
            if value is not None:
                self._log_metric(metric_name, value)
                print(f"[task2] {metric_name}: {value:.4f}")

        # --- Soft labels: weighted vote, all protos get a label ---
        soft_scalars = {}
        if soft_metrics and getattr(self, 'umap_similarity', 'embedding') == 'proto':
            soft_proto_labels = self.label_prototypes_soft(lk)['labels']  # all K protos labeled
            mc_ad_soft = mc_ad_all.copy()
            mc_ad_soft.obs[lk] = mc_ad_soft.obs['prototype_id'].map(soft_proto_labels)
            sc.tl.pca(mc_ad_soft)
            mc_ad_soft.obsm[obsm_key] = mc_ad_soft.obsm["X_pca"]
            soft_dump = os.path.join(dump, 'soft')
            os.makedirs(soft_dump, exist_ok=True)
            soft_scalars = calc_task2_metrics(ad, mc_ad_soft, lk, bk, [obsm_key], name + '_soft', soft_dump)
            for metric_name, value in soft_scalars.items():
                if value is not None:
                    self._log_metric(f'soft_{metric_name}', value)
                    print(f"[task2/soft] {metric_name}: {value:.4f}")

        return {**scalars, **{f'soft_{k}': v for k, v in soft_scalars.items()}}

    def eval_task3_metrics(self, mc_ad=None):
        """Compute and save task 3 spatial metrics (niche DGE consistency).

        Only runs if dataset has a niche_key defined. Saves:
            ct_niche_rbo_full.csv  — raw per-(cell_type, niche) RBO
            ct_niche_rbo.csv       — mean RBO per cell type

        Returns:
            dict with key 'ct_niche_rbo_avg', or empty dict if no niche_key.
        """
        nk = getattr(self.dataset, 'niche_key', None)
        if not nk:
            print("[task3] no niche_key defined, skipped")
            return {}

        from interpretable_ssl.evaluation.de_helper import celltype_niche_dge
        import anndata

        ad   = self.train_ds.adata
        lk   = self.dataset.label_key
        name = self.get_model_name()
        dump = self.get_dump_path()

        if mc_ad is None:
            mc_path = os.path.join(dump, 'metacells.h5ad')
            mc_ad = anndata.read_h5ad(mc_path) if os.path.exists(mc_path) else self.save_metacells()

        proto_labels = self.label_prototypes(lk)['labels']
        mc_ad = mc_ad.copy()
        mc_ad.obs[lk] = mc_ad.obs['prototype_id'].map(proto_labels)
        mc_ad = mc_ad[mc_ad.obs[lk].notna()].copy()

        # propagate niche labels to metacells via majority vote
        obs = self.train_ds.adata.obs.copy()
        assignments, _ = self._get_assignments(None, 'proto')
        obs['_mc'] = assignments
        from interpretable_ssl.evaluation.mc_metric_utils import calc_purity
        _, major_niche = calc_purity(obs, label_key=nk, mc_key='_mc',
                                     return_per_mc=True, return_major_label=True)
        mc_ad.obs[nk] = mc_ad.obs['prototype_id'].astype(str).map(major_niche)

        summary, _ = celltype_niche_dge(ad, mc_ad, lk, nk, name, dump)
        ct_niche_rbo_avg = float(summary.values.mean())

        self._log_metric('ct_niche_rbo_avg', ct_niche_rbo_avg)
        print(f"[task3] mean ct-niche RBO: {ct_niche_rbo_avg:.4f}")

        return {'ct_niche_rbo_avg': ct_niche_rbo_avg}

    def _log_metric(self, name, value):
        """Store a scalar or dict metric value in the internal log and flush to disk."""
        self._metrics_log[name] = value
        self._save_metrics()

    def _save_metrics(self):
        """Flush current metrics log to a small JSON file on disk."""
        import json
        try:
            # When running inside _eval_baseline, write to baseline folder
            if hasattr(self, '_baseline_save_dir'):
                save_dir = self._baseline_save_dir
            else:
                save_dir = self.get_dump_path()
            path = os.path.join(save_dir, 'metrics.json')
            with open(path, 'w') as f:
                json.dump(self._metrics_log, f, indent=2)
        except Exception:
            pass  # don't break metric computation if save fails

    def save_clusters(self, assignments=None, label='proto', path=None):
        """Save cluster assignments and all accumulated metrics to disk.

        Saves a .npz file with:
          - assignments: int array (n_cells,)
          - label: assignment method name
          - metrics: dict of all scalar metrics recorded so far

        Args:
            assignments: cluster labels, or None to auto-resolve.
            label: assignment method name.
            path: save path. Defaults to <dump_path>/clusters.npz.

        Returns:
            path where file was saved.
        """
        import json

        assignments, label = self._get_assignments(assignments, label)

        if path is None:
            path = os.path.join(self.get_dump_path(), 'clusters.npz')

        # Use smallest int dtype that fits
        max_val = int(assignments.max()) if len(assignments) else 0
        if max_val < 128:
            dtype = np.int8
        elif max_val < 32768:
            dtype = np.int16
        else:
            dtype = np.int32

        # Serialise metrics to JSON string (np.savez doesn't handle nested dicts)
        metrics_json = json.dumps(self._metrics_log)

        np.savez_compressed(
            path,
            assignments=assignments.astype(dtype),
            label=np.array(label),
            metrics_json=np.array(metrics_json),
        )
        print(f"Saved clusters ({len(assignments)} cells, label='{label}') "
              f"and {len(self._metrics_log)} metrics to {path}")
        return path

    @staticmethod
    def load_metrics(path):
        """Load cluster assignments and metrics saved by save_clusters.

        Args:
            path: path to .npz file saved by save_clusters.

        Returns:
            dict with 'assignments', 'label', 'metrics'.
        """
        import json

        data = np.load(path, allow_pickle=True)
        metrics = json.loads(str(data['metrics_json']))
        return {
            'assignments': data['assignments'],
            'label': str(data['label']),
            'metrics': metrics,
        }

    def _get_assignments(self, assignments=None, label=None):
        """Resolve cluster assignments and label.

        If assignments is None:
          - if prototypes were trained (umap_similarity == 'proto'), use argmax
          - otherwise, run k-means on the embedding space

        Returns:
            (assignments, label) — numpy int array and string label.
        """
        from sklearn.cluster import KMeans

        with torch.no_grad():
            z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)

        if assignments is None:
            proto_trained = getattr(self, 'umap_similarity', 'embedding') == 'proto'
            protos = self.model.get_prototypes()
            if proto_trained and protos is not None and protos.shape[0] > 0:
                scores = self.model.prototypes(z)
                assignments = scores.argmax(dim=1).cpu().numpy()
                label = label or 'proto'
            else:
                z_np = z.cpu().numpy()
                K = self.nmb_prototypes
                km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(z_np)
                assignments = km.labels_
                label = label or 'kmeans'
        else:
            label = label or 'custom'

        return assignments, label

    def clustering_score(self, assignments=None, label='proto'):
        """Compute weighted same-cluster rate for a given assignment.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with score, expected_random, ratio, per_cluster scores, assignments.
        """
        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A_coo = A.tocoo()

        # Weighted same-cluster rate
        heads, tails, weights = A_coo.row, A_coo.col, A_coo.data.astype(np.float64)
        same = (assignments[heads] == assignments[tails])
        score = (weights * same).sum() / weights.sum()

        # Expected random baseline: Sigma_k (n_k / n)^2
        _, counts = np.unique(assignments, return_counts=True)
        n = len(assignments)
        expected = ((counts / n) ** 2).sum()

        # Per-cluster: fraction of member affinity that stays within cluster
        cluster_ids = np.unique(assignments)
        per_cluster = {}
        for c in cluster_ids:
            mask = assignments[heads] == c
            if mask.sum() == 0:
                continue
            c_weights = weights[mask]
            c_same = assignments[tails[mask]] == c
            per_cluster[c] = (c_weights * c_same).sum() / c_weights.sum()

        print(f"[{label}] weighted same-cluster rate: {score:.4f} "
              f"(random baseline: {expected:.4f}, ratio: {score/expected:.1f}x)")

        self._log_metric('clustering_score', score)
        self._log_metric('clustering_score/ratio', score / expected)
        self._log_metric('clustering_score/per_cluster', {int(k): float(v) for k, v in per_cluster.items()})

        return {
            'score': score,
            'expected_random': expected,
            'ratio': score / expected,
            'per_cluster': per_cluster,
            'assignments': assignments,
            'label': label,
        }

    def modularity(self, assignments=None, label='proto'):
        """Compute weighted Newman modularity for a given assignment.

        Q = (1/2m) * sum_k [ e_k - d_k^2 / (2m) ]

        where e_k = sum of edge weights within cluster k,
              d_k = sum of degrees of nodes in cluster k,
              2m  = total edge weight.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with modularity, assignments, label, per_cluster_contribution.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        # Degree vector and total weight
        degrees = np.array(A.sum(axis=1)).ravel()
        two_m = degrees.sum()

        if two_m == 0:
            print(f"[{label}] weighted modularity: 0.0000 (no edges)")
            return {
                'modularity': 0.0,
                'assignments': assignments,
                'label': label,
                'per_cluster_contribution': {},
            }

        # Per-cluster modularity contribution
        cluster_ids = np.unique(assignments)
        per_cluster = {}
        Q = 0.0
        for c in cluster_ids:
            mask = (assignments == c)
            # e_k: sum of edge weights within cluster k (both directions counted)
            A_sub = A[mask][:, mask]
            e_k = A_sub.sum()
            # d_k: sum of degrees of nodes in cluster k
            d_k = degrees[mask].sum()
            contrib = (e_k - d_k * d_k / two_m) / two_m
            per_cluster[int(c)] = float(contrib)
            Q += contrib

        Q = float(Q)
        print(f"[{label}] weighted modularity: {Q:.4f}")

        self._log_metric('modularity', Q)
        self._log_metric('modularity/per_cluster', per_cluster)

        return {
            'modularity': Q,
            'assignments': assignments,
            'label': label,
            'per_cluster_contribution': per_cluster,
        }

    def ncut(self, assignments=None, label='proto'):
        """Compute weighted normalized cut for a given assignment.

        NCut = sum_k  cut(C_k, V\\C_k) / vol(C_k)

        where cut(C_k, V\\C_k) = sum of edge weights crossing cluster k,
              vol(C_k)          = sum of degrees of nodes in cluster k.

        Lower is better.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with ncut, assignments, label, per_cluster_contribution.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        # Symmetrise — NCut formula assumes undirected graph
        A = (A + A.T) / 2

        degrees = np.array(A.sum(axis=1)).ravel()

        cluster_ids = np.unique(assignments)
        per_cluster = {}
        ncut_val = 0.0
        for c in cluster_ids:
            mask = (assignments == c)
            vol_k = degrees[mask].sum()
            if vol_k == 0:
                continue
            # Intra-cluster edge weight
            e_k = A[mask][:, mask].sum()
            # cut = vol_k - e_k  (edges leaving cluster = total degree - internal edges)
            cut_k = vol_k - e_k
            contrib = cut_k / vol_k
            per_cluster[int(c)] = float(contrib)
            ncut_val += contrib

        ncut_val = float(ncut_val)
        print(f"[{label}] weighted normalized cut: {ncut_val:.4f}")

        self._log_metric('ncut', ncut_val)
        self._log_metric('ncut/per_cluster', per_cluster)

        return {
            'ncut': ncut_val,
            'assignments': assignments,
            'label': label,
            'per_cluster_contribution': per_cluster,
        }

    def conductance(self, assignments=None, label='proto'):
        """Compute mean weighted conductance across clusters.

        conductance(C) = cut(C, V\\C) / min(vol(C), vol(V\\C))

        Lower is better (0 = perfect, 1 = worst).

        Returns:
            dict with mean_conductance, per_cluster, assignments, label.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        degrees = np.array(A.sum(axis=1)).ravel()
        total_vol = degrees.sum()

        cluster_ids = np.unique(assignments)
        per_cluster = {}
        vals = []
        for c in cluster_ids:
            mask = (assignments == c)
            vol_c = degrees[mask].sum()
            if vol_c == 0:
                continue
            e_c = A[mask][:, mask].sum()
            cut_c = vol_c - e_c
            denom = min(vol_c, total_vol - vol_c)
            if denom == 0:
                continue
            cond = cut_c / denom
            per_cluster[int(c)] = float(cond)
            vals.append(cond)

        mean_cond = float(np.mean(vals)) if vals else 0.0
        print(f"[{label}] mean conductance: {mean_cond:.4f}  (lower is better)")

        self._log_metric('conductance', mean_cond)
        self._log_metric('conductance/per_cluster', per_cluster)

        return {
            'mean_conductance': mean_cond,
            'per_cluster': per_cluster,
            'assignments': assignments,
            'label': label,
        }

    def edge_homophily(self, assignments=None, label='proto'):
        """Compute weighted edge homophily.

        homophily = sum of edge weights within clusters / total edge weight

        Higher is better (1 = all edges within clusters, 0 = all edges cut).

        Returns:
            dict with homophily, assignments, label.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        total_weight = A.sum()
        if total_weight == 0:
            print(f"[{label}] edge homophily: 0.0000 (no edges)")
            return {'homophily': 0.0, 'assignments': assignments, 'label': label}

        intra_weight = 0.0
        for c in np.unique(assignments):
            mask = (assignments == c)
            intra_weight += A[mask][:, mask].sum()

        h = float(intra_weight / total_weight)
        print(f"[{label}] edge homophily: {h:.4f}  (higher is better)")

        self._log_metric('edge_homophily', h)

        return {
            'homophily': h,
            'assignments': assignments,
            'label': label,
        }

    def eval_graph_structure(self, assignments=None, label='proto'):
        """Run graph structure preservation metrics and log them.

        Computes: modularity, edge_homophily.

        Returns:
            dict with all scalar metrics.
        """
        print(f"\n{'='*50}")
        print(f"Graph Structure Preservation Metrics [{label}]")
        print(f"{'='*50}")

        r_mod = self.modularity(assignments, label)
        r_hom = self.edge_homophily(assignments, label)

        summary = {
            'modularity':     r_mod['modularity'],
            'edge_homophily': r_hom['homophily'],
        }

        print(f"\nSummary:")
        print(f"  Modularity:      {summary['modularity']:.4f}  (higher is better)")
        print(f"  Edge Homophily:  {summary['edge_homophily']:.4f}  (higher is better)")
        print(f"{'='*50}\n")

        return summary

    def metacell_f1(self, assignments=None, label='proto'):
        """Compute F1 score treating clusters as metacells with majority-vote labels.

        Each cluster is assigned the majority ground-truth label of its member
        cells. Every cell then inherits its cluster's majority label as its
        prediction. F1 is computed between ground-truth and predicted labels.

        Evaluates on each available label key:
          - self.dataset.label_key (e.g. "celltype", "final_annotation")
          - "niches_2D" (if present in obs)

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with label, assignments, and results per label key containing
            per_class_f1, macro_f1, weighted_f1.
        """
        from sklearn.metrics import f1_score

        assignments, label = self._get_assignments(assignments, label)
        obs = self.train_ds.adata.obs

        # Gather label keys to evaluate
        keys_to_eval = []
        lk = self.dataset.label_key
        if lk in obs.columns:
            keys_to_eval.append(lk)
        if "niches_2D" in obs.columns:
            keys_to_eval.append("niches_2D")

        if not keys_to_eval:
            print(f"[{label}] metacell_f1: no label keys found in obs "
                  f"(checked '{lk}', 'niches_2D')")
            return {'label': label, 'assignments': assignments, 'results': {}}

        results = {}
        for key in keys_to_eval:
            gt = obs[key].values
            cluster_ids = np.unique(assignments)

            # Majority-vote label per cluster
            cluster_label = {}
            for c in cluster_ids:
                member_labels = gt[assignments == c]
                vals, counts = np.unique(member_labels, return_counts=True)
                cluster_label[c] = vals[counts.argmax()]

            # Predicted label for each cell = its cluster's majority label
            pred = np.array([cluster_label[c] for c in assignments])

            # Unique classes present in ground truth
            classes = np.unique(gt)

            per_class = {}
            f1_per = f1_score(gt, pred, labels=classes, average=None, zero_division=0)
            for cls, f in zip(classes, f1_per):
                per_class[cls] = float(f)

            macro = float(f1_score(gt, pred, labels=classes, average='macro', zero_division=0))
            weighted = float(f1_score(gt, pred, labels=classes, average='weighted', zero_division=0))

            results[key] = {
                'per_class_f1': per_class,
                'macro_f1': macro,
                'weighted_f1': weighted,
            }

            self._log_metric(f'f1_macro/{key}', macro)
            self._log_metric(f'f1_weighted/{key}', weighted)
            self._log_metric(f'f1_per_class/{key}', {str(k): float(v) for k, v in per_class.items()})

            print(f"[{label} | {key}] macro_f1: {macro:.4f}  weighted_f1: {weighted:.4f}")

        return {
            'label': label,
            'assignments': assignments,
            'results': results,
        }

    def spectral_assignments(self, n_clusters=None):
        """Run spectral clustering on the affinity graph.

        Args:
            n_clusters: number of clusters. Defaults to self.nmb_prototypes.

        Returns:
            np.ndarray of integer cluster labels (n_cells,).
        """
        from sklearn.cluster import SpectralClustering
        import scipy.sparse as sp

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)

        # Symmetrise just in case
        A = (A + A.T) / 2

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=3,
        )
        return sc.fit_predict(A)

    def _run_metrics(self, assignments, label):
        """Run all four metrics on given assignments. Returns results dict."""
        cs = self.clustering_score(assignments=assignments, label=label)
        mod = self.modularity(assignments=assignments, label=label)
        nc = self.ncut(assignments=assignments, label=label)
        f1 = self.metacell_f1(assignments=assignments, label=label)
        return {
            'clustering_score': cs,
            'modularity': mod,
            'ncut': nc,
            'metacell_f1': f1,
        }

    def eval_all(self, assignments=None, label=None):
        """Run all metrics, save clusters and metrics to disk.

        Args:
            assignments: cluster labels, or None for auto (proto/kmeans).
            label: name for the primary assignment. Auto-detected if None.

        Returns:
            dict with clustering_score, modularity, ncut, metacell_f1,
            assignments, label.
        """
        assignments, label = self._get_assignments(assignments, label)

        # Save model info for comparison tables
        self._log_metric('_info/umap_similarity', getattr(self, 'umap_similarity', 'embedding'))
        self._log_metric('_info/assignment_method', label)
        self._log_metric('_info/n_clusters', int(len(np.unique(assignments))))

        results = self._run_metrics(assignments, label)

        self.save_clusters(assignments=assignments, label=label)

        return {
            **results,
            'assignments': assignments,
            'label': label,
        }

    def _eval_baseline(self, assignments, name, n_clusters):
        """Evaluate a baseline and save to its own folder under get_save_dir().

        Args:
            assignments: np.ndarray of integer cluster labels.
            name: baseline name (used as folder name and label).
            n_clusters: number of clusters (for info logging).

        Returns:
            dict of metric results.
        """
        import json

        save_dir = os.path.join(self.get_save_dir(), name)
        os.makedirs(save_dir, exist_ok=True)

        # Swap metrics log AND disable auto-flush to model's folder
        orig_log = self._metrics_log
        self._metrics_log = {}
        self._baseline_save_dir = save_dir  # redirect _save_metrics

        self._log_metric('_info/assignment_method', name)
        self._log_metric('_info/n_clusters', int(n_clusters))

        results = self._run_metrics(assignments, name)

        # Write clusters.npz
        max_val = int(assignments.max()) if len(assignments) else 0
        dtype = np.int8 if max_val < 128 else (np.int16 if max_val < 32768 else np.int32)
        np.savez_compressed(
            os.path.join(save_dir, 'clusters.npz'),
            assignments=assignments.astype(dtype),
            label=np.array(name),
            metrics_json=np.array(json.dumps(self._metrics_log)),
        )

        print(f"Saved {name} baseline to {save_dir}")

        # Restore
        del self._baseline_save_dir
        self._metrics_log = orig_log
        return results

    def eval_spectral(self, n_clusters=None):
        """Evaluate spectral clustering baseline and save to its own folder.

        Args:
            n_clusters: number of clusters. Defaults to self.nmb_prototypes.

        Returns:
            dict of metric results.
        """
        if n_clusters is None:
            n_clusters = self.nmb_prototypes
        assignments = self.spectral_assignments(n_clusters)
        return self._eval_baseline(assignments, f'spectral_K{n_clusters}', n_clusters)

    def eval_seacells(self, n_clusters=None, max_iter=100):
        """Run SEACells archetypal analysis on our affinity and evaluate.

        Steps:
          1. Load affinity matrix, cast to symmetric CSR kernel
          2. Initialize SEACells model + archetypes
          3. Run archetypal analysis (fit)
          4. Extract hard assignments
          5. Save clustering + compute metrics for comparison

        Args:
            n_clusters: number of archetypes. Defaults to self.nmb_prototypes.
            max_iter: max fitting iterations.

        Returns:
            dict of metric results.
        """
        import SEACells.core
        import scipy.sparse as sp

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        # 1. Load affinity, cast to symmetric kernel
        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        K = sp.csr_matrix(A)
        K = (K + K.T) / 2

        # 2. Init model with our kernel
        ad = self.train_ds.adata
        model = SEACells.core.SEACells(
            ad,
            build_kernel_on='X_pca',  # ignored when precomputed
            n_SEACells=n_clusters,
            use_gpu=False,
            verbose=True,
        )
        model.add_precomputed_kernel_matrix(K)

        # 3. Run archetypal analysis — random init (skip PCA-based waypoints)
        init_idx = np.random.choice(K.shape[0], n_clusters, replace=False)
        model.fit(max_iter=max_iter, initial_archetypes=init_idx)

        # 4. Hard assignments (A_: shape k x n → argmax per cell)
        assignments = np.array(model.A_.argmax(axis=0)).ravel()
        n_unique = len(np.unique(assignments))
        print(f"SEACells: {n_unique} unique clusters from {n_clusters} archetypes, "
              f"A_ shape={model.A_.shape}, RSS={getattr(model, 'RSS_iters', ['?'])[-1]}")

        # 5. Save + metrics
        return self._eval_baseline(assignments, f'seacells_K{n_clusters}', n_clusters)

    def eval_seacells_native(self, n_clusters=None, n_waypoint_eigs=10, max_iter=100):
        """Run SEACells with its own affinity computation (native mode).

        Unlike eval_seacells which uses our precomputed affinity, this runs
        SEACells end-to-end with its own kernel built from X_pca.

        Args:
            n_clusters: number of archetypes. Defaults to self.nmb_prototypes.
            n_waypoint_eigs: number of eigenvectors for waypoint initialization.
            max_iter: max fitting iterations.

        Returns:
            dict of metric results.
        """
        import SEACells.core
        import scanpy as sc

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        ad = self.train_ds.adata.copy()

        # Ensure PCA exists
        if 'X_pca' not in ad.obsm:
            print("Computing PCA for SEACells...")
            sc.pp.pca(ad, n_comps=50)

        # Build SEACells model with its own kernel
        model = SEACells.core.SEACells(
            ad,
            build_kernel_on='X_pca',
            n_SEACells=n_clusters,
            n_waypoint_eigs=n_waypoint_eigs,
            use_gpu=False,
            verbose=True,
        )

        # Build kernel (SEACells native)
        print("Building SEACells kernel...")
        model.construct_kernel_matrix()

        # Initialize archetypes
        print("Initializing archetypes...")
        model.initialize_archetypes()

        # Fit
        print(f"Fitting SEACells (max_iter={max_iter})...")
        model.fit(max_iter=max_iter)

        # Hard assignments
        assignments = np.array(model.A_.argmax(axis=0)).ravel()
        n_unique = len(np.unique(assignments))
        print(f"SEACells native: {n_unique} unique clusters from {n_clusters} archetypes")

        # Save + metrics
        return self._eval_baseline(assignments, f'seacells_native_K{n_clusters}', n_clusters)

    def compare_runs(self, base_dir=None):
        """Auto-discover and compare metrics from all model runs in a base directory.

        Scans every subdirectory of base_dir for metrics.json or clusters.npz.
        Only loads runs that have saved metrics. Model folder names are
        auto-shortened by stripping the longest common prefix so only the
        differing parts are shown.

        Args:
            base_dir: parent directory containing one folder per run.
                      Defaults to self.get_save_dir() (e.g. MODEL_DIR/<dataset_id>/).

        Returns:
            (df, all_metrics) — pandas DataFrame of scalar metrics + info,
            and full metrics dict.
        """
        import json
        import pandas as pd

        if base_dir is None:
            base_dir = self.get_save_dir()

        # Auto-discover runs
        all_metrics = {}
        for entry in sorted(os.listdir(base_dir)):
            run_dir = os.path.join(base_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            json_path = os.path.join(run_dir, 'metrics.json')
            npz_path = os.path.join(run_dir, 'clusters.npz')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    all_metrics[entry] = json.load(f)
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                all_metrics[entry] = json.loads(str(data['metrics_json']))

        if not all_metrics:
            print(f"No runs with metrics found in {base_dir}")
            return None, {}

        # Auto-shorten names: strip longest common prefix up to last separator
        full_names = list(all_metrics.keys())
        if len(full_names) > 1:
            prefix = os.path.commonprefix(full_names)
            # Trim to last underscore/hyphen so we don't cut mid-word
            for sep in ('_', '-'):
                idx = prefix.rfind(sep)
                if idx > 0:
                    prefix = prefix[:idx + 1]
                    break
            short_names = [n[len(prefix):] or n for n in full_names]
        else:
            short_names = full_names
        name_map = dict(zip(full_names, short_names))

        print(f"Found {len(all_metrics)} runs in {base_dir}")

        # Collect scalar metrics, per-cluster dicts, and info fields
        scalar_keys = set()
        per_cluster_keys = set()
        info_keys = set()
        for m in all_metrics.values():
            for k, v in m.items():
                if k.startswith('_info/'):
                    info_keys.add(k)
                elif isinstance(v, (int, float)):
                    scalar_keys.add(k)
                elif isinstance(v, dict):
                    per_cluster_keys.add(k)
        scalar_keys = sorted(scalar_keys)
        per_cluster_keys = sorted(per_cluster_keys)
        info_keys = sorted(info_keys)

        # --- Build DataFrame ---
        rows = {}
        for full, short in name_map.items():
            row = {}
            for k in info_keys:
                row[k.replace('_info/', '')] = all_metrics[full].get(k, '')
            for k in scalar_keys:
                v = all_metrics[full].get(k)
                row[k] = v if isinstance(v, (int, float)) else None
            rows[short] = row
        df = pd.DataFrame(rows).T
        df.index.name = 'run'

        return df, all_metrics

    def compare_clusterings(self, assignments_dict=None, figsize=(14, 4)):
        """Compare clustering quality across methods.

        Args:
            assignments_dict: dict of {name: np.ndarray} cluster assignments.
                If None, compares proto assignments vs k-means on embeddings.
            figsize: figure size.

        Returns:
            dict of results per method, and the figure.
        """
        import matplotlib.pyplot as plt

        if assignments_dict is None:
            # Auto: proto vs kmeans
            from sklearn.cluster import KMeans
            with torch.no_grad():
                z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            z_np = z.cpu().numpy()

            # Proto assignments
            scores = self.model.prototypes(z)
            proto_assign = scores.argmax(dim=1).cpu().numpy()

            # K-means on same embeddings
            K = self.nmb_prototypes
            km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(z_np)

            assignments_dict = {
                'proto': proto_assign,
                'kmeans': km.labels_,
            }

        results = {}
        for name, assigns in assignments_dict.items():
            results[name] = self.clustering_score(assigns, label=name)

        # Plot
        names = list(results.keys())
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: overall score bar
        scores = [results[n]['score'] for n in names]
        baselines = [results[n]['expected_random'] for n in names]
        x = np.arange(len(names))
        axes[0].bar(x, scores, alpha=0.8, label='score')
        axes[0].bar(x, baselines, alpha=0.3, color='gray', label='random baseline')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names)
        axes[0].set_ylabel('weighted same-cluster rate')
        axes[0].set_title('Overall score')
        axes[0].legend(fontsize=8)

        # Panel 2: box plot of per-cluster intra-affinity
        box_data = []
        box_labels = []
        for n in names:
            vals = list(results[n]['per_cluster'].values())
            box_data.append(vals)
            box_labels.append(n)
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('intra-cluster affinity fraction')
        axes[1].set_title('Per-cluster score')

        # Panel 3: ratio bar
        ratios = [results[n]['ratio'] for n in names]
        axes[2].bar(x, ratios, alpha=0.8, color='steelblue')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names)
        axes[2].set_ylabel('score / random baseline')
        axes[2].set_title('Ratio (higher = better)')
        axes[2].axhline(1.0, color='gray', ls='--', lw=0.8)

        fig.tight_layout()
        plt.show()

        return results, fig

    def calc_r1r2_loss(self, z):
        """
        Li et al. style R1/R2 prototype coverage losses.
        Uses scores (same as assignment) to ensure consistency.

        R1: each proto should be "best" for at least 1 cell -> move proto
        R2: each cell should have high score for at least 1 proto -> move encoder

        Ensures: no orphan protos, no uncovered cells.

        r1r2_log=1: use -log(max) instead of -max.
            Dead protos/cells get unbounded penalty (log(x)->-inf as x->0),
            so no proto or cell can free-ride on the average.
        """
        use_log = getattr(self, 'r1r2_log', 0)
        protos = self.model.get_prototypes()  # (K, D)

        def _soft_assign(z_, protos_):
            if self.assignment_metric == 'sneuc':
                d2 = torch.cdist(z_, protos_, p=2) ** 2
                s = -d2
                s = s - s.max(dim=1, keepdim=True)[0]
                s = s.clamp(min=-75)
            elif self.assignment_metric == 'dotp':
                s = z_ @ protos_.T
            else:
                s = -torch.cdist(z_, protos_, p=2)
            return F.softmax(s / self.epsilon, dim=1)  # (B, K) in (0,1)

        # R1: protos get gradient (z detached) — protos move toward cells
        max_r1 = _soft_assign(z.detach(), protos).max(dim=0).values   # (K,)

        # R2: z gets gradient (protos detached) — cells move toward protos
        max_r2 = _soft_assign(z, protos.detach()).max(dim=1).values    # (B,)

        if use_log:
            r1 = -torch.log(max_r1.clamp(min=1e-6)).mean()
            r2 = -torch.log(max_r2.clamp(min=1e-6)).mean()
        else:
            r1 = -max_r1.mean()
            r2 = -max_r2.mean()

        return r1 + r2

    @torch.no_grad()
    def _sample_edges(self, n_edges=10000):
        """Sample positive edges and negative pairs, return indices and weights."""
        aff = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        coo = aff.tocoo()
        heads, tails, weights = coo.row, coo.col, coo.data.astype(np.float64)
        n = min(n_edges, len(heads))
        idx = np.random.choice(len(heads), size=n, replace=False)
        neg_tails = np.random.randint(0, aff.shape[0], size=n)
        return heads[idx], tails[idx], weights[idx], neg_tails

    @torch.no_grad()
    def plot_p_distribution(self, n_edges=10000, bins=80, figsize=(7, 4)):
        """Plot p (input-space affinity weight) distribution for sampled edges."""
        import matplotlib.pyplot as plt
        heads, tails, weights, _ = self._sample_edges(n_edges)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(weights, bins=bins, alpha=0.7, density=True, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('p (affinity weight)')
        ax.set_ylabel('density')
        ax.set_title(f'p distribution (input space) — {len(weights)} edges')
        ax.axvline(np.median(weights), color='red', ls='--', label=f'median={np.median(weights):.4f}')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig

    @torch.no_grad()
    def plot_q_distribution(self, n_edges=10000, bins=80, figsize=(10, 4)):
        """Plot q (latent-space similarity) for positive and negative pairs."""
        import matplotlib.pyplot as plt

        heads, tails, weights, neg_tails = self._sample_edges(n_edges)

        # Build X tensor if _umap_state not available
        if hasattr(self, '_umap_state'):
            X = self._umap_state['X']
            loss_fn = self._umap_state['loss_fn']
        else:
            adata = self.train_ds.adata
            if hasattr(adata.X, 'toarray'):
                X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
            else:
                X = torch.tensor(adata.X, dtype=torch.float32).to(self.device)
            min_dist = getattr(self, 'umap_min_dist', 0.5)
            spread = getattr(self, 'umap_spread', 1.0)
            neg_rate = getattr(self, 'umap_neg_rate', 5)
            loss_fn = ParametricUMAPLoss(min_dist=min_dist, spread=spread, negative_sample_rate=neg_rate)

        use_proto_sim = getattr(self, 'umap_similarity', 'embedding') == 'proto'
        proto_metric = getattr(self, 'umap_proto_metric', 'dotp')

        all_idx = np.unique(np.concatenate([heads, tails, neg_tails]))
        X_batch = X[all_idx]

        if hasattr(self.train_ds, 'conditions'):
            batch_cond = self.train_ds.conditions[torch.tensor(all_idx)].to(self.device)
        else:
            n_conds = len(self.model.scpoli_cvae.n_conditions)
            batch_cond = torch.zeros(len(all_idx), n_conds, dtype=torch.long, device=self.device)

        self.model.eval()
        z_all, _, _ = self.model.encoder_out({'x': X_batch, 'batch': batch_cond})
        idx_map = {int(v): i for i, v in enumerate(all_idx)}
        def _map(arr):
            return torch.tensor([idx_map[int(x)] for x in arr], device=self.device)

        if use_proto_sim:
            logits = self.model.prototypes(z_all)
            sa = F.softmax(logits / self.epsilon, dim=1)
            s_h = sa[_map(heads)]
            s_t = sa[_map(tails)]
            s_n = sa[_map(neg_tails)]
            _eps = 1e-4
            q_pos = self._proto_sim(s_h, s_t, proto_metric).clamp(_eps, 1 - _eps).cpu().numpy()
            q_neg = self._proto_sim(s_h, s_n, proto_metric).clamp(_eps, 1 - _eps).cpu().numpy()
        else:
            z_h = z_all[_map(heads)]
            z_t = z_all[_map(tails)]
            z_n = z_all[_map(neg_tails)]
            q_pos = loss_fn.compute_q(z_h, z_t).cpu().numpy()
            q_neg = loss_fn.compute_q(z_h, z_n).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: q_pos vs q_neg density histograms
        ax1.hist(q_pos, bins=bins, alpha=0.6, density=True, label=f'q_pos (mean={q_pos.mean():.4f})', color='steelblue', edgecolor='black', linewidth=0.3)
        ax1.hist(q_neg, bins=bins, alpha=0.6, density=True, label=f'q_neg (mean={q_neg.mean():.4f})', color='salmon', edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('q (latent similarity)')
        ax1.set_ylabel('density')
        ax1.set_title(f'q distribution — {len(q_pos)} edges')
        ax1.legend()

        # Right: p vs q scatter for positive edges
        ax2.scatter(weights, q_pos, s=1, alpha=0.3, c='steelblue', rasterized=True)
        ax2.set_xlabel('p (input affinity)')
        ax2.set_ylabel('q (latent similarity)')
        ax2.set_title('p vs q (positive edges)')
        # diagonal reference
        lims = [0, max(weights.max(), q_pos.max())]
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)

        plt.tight_layout()
        plt.show()
        return fig

    # ==================== Prototype Interpretation ====================

    @torch.no_grad()
    def decode_prototypes(self, batch_mode='mean'):
        """Decode all prototypes to gene expression space.

        Args:
            batch_mode: how to handle batch condition.
                'mean' - average over all batch conditions (default)
                'first' - use first batch condition
                'all' - return decoded for each batch condition

        Returns:
            If batch_mode='all': dict of {batch_id: np.ndarray [K, genes]}
            Otherwise: np.ndarray [K, genes]
        """
        protos = self.model.get_prototypes()  # [K, latent_dim]
        K = protos.shape[0]

        # Get unique batch conditions
        if hasattr(self.train_ds, 'conditions'):
            all_conds = self.train_ds.conditions.unique(dim=0)
        else:
            n_conds = len(self.model.scpoli_cvae.n_conditions)
            all_conds = torch.zeros(1, n_conds, dtype=torch.long, device=self.device)

        all_conds = all_conds.to(self.device)
        protos = protos.to(self.device)

        if batch_mode == 'all':
            results = {}
            for i, cond in enumerate(all_conds):
                cond_expanded = cond.unsqueeze(0).expand(K, -1)
                decoded = self.model.decode(protos, cond_expanded)
                results[i] = decoded.cpu().numpy()
            return results

        elif batch_mode == 'first':
            cond = all_conds[0].unsqueeze(0).expand(K, -1)
            decoded = self.model.decode(protos, cond)
            return decoded.cpu().numpy()

        else:  # mean
            decoded_sum = None
            for cond in all_conds:
                cond_expanded = cond.unsqueeze(0).expand(K, -1)
                decoded = self.model.decode(protos, cond_expanded)
                if decoded_sum is None:
                    decoded_sum = decoded
                else:
                    decoded_sum = decoded_sum + decoded
            return (decoded_sum / len(all_conds)).cpu().numpy()

    @torch.no_grad()
    def save_metacells(self, path=None):
        """Decode all prototypes using the globally dominant batch (most cells).

        1. Count total cells per batch across all cells
        2. Pick the batch with the most cells
        3. Decode all K prototypes with that single batch embedding
        4. Save as AnnData to dump_path/metacells.h5ad

        Returns:
            AnnData of shape [K, genes]
        """
        import anndata as ad
        import pandas as pd

        if path is None:
            path = os.path.join(self.get_dump_path(), 'metacells.h5ad')

        adata = self.train_ds.adata
        K = self.nmb_prototypes

        # Find globally dominant batch condition
        if hasattr(self.train_ds, 'conditions'):
            conditions = self.train_ds.conditions  # [N, n_cond_keys]
            unique_conds, counts = torch.unique(conditions, dim=0, return_counts=True)
            dominant_cond = unique_conds[counts.argmax()].unsqueeze(0).to(self.device)
            print(f"Dominant batch: {dominant_cond.cpu().tolist()} ({counts.max().item()} cells)")
        else:
            n_conds = len(self.model.scpoli_cvae.n_conditions)
            dominant_cond = torch.zeros(1, n_conds, dtype=torch.long, device=self.device)

        # Decode all prototypes with the dominant batch
        protos = self.model.get_prototypes().to(self.device)  # [K, latent_dim]
        cond_expanded = dominant_cond.expand(K, -1)           # [K, n_cond_keys]
        X = self.model.decode(protos, cond_expanded).cpu().numpy()  # [K, genes]

        # Build obs metadata
        assignments, _ = self._get_assignments()
        cell_counts = [(assignments == k).sum() for k in range(K)]
        obs = pd.DataFrame({
            'prototype_id': np.arange(K),
            'n_cells': cell_counts,
        }, index=[f'proto_{k}' for k in range(K)])

        mc_adata = ad.AnnData(
            X=X,
            obs=obs,
            var=adata.var.copy(),
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write to a temp path then rename to avoid h5py "file already open" errors
        # (stale handles from prior Colab cell runs cause ACC_TRUNC to fail on the same path)
        tmp_path = path + '.tmp'
        mc_adata.write_h5ad(tmp_path)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp_path, path)
        print(f"Saved metacells ({K} prototypes) to {path}")
        return mc_adata

    def label_prototypes(self, label_key):
        """Label each prototype based on majority vote of assigned cells.

        Args:
            label_key: column name in adata.obs for labels

        Returns:
            dict with:
                'labels': pd.Series mapping proto_id -> majority label
                'purity': pd.Series mapping proto_id -> purity (fraction of majority)
                'counts': pd.DataFrame with full crosstab
        """
        assignments, _ = self._get_assignments()
        labels = self.train_ds.adata.obs[label_key].values

        # Build crosstab: proto x label
        df = pd.DataFrame({'proto': assignments, 'label': labels})
        counts = pd.crosstab(df['proto'], df['label'])

        # Majority vote
        majority_labels = counts.idxmax(axis=1)
        purity = counts.max(axis=1) / counts.sum(axis=1)

        return {
            'labels': majority_labels,
            'purity': purity,
            'counts': counts,
        }

    def label_prototypes_soft(self, label_key):
        """Label every prototype via soft-assignment weighted voting.

        S = softmax(scores / epsilon).  Every proto gets a label (no filtering).
        Purity = max class weight / effective size of that proto.
        """
        import torch.nn.functional as F
        with torch.no_grad():
            z      = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            scores = self.model.prototypes(z)
            print(f"[label_prototypes_soft] using epsilon={self.epsilon:.6f}")
            S      = F.softmax(scores / self.epsilon, dim=1).cpu().numpy()  # (N, K)

        labels  = self.train_ds.adata.obs[label_key].values
        K       = S.shape[1]
        lbl_cat = pd.Categorical(labels)
        onehot  = np.eye(len(lbl_cat.categories))[lbl_cat.codes]  # (N, L)
        lbl_weights    = S.T @ onehot                              # (K, L)
        effective_sizes = S.sum(axis=0)                            # (K,)

        winner_codes   = lbl_weights.argmax(axis=1)
        majority_labels = pd.Series(
            lbl_cat.categories[winner_codes],
            index=np.arange(K),
            name=label_key,
        )
        soft_purity = lbl_weights.max(axis=1) / np.maximum(effective_sizes, 1e-10)
        return {
            'labels':          majority_labels,
            'purity':          pd.Series(soft_purity, index=np.arange(K)),
            'effective_sizes': effective_sizes,
        }

    def prototype_dge(self, niche_key, celltype_key, method='wilcoxon'):
        """Compute DGE on decoded prototypes per niche within each celltype.

        For each celltype, compares prototypes labeled as niche X vs rest.
        Saves ALL genes (no filtering).

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            method: scanpy rank_genes_groups method ('wilcoxon', 't-test', etc.)

        Returns:
            pd.DataFrame with columns:
                celltype, niche, gene, logfoldchange, pval, pval_adj, score
        """
        import scanpy as sc

        # Decode prototypes and create AnnData
        decoded = self.decode_prototypes(batch_mode='mean')
        proto_ad = sc.AnnData(decoded)
        proto_ad.var_names = self.train_ds.adata.var_names

        # Label prototypes
        niche_labels = self.label_prototypes(niche_key)['labels']
        ct_labels = self.label_prototypes(celltype_key)['labels']

        proto_ad.obs['niche'] = niche_labels.values
        proto_ad.obs['celltype'] = ct_labels.values
        proto_ad.obs['proto_id'] = niche_labels.index.values

        rows = []
        for ct in proto_ad.obs['celltype'].unique():
            ct_ad = proto_ad[proto_ad.obs['celltype'] == ct].copy()
            if ct_ad.n_obs < 3:
                continue

            niches = ct_ad.obs['niche'].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs['niche'] == niche).sum()
                n_neg = (ct_ad.obs['niche'] != niche).sum()
                if n_pos < 1 or n_neg < 1:
                    continue

                # Create binary group
                ct_ad.obs['_group'] = np.where(ct_ad.obs['niche'] == niche, 'pos', 'rest')

                try:
                    sc.tl.rank_genes_groups(
                        ct_ad, '_group', groups=['pos'], reference='rest',
                        method=method, use_raw=False
                    )
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")
                    continue

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        result = result.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        return result[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

    def singlecell_dge(self, niche_key, celltype_key, method='wilcoxon'):
        """Compute DGE on single cells per niche within each celltype.

        Same format as prototype_dge for direct comparison.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            method: scanpy rank_genes_groups method

        Returns:
            pd.DataFrame with same columns as prototype_dge
        """
        import scanpy as sc

        adata = self.train_ds.adata.copy()

        rows = []
        for ct in adata.obs[celltype_key].unique():
            ct_ad = adata[adata.obs[celltype_key] == ct].copy()
            if ct_ad.n_obs < 3:
                continue

            niches = ct_ad.obs[niche_key].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs[niche_key] == niche).sum()
                n_neg = (ct_ad.obs[niche_key] != niche).sum()
                if n_pos < 2 or n_neg < 2:
                    continue

                ct_ad.obs['_group'] = np.where(ct_ad.obs[niche_key] == niche, 'pos', 'rest')

                try:
                    sc.tl.rank_genes_groups(
                        ct_ad, '_group', groups=['pos'], reference='rest',
                        method=method, use_raw=False
                    )
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")
                    continue

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        result = result.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        return result[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

    def prototype_stats(self, niche_key, celltype_key):
        """Compute statistical metrics for prototype clustering.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels

        Returns:
            dict with:
                'niche_purity': purity of niche labels per prototype
                'celltype_purity': purity of celltype labels per prototype
                'chi2_niche': chi2 test for niche vs prototype association
                'chi2_celltype': chi2 test for celltype vs prototype association
                'silhouette_niche': silhouette score for niche separation (per celltype)
                'silhouette_celltype': silhouette score for celltype separation
        """
        from scipy.stats import chi2_contingency
        from sklearn.metrics import silhouette_score

        assignments, _ = self._get_assignments()
        obs = self.train_ds.adata.obs

        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        # Purity
        niche_info = self.label_prototypes(niche_key)
        ct_info = self.label_prototypes(celltype_key)

        # Chi2 tests
        niche_tab = pd.crosstab(assignments, niche_labels)
        ct_tab = pd.crosstab(assignments, ct_labels)

        chi2_niche = chi2_contingency(niche_tab)[1] if niche_tab.shape[0] > 1 and niche_tab.shape[1] > 1 else np.nan
        chi2_ct = chi2_contingency(ct_tab)[1] if ct_tab.shape[0] > 1 and ct_tab.shape[1] > 1 else np.nan

        # Silhouette on embeddings
        with torch.no_grad():
            z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
        z_np = z.cpu().numpy()

        # Overall celltype silhouette
        if len(np.unique(ct_labels)) >= 2:
            sil_ct = silhouette_score(z_np, ct_labels)
        else:
            sil_ct = np.nan

        # Niche silhouette per celltype
        sil_niche_per_ct = {}
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            niches_ct = niche_labels[mask]
            if len(np.unique(niches_ct)) >= 2:
                sil_niche_per_ct[ct] = silhouette_score(z_np[mask], niches_ct)
            else:
                sil_niche_per_ct[ct] = np.nan

        return {
            'niche_purity_mean': niche_info['purity'].mean(),
            'niche_purity_per_proto': niche_info['purity'].to_dict(),
            'celltype_purity_mean': ct_info['purity'].mean(),
            'celltype_purity_per_proto': ct_info['purity'].to_dict(),
            'chi2_niche_pval': chi2_niche,
            'chi2_celltype_pval': chi2_ct,
            'silhouette_celltype': sil_ct,
            'silhouette_niche_per_ct': sil_niche_per_ct,
            'silhouette_niche_mean': np.nanmean(list(sil_niche_per_ct.values())),
        }

    def eval_prototypes(self, niche_key=None, celltype_key=None, save_dir=None):
        """Run all prototype evaluation and save results.

        Args:
            niche_key: column for niche labels. Defaults to 'niches_2D' if present.
            celltype_key: column for celltype labels. Defaults to dataset.label_key.
            save_dir: directory to save results. Defaults to get_dump_path().

        Returns:
            dict with all results
        """
        import json

        if save_dir is None:
            save_dir = self.get_dump_path()
        os.makedirs(save_dir, exist_ok=True)

        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating prototypes: niche_key={niche_key}, celltype_key={celltype_key}")

        results = {}

        # 1. Decode prototypes
        print("Decoding prototypes...")
        decoded = self.decode_prototypes(batch_mode='mean')
        np.save(os.path.join(save_dir, 'decoded_prototypes.npy'), decoded)
        results['decoded_shape'] = decoded.shape

        # 2. Label prototypes
        print("Labeling prototypes...")
        niche_labels = self.label_prototypes(niche_key)
        ct_labels = self.label_prototypes(celltype_key)

        label_df = pd.DataFrame({
            'proto_id': niche_labels['labels'].index,
            'niche': niche_labels['labels'].values,
            'niche_purity': niche_labels['purity'].values,
            'celltype': ct_labels['labels'].values,
            'celltype_purity': ct_labels['purity'].values,
        })
        label_df.to_csv(os.path.join(save_dir, 'prototype_labels.csv'), index=False)
        results['n_prototypes'] = len(label_df)

        # 3. Prototype DGE
        print("Computing prototype DGE...")
        proto_dge = self.prototype_dge(niche_key, celltype_key)
        proto_dge.to_csv(os.path.join(save_dir, 'prototype_dge.csv'), index=False)
        results['proto_dge_rows'] = len(proto_dge)

        # 4. Single-cell DGE
        print("Computing single-cell DGE...")
        sc_dge = self.singlecell_dge(niche_key, celltype_key)
        sc_dge.to_csv(os.path.join(save_dir, 'singlecell_dge.csv'), index=False)
        results['sc_dge_rows'] = len(sc_dge)

        # 5. Statistical tests
        print("Computing statistics...")
        stats = self.prototype_stats(niche_key, celltype_key)
        results.update({k: v for k, v in stats.items() if not isinstance(v, dict)})

        # Save stats
        stats_save = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                      for k, v in stats.items()}
        with open(os.path.join(save_dir, 'prototype_stats.json'), 'w') as f:
            json.dump(stats_save, f, indent=2, default=float)

        # 6. Save summary
        with open(os.path.join(save_dir, 'eval_summary.json'), 'w') as f:
            json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                       for k, v in results.items()}, f, indent=2)

        print(f"Results saved to {save_dir}")
        print(f"  - decoded_prototypes.npy: {decoded.shape}")
        print(f"  - prototype_labels.csv: {len(label_df)} prototypes")
        print(f"  - prototype_dge.csv: {len(proto_dge)} rows")
        print(f"  - singlecell_dge.csv: {len(sc_dge)} rows")
        print(f"  - prototype_stats.json")

        return results

    def eval_singlecell(self, niche_key=None, celltype_key=None):
        """Run DGE and stats on raw single cells (ground truth baseline).

        Saves results to <save_dir>/gt/ folder.

        Args:
            niche_key: column for niche labels. Defaults to 'niches_2D'.
            celltype_key: column for celltype labels. Defaults to dataset.label_key.

        Returns:
            dict with dge DataFrame and stats
        """
        import json
        import scanpy as sc
        from scipy.stats import chi2_contingency
        from sklearn.metrics import silhouette_score

        save_dir = os.path.join(self.get_save_dir(), 'gt')
        os.makedirs(save_dir, exist_ok=True)

        adata = self.train_ds.adata
        obs = adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating single cells: niche_key={niche_key}, celltype_key={celltype_key}")

        # 1. DGE
        print("Computing single-cell DGE...")
        sc_dge = self.singlecell_dge(niche_key, celltype_key)
        sc_dge.to_csv(os.path.join(save_dir, 'singlecell_dge.csv'), index=False)

        # 2. Stats
        print("Computing statistics...")
        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        # Chi2
        niche_tab = pd.crosstab(ct_labels, niche_labels)
        chi2_pval = chi2_contingency(niche_tab)[1] if niche_tab.shape[0] > 1 and niche_tab.shape[1] > 1 else np.nan

        # Silhouette on PCA
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
        X = adata.obsm['X_pca']

        sil_ct = silhouette_score(X, ct_labels) if len(np.unique(ct_labels)) >= 2 else np.nan
        sil_niche_per_ct = {}
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            if len(np.unique(niche_labels[mask])) >= 2:
                sil_niche_per_ct[ct] = silhouette_score(X[mask], niche_labels[mask])

        stats = {
            'chi2_niche_celltype_pval': chi2_pval,
            'silhouette_celltype': sil_ct,
            'silhouette_niche_per_ct': {str(k): v for k, v in sil_niche_per_ct.items()},
            'silhouette_niche_mean': np.nanmean(list(sil_niche_per_ct.values())),
            'n_cells': len(adata),
        }
        with open(os.path.join(save_dir, 'singlecell_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=float)

        print(f"Saved to {save_dir}: singlecell_dge.csv ({len(sc_dge)} rows), singlecell_stats.json")
        return {'dge': sc_dge, 'stats': stats}

    def compare_dge_plots(self, ct=None, niche=None):
        """Load DGE results and plot volcano + logFC distribution comparison.

        Args:
            ct: celltype to show in volcano (if None, picks first available)
            niche: niche to show in volcano (if None, picks first available)

        Returns:
            dict with proto_dge, sc_dge DataFrames
        """
        import matplotlib.pyplot as plt

        # Load DGE files
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path):
            print(f"Run t.eval_prototypes() first. Missing: {proto_path}")
            return None
        if not os.path.exists(sc_path):
            print(f"Run t.eval_singlecell() first. Missing: {sc_path}")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        # Pick ct/niche if not specified
        if ct is None:
            ct = proto_dge['celltype'].iloc[0]
        if niche is None:
            niche = proto_dge[proto_dge['celltype'] == ct]['niche'].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # --- Row 1: Volcano plots ---
        for ax, dge, title in [(axes[0, 0], sc_dge, 'Single-cell'), (axes[0, 1], proto_dge, 'Metacell')]:
            d = dge[(dge['celltype'] == ct) & (dge['niche'] == niche)].copy()
            if len(d) == 0:
                ax.set_title(f'{title}: No data for {ct}/{niche}')
                continue
            d['neg_log_p'] = -np.log10(d['pval'].clip(lower=1e-300))

            ax.scatter(d['logfoldchange'], d['neg_log_p'], s=3, alpha=0.4, c='gray')
            sig = d[(d['pval_adj'] < 0.05) & (abs(d['logfoldchange']) > 0.5)]
            ax.scatter(sig['logfoldchange'], sig['neg_log_p'], s=8, c='red', alpha=0.7)

            ax.set_xlabel('Log Fold Change')
            ax.set_ylabel('-log10(p)')
            ax.set_title(f'{title}: {ct} / {niche}\n({len(sig)} sig genes)')
            ax.axhline(-np.log10(0.05), ls='--', c='blue', alpha=0.3)
            ax.axvline(0, ls='--', c='black', alpha=0.3)

        # --- Row 2: LogFC distributions ---
        ax = axes[1, 0]
        ax.hist(sc_dge['logfoldchange'].dropna(), bins=50, alpha=0.6,
                label=f'Single-cell (std={sc_dge["logfoldchange"].std():.2f})', density=True)
        ax.hist(proto_dge['logfoldchange'].dropna(), bins=50, alpha=0.6,
                label=f'Metacell (std={proto_dge["logfoldchange"].std():.2f})', density=True)
        ax.set_xlabel('Log Fold Change')
        ax.set_ylabel('Density')
        ax.set_title('LogFC Distribution (all genes)')
        ax.legend()

        # --- Row 2 right: Signal comparison scatter ---
        ax = axes[1, 1]
        rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 10 or len(p_grp) < 5:
                continue
            rows.append({
                'sc_signal': s_grp.nlargest(20, 'logfoldchange')['logfoldchange'].mean(),
                'proto_signal': p_grp.nlargest(20, 'logfoldchange')['logfoldchange'].mean(),
            })
        if rows:
            df_sig = pd.DataFrame(rows)
            ax.scatter(df_sig['sc_signal'], df_sig['proto_signal'], s=40, alpha=0.7)
            lim = max(df_sig['sc_signal'].max(), df_sig['proto_signal'].max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
            pct = (df_sig['proto_signal'] > df_sig['sc_signal']).mean() * 100
            ax.set_title(f'Signal Strength\n(Metacell stronger in {pct:.0f}% cases)')
        ax.set_xlabel('Single-cell signal (mean top20 logFC)')
        ax.set_ylabel('Metacell signal (mean top20 logFC)')

        plt.tight_layout()
        plt.show()

        # Compute comparison scores
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo
        score_rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 10 or len(p_grp) < 5:
                continue
            p_genes = p_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            merged = p_grp.merge(s_grp, on='gene', suffixes=('_proto', '_sc'))
            corr = merged['logfoldchange_proto'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan
            score_rows.append({
                'celltype': c, 'niche': n,
                'jaccard': jaccard(p_genes, s_genes),
                'rbo': rbo(p_genes, s_genes),
                'logfc_corr': corr,
                'n_proto': len(p_grp), 'n_sc': len(s_grp),
            })
        scores_df = pd.DataFrame(score_rows)

        print(f"\nDGE Comparison Scores:")
        print(f"  Jaccard:   mean={scores_df['jaccard'].mean():.3f}, median={scores_df['jaccard'].median():.3f}")
        print(f"  RBO:       mean={scores_df['rbo'].mean():.3f}, median={scores_df['rbo'].median():.3f}")
        print(f"  LogFC corr: mean={scores_df['logfc_corr'].mean():.3f}, median={scores_df['logfc_corr'].median():.3f}")

        return {'proto_dge': proto_dge, 'sc_dge': sc_dge, 'scores': scores_df}

    def compare_dge_all_niches(self, celltype=None):
        """Plot volcano comparison for all niches (one row per niche).

        Args:
            celltype: filter to specific celltype (None = all)

        Returns:
            dict with proto_dge, sc_dge, scores DataFrames
        """
        import matplotlib.pyplot as plt

        # Load DGE files
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path) or not os.path.exists(sc_path):
            print("Run t.eval_prototypes() and t.eval_singlecell() first")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        # Get unique celltype/niche combos
        if celltype:
            proto_dge = proto_dge[proto_dge['celltype'] == celltype]
            sc_dge = sc_dge[sc_dge['celltype'] == celltype]

        combos = proto_dge.groupby(['celltype', 'niche']).size().reset_index()[['celltype', 'niche']]
        n_combos = len(combos)

        if n_combos == 0:
            print("No data found")
            return None

        # Create figure: 2 columns (single-cell, metacell), n_combos rows
        fig, axes = plt.subplots(n_combos, 2, figsize=(10, 3 * n_combos))
        if n_combos == 1:
            axes = axes.reshape(1, -1)

        for i, (_, row) in enumerate(combos.iterrows()):
            ct, niche = row['celltype'], row['niche']

            for j, (dge, title) in enumerate([(sc_dge, 'Single-cell'), (proto_dge, 'Metacell')]):
                ax = axes[i, j]
                d = dge[(dge['celltype'] == ct) & (dge['niche'] == niche)].copy()

                if len(d) == 0:
                    ax.set_title(f'{title}: No data')
                    continue

                d['neg_log_p'] = -np.log10(d['pval'].clip(lower=1e-300))

                ax.scatter(d['logfoldchange'], d['neg_log_p'], s=2, alpha=0.4, c='gray')
                sig = d[(d['pval_adj'] < 0.05) & (abs(d['logfoldchange']) > 0.5)]
                ax.scatter(sig['logfoldchange'], sig['neg_log_p'], s=6, c='red', alpha=0.7)

                ax.set_xlabel('LogFC')
                ax.set_ylabel('-log10(p)')
                ax.set_title(f'{title}: {ct[:15]} / {niche[:20]}\n({len(sig)} sig)')
                ax.axhline(-np.log10(0.05), ls='--', c='blue', alpha=0.3)
                ax.axvline(0, ls='--', c='black', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Compute scores
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo
        score_rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 5 or len(p_grp) < 3:
                continue
            p_genes = p_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            merged = p_grp.merge(s_grp, on='gene', suffixes=('_proto', '_sc'))
            corr = merged['logfoldchange_proto'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan
            score_rows.append({
                'celltype': c, 'niche': n,
                'jaccard': jaccard(p_genes, s_genes),
                'rbo': rbo(p_genes, s_genes),
                'logfc_corr': corr,
            })
        scores_df = pd.DataFrame(score_rows)

        print(f"\nScores: jaccard={scores_df['jaccard'].mean():.3f}, rbo={scores_df['rbo'].mean():.3f}, corr={scores_df['logfc_corr'].mean():.3f}")

        return {'proto_dge': proto_dge, 'sc_dge': sc_dge, 'scores': scores_df}

    def eval_baseline_clustering(self, name, niche_key=None, celltype_key=None):
        """Evaluate a baseline clustering method (spectral, seacells, etc.)

        Loads clustering from baseline folder, computes average gene expression
        per cluster, runs DGE, and saves results.

        Args:
            name: baseline name (e.g., 'spectral_K50', 'seacells_K50')
            niche_key: column for niche labels
            celltype_key: column for celltype labels

        Returns:
            dict with dge DataFrame and stats
        """
        import scanpy as sc

        # Paths
        baseline_dir = os.path.join(self.get_save_dir(), name)
        clusters_path = os.path.join(baseline_dir, 'clusters.npz')

        if not os.path.exists(clusters_path):
            print(f"Baseline not found: {clusters_path}")
            print(f"Run t.eval_spectral() or t.eval_seacells() first")
            return None

        # Load clustering
        data = np.load(clusters_path, allow_pickle=True)
        assignments = data['assignments']

        # Setup keys
        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating baseline '{name}': niche_key={niche_key}, celltype_key={celltype_key}")

        # Compute average gene expression per cluster
        X = self.train_ds.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        n_clusters = len(np.unique(assignments))
        avg_expr = np.zeros((n_clusters, X.shape[1]))
        cluster_ids = np.unique(assignments)

        for i, c in enumerate(cluster_ids):
            mask = assignments == c
            avg_expr[i] = X[mask].mean(axis=0)

        # Create AnnData for clusters
        cluster_ad = sc.AnnData(avg_expr)
        cluster_ad.var_names = self.train_ds.adata.var_names

        # Label clusters by majority vote
        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        cluster_niche = []
        cluster_ct = []
        for c in cluster_ids:
            mask = assignments == c
            # Majority vote
            vals, counts = np.unique(niche_labels[mask], return_counts=True)
            cluster_niche.append(vals[counts.argmax()])
            vals, counts = np.unique(ct_labels[mask], return_counts=True)
            cluster_ct.append(vals[counts.argmax()])

        cluster_ad.obs['niche'] = cluster_niche
        cluster_ad.obs['celltype'] = cluster_ct
        cluster_ad.obs['cluster_id'] = cluster_ids

        # Run DGE (same logic as prototype_dge)
        rows = []
        for ct in cluster_ad.obs['celltype'].unique():
            ct_ad = cluster_ad[cluster_ad.obs['celltype'] == ct].copy()
            if ct_ad.n_obs < 3:
                continue
            niches = ct_ad.obs['niche'].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs['niche'] == niche).sum()
                n_neg = (ct_ad.obs['niche'] != niche).sum()
                if n_pos < 1 or n_neg < 1:
                    continue

                ct_ad.obs['_group'] = np.where(ct_ad.obs['niche'] == niche, 'pos', 'rest')
                try:
                    sc.tl.rank_genes_groups(ct_ad, '_group', groups=['pos'], reference='rest',
                                            method='wilcoxon', use_raw=False)
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")

        if not rows:
            print("No DGE results")
            return None

        dge_df = pd.concat(rows, ignore_index=True)
        dge_df = dge_df.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        dge_df = dge_df[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

        # Save
        dge_df.to_csv(os.path.join(baseline_dir, 'baseline_dge.csv'), index=False)
        np.save(os.path.join(baseline_dir, 'avg_expression.npy'), avg_expr)

        print(f"Saved to {baseline_dir}: baseline_dge.csv ({len(dge_df)} rows), avg_expression.npy")

        return {'dge': dge_df, 'avg_expr': avg_expr, 'cluster_ad': cluster_ad}

    def compare_all_methods(self, niche_key=None, celltype_key=None, name_map=None):
        """Compare DGE results across all methods: prototype, baselines, single-cell.

        Computes jaccard/rbo/logfc_corr for each method vs single-cell ground truth.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            name_map: dict mapping folder names to display names, e.g.
                      {'scproto_ds-s28f_v31': 'Proto UMAP', 'spectral_K50': 'Spectral'}

        Returns:
            DataFrame with method, mean_jaccard, mean_rbo, mean_logfc_corr, logfc_std
        """
        import matplotlib.pyplot as plt
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo

        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        # Load single-cell ground truth
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')
        if not os.path.exists(sc_path):
            print("Run t.eval_singlecell() first")
            return None
        sc_dge = pd.read_csv(sc_path)

        # Find all methods in model dir
        methods = {}

        for entry in os.listdir(self.get_save_dir()):
            entry_dir = os.path.join(self.get_save_dir(), entry)
            if not os.path.isdir(entry_dir):
                continue

            # Check for prototype_dge.csv (trained models)
            proto_path = os.path.join(entry_dir, 'prototype_dge.csv')
            if os.path.exists(proto_path):
                methods[entry] = pd.read_csv(proto_path)
                continue

            # Check for baseline_dge.csv (baselines)
            baseline_path = os.path.join(entry_dir, 'baseline_dge.csv')
            if os.path.exists(baseline_path):
                methods[entry] = pd.read_csv(baseline_path)

        if not methods:
            print("No methods found. Run t.eval_prototypes() and t.eval_baseline_clustering() first")
            return None

        # Compute scores for each method
        results = []
        for method_name, method_dge in methods.items():
            scores = []
            for (ct, niche), m_grp in method_dge.groupby(['celltype', 'niche']):
                s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
                if len(s_grp) < 10 or len(m_grp) < 3:
                    continue

                m_genes = m_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
                s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
                merged = m_grp.merge(s_grp, on='gene', suffixes=('_method', '_sc'))
                corr = merged['logfoldchange_method'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan

                scores.append({
                    'jaccard': jaccard(m_genes, s_genes),
                    'rbo': rbo(m_genes, s_genes),
                    'logfc_corr': corr,
                })

            if scores:
                scores_df = pd.DataFrame(scores)
                results.append({
                    'method': method_name,
                    'mean_jaccard': scores_df['jaccard'].mean(),
                    'mean_rbo': scores_df['rbo'].mean(),
                    'mean_logfc_corr': scores_df['logfc_corr'].mean(),
                    'logfc_std': method_dge['logfoldchange'].std(),
                    'n_comparisons': len(scores),
                })

        results_df = pd.DataFrame(results).sort_values('mean_jaccard', ascending=False)

        # Apply name mapping
        if name_map:
            results_df['display_name'] = results_df['method'].map(lambda x: name_map.get(x, x))
        else:
            results_df['display_name'] = results_df['method']

        # Plot comparison
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        x = range(len(results_df))
        names = results_df['display_name'].tolist()

        for ax, col, title in zip(axes, ['mean_jaccard', 'mean_rbo', 'mean_logfc_corr', 'logfc_std'],
                                   ['Jaccard ↑', 'RBO ↑', 'LogFC Corr ↑', 'LogFC Std ↑']):
            bars = ax.bar(x, results_df[col], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_title(title)
            # Highlight best
            best_idx = results_df[col].idxmax() if '↑' in title else results_df[col].idxmin()
            best_pos = results_df.index.get_loc(best_idx)
            bars[best_pos].set_color('green')

        plt.tight_layout()
        plt.show()

        print("\nMethod Comparison (vs single-cell ground truth):")
        print(results_df.to_string(index=False))

        return results_df

    def find_discovered_genes(self, pval_thr=0.05, logfc_thr=0.5):
        """Find genes significant in metacell but not in single-cell.

        These are genes that metacells "discover" by reducing noise/sparsity.

        Args:
            pval_thr: adjusted p-value threshold for significance
            logfc_thr: absolute log fold change threshold

        Returns:
            DataFrame with discovered genes and their stats
        """
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path) or not os.path.exists(sc_path):
            print("Run t.eval_prototypes() and t.eval_singlecell() first")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        rows = []
        for (ct, niche), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
            if len(s_grp) == 0:
                continue

            # Significant in prototype
            p_sig = p_grp[(p_grp['pval_adj'] < pval_thr) & (abs(p_grp['logfoldchange']) > logfc_thr)]

            # Significant genes in single-cell
            s_sig_genes = set(s_grp[(s_grp['pval_adj'] < pval_thr) & (abs(s_grp['logfoldchange']) > logfc_thr)]['gene'])

            # Metacell-only genes
            for _, gene_row in p_sig.iterrows():
                if gene_row['gene'] not in s_sig_genes:
                    sc_match = s_grp[s_grp['gene'] == gene_row['gene']]
                    rows.append({
                        'celltype': ct,
                        'niche': niche,
                        'gene': gene_row['gene'],
                        'proto_logfc': gene_row['logfoldchange'],
                        'proto_pval': gene_row['pval_adj'],
                        'sc_logfc': sc_match['logfoldchange'].values[0] if len(sc_match) > 0 else np.nan,
                        'sc_pval': sc_match['pval_adj'].values[0] if len(sc_match) > 0 else np.nan,
                    })

        df = pd.DataFrame(rows)

        if len(df) == 0:
            print("No metacell-discovered genes found")
            return df

        # Sort by proto_logfc
        df = df.sort_values('proto_logfc', ascending=False).reset_index(drop=True)

        # Save
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        df.to_csv(save_path, index=False)

        print(f"Found {len(df)} metacell-discovered genes")
        print(f"Saved to {save_path}")
        print(f"\nTop 15 by logFC:")
        print(df.head(15).to_string(index=False))

        return df

    def plot_discovered_genes(self, top_n=20, figsize=(14, 10)):
        """Plot discovered genes: bar chart + volcano highlighting.

        Args:
            top_n: number of top genes to show in bar chart
            figsize: figure size

        Returns:
            Figure and discovered genes DataFrame
        """
        import matplotlib.pyplot as plt

        # Load discovered genes
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- Top-left: Top upregulated genes ---
        ax = axes[0, 0]
        top_up = df[df['proto_logfc'] > 0].nlargest(top_n, 'proto_logfc')
        colors = ['#2ecc71' if 'Tumor' in n else '#3498db' for n in top_up['niche']]
        bars = ax.barh(range(len(top_up)), top_up['proto_logfc'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_up)))
        ax.set_yticklabels([f"{row['gene']} ({row['niche'][:15]})" for _, row in top_up.iterrows()], fontsize=8)
        ax.set_xlabel('Log Fold Change (metacell)')
        ax.set_title(f'Top {len(top_up)} Upregulated\n(green=Tumor surface, blue=Desmoplastic)')
        ax.invert_yaxis()

        # --- Top-right: Top downregulated genes ---
        ax = axes[0, 1]
        top_down = df[df['proto_logfc'] < 0].nsmallest(top_n, 'proto_logfc')
        colors = ['#e74c3c' if 'Desmo' in n else '#9b59b6' for n in top_down['niche']]
        bars = ax.barh(range(len(top_down)), top_down['proto_logfc'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_down)))
        ax.set_yticklabels([f"{row['gene']} ({row['niche'][:15]})" for _, row in top_down.iterrows()], fontsize=8)
        ax.set_xlabel('Log Fold Change (metacell)')
        ax.set_title(f'Top {len(top_down)} Downregulated\n(red=Desmoplastic, purple=other)')
        ax.invert_yaxis()

        # --- Bottom-left: Metacell vs Single-cell logFC scatter ---
        ax = axes[1, 0]
        valid = df.dropna(subset=['sc_logfc'])
        ax.scatter(valid['sc_logfc'], valid['proto_logfc'], s=30, alpha=0.6, c='steelblue')

        # Label top genes
        for _, row in valid.nlargest(5, 'proto_logfc').iterrows():
            ax.annotate(row['gene'], (row['sc_logfc'], row['proto_logfc']), fontsize=7)
        for _, row in valid.nsmallest(5, 'proto_logfc').iterrows():
            ax.annotate(row['gene'], (row['sc_logfc'], row['proto_logfc']), fontsize=7)

        lim = max(abs(valid['sc_logfc']).max(), abs(valid['proto_logfc']).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        ax.axhline(0, c='gray', lw=0.5)
        ax.axvline(0, c='gray', lw=0.5)
        ax.set_xlabel('Single-cell logFC (not significant)')
        ax.set_ylabel('Metacell logFC (significant)')
        ax.set_title('Metacell amplifies weak signals')

        # --- Bottom-right: P-value comparison ---
        ax = axes[1, 1]
        valid = df.dropna(subset=['sc_pval'])
        ax.scatter(-np.log10(valid['sc_pval'].clip(lower=1e-10)),
                   -np.log10(valid['proto_pval'].clip(lower=1e-10)),
                   s=30, alpha=0.6, c='coral')

        ax.axhline(-np.log10(0.05), c='red', ls='--', alpha=0.5, label='p=0.05')
        ax.axvline(-np.log10(0.05), c='blue', ls='--', alpha=0.5)
        ax.set_xlabel('Single-cell -log10(p)')
        ax.set_ylabel('Metacell -log10(p)')
        ax.set_title('Metacell finds significance\n(top-left = discovered)')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        return fig, df

    def discovered_genes_table(self, top_n=30):
        """Generate a formatted table of discovered genes for presentation.

        Args:
            top_n: number of genes to include

        Returns:
            DataFrame formatted for presentation
        """
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        # Select top up and down
        top_up = df[df['proto_logfc'] > 0].nlargest(top_n // 2, 'proto_logfc')
        top_down = df[df['proto_logfc'] < 0].nsmallest(top_n // 2, 'proto_logfc')
        selected = pd.concat([top_up, top_down])

        # Format for presentation
        table = selected[['gene', 'niche', 'proto_logfc', 'proto_pval', 'sc_logfc', 'sc_pval']].copy()
        table['proto_logfc'] = table['proto_logfc'].round(2)
        table['sc_logfc'] = table['sc_logfc'].round(2)
        table['proto_pval'] = table['proto_pval'].apply(lambda x: f"{x:.1e}")
        table['sc_pval'] = table['sc_pval'].apply(lambda x: f"{x:.1e}" if pd.notna(x) else "NA")
        table['direction'] = table['proto_logfc'].apply(lambda x: '↑' if x > 0 else '↓')

        table = table.rename(columns={
            'gene': 'Gene',
            'niche': 'Niche',
            'proto_logfc': 'Metacell LogFC',
            'proto_pval': 'Metacell p-val',
            'sc_logfc': 'Single-cell LogFC',
            'sc_pval': 'Single-cell p-val',
            'direction': '↑↓'
        })

        return table[['Gene', '↑↓', 'Niche', 'Metacell LogFC', 'Metacell p-val', 'Single-cell LogFC', 'Single-cell p-val']]

    def plot_gene_comparison(self, genes=None, top_n=10):
        """Plot side-by-side comparison of specific genes: metacell vs single-cell.

        Args:
            genes: list of gene names to plot. If None, uses top discovered genes.
            top_n: if genes is None, use top_n discovered genes

        Returns:
            Figure
        """
        import matplotlib.pyplot as plt

        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        if genes is None:
            # Get top by absolute logfc
            top_up = df[df['proto_logfc'] > 0].nlargest(top_n // 2, 'proto_logfc')
            top_down = df[df['proto_logfc'] < 0].nsmallest(top_n // 2, 'proto_logfc')
            selected = pd.concat([top_up, top_down])
        else:
            selected = df[df['gene'].isin(genes)]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(selected))
        width = 0.35

        bars1 = ax.bar(x - width/2, selected['proto_logfc'], width, label='Metacell (significant)', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, selected['sc_logfc'], width, label='Single-cell (not sig)', color='#95a5a6', alpha=0.8)

        ax.set_ylabel('Log Fold Change')
        ax.set_title('Discovered Genes: Metacell vs Single-cell')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['gene']}\n({row['niche'][:10]})" for _, row in selected.iterrows()],
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.axhline(0, c='black', lw=0.5)

        # Add significance markers
        for i, (_, row) in enumerate(selected.iterrows()):
            if row['proto_pval'] < 0.01:
                ax.text(i - width/2, row['proto_logfc'] + 0.05 * np.sign(row['proto_logfc']), '**', ha='center', fontsize=10)
            elif row['proto_pval'] < 0.05:
                ax.text(i - width/2, row['proto_logfc'] + 0.05 * np.sign(row['proto_logfc']), '*', ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

        return fig

    def plot_method_comparison_summary(self):
        """Create a summary figure comparing all methods for presentation.

        Returns:
            Figure
        """
        import matplotlib.pyplot as plt

        # Run compare_all_methods but capture the data
        result = self.compare_all_methods()
        if result is None:
            return None

        # Additional summary plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Load discovered genes count
        disc_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        n_discovered = len(pd.read_csv(disc_path)) if os.path.exists(disc_path) else 0

        # Panel 1: Method comparison bar
        ax = axes[0]
        methods = result['method'].tolist()
        jaccard = result['mean_jaccard'].tolist()
        colors = ['#2ecc71' if 'proto' in m.lower() else '#3498db' for m in methods]
        ax.bar(methods, jaccard, color=colors, alpha=0.8)
        ax.set_ylabel('Jaccard Score')
        ax.set_title('Agreement with Ground Truth')
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Panel 2: Signal strength
        ax = axes[1]
        logfc_std = result['logfc_std'].tolist()
        ax.bar(methods, logfc_std, color=colors, alpha=0.8)
        ax.set_ylabel('LogFC Std')
        ax.set_title('Signal Strength (higher=cleaner)')
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Panel 3: Discovered genes summary
        ax = axes[2]
        ax.text(0.5, 0.7, f"{n_discovered}", fontsize=48, ha='center', va='center', transform=ax.transAxes, fontweight='bold', color='#2ecc71')
        ax.text(0.5, 0.35, "Genes Discovered", fontsize=14, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.15, "(significant in metacell,\nnot in single-cell)", fontsize=10, ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.axis('off')

        plt.tight_layout()
        plt.show()

        return fig