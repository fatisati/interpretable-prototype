import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------

def responsibilities_from_gaussians(z_mu, p_mu, pi=None, log_sigma2_p=0.0, return_logits=False):
    """
    q(c|x) ∝ π_k * N(z_mu; p_mu_k, σ_p^2 I)
    z_mu: (B,d), p_mu: (K,d), pi: (K,)
    returns: a (B,K)
    """
    # squared euclidean distances between each sample mean and each prototype mean
    sqdist = torch.cdist(z_mu, p_mu) ** 2            # (B,K)
    logits = -0.5 * sqdist / torch.exp(log_sigma2_p) # (B,K)  (log-likelihood up to const)
    if pi is not None:
        logits = logits + torch.log(pi[None, :] + 1e-8)
    if return_logits:
        return logits
    return F.softmax(logits, dim=1)                  # (B,K)

def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) )
    all shapes (..., d) -> returns (...,) (sum over d)
    """
    var_q = torch.exp(logvar_q)
    inv_var_p = torch.exp(-logvar_p)
    d = mu_q.shape[-1]
    tr   = (var_q * inv_var_p).sum(-1)
    quad = ((mu_p - mu_q)**2 * inv_var_p).sum(-1)
    logd = (logvar_p - logvar_q).sum(-1)
    return 0.5 * (tr + quad - d + logd)

def rbf_similarity(z, p, sigma2=1.0, normalize=False):
    """
    Compute Gaussian RBF similarities between embeddings and prototypes.

    Args:
        z: Tensor of shape (B, d)   - batch of embeddings
        p: Tensor of shape (K, d)   - prototypes
        sigma2: float or tensor     - variance parameter (σ²)
        normalize: bool             - if True, row-normalize to sum=1

    Returns:
        sims: Tensor of shape (B, K), positive similarities
    """
    # Pairwise squared Euclidean distances (B, K)
    sqdist = torch.cdist(z, p) ** 2

    # Gaussian RBF: exp(-||z - p||^2 / (2σ²))
    sims = torch.exp(-0.5 * sqdist / sigma2)

    if normalize:
        sims = sims / sims.sum(dim=1, keepdim=True)

    return sims