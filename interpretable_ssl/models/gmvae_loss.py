import torch
import torch.nn.functional as F


def responsibilities(mu_x, proto_mu, proto_vparam, T=1.0, return_logits=False):
    """
    Responsibilities with optional temperature T.
    mu_x: (B,D) encoder means
    proto_mu: (K,D) prototype means
    proto_vparam: (K,D) unconstrained variance parameters
    T: temperature (default=1.0)
    """
    device = mu_x.device
    B, D = mu_x.shape
    # K = proto_mu.size(0)

    proto_var = F.softplus(proto_vparam) + 1e-4

    diff = mu_x[:, None, :] - proto_mu[None, :, :]  # (B,K,D)
    const = D * torch.log(torch.tensor(2.0 * torch.pi, device=device))
    log_prob = -0.5 * (
        (diff**2 / proto_var[None]).sum(-1)
        + torch.log(proto_var).sum(-1)[None, :]
        + const
    )  # (B,K)

    if return_logits:
        return log_prob
    r = torch.softmax(log_prob / T, dim=1)  # apply temperature
    return r


def gaussian_kl(mu_q, vparam_q, mu_p, vparam_p, eps=1e-4):
    """
    KL divergence KL( N(mu_q, Σ_q) || N(mu_p, Σ_p) )
    for diagonal covariances, batch-to-prototype comparison.

    Args:
        mu_q:     (B,D) encoder means
        vparam_q: (B,D) unconstrained variance params for q
        mu_p:     (K,D) prototype means
        vparam_p: (K,D) unconstrained variance params for p
        eps:      float, stability constant

    Returns:
        kl: (B,K) matrix of KL divergences
    """
    # map unconstrained params -> positive variances
    var_q = F.softplus(vparam_q) + eps  # (B,D)
    var_p = F.softplus(vparam_p) + eps  # (K,D)

    B, D = mu_q.shape
    K = mu_p.shape[0]

    # expand for broadcasting (B,K,D)
    mu_q = mu_q[:, None, :].expand(B, K, D)
    var_q = var_q[:, None, :].expand(B, K, D)

    mu_p = mu_p[None, :, :].expand(B, K, D)
    var_p = var_p[None, :, :].expand(B, K, D)

    # log|Σ_p| - log|Σ_q|
    log_var_ratio = (torch.log(var_p) - torch.log(var_q)).sum(-1)

    # tr(Σ_p^-1 Σ_q)
    trace_term = (var_q / var_p).sum(-1)

    # (μ_q-μ_p)^T Σ_p^-1 (μ_q-μ_p)
    mean_diff = ((mu_q - mu_p) ** 2 / var_p).sum(-1)

    # KL divergence
    kl = 0.5 * (log_var_ratio - D + trace_term + mean_diff)
    return kl  # (B,K)


def gm_kl(mu_x, vparam_x, proto_mu, vparam_p, resp):
    """
    Full GMVAE KL with uniform prior.

    Args:
        mu_x:     (B,D) encoder means
        vparam_x: (B,D) encoder variance params (unconstrained)
        proto_mu: (K,D) prototype means
        vparam_p: (K,D) prototype variance params (unconstrained)
        resp:     (B,K) responsibilities (soft assignment)

    Returns:
        kl_total: scalar KL (averaged over batch)
        parts:    dict with {"kl_gauss", "kl_balance"}
    """
    # responsibility-weighted Gaussian KL
    kl_g = gaussian_kl(mu_x, vparam_x, proto_mu, vparam_p)  # (B,K)
    kl_weighted = (resp * kl_g).sum(-1).mean()  # scalar

    # categorical KL vs uniform prior over K components
    B, K = resp.shape
    resp_mean = resp.mean(dim=0)  # (K,)
    kl_balance = (resp_mean * torch.log(resp_mean * K + 1e-12)).sum()

    return kl_weighted, {
        "kl_wgauss": kl_weighted.item(),
        "kl_balance": kl_balance.item(),
    }
