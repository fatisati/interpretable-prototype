import torch

def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL(N(mu_q, var_q) || N(mu_p, var_p)) for diagonal covariances.
    mu_q: (B,D), logvar_q: (B,D)
    mu_p: (K,D), logvar_p: (K,D)
    returns: (B,K) KLs
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    B, D = mu_q.shape
    K, _ = mu_p.shape

    mu_q = mu_q[:, None, :].expand(B, K, D)
    logvar_q = logvar_q[:, None, :].expand(B, K, D)
    var_q = var_q[:, None, :].expand(B, K, D)

    mu_p = mu_p[None, :, :].expand(B, K, D)
    logvar_p = logvar_p[None, :, :].expand(B, K, D)
    var_p = var_p[None, :, :].expand(B, K, D)

    term1 = (logvar_p - logvar_q).sum(-1)
    term2 = (var_q / var_p).sum(-1)
    term3 = ((mu_q - mu_p) ** 2 / var_p).sum(-1)
    return 0.5 * (term1 - D + term2 + term3)   # (B,K)


def kl_gmvae(mu_x, logvar_x, proto_mu, proto_logvar, resp):
    """
    Full GMVAE KL with uniform pi.
    mu_x, logvar_x: (B,D) encoder outputs
    proto_mu, proto_logvar: (K,D) prototype params
    resp: (B,K) responsibilities (sum to 1 per row)
    returns: scalar KL (mean over batch), dict with parts
    """
    B, K = resp.shape

    # 1) responsibility-weighted Gaussian KL
    kl_g = gaussian_kl(mu_x, logvar_x, proto_mu, proto_logvar)  # (B,K)
    kl_weighted = (resp * kl_g).sum(-1).mean()

    # 2) categorical KL to uniform prior
    kl_cat = (resp * torch.log(resp * K + 1e-12)).sum(-1).mean()

    return kl_weighted + kl_cat, {"kl_gauss": kl_weighted.item(),
                                  "kl_cat": kl_cat.item()}
