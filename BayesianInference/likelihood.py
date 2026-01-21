from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import logsumexp


@dataclass
class LikelihoodParams:
    G_gt: np.ndarray
    mu: float
    sigma: float
    t_grid: np.ndarray
    eps: float = 1e-12


def compute_log_p0(
    total_counts: np.ndarray,
    t_grid: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    total_counts = np.asarray(total_counts, dtype=np.float64)
    t = np.asarray(t_grid, dtype=np.float64)
    ln2 = np.log(2.0)
    sigma = max(float(sigma), 1e-8)
    alpha = (mu + sigma * sigma) / (sigma * sigma)

    logS = np.zeros_like(total_counts, dtype=np.float64)
    mask = total_counts > 0
    logS[mask] = np.log(total_counts[mask])

    diff = logS[:, None] - t[None, :] * ln2 - float(mu)
    logP0 = -alpha * t[None, :] * ln2 - (diff * diff) / (2.0 * sigma * sigma)
    if np.any(~mask):
        logP0[~mask, :] = 0.0
    return logP0.astype(np.float32)


def log_likelihood_collapsed_diagnostic(
    counts: sparse.spmatrix,
    M_hat: np.ndarray,
    eps: float = 1e-12,
) -> float:
    if sparse.issparse(counts):
        counts = counts.tocsr()
    else:
        counts = sparse.csr_matrix(counts)

    M = np.asarray(M_hat, dtype=np.float64)

    row_sums = np.sum(M, axis=1, keepdims=True)
    psi = M / np.maximum(row_sums, eps)

    logpsi = np.log(np.maximum(psi, eps))

    ll = 0.0
    for c in range(counts.shape[0]):
        start, end = counts.indptr[c], counts.indptr[c + 1]
        g_idx = counts.indices[start:end]
        n = counts.data[start:end]
        ll += float(np.dot(n, logpsi[c, g_idx]))

    return float(ll)


def log_likelihood_bayes(
    counts: sparse.spmatrix,
    P0_ct: np.ndarray,
    G_gt: np.ndarray,
    total_counts: np.ndarray,
    t_grid: np.ndarray,
    eps: float = 1e-12,
    tau: float = 1.0,
) -> float:
    if sparse.issparse(counts):
        counts = counts.tocsr()
    else:
        counts = sparse.csr_matrix(counts)

    logP0 = np.asarray(P0_ct, dtype=np.float64)
    M = np.asarray(G_gt, dtype=np.float64)
    M = np.maximum(M, eps)
    logm = np.log(M)
    sum_logm = counts.dot(logm)
    sum_logm = np.asarray(sum_logm, dtype=np.float64)

    logL_mat = logP0 + sum_logm
    logL = logsumexp(logL_mat, axis=1)
    return float(np.sum(logL))


def log_likelihood(
    counts: sparse.spmatrix,
    P0_ct: np.ndarray,
    G_gt: np.ndarray,
    total_counts: np.ndarray,
    t_grid: np.ndarray,
    eps: float = 1e-12,
    tau: float = 1.0,
) -> float:
    return log_likelihood_bayes(counts, P0_ct, G_gt, total_counts, t_grid, eps=eps, tau=tau)


def perturb_params(
    params: LikelihoodParams,
    rng: np.random.Generator,
    scale: float = 0.05,
) -> LikelihoodParams:
    G = np.asarray(params.G_gt, dtype=np.float64)
    noise = rng.normal(loc=0.0, scale=scale, size=G.shape)
    G_new = np.maximum(G * np.exp(noise), params.eps)
    mu_new = float(params.mu + rng.normal(0.0, scale * 0.5))
    sigma_new = float(max(params.sigma + rng.normal(0.0, scale * 0.5), 1e-6))
    return LikelihoodParams(G_gt=G_new, mu=mu_new, sigma=sigma_new, t_grid=params.t_grid, eps=params.eps)


# =============================================================================
# NEW FUNCTIONS FOR THETA-BASED INFERENCE
# =============================================================================


def log_likelihood_from_theta(
    counts: sparse.spmatrix,
    theta,  # BiophysTheta - avoid circular import
    t_grid: np.ndarray,
    total_counts: np.ndarray,
    mu: float,
    sigma: float,
    eps: float = 1e-12,
    tau: float = 1.0,
) -> float:
    """Compute log-likelihood using biophysical parameters theta.

    This function computes m_g(t) from theta using the one-to-one mapping,
    then uses it to compute the log-likelihood.

    Args:
        counts: Sparse count matrix (n_cells, n_genes).
        theta: BiophysTheta parameter set.
        t_grid: Time grid (n_time,).
        total_counts: Total counts per cell (n_cells,).
        mu: Mean of log-normal size prior.
        sigma: Std of log-normal size prior.
        eps: Numerical stability constant.
        tau: Division time.

    Returns:
        Total log-likelihood.
    """
    # Import here to avoid circular dependency
    from BayesianInference.biophys_model import compute_m_from_theta

    # Compute m_g(t) from theta
    G_gt = compute_m_from_theta(theta, t_grid, eps=eps)

    # Compute log P(t|S) prior
    logP0 = compute_log_p0(total_counts, t_grid, mu, sigma)

    # Compute full log-likelihood
    return log_likelihood_bayes(counts, logP0, G_gt, total_counts, t_grid, eps=eps, tau=tau)
