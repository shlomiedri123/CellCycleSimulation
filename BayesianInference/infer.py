"""Main inference loop for cell cycle Bayesian inference.

This module implements the iterative Bayesian inference algorithm for
estimating cell ages and gene expression profiles from snapshot RNA-seq data.

The key principle is the "one-to-one" constraint:
    θ → m_g(t) via physics → p_g(t) → P_ct via Bayes

Where θ = {a_g, N_f_t, x_g, C, D, regimes} are the ONLY free parameters.
Gene profiles m_g(t) are DERIVED from θ, not fitted freely.

Mathematical basis (from CellSizeNonlinearScaling-3.pdf):
- Equations 1-7: Biophysical model for m_g(t)
- Equations 16-25: Bayesian inference for cell ages
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from BayesianInference.biophys_model import (
    BiophysTheta,
    compute_dosage_grid,
    compute_m_from_theta,
    compute_psi_from_m,
    initialize_theta,
    validate_theta,
)
from BayesianInference.nmf_fit import (
    fit_theta_from_empirical,
    initialize_theta_from_empirical,
    update_theta_from_nmf,
)
from BayesianInference.position_inference import (
    infer_positions_and_cd,
)
from BayesianInference.likelihood import compute_log_p0, log_likelihood
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of Bayesian inference.

    Attributes:
        P_ct: Posterior age distributions (n_cells, n_time).
        G_gt: Inferred gene profiles m_g(t) (n_genes, n_time).
        psi_gt: Normalized gene fractions (n_genes, n_time).
        t_grid: Time grid used for inference.
        t_map: MAP age estimate per cell.
        mu: Fitted sequencing depth mean parameter.
        sigma: Fitted sequencing depth std parameter.
        theta: Complete biophysical parameter set (NEW).
        params: Legacy parameter dict (for backward compatibility).
        metrics: Per-iteration metrics.
        timings: Per-iteration timing info.
        loglik_trace: Log-likelihood per iteration.
    """

    P_ct: np.ndarray
    G_gt: np.ndarray
    psi_gt: np.ndarray
    t_grid: np.ndarray
    t_map: np.ndarray
    mu: float
    sigma: float
    theta: BiophysTheta | None = None
    params: dict[str, np.ndarray] | None = None
    metrics: list[dict[str, float]] = field(default_factory=list)
    timings: list[dict[str, float]] = field(default_factory=list)
    loglik_trace: list[float] = field(default_factory=list)


@dataclass
class InferenceConfig:
    """Configuration for inference.

    Attributes:
        nt: Number of time bins.
        max_iters: Maximum outer iterations.
        tol: Convergence tolerance for log-likelihood.
        eps: Numerical stability constant.
        mu: Fixed sequencing depth mean (if known).
        sigma: Fixed sequencing depth std (if known).
        seed: Random seed for reproducibility.
        max_cells_fit: Max cells for depth fitting.
        tau: Division time (1.0 for normalized time).
        nmf_iters: NMF iterations per outer iteration.
        nmf_tol: NMF convergence tolerance.
        regime_threshold: Threshold for regime assignment.
        infer_positions: Whether to infer gene positions.
        infer_cd: Whether to infer C, D parameters.
        C_init: Initial C-period value.
        D_init: Initial D-period value.
    """
    nt: int = 40
    max_iters: int = 20
    tol: float = 1e-4
    eps: float = 1e-12
    mu: float | None = None
    sigma: float | None = None
    seed: int = 0
    max_cells_fit: int = 500000
    tau: float = 1.0
    nmf_iters: int = 50
    nmf_tol: float = 1e-5
    regime_threshold: float = 0.05
    infer_positions: bool = True
    infer_cd: bool = True
    C_init: float = 0.6
    D_init: float = 0.2


def time_grid(nt: int) -> np.ndarray:
    """Create a uniform time grid on [0, 1].

    Args:
        nt: Number of time bins.

    Returns:
        t_grid: Time grid with bin centers.
    """
    if nt <= 0:
        raise ValueError("nt must be positive")
    edges = np.linspace(0.0, 1.0, nt + 1, dtype=np.float32)
    return 0.5 * (edges[:-1] + edges[1:])


def compute_empirical_G(
    counts: sparse.csr_matrix,
    P_ct: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Compute empirical gene profiles from counts and age posteriors.

    Formula: G_g(t) = [Σ_c n_cg P(t|c)] / [Σ_c P(t|c)]


    Args:
        counts: Count matrix (n_cells, n_genes).
        P_ct: Age posteriors (n_cells, n_time).
        eps: Numerical stability constant.

    Returns:
        G_gt: Empirical gene profiles (n_genes, n_time).
    """
    counts_t = counts.transpose().tocsr()
    N_gt = counts_t.dot(P_ct)
    N_gt = np.asarray(N_gt, dtype=np.float64)
    denom = np.sum(P_ct, axis=0, keepdims=True)
    return (N_gt / np.maximum(denom, eps)).astype(np.float64)


def update_gene_profiles(
    counts: sparse.csr_matrix,
    P_ct: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Update empirical gene profiles (alias for compute_empirical_G)."""
    return compute_empirical_G(counts, P_ct, eps)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to each row of a matrix."""
    max_log = np.max(logits, axis=1, keepdims=True)
    np.exp(logits - max_log, out=logits)
    denom = np.sum(logits, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    logits /= denom
    return logits


def update_posteriors(
    counts: sparse.csr_matrix,
    logP0: np.ndarray,
    logM: np.ndarray,
    total_counts: np.ndarray,
    sum_m: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Update cell age posteriors given gene profiles.

    Formula: P(t|n,S) ∝ P(t|S) × Π_g m_g(t)^{n_cg}

    Args:
        counts: Count matrix (n_cells, n_genes).
        logP0: Log prior P(t|S) (n_cells, n_time).
        logM: Log gene profiles (n_genes, n_time).
        total_counts: Total counts per cell (n_cells,).
        sum_m: Sum of m_g(t) over genes (n_time,).
        eps: Numerical stability constant.

    Returns:
        P_new: Updated posteriors (n_cells, n_time).
    """
    sum_log = counts.dot(logM)
    sum_log = np.asarray(sum_log, dtype=np.float32)

    log_sum_m = np.log(np.maximum(sum_m, eps)).astype(np.float32)
    logP = logP0 + sum_log - total_counts[:, None] * log_sum_m[None, :]
    logP = _softmax_rows(logP)
    return logP.astype(np.float32)


def fit_log_depth_params(
    total_counts: np.ndarray,
    t_grid: np.ndarray,
    max_cells: int = 50000,
) -> tuple[float, float]:
    """Fit log-normal sequencing depth parameters μ and σ.

    Args:
        total_counts: Total counts per cell.
        t_grid: Time grid.
        max_cells: Maximum cells to use for fitting.

    Returns:
        mu: Fitted mean parameter.
        sigma: Fitted std parameter.
    """
    from scipy import optimize

    ln2 = np.log(2.0)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    mask = total_counts > 0
    if not np.any(mask):
        raise ValueError("total_counts must contain positive values")
    total_counts = total_counts[mask]
    if total_counts.size > max_cells:
        rng = np.random.default_rng(0)
        total_counts = rng.choice(total_counts, size=max_cells, replace=False)

    lnS = np.log(total_counts)
    mu0 = float(lnS.mean() - 0.5 * ln2)
    sigma0 = float(max(lnS.std(), 1e-3))
    t = np.asarray(t_grid, dtype=np.float64)
    dt = 1.0 / float(t.size)

    def _neg_log_like(x: np.ndarray) -> float:
        mu = float(x[0])
        sigma = float(np.exp(x[1]))
        log_pref = -np.log(total_counts) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        diff = lnS[:, None] - t[None, :] * ln2 - mu
        log_p = log_pref[:, None] - (diff * diff) / (2.0 * sigma * sigma) - t[None, :] * ln2
        log_int = logsumexp(log_p, axis=1) + np.log(dt)
        return -float(np.sum(log_int))

    x0 = np.array([mu0, np.log(sigma0)], dtype=np.float64)
    # Optimize with BFGS algorithm
    res = optimize.minimize(_neg_log_like, x0, method="L-BFGS-B") 
    mu_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    return mu_hat, sigma_hat


# =============================================================================
# THETA-BASED INFERENCE
# =============================================================================


def run_inference(
    counts: sparse.spmatrix,
    config: InferenceConfig | None = None,
    chrom_pos: np.ndarray | None = None,
) -> InferenceResult:
    """Run Bayesian inference for cell ages and gene profiles.

    This uses theta-based inference where all parameters are inferred from
    count data using NMF-based fitting.

    The key principle is the "one-to-one" constraint:
        θ → m_g(t) via physics → p_g(t) → P_ct via Bayes

    Where θ = {a_g, N_f_t, x_g, C, D, regimes} are the ONLY free parameters.
    Gene profiles m_g(t) are DERIVED from θ, not fitted freely.

    Args:
        counts: Count matrix (n_cells, n_genes), sparse.
        config: Inference configuration.
        chrom_pos: Gene chromosomal positions (optional, can be inferred).

    Returns:
        InferenceResult with posteriors, profiles, and parameters.
    """
    if config is None:
        config = InferenceConfig()

    counts = counts.tocsr()
    total_counts = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    counts = counts.astype(np.float32, copy=False)
    t_grid_arr = time_grid(config.nt)

    logger.info("Running theta-based inference")
    return _run_inference_theta(
        counts, config, chrom_pos, total_counts, t_grid_arr
    )


def _run_inference_theta(
    counts: sparse.csr_matrix,
    config: InferenceConfig,
    chrom_pos: np.ndarray | None,
    total_counts: np.ndarray,
    t_grid: np.ndarray,
) -> InferenceResult:
    """Run inference using new theta-based approach.

    All parameters are inferred from count data.
    """
    n_cells, n_genes = counts.shape
    n_time = len(t_grid)

    # Step 1: Fit sequencing depth parameters μ, σ
    mu = config.mu
    sigma = config.sigma
    if mu is None or sigma is None:
        mu, sigma = fit_log_depth_params(
            total_counts, t_grid, max_cells=config.max_cells_fit
        )

    # Step 2: Initialize age posteriors from total counts
    logP0 = compute_log_p0(total_counts, t_grid, mu, sigma)
    P_ct = _softmax_rows(logP0.copy())

    # Step 3: Compute initial empirical profiles
    G_emp = update_gene_profiles(counts, P_ct, config.eps)

    # Step 4: Initialize theta from empirical profiles
    theta = initialize_theta_from_empirical(
        G_emp, t_grid,
        C_init=config.C_init,
        D_init=config.D_init,
        tau=config.tau,
        eps=config.eps,
    )

    # If chromosomal positions are provided, use them
    if chrom_pos is not None:
        theta.x_g = np.asarray(chrom_pos, dtype=np.float64)

    metrics: list[dict[str, float]] = []
    timings: list[dict[str, float]] = []
    loglik_trace: list[float] = []
    prev_loglik = float("-inf")

    # Track best state
    best_loglik = float("-inf")
    best_P_ct = P_ct.copy()
    best_G_gt: np.ndarray | None = None
    best_theta = theta.copy()

    # Main iteration loop
    for iteration in range(config.max_iters):
        t_start = perf_counter()

        # Step 5: Update empirical profiles
        G_emp = update_gene_profiles(counts, P_ct, config.eps)

        # Step 6: Compute dosage from current positions
        dosage_gt = compute_dosage_grid(
            t_grid, theta.x_g, theta.C, theta.D
        )

        # Step 7: NMF fit to update a_g, N_f_t, and regimes
        nmf_result = fit_theta_from_empirical(
            G_emp, t_grid, theta,
            n_outer_iters=5,  # Inner iterations per outer iteration
            n_nmf_iters=config.nmf_iters,
            tol=config.nmf_tol,
            regime_threshold=config.regime_threshold,
            eps=config.eps,
        )
        theta = update_theta_from_nmf(theta, nmf_result)

        # Step 8: Infer gene positions and C, D (if enabled)
        if config.infer_positions or config.infer_cd:
            pos_result = infer_positions_and_cd(
                G_emp, t_grid,
                x_g_prior=chrom_pos,  # Use provided positions as prior
                C_prior=theta.C,
                D_prior=theta.D,
                min_jumps=max(5, n_genes // 100),
            )

            if config.infer_positions and chrom_pos is None:
                # Only update positions if not provided externally
                theta.x_g = pos_result.x_g

            if config.infer_cd:
                theta.C = pos_result.C
                theta.D = pos_result.D

        # Step 9: Compute m_g(t) from theta
        G_gt = compute_m_from_theta(theta, t_grid, eps=config.eps)

        # Step 10: Update posteriors
        logM = np.log(np.maximum(G_gt, config.eps)).astype(np.float32)
        sum_m = np.sum(G_gt, axis=0, keepdims=False).astype(np.float32)
        P_new = update_posteriors(counts, logP0, logM, total_counts, sum_m, config.eps)

        t_end = perf_counter()

        # Compute log-likelihood
        loglik = log_likelihood(
            counts, P0_ct=logP0, G_gt=G_gt,
            total_counts=total_counts, t_grid=t_grid,
            eps=config.eps, tau=config.tau,
        )

        # Track metrics
        delta_ll = float(abs(loglik - prev_loglik)) if prev_loglik > float("-inf") else 0.0
        delta_P = float(np.mean(np.abs(P_new - P_ct)))
        delta_m = float(np.mean(np.abs(G_gt - best_G_gt))) if best_G_gt is not None else float("nan")

        metrics.append({
            "delta_P": delta_P,
            "delta_m": delta_m,
            "delta_ll": delta_ll,
            "nmf_error": nmf_result.reconstruction_error,
            "n_regime_I": int(np.sum(theta.regimes == "regime_I")),
            "n_regime_II": int(np.sum(theta.regimes == "regime_II")),
            "n_regime_full": int(np.sum(theta.regimes == "full")),
        })
        timings.append({"iteration_seconds": t_end - t_start})
        loglik_trace.append(float(loglik))

        # Update best state if improved
        if loglik > best_loglik:
            best_loglik = float(loglik)
            best_P_ct = P_new.copy()
            best_G_gt = G_gt.copy()
            best_theta = theta.copy()

        # Always accept the update for next iteration (EM-style)
        P_ct = P_new
        prev_loglik = float(loglik)

        # Check convergence
        if iteration > 0 and delta_ll < config.tol:
            break

    # Use best state for final result
    P_ct = best_P_ct
    G_gt = best_G_gt if best_G_gt is not None else G_gt
    theta = best_theta

    # Compute MAP ages and normalized profiles
    t_map = t_grid[np.argmax(P_ct, axis=1)].astype(np.float32)
    psi_gt = compute_psi_from_m(G_gt, eps=config.eps)

    # Create legacy params dict for backward compatibility
    params = {
        "a_g": theta.a_g.astype(np.float64),
        "b_g": theta.b_g.astype(np.float64) if theta.b_g is not None else None,
        "N_f_t": theta.N_f_t.astype(np.float64),
        "B_t": theta.B_t.astype(np.float64) if theta.B_t is not None else None,
        "regime": theta.regimes,
        "x_g": theta.x_g.astype(np.float64),
        "t_grid": t_grid.astype(np.float64),
        "t_rep": theta.get_t_rep().astype(np.float64),
        "gamma_g": theta.gamma_g.astype(np.float64) if theta.gamma_g is not None else None,
    }
    if theta.C is not None:
        params["C"] = np.array([theta.C], dtype=np.float64)
    if theta.D is not None:
        params["D"] = np.array([theta.D], dtype=np.float64)

    return InferenceResult(
        P_ct=P_ct,
        G_gt=G_gt,
        psi_gt=psi_gt,
        t_grid=t_grid,
        t_map=t_map,
        mu=float(mu),
        sigma=float(sigma),
        theta=theta,
        params=params,
        metrics=metrics,
        timings=timings,
        loglik_trace=loglik_trace,
    )


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_with_truth(
    t_map: np.ndarray,
    t_true: np.ndarray,
    P_ct: np.ndarray,
    t_grid: np.ndarray,
) -> dict[str, Any]:
    """Validate inferred ages against ground truth."""
    t_true = np.asarray(t_true, dtype=np.float32)
    t_map = np.asarray(t_map, dtype=np.float32)
    if t_true.shape[0] != t_map.shape[0]:
        raise ValueError("t_true length must match t_map")

    corr = float(np.corrcoef(t_true, t_map)[0, 1])

    cdf = np.cumsum(P_ct, axis=1)
    idx = np.searchsorted(t_grid, t_true, side="right") - 1
    idx = np.clip(idx, 0, t_grid.size - 1)
    cdf_at_true = cdf[np.arange(t_true.size), idx]

    quantiles = np.quantile(cdf_at_true, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(float)

    return {
        "t_map_corr": corr,
        "cdf_quantiles": quantiles.tolist(),
    }


def r2_per_gene(G_gt: np.ndarray, G_true: np.ndarray) -> np.ndarray:
    """Compute R-squared per gene between inferred and true profiles."""
    G_gt = np.asarray(G_gt, dtype=np.float64)
    G_true = np.asarray(G_true, dtype=np.float64)
    if G_gt.shape != G_true.shape:
        raise ValueError("G_gt and G_true must have the same shape")
    mean_true = np.mean(G_true, axis=1, keepdims=True)
    ss_res = np.sum((G_true - G_gt) ** 2, axis=1)
    ss_tot = np.sum((G_true - mean_true) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return r2.astype(np.float32)
