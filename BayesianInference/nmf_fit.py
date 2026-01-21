"""NMF-based parameter fitting for biophysical model.

This module implements Non-negative Matrix Factorization (NMF) for fitting
the biophysical parameters θ = {a, b, N_f, regimes} from empirical gene profiles.

Mathematical basis (from CellSizeNonlinearScaling-3.pdf, Equations 14-15):

Equation 14: The inverted model is:
    g_i(t) / (2^{t/τ} ψ_i(t)) ≈ a_i + B(t) × b_i = H × W^T

Where:
    a_i = M_0 γ_i / Γ_i
    b_i = (M_0 γ_i / Γ_i) × (k_off + Γ_i) / k_on
    B(t) = 1 / N_f(t)
    W = [a_i, b_i]^T  (2 × n_genes)
    H = [1, B(t)]     (n_time × 2, first column all ones)

This is a RANK-2 NMF problem constrained by the first column of H being all ones.

The fitting identifies:
1. Per-gene parameters: a_i, b_i
2. Time-dependent: B(t) = 1/N_f(t)
3. Regime assignment based on ratio b_i / a_i
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from BayesianInference.biophys_model import BiophysTheta, compute_dosage_grid


@dataclass
class NMFFitResult:
    """Result of rank-2 NMF fitting.

    Attributes:
        a_g: Fitted a_i parameters (n_genes,) - baseline amplitude.
        b_g: Fitted b_i parameters (n_genes,) - N_f-dependent amplitude.
        N_f_t: Fitted free RNAP trajectory (n_time,).
        B_t: Fitted 1/N_f(t) trajectory (n_time,).
        regimes: Regime assignments per gene (n_genes,).
        residuals_full: Reconstruction error under full model (n_genes,).
        residuals_I: Reconstruction error under Regime I only (n_genes,).
        residuals_II: Reconstruction error under Regime II only (n_genes,).
        reconstruction_error: Total reconstruction error.
        n_iterations: Number of iterations performed.
    """

    a_g: np.ndarray
    b_g: np.ndarray
    N_f_t: np.ndarray
    B_t: np.ndarray
    regimes: np.ndarray
    residuals_full: np.ndarray
    residuals_I: np.ndarray
    residuals_II: np.ndarray
    reconstruction_error: float
    n_iterations: int


def _fit_rank2_nmf_constrained(
    Y: np.ndarray,
    n_iters: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit constrained rank-2 NMF: Y ≈ W @ H^T.

    The constraint is that H[:, 0] = 1 (first column all ones).
    This corresponds to Equation 15 in the paper:
        H = [1, B(t)]
        W = [a_i, b_i]

    So Y[g, t] ≈ a_g × 1 + b_g × B(t) = a_g + b_g × B(t)

    Uses alternating least squares with non-negativity constraints.

    Args:
        Y: Data matrix (n_genes, n_time).
        n_iters: Maximum number of iterations.
        tol: Convergence tolerance.
        eps: Small constant for numerical stability.

    Returns:
        a_g: First column of W (n_genes,) - constant term.
        b_g: Second column of W (n_genes,) - B(t) coefficient.
        B_t: Second column of H (n_time,) - 1/N_f(t).
    """
    Y = np.asarray(Y, dtype=np.float64)
    Y = np.maximum(Y, eps)  # Ensure non-negative
    n_genes, n_time = Y.shape

    # Initialize factors
    # a_g: estimate from minimum of each gene (baseline when B(t) is low)
    a_g = np.percentile(Y, 25, axis=1)
    a_g = np.maximum(a_g, eps)

    # b_g: estimate from variation
    b_g = np.std(Y, axis=1)
    b_g = np.maximum(b_g, eps)

    # B_t: estimate from total variation pattern
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    B_t = np.mean(Y_centered, axis=0)
    B_t = B_t - np.min(B_t) + eps
    B_t = np.maximum(B_t, eps)

    prev_error = float("inf")

    for _ in range(n_iters):
        # Step 1: Update W = [a_g, b_g] given H = [1, B_t]
        # Y[g, t] ≈ a_g + b_g × B_t
        # For each gene, solve least squares:
        # min_a,b ||Y[g,:] - a - b*B_t||^2

        # Build design matrix: X = [ones, B_t] (n_time × 2)
        X = np.column_stack([np.ones(n_time), B_t])  # (n_time, 2)

        # Solve Y.T = X @ W.T for W (least squares per gene)
        # W.T = (X^T X)^{-1} X^T Y.T
        XTX = X.T @ X  # (2, 2)
        XTX_inv = np.linalg.inv(XTX + eps * np.eye(2))
        W_T = XTX_inv @ (X.T @ Y.T)  # (2, n_genes)

        a_g_new = W_T[0, :]
        b_g_new = W_T[1, :]

        # Apply non-negativity constraint
        a_g = np.maximum(a_g_new, eps)
        b_g = np.maximum(b_g_new, eps)

        # Step 2: Update B_t given W = [a_g, b_g]
        # Y[g, t] ≈ a_g + b_g × B_t
        # Rearrange: (Y[g, t] - a_g) / b_g ≈ B_t
        # Average over genes to get B_t

        residual = Y - a_g[:, None]  # (n_genes, n_time)
        # B_t ≈ mean_g((Y[g,t] - a_g) / b_g)
        B_t_estimates = residual / b_g[:, None]  # (n_genes, n_time)

        # Weighted average (weight by b_g to give more weight to genes with strong signal)
        weights = b_g / (np.sum(b_g) + eps)
        B_t = np.sum(weights[:, None] * B_t_estimates, axis=0)
        B_t = np.maximum(B_t, eps)

        # Check convergence
        reconstruction = a_g[:, None] + b_g[:, None] * B_t[None, :]
        error = np.sum((Y - reconstruction) ** 2)

        if abs(error - prev_error) < tol:
            break
        prev_error = error

    return a_g, b_g, B_t


def fit_rank2_nmf(
    G_emp: np.ndarray,
    dosage_gt: np.ndarray,
    t_grid: np.ndarray,
    tau: float = 1.0,
    n_iters: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit rank-2 NMF using Equation 14 formulation.

    The model is:
        g_i(t) / (2^{t/τ} ψ_i(t)) ≈ a_i + B(t) × b_i

    Where the LHS is computed from empirical data:
        LHS = dosage × expression / (2^{t/τ} × fraction)

    We simplify by working with:
        Y[g, t] = G_emp[g, t] / dosage[g, t]

    Then Y ≈ a_g + b_g × B(t) represents the dosage-corrected model.

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        dosage_gt: Gene dosage (n_genes, n_time).
        t_grid: Time grid (n_time,).
        tau: Division time (normalized to 1.0).
        n_iters: Maximum NMF iterations.
        tol: Convergence tolerance.
        eps: Numerical stability constant.

    Returns:
        a_g: Baseline amplitude parameters (n_genes,).
        b_g: N_f-dependent amplitude parameters (n_genes,).
        B_t: 1/N_f(t) trajectory (n_time,).
        N_f_t: Free RNAP trajectory (n_time,).
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    dosage_gt = np.asarray(dosage_gt, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    # Compute the transformed data matrix
    # Y = G_emp / dosage removes dosage effect
    Y = G_emp / np.maximum(dosage_gt, eps)

    # Fit rank-2 NMF: Y ≈ a_g + b_g × B(t)
    a_g, b_g, B_t = _fit_rank2_nmf_constrained(Y, n_iters=n_iters, tol=tol, eps=eps)

    # Convert B(t) to N_f(t) = 1/B(t)
    N_f_t = 1.0 / np.maximum(B_t, eps)

    return a_g, b_g, B_t, N_f_t


def compute_model_predictions(
    a_g: np.ndarray,
    b_g: np.ndarray,
    B_t: np.ndarray,
    dosage_gt: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute full model prediction m_g(t) from fitted parameters.

    Full model (Equation 7 inverted):
        m_g(t) = g_g(t) / (a_g + b_g × B(t))

    Where:
        - g_g(t) is dosage
        - a_g + b_g × B(t) is the denominator from Equation 14

    Actually from Equation 7:
        m_i(t) = g_i(t) × Γ_i / γ_i / (1 + (k_off + Γ_i) / (N_f(t) × k_on))

    With definitions from Eq 15:
        a_i = M_0 γ_i / Γ_i
        b_i = a_i × (k_off + Γ_i) / k_on

    The fitted Y ≈ a + b × B(t) actually represents the inverted form.
    So m_g(t) ∝ g_g(t) / (a_g + b_g × B(t)) when relating back to counts.

    However, for direct prediction:
        m_g(t) = dosage[g,t] / (a_g + b_g × B(t))

    Args:
        a_g: Baseline amplitude (n_genes,).
        b_g: N_f-dependent amplitude (n_genes,).
        B_t: 1/N_f(t) trajectory (n_time,).
        dosage_gt: Gene dosage (n_genes, n_time).
        eps: Numerical stability.

    Returns:
        m_gt: Predicted expression (n_genes, n_time).
    """
    # Denominator: a_g + b_g × B(t)
    denom = a_g[:, None] + b_g[:, None] * B_t[None, :]
    denom = np.maximum(denom, eps)

    # m_g(t) = dosage / denominator
    # This gives expression proportional to the biological model
    m_gt = dosage_gt / denom

    return np.maximum(m_gt, eps)


def assign_regimes_from_params(
    a_g: np.ndarray,
    b_g: np.ndarray,
    G_emp: np.ndarray,
    dosage_gt: np.ndarray,
    B_t: np.ndarray,
    regime_threshold: float = 0.1,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign genes to regimes based on fitted parameters and model comparison.

    Regime determination from Equation 7:
    - Regime I (linear scaling): When (k_off + Γ) / (N_f × k_on) >> 1
      This means b_g >> a_g × B(t)_avg, so expression ∝ N_f(t)
    - Regime II (saturation): When (k_off + Γ) / (N_f × k_on) << 1
      This means b_g << a_g × B(t)_avg, so expression ∝ constant

    We also compute residuals for each model to make the final decision.

    Args:
        a_g: Baseline amplitude (n_genes,).
        b_g: N_f-dependent amplitude (n_genes,).
        G_emp: Empirical profiles (n_genes, n_time).
        dosage_gt: Gene dosage (n_genes, n_time).
        B_t: 1/N_f(t) trajectory (n_time,).
        regime_threshold: Threshold for regime assignment.
        eps: Numerical stability.

    Returns:
        regimes: Regime assignments ("regime_I", "regime_II", or "full").
        residuals_full: Error under full model.
        residuals_I: Error under Regime I approximation.
        residuals_II: Error under Regime II approximation.
    """
    n_genes = len(a_g)
    Y = G_emp / np.maximum(dosage_gt, eps)

    # Full model: Y ≈ a_g + b_g × B(t)
    pred_full = a_g[:, None] + b_g[:, None] * B_t[None, :]
    residuals_full = np.sum((Y - pred_full) ** 2, axis=1)

    # Regime I: Y ≈ b_g × B(t) (linear in B(t), i.e., linear in 1/N_f)
    # Expression ∝ N_f(t) when this dominates
    pred_I = b_g[:, None] * B_t[None, :]
    residuals_I = np.sum((Y - pred_I) ** 2, axis=1)

    # Regime II: Y ≈ a_g (constant, independent of N_f)
    pred_II = a_g[:, None] * np.ones_like(B_t)[None, :]
    residuals_II = np.sum((Y - pred_II) ** 2, axis=1)

    # Assign regimes based on best fit
    regimes = np.array(["full"] * n_genes, dtype=object)

    for g in range(n_genes):
        # Compare residuals
        min_residual = min(residuals_full[g], residuals_I[g], residuals_II[g])

        # Check ratio of a_g to b_g × B_t_mean
        B_mean = np.mean(B_t)
        ratio = (b_g[g] * B_mean) / (a_g[g] + eps)

        if residuals_I[g] < residuals_full[g] * (1 - regime_threshold):
            # Regime I fits better (N_f-dependent, linear scaling)
            if ratio > 1.0:  # b term dominates
                regimes[g] = "regime_I"
        elif residuals_II[g] < residuals_full[g] * (1 - regime_threshold):
            # Regime II fits better (constant, saturation)
            if ratio < 1.0:  # a term dominates
                regimes[g] = "regime_II"
        # Otherwise keep "full" model

    return regimes, residuals_full, residuals_I, residuals_II


def fit_theta_from_empirical(
    G_emp: np.ndarray,
    t_grid: np.ndarray,
    theta_init: BiophysTheta,
    n_outer_iters: int = 10,
    n_nmf_iters: int = 200,
    tol: float = 1e-5,
    regime_threshold: float = 0.05,
    eps: float = 1e-12,
) -> NMFFitResult:
    """Fit biophysical parameters θ from empirical gene profiles using rank-2 NMF.

    This is the main NMF fitting function implementing Equations 14-15:
    1. Transform data: Y = G_emp / dosage
    2. Fit rank-2 NMF: Y ≈ a_g + b_g × B(t)
    3. Assign regimes based on fit quality and parameter ratios
    4. Refine fit iteratively

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        t_grid: Time grid (n_time,).
        theta_init: Initial parameter estimates.
        n_outer_iters: Maximum outer iterations for refinement.
        n_nmf_iters: NMF iterations per outer iteration.
        tol: Convergence tolerance.
        regime_threshold: Threshold for regime assignment.
        eps: Numerical stability constant.

    Returns:
        NMFFitResult with fitted parameters and diagnostics.
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_genes, n_time = G_emp.shape

    # Compute dosage from current position estimates
    dosage_gt = compute_dosage_grid(
        t_grid, theta_init.x_g, theta_init.C, theta_init.D
    )

    # Initialize from previous theta if available
    a_g = theta_init.a_g.copy()
    N_f_t = theta_init.N_f_t.copy()
    B_t = 1.0 / np.maximum(N_f_t, eps)

    # Initialize b_g based on regime
    b_g = np.zeros_like(a_g)
    regime_I_mask = theta_init.regimes == "regime_I"
    if np.any(regime_I_mask):
        # For regime I genes, estimate b_g from variation
        Y = G_emp / np.maximum(dosage_gt, eps)
        b_g[regime_I_mask] = np.std(Y[regime_I_mask], axis=1)
    b_g = np.maximum(b_g, eps)

    prev_error = float("inf")
    prev_regimes = None

    for iteration in range(n_outer_iters):
        # Step 1: Fit rank-2 NMF
        a_g, b_g, B_t, N_f_t = fit_rank2_nmf(
            G_emp, dosage_gt, t_grid,
            tau=theta_init.tau,
            n_iters=n_nmf_iters,
            tol=tol,
            eps=eps,
        )

        # Step 2: Assign regimes
        regimes, residuals_full, residuals_I, residuals_II = assign_regimes_from_params(
            a_g, b_g, G_emp, dosage_gt, B_t,
            regime_threshold=regime_threshold,
            eps=eps,
        )

        # Step 3: Compute total reconstruction error
        Y = G_emp / np.maximum(dosage_gt, eps)
        pred = a_g[:, None] + b_g[:, None] * B_t[None, :]
        total_error = np.sum((Y - pred) ** 2)

        # Check convergence
        if prev_regimes is not None:
            if np.all(regimes == prev_regimes) and abs(total_error - prev_error) < tol:
                break

        prev_regimes = regimes.copy()
        prev_error = total_error

    # For backward compatibility, convert a_g to combined amplitude
    # For regime I: effective amplitude is related to b_g
    # For regime II: effective amplitude is a_g
    a_g_combined = np.where(
        regimes == "regime_I",
        b_g,  # N_f-dependent term dominates
        a_g,  # Constant term dominates
    )

    return NMFFitResult(
        a_g=a_g_combined,
        b_g=b_g,
        N_f_t=N_f_t,
        B_t=B_t,
        regimes=regimes,
        residuals_full=residuals_full,
        residuals_I=residuals_I,
        residuals_II=residuals_II,
        reconstruction_error=float(total_error),
        n_iterations=iteration + 1,
    )


def update_theta_from_nmf(
    theta: BiophysTheta,
    nmf_result: NMFFitResult,
) -> BiophysTheta:
    """Update BiophysTheta with NMF fitting results.

    Args:
        theta: Current parameter estimates.
        nmf_result: NMF fitting results.

    Returns:
        Updated BiophysTheta.
    """
    return BiophysTheta(
        a_g=nmf_result.a_g.copy(),
        b_g=nmf_result.b_g.copy(),
        regimes=nmf_result.regimes.copy(),
        N_f_t=nmf_result.N_f_t.copy(),
        B_t=nmf_result.B_t.copy(),
        x_g=theta.x_g.copy(),
        C=theta.C,
        D=theta.D,
        tau=theta.tau,
    )


def initialize_theta_from_empirical(
    G_emp: np.ndarray,
    t_grid: np.ndarray,
    C_init: float = 0.6,
    D_init: float = 0.2,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> BiophysTheta:
    """Initialize BiophysTheta from empirical gene profiles.

    Uses heuristics to get reasonable initial parameter values:
    - a_g: Proportional to mean expression level
    - b_g: Proportional to expression variation
    - N_f_t: Proportional to total expression over time
    - x_g: Random (will be refined by position inference)

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        t_grid: Time grid.
        C_init: Initial C-period value.
        D_init: Initial D-period value.
        tau: Division time.
        eps: Numerical stability constant.

    Returns:
        Initialized BiophysTheta.
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    n_genes, n_time = G_emp.shape

    # Initialize N_f(t) from total expression pattern
    total_expr = np.sum(G_emp, axis=0)
    N_f_t = total_expr / np.mean(total_expr)
    N_f_t = np.maximum(N_f_t, eps)

    # Initialize B(t) = 1/N_f(t)
    B_t = 1.0 / np.maximum(N_f_t, eps)

    # Initialize a_g from mean expression (adjusted for estimated N_f)
    mean_expr = np.mean(G_emp, axis=1)
    mean_Nf = np.mean(N_f_t)
    a_g = mean_expr / mean_Nf
    a_g = np.maximum(a_g, eps)

    # Initialize b_g from expression variation
    b_g = np.std(G_emp, axis=1)
    b_g = np.maximum(b_g, eps)

    # Initialize all as Regime I (most general)
    regimes = np.array(["regime_I"] * n_genes, dtype=object)

    # Random gene positions
    rng = np.random.default_rng(0)
    x_g = rng.uniform(0.0, 1.0, size=n_genes)

    return BiophysTheta(
        a_g=a_g,
        b_g=b_g,
        regimes=regimes,
        N_f_t=N_f_t,
        B_t=B_t,
        x_g=x_g,
        C=C_init,
        D=D_init,
        tau=tau,
    )


# Legacy functions for backward compatibility


def _fit_rank1_nmf(
    Y: np.ndarray,
    n_iters: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a rank-1 NMF: Y ≈ w @ h.T (legacy function).

    Uses multiplicative updates for non-negative least squares.
    Kept for backward compatibility with regime-specific fitting.

    Args:
        Y: Data matrix (n_genes, n_time).
        n_iters: Maximum number of iterations.
        tol: Convergence tolerance.
        eps: Small constant for numerical stability.

    Returns:
        w: Column factor (n_genes,).
        h: Row factor (n_time,).
    """
    Y = np.asarray(Y, dtype=np.float64)
    Y = np.maximum(Y, eps)
    n_genes, n_time = Y.shape

    w = np.mean(Y, axis=1)
    w = np.maximum(w, eps)
    h = np.ones(n_time, dtype=np.float64)

    prev_error = float("inf")

    for _ in range(n_iters):
        Yw = Y.T @ w
        w_norm_sq = np.dot(w, w)
        h = h * Yw / (w_norm_sq * h + eps)
        h = np.maximum(h, eps)

        Yh = Y @ h
        h_norm_sq = np.dot(h, h)
        w = w * Yh / (h_norm_sq * w + eps)
        w = np.maximum(w, eps)

        reconstruction = np.outer(w, h)
        error = np.sum((Y - reconstruction) ** 2)
        if abs(error - prev_error) < tol:
            break
        prev_error = error

    return w, h


def fit_regime_I_params(
    G_emp: np.ndarray,
    dosage_gt: np.ndarray,
    eps: float = 1e-12,
    n_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Regime I parameters (legacy function).

    After dividing by dosage: Y = G_emp / dosage ≈ a_I[:, None] @ N_f[None, :]
    This is a rank-1 NMF problem.

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        dosage_gt: Gene dosage (n_genes, n_time).
        eps: Numerical stability constant.
        n_iters: Maximum NMF iterations.
        tol: Convergence tolerance.

    Returns:
        a_I: Amplitude parameters (n_genes,).
        N_f_t: Free RNAP trajectory (n_time,).
        residuals: Per-gene reconstruction error (n_genes,).
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    dosage_gt = np.asarray(dosage_gt, dtype=np.float64)

    Y = G_emp / np.maximum(dosage_gt, eps)
    a_I, N_f_t = _fit_rank1_nmf(Y, n_iters=n_iters, tol=tol, eps=eps)

    reconstruction = np.outer(a_I, N_f_t) * dosage_gt
    residuals = np.sum((G_emp - reconstruction) ** 2, axis=1)

    return a_I, N_f_t, residuals


def fit_regime_II_params(
    G_emp: np.ndarray,
    dosage_gt: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Regime II parameters (legacy function).

    Since Regime II is N_f-independent, after dividing by dosage the
    expression should be constant. We estimate a_II as the mean.

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        dosage_gt: Gene dosage (n_genes, n_time).
        eps: Numerical stability constant.

    Returns:
        a_II: Amplitude parameters (n_genes,).
        residuals: Per-gene reconstruction error (n_genes,).
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    dosage_gt = np.asarray(dosage_gt, dtype=np.float64)

    Y = G_emp / np.maximum(dosage_gt, eps)
    a_II = np.mean(Y, axis=1)
    a_II = np.maximum(a_II, eps)

    reconstruction = a_II[:, None] * dosage_gt
    residuals = np.sum((G_emp - reconstruction) ** 2, axis=1)

    return a_II, residuals


def assign_regimes(
    residuals_I: np.ndarray,
    residuals_II: np.ndarray,
    regime_threshold: float = 0.0,
) -> np.ndarray:
    """Assign genes to regimes based on reconstruction error (legacy function).

    Args:
        residuals_I: Per-gene error under Regime I (n_genes,).
        residuals_II: Per-gene error under Regime II (n_genes,).
        regime_threshold: Regime II must be this much better (relative).

    Returns:
        regimes: Regime assignments ("regime_I" or "regime_II").
    """
    n_genes = len(residuals_I)
    regimes = np.array(["regime_I"] * n_genes, dtype=object)

    regime_II_better = residuals_II < residuals_I * (1.0 - regime_threshold)
    regimes[regime_II_better] = "regime_II"

    return regimes
