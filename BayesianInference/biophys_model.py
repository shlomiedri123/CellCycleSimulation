"""Biophysical model for cell cycle gene expression.

This module defines the core parameter representation and model computation
for the Bayesian inference of cell ages and gene expression profiles.

The key principle is the "one-to-one" constraint: θ → m_g(t) via physics.
Gene profiles are DERIVED from biophysical parameters, not fitted freely.

Mathematical basis (from CellSizeNonlinearScaling-3.pdf):

Equation 7 (full model):
    m_i(t) = g_i(t) × Γ_i / γ_i / (1 + (k_off + Γ_i) / (N_f(t) × k_on))

Equations 14-15 (NMF formulation):
    g_i(t) / (2^{t/τ} ψ_i(t)) ≈ a_i + B(t) × b_i

    Where:
        a_i = M_0 γ_i / Γ_i
        b_i = a_i × (k_off + Γ_i) / k_on
        B(t) = 1 / N_f(t)

Equation 4 (regime approximations):
    - Regime I: m_g(t) ∝ g_g(t) × N_f(t)  (when b_i × B(t) >> a_i)
    - Regime II: m_g(t) ∝ g_g(t)          (when a_i >> b_i × B(t))

Equation 2 (integral model for non-quasi-steady-state):
    m(t) = ∫₀^τ e^{γ(t'-t)} g(t') / (1 + b/N_f(t')) dt' / (2e^{γτ} - 1)
         + ∫₀^t e^{γ(t'-t)} g(t') / (1 + b/N_f(t')) dt'

Cooper-Helmstetter model for gene dosage:
    - t_rep(x) = B + C × x = (1 - C - D) + C × x
    - g_g(t) = 1 if t < t_rep, else 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


RegimeType = Literal["regime_I", "regime_II", "full"]


@dataclass
class BiophysTheta:
    """Complete biophysical parameter set θ for rank-2 NMF model.

    This is the ONLY set of free parameters in the inference.
    All gene expression profiles m_g(t) are DERIVED from these parameters
    using the model: Y[g,t] = a_g + b_g × B(t), where Y is dosage-corrected.

    Attributes:
        a_g: Per-gene baseline amplitude (n_genes,).
            From Eq. 15: a_i = M_0 γ_i / Γ_i
        b_g: Per-gene N_f-dependent amplitude (n_genes,).
            From Eq. 15: b_i = a_i × (k_off + Γ_i) / k_on
        regimes: Per-gene regime assignment (n_genes,).
            "regime_I" (N_f-dependent), "regime_II" (constant), or "full".
        N_f_t: Free RNAP trajectory (n_time,).
        B_t: 1/N_f(t) trajectory (n_time,).
        x_g: Gene chromosomal positions (n_genes,), normalized to [0, 1].
        C: C-period duration as fraction of cell cycle.
        D: D-period duration as fraction of cell cycle.
        tau: Division time (default 1.0 for normalized time).
        gamma_g: Per-gene mRNA degradation rate (n_genes,), optional.
    """

    # Per-gene parameters from rank-2 NMF (Eq. 15)
    a_g: np.ndarray  # shape (n_genes,) - baseline term
    regimes: np.ndarray  # shape (n_genes,), dtype=object

    # Time-dependent global parameter
    N_f_t: np.ndarray  # shape (n_time,)

    # Gene position parameters
    x_g: np.ndarray  # shape (n_genes,)
    C: float
    D: float

    # Time parameters
    tau: float = 1.0

    # Optional parameters with defaults
    b_g: np.ndarray | None = None  # shape (n_genes,) - N_f-dependent term
    B_t: np.ndarray | None = None  # shape (n_time,) - 1/N_f(t)
    gamma_g: np.ndarray | None = None  # shape (n_genes,) - degradation rates

    # Cached derived quantities (computed on demand)
    _dosage_gt: np.ndarray | None = field(default=None, repr=False)
    _t_grid: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize derived parameters if not provided."""
        eps = 1e-12
        # Initialize b_g if not provided
        if self.b_g is None:
            self.b_g = np.ones_like(self.a_g) * eps

        # Initialize B_t from N_f_t if not provided
        if self.B_t is None:
            self.B_t = 1.0 / np.maximum(self.N_f_t, eps)

        # Initialize gamma_g if not provided (default to 1.0)
        if self.gamma_g is None:
            self.gamma_g = np.ones_like(self.a_g)

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return len(self.a_g)

    @property
    def n_time(self) -> int:
        """Number of time points."""
        return len(self.N_f_t)

    @property
    def B_period(self) -> float:
        """B-period (G1) duration as fraction of cell cycle."""
        return max(1.0 - self.C - self.D, 0.0)

    def get_regime_I_mask(self) -> np.ndarray:
        """Return boolean mask for Regime I genes."""
        return self.regimes == "regime_I"

    def get_regime_II_mask(self) -> np.ndarray:
        """Return boolean mask for Regime II genes."""
        return self.regimes == "regime_II"

    def get_full_mask(self) -> np.ndarray:
        """Return boolean mask for genes using full model."""
        return self.regimes == "full"

    def count_regimes(self) -> dict[str, int]:
        """Count genes in each regime."""
        return {
            "regime_I": int(np.sum(self.regimes == "regime_I")),
            "regime_II": int(np.sum(self.regimes == "regime_II")),
            "full": int(np.sum(self.regimes == "full")),
        }

    def copy(self) -> BiophysTheta:
        """Create a deep copy of the parameter set."""
        return BiophysTheta(
            a_g=self.a_g.copy(),
            b_g=self.b_g.copy() if self.b_g is not None else None,
            regimes=self.regimes.copy(),
            N_f_t=self.N_f_t.copy(),
            B_t=self.B_t.copy() if self.B_t is not None else None,
            x_g=self.x_g.copy(),
            C=self.C,
            D=self.D,
            tau=self.tau,
            gamma_g=self.gamma_g.copy() if self.gamma_g is not None else None,
        )

    def get_t_rep(self) -> np.ndarray:
        """Get replication time for each gene."""
        return compute_t_rep(self.x_g, self.C, self.D)


def compute_t_rep(x_g: np.ndarray, C: float, D: float) -> np.ndarray:
    """Compute replication time for each gene using Cooper-Helmstetter model.

    The replication time is when the replication fork reaches each gene,
    causing the gene dosage to double from 1 to 2.

    Formula: t_rep(x) = B + C × x = (1 - C - D) + C × x

    Args:
        x_g: Gene chromosomal positions, normalized to [0, 1].
        C: C-period duration as fraction of cell cycle.
        D: D-period duration as fraction of cell cycle.

    Returns:
        t_rep: Replication time for each gene, in [0, 1].
    """
    x_g = np.asarray(x_g, dtype=np.float64)
    B = max(1.0 - C - D, 0.0)
    t_rep = B + C * x_g
    return np.clip(t_rep, 0.0, 1.0)


def compute_dosage_grid(
    t_grid: np.ndarray,
    x_g: np.ndarray,
    C: float,
    D: float,
) -> np.ndarray:
    """Compute gene dosage g_g(t) for all genes and time points.

    Gene dosage is 1 before replication and 2 after.

    Args:
        t_grid: Time grid, normalized to [0, 1].
        x_g: Gene chromosomal positions, normalized to [0, 1].
        C: C-period duration.
        D: D-period duration.

    Returns:
        dosage_gt: Gene dosage matrix (n_genes, n_time).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    t_rep = compute_t_rep(x_g, C, D)

    # dosage[g, t] = 2 if t >= t_rep[g] else 1
    dosage_gt = np.where(
        t_grid[None, :] >= t_rep[:, None],
        2.0,
        1.0,
    )
    return dosage_gt.astype(np.float64)


def compute_m_from_theta(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute expected gene expression m_g(t) from biophysical parameters.

    This is the core "one-to-one" function: θ → m_g(t).
    Gene profiles are DERIVED, not fitted freely.

    Uses the rank-2 NMF formulation (Eq. 14-15):
        Y[g,t] = a_g + b_g × B(t), where Y = G_emp / dosage

    So m_g(t) = dosage[g,t] × (a_g + b_g × B(t)) for the full model,
    or the regime-specific approximations.

    For regime I (b term dominates): m_g(t) ≈ dosage × b_g × B(t) ∝ dosage × N_f(t)
    For regime II (a term dominates): m_g(t) ≈ dosage × a_g (constant in N_f)

    Args:
        theta: Biophysical parameter set.
        t_grid: Time grid (n_time,).
        eps: Small constant for numerical stability.

    Returns:
        m_gt: Expected gene expression (n_genes, n_time).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_genes = theta.n_genes
    n_time = len(t_grid)

    # Compute dosage
    dosage_gt = compute_dosage_grid(t_grid, theta.x_g, theta.C, theta.D)

    # Get B(t) = 1/N_f(t)
    if theta.B_t is not None:
        B_t = theta.B_t
    else:
        B_t = 1.0 / np.maximum(theta.N_f_t, eps)

    # Initialize output
    m_gt = np.zeros((n_genes, n_time), dtype=np.float64)

    # Full model genes: m_g(t) = dosage × (a_g + b_g × B(t))
    # Note: This represents the denominator in Eq. 7, so we might invert
    # For now, use it as the fitted representation
    full_mask = theta.get_full_mask()
    if np.any(full_mask):
        if theta.b_g is not None:
            m_gt[full_mask] = dosage_gt[full_mask] * (
                theta.a_g[full_mask, None] +
                theta.b_g[full_mask, None] * B_t[None, :]
            )
        else:
            m_gt[full_mask] = dosage_gt[full_mask] * theta.a_g[full_mask, None]

    # Regime I genes: m_g(t) = g_g(t) × a_I × N_f(t)
    # Expression proportional to N_f(t) (linear scaling regime)
    regime_I_mask = theta.get_regime_I_mask()
    if np.any(regime_I_mask):
        m_gt[regime_I_mask] = (
            dosage_gt[regime_I_mask] *
            theta.a_g[regime_I_mask, None] *
            theta.N_f_t[None, :]
        )

    # Regime II genes: m_g(t) = g_g(t) × a_II
    # Expression constant in N_f(t) (saturation regime)
    regime_II_mask = theta.get_regime_II_mask()
    if np.any(regime_II_mask):
        m_gt[regime_II_mask] = (
            dosage_gt[regime_II_mask] *
            theta.a_g[regime_II_mask, None]
        )

    # Ensure non-negative
    return np.maximum(m_gt, eps)


def compute_m_equation7(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute expected gene expression using full Equation 7.

    Equation 7:
        m_i(t) = g_i(t) × Γ_i / γ_i / (1 + (k_off + Γ_i) / (N_f(t) × k_on))

    Using the NMF parameters (Eq. 15):
        a_i = M_0 γ_i / Γ_i
        b_i = a_i × (k_off + Γ_i) / k_on

    We can write:
        m_i(t) ∝ g_i(t) / (a_i + b_i × B(t))

    Args:
        theta: Biophysical parameter set.
        t_grid: Time grid (n_time,).
        eps: Small constant for numerical stability.

    Returns:
        m_gt: Expected gene expression (n_genes, n_time).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_genes = theta.n_genes
    n_time = len(t_grid)

    # Compute dosage
    dosage_gt = compute_dosage_grid(t_grid, theta.x_g, theta.C, theta.D)

    # Get B(t) = 1/N_f(t)
    if theta.B_t is not None:
        B_t = theta.B_t
    else:
        B_t = 1.0 / np.maximum(theta.N_f_t, eps)

    # Ensure b_g is available
    b_g = theta.b_g if theta.b_g is not None else np.ones_like(theta.a_g) * eps

    # Denominator: a_i + b_i × B(t)
    denom = theta.a_g[:, None] + b_g[:, None] * B_t[None, :]
    denom = np.maximum(denom, eps)

    # m_g(t) = dosage / denom
    # This inverts the relationship from Eq. 14
    m_gt = dosage_gt / denom

    return np.maximum(m_gt, eps)


def compute_m_equation2_integral(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute expected gene expression using full Equation 2 (integral form).

    Equation 2 (for non-quasi-steady-state dynamics):
        m(t) = [∫₀^τ e^{γ(t'-t)} g(t')/(1+b/N_f(t')) dt'] / (2e^{γτ}-1)
             + ∫₀^t e^{γ(t'-t)} g(t')/(1+b/N_f(t')) dt'

    This is used when the quasi-steady-state assumption doesn't hold.

    Args:
        theta: Biophysical parameter set.
        t_grid: Time grid (n_time,).
        eps: Small constant for numerical stability.

    Returns:
        m_gt: Expected gene expression (n_genes, n_time).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_genes = theta.n_genes
    n_time = len(t_grid)
    tau = theta.tau

    # Compute dosage
    dosage_gt = compute_dosage_grid(t_grid, theta.x_g, theta.C, theta.D)

    # Get gamma_g (degradation rates)
    gamma_g = theta.gamma_g if theta.gamma_g is not None else np.ones(n_genes)

    # Get b parameter for Eq. 2
    # In Eq. 2, b = (k_off + Γ) / k_on, which relates to b_g/a_g
    b_g = theta.b_g if theta.b_g is not None else np.ones(n_genes) * eps
    b_ratio = b_g / np.maximum(theta.a_g, eps)  # b_i / a_i = (k_off + Γ) / k_on

    m_gt = np.zeros((n_genes, n_time), dtype=np.float64)
    dt = t_grid[1] - t_grid[0] if n_time > 1 else 1.0 / n_time

    for g in range(n_genes):
        gamma = gamma_g[g]
        b = b_ratio[g]
        dosage = dosage_gt[g]

        # Compute the periodic integral term (from t'=0 to τ)
        prefactor = 1.0 / (2 * np.exp(gamma * tau) - 1)

        for t_idx, t in enumerate(t_grid):
            # First integral: ∫₀^τ e^{γ(t'-t)} g(t')/(1+b/N_f(t')) dt'
            integral1 = 0.0
            for tp_idx in range(n_time):
                tp = t_grid[tp_idx]
                exp_factor = np.exp(gamma * (tp - t))
                denom = 1.0 + b / np.maximum(theta.N_f_t[tp_idx], eps)
                integral1 += exp_factor * dosage[tp_idx] / denom * dt

            # Second integral: ∫₀^t e^{γ(t'-t)} g(t')/(1+b/N_f(t')) dt'
            integral2 = 0.0
            for tp_idx in range(t_idx + 1):
                tp = t_grid[tp_idx]
                exp_factor = np.exp(gamma * (tp - t))
                denom = 1.0 + b / np.maximum(theta.N_f_t[tp_idx], eps)
                integral2 += exp_factor * dosage[tp_idx] / denom * dt

            m_gt[g, t_idx] = prefactor * integral1 + integral2

    return np.maximum(m_gt, eps)


def compute_psi_from_m(
    m_gt: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute gene fractions ψ_g(t) = m_g(t) / Σ_j m_j(t).

    Args:
        m_gt: Expected gene expression (n_genes, n_time).
        eps: Small constant for numerical stability.

    Returns:
        psi_gt: Gene fractions (n_genes, n_time), each column sums to 1.
    """
    m_gt = np.asarray(m_gt, dtype=np.float64)
    sum_m = np.sum(m_gt, axis=0, keepdims=True)
    psi_gt = m_gt / np.maximum(sum_m, eps)
    return psi_gt


def theta_to_fractions(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute gene fractions p_g(t) from biophysical parameters.

    Convenience function that combines compute_m_from_theta and compute_psi_from_m.

    Args:
        theta: Biophysical parameter set.
        t_grid: Time grid.
        eps: Small constant for numerical stability.

    Returns:
        psi_gt: Gene fractions (n_genes, n_time).
    """
    m_gt = compute_m_from_theta(theta, t_grid, eps=eps)
    return compute_psi_from_m(m_gt, eps=eps)


def initialize_theta(
    n_genes: int,
    n_time: int,
    C_init: float = 0.6,
    D_init: float = 0.2,
    tau: float = 1.0,
    seed: int | None = None,
) -> BiophysTheta:
    """Initialize biophysical parameters randomly.

    Args:
        n_genes: Number of genes.
        n_time: Number of time points.
        C_init: Initial C-period value.
        D_init: Initial D-period value.
        tau: Division time.
        seed: Random seed for reproducibility.

    Returns:
        theta: Initialized parameter set.
    """
    rng = np.random.default_rng(seed)

    # Initialize all genes as Regime I (will be refined during inference)
    regimes = np.array(["regime_I"] * n_genes, dtype=object)

    # Random gene positions (uniform on chromosome)
    x_g = rng.uniform(0.0, 1.0, size=n_genes)

    # Random amplitude parameters (positive)
    a_g = rng.uniform(0.5, 2.0, size=n_genes)

    # Random b_g parameters (N_f-dependent term)
    b_g = rng.uniform(0.1, 1.0, size=n_genes)

    # Initialize N_f(t) as roughly constant with small variation
    N_f_t = rng.uniform(0.8, 1.2, size=n_time)

    # B(t) = 1/N_f(t)
    B_t = 1.0 / N_f_t

    # Default gamma_g (degradation rates)
    gamma_g = np.ones(n_genes)

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
        gamma_g=gamma_g,
    )


def validate_theta(theta: BiophysTheta) -> None:
    """Validate that theta parameters are consistent and valid.

    Raises:
        ValueError: If any parameters are invalid.
    """
    # Check shapes
    if theta.a_g.shape[0] != theta.x_g.shape[0]:
        raise ValueError("a_g and x_g must have same length")
    if theta.a_g.shape[0] != theta.regimes.shape[0]:
        raise ValueError("a_g and regimes must have same length")
    if theta.b_g is not None and theta.a_g.shape[0] != theta.b_g.shape[0]:
        raise ValueError("a_g and b_g must have same length")
    if theta.gamma_g is not None and theta.a_g.shape[0] != theta.gamma_g.shape[0]:
        raise ValueError("a_g and gamma_g must have same length")

    # Check values
    if np.any(theta.a_g < 0):
        raise ValueError("a_g must be non-negative")
    if theta.b_g is not None and np.any(theta.b_g < 0):
        raise ValueError("b_g must be non-negative")
    if np.any(theta.N_f_t < 0):
        raise ValueError("N_f_t must be non-negative")
    if theta.B_t is not None and np.any(theta.B_t < 0):
        raise ValueError("B_t must be non-negative")
    if np.any((theta.x_g < 0) | (theta.x_g > 1)):
        raise ValueError("x_g must be in [0, 1]")
    if theta.C < 0 or theta.C > 1:
        raise ValueError("C must be in [0, 1]")
    if theta.D < 0 or theta.D > 1:
        raise ValueError("D must be in [0, 1]")
    if theta.C + theta.D > 1:
        raise ValueError("C + D must be <= 1")
    if theta.gamma_g is not None and np.any(theta.gamma_g <= 0):
        raise ValueError("gamma_g must be positive")

    # Check regimes
    valid_regimes = {"regime_I", "regime_II", "full"}
    invalid = set(theta.regimes) - valid_regimes
    if invalid:
        raise ValueError(f"Invalid regimes: {invalid}")
