"""Biophysical model computation and regime selection.

This module contains functions for computing gene expression profiles m_g(t)
using the identifiable parameter representation.

For theta-based inference, use:
- biophys_model.compute_m_from_theta() for computing m_g(t)
- nmf_fit.fit_theta_from_empirical() for parameter fitting

Key principle: We only work with identifiable parameters (a_g, N_f_t, regimes),
not the raw biophysical parameters (gamma, Gamma, k_on, k_off).
"""

from __future__ import annotations

import numpy as np


def infer_regimes_by_likelihood(
    G_emp: np.ndarray,
    dosage_gt: np.ndarray,
    a_g: np.ndarray,
    N_f_t: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Infer gene regimes by comparing likelihood under each model.

    This function computes the expected expression under both regime
    assumptions and selects the regime that maximizes likelihood.

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        dosage_gt: Gene dosage (n_genes, n_time).
        a_g: Amplitude parameters (n_genes,).
        N_f_t: Free RNAP trajectory (n_time,).
        eps: Numerical stability constant.

    Returns:
        regimes: Regime assignments ("regime_I" or "regime_II").
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    dosage_gt = np.asarray(dosage_gt, dtype=np.float64)
    n_genes = G_emp.shape[0]

    regimes = np.array(["regime_I"] * n_genes, dtype=object)

    for g in range(n_genes):
        # Regime I: m_g(t) = g_g(t) × a_I × N_f(t)
        m_I = dosage_gt[g] * a_g[g] * N_f_t
        m_I = np.maximum(m_I, eps)

        # Regime II: m_g(t) = g_g(t) × a_II (constant, use same a for fair comparison)
        # For Regime II, we estimate the best constant a_II
        Y = G_emp[g] / np.maximum(dosage_gt[g], eps)
        a_II_opt = np.mean(Y)
        m_II = dosage_gt[g] * a_II_opt
        m_II = np.maximum(m_II, eps)

        # Compute log-likelihood (proportional) for multinomial
        # LL ∝ Σ_t G_emp[g,t] × log(m[g,t])
        ll_I = float(np.sum(G_emp[g] * np.log(m_I)))
        ll_II = float(np.sum(G_emp[g] * np.log(m_II)))

        if ll_II > ll_I:
            regimes[g] = "regime_II"

    return regimes


def compute_m_from_identifiable_params(
    t_grid: np.ndarray,
    a_g: np.ndarray,
    N_f_t: np.ndarray,
    dosage_gt: np.ndarray,
    regimes: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute m_g(t) from identifiable parameters.

    This is the "one-to-one" function: θ → m_g(t).

    Args:
        t_grid: Time grid (n_time,).
        a_g: Amplitude parameters (n_genes,).
        N_f_t: Free RNAP trajectory (n_time,).
        dosage_gt: Gene dosage (n_genes, n_time).
        regimes: Regime per gene ("regime_I" or "regime_II").
        eps: Numerical stability constant.

    Returns:
        m_gt: Expected gene expression (n_genes, n_time).
    """
    n_genes, n_time = dosage_gt.shape
    m_gt = np.zeros((n_genes, n_time), dtype=np.float64)

    regime_I_mask = regimes == "regime_I"
    regime_II_mask = regimes == "regime_II"

    # Regime I: m_g(t) = g_g(t) × a_I × N_f(t)
    if np.any(regime_I_mask):
        m_gt[regime_I_mask] = (
            dosage_gt[regime_I_mask] *
            a_g[regime_I_mask, None] *
            N_f_t[None, :]
        )

    # Regime II: m_g(t) = g_g(t) × a_II
    if np.any(regime_II_mask):
        m_gt[regime_II_mask] = (
            dosage_gt[regime_II_mask] *
            a_g[regime_II_mask, None]
        )

    return np.maximum(m_gt, eps)


__all__ = [
    "infer_regimes_by_likelihood",
    "compute_m_from_identifiable_params",
]
