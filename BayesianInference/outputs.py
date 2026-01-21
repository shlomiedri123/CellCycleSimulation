"""Output generation module for Bayesian inference results.

This module provides functions to:
1. Plot gene expression profiles
2. Plot estimated age distribution
3. Plot N_f(t) function
4. Save inferred parameters to JSON
5. Compare inferred vs true parameters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from BayesianInference.biophys_model import BiophysTheta, compute_t_rep


def save_inference_results(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    t_map: np.ndarray,
    P_ct: np.ndarray,
    out_path: Path | str,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """Save inference results to JSON file.

    Creates a JSON file containing:
    - Per-gene parameters: a_i, b_i, gamma_i, t_rep
    - Global parameters: C, D, N_f(t)
    - Per-cell ages

    Args:
        theta: Fitted biophysical parameters.
        t_grid: Time grid.
        t_map: MAP age estimate per cell.
        P_ct: Posterior age distributions.
        out_path: Output file path.
        extra_params: Additional parameters to include.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute derived quantities
    t_rep = compute_t_rep(theta.x_g, theta.C, theta.D)

    # Build per-gene parameters
    genes = []
    for g in range(theta.n_genes):
        gene_params = {
            "gene_idx": g,
            "a": float(theta.a_g[g]),
            "b": float(theta.b_g[g]) if theta.b_g is not None else None,
            "gamma": float(theta.gamma_g[g]) if theta.gamma_g is not None else None,
            "t_rep": float(t_rep[g]),
            "x": float(theta.x_g[g]),
            "regime": str(theta.regimes[g]),
        }
        genes.append(gene_params)

    # Build global parameters
    global_params = {
        "C": float(theta.C),
        "D": float(theta.D),
        "B_period": float(theta.B_period),
        "tau": float(theta.tau),
        "N_f_t": theta.N_f_t.tolist(),
        "B_t": theta.B_t.tolist() if theta.B_t is not None else None,
        "t_grid": t_grid.tolist(),
    }

    # Build per-cell ages
    cells = []
    for c in range(len(t_map)):
        cell_params = {
            "cell_idx": c,
            "t_map": float(t_map[c]),
            "posterior_mean": float(np.sum(P_ct[c] * t_grid)),
            "posterior_std": float(np.sqrt(np.sum(P_ct[c] * (t_grid - np.sum(P_ct[c] * t_grid))**2))),
        }
        cells.append(cell_params)

    # Combine all
    result = {
        "genes": genes,
        "global_params": global_params,
        "cells": cells,
        "n_genes": theta.n_genes,
        "n_cells": len(t_map),
        "n_time": theta.n_time,
        "regime_counts": theta.count_regimes(),
    }

    if extra_params:
        result["extra"] = extra_params

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def plot_gene_profiles(
    G_gt: np.ndarray,
    t_grid: np.ndarray,
    theta: BiophysTheta | None = None,
    max_genes: int = 20,
    n_cols: int = 2,
    out_path: Path | str | None = None,
    G_true: np.ndarray | None = None,
) -> None:
    """Plot gene expression profiles in a grid.

    Args:
        G_gt: Inferred gene profiles (n_genes, n_time).
        t_grid: Time grid.
        theta: Optional theta for adding regime info.
        max_genes: Maximum number of genes to plot.
        n_cols: Number of columns in the grid.
        out_path: If provided, save figure to this path.
        G_true: Optional true profiles for comparison.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_genes = min(G_gt.shape[0], max_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_genes):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        ax.plot(t_grid, G_gt[i], "b-", linewidth=1.5, label="Inferred")

        if G_true is not None:
            ax.plot(t_grid, G_true[i], "r--", linewidth=1, alpha=0.7, label="True")

        title = f"Gene {i}"
        if theta is not None:
            regime = theta.regimes[i]
            t_rep = compute_t_rep(theta.x_g[i:i+1], theta.C, theta.D)[0]
            title += f" ({regime}, t_rep={t_rep:.2f})"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Cell age (t)", fontsize=8)
        ax.set_ylabel("m(t)", fontsize=8)
        ax.tick_params(labelsize=7)

        if i == 0 and G_true is not None:
            ax.legend(fontsize=7)

    # Hide empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_age_distribution(
    t_map: np.ndarray,
    t_grid: np.ndarray,
    P_ct: np.ndarray | None = None,
    t_true: np.ndarray | None = None,
    out_path: Path | str | None = None,
) -> None:
    """Plot estimated age distribution of the dataset.

    Args:
        t_map: MAP age estimates per cell.
        t_grid: Time grid.
        P_ct: Optional posterior distributions for showing uncertainty.
        t_true: Optional true ages for comparison.
        out_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Histogram of MAP ages
    ax1 = axes[0]
    ax1.hist(t_map, bins=30, density=True, alpha=0.7, label="Inferred")

    if t_true is not None:
        ax1.hist(t_true, bins=30, density=True, alpha=0.5, label="True")
        ax1.legend()

    # Add theoretical exponential distribution
    t_theory = np.linspace(0, 1, 100)
    p_theory = 2 * np.log(2) * 2**(-t_theory)  # Eq. 16
    ax1.plot(t_theory, p_theory, "k--", label="Theory (Eq. 16)")
    ax1.legend()

    ax1.set_xlabel("Cell age (t)")
    ax1.set_ylabel("Density")
    ax1.set_title("Age Distribution")

    # Plot 2: Mean posterior vs MAP
    ax2 = axes[1]
    if P_ct is not None:
        posterior_mean = np.sum(P_ct * t_grid, axis=1)
        ax2.scatter(t_map, posterior_mean, alpha=0.3, s=5)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax2.set_xlabel("MAP age")
        ax2.set_ylabel("Posterior mean age")
        ax2.set_title("MAP vs Posterior Mean")
    else:
        ax2.hist2d(t_map, t_map, bins=30)
        ax2.set_xlabel("Cell index")
        ax2.set_ylabel("MAP age")
        ax2.set_title("MAP Ages")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_nf_trajectory(
    N_f_t: np.ndarray,
    t_grid: np.ndarray,
    N_f_true: np.ndarray | None = None,
    out_path: Path | str | None = None,
) -> None:
    """Plot estimated N_f(t) function.

    Args:
        N_f_t: Inferred free RNAP trajectory.
        t_grid: Time grid.
        N_f_true: Optional true N_f(t) for comparison.
        out_path: If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot N_f(t)
    ax1 = axes[0]
    ax1.plot(t_grid, N_f_t, "b-", linewidth=2, label="Inferred")

    if N_f_true is not None:
        ax1.plot(t_grid, N_f_true, "r--", linewidth=1.5, alpha=0.7, label="True")
        ax1.legend()

    ax1.set_xlabel("Cell age (t)")
    ax1.set_ylabel("N_f(t)")
    ax1.set_title("Free RNAP Trajectory")

    # Plot B(t) = 1/N_f(t)
    ax2 = axes[1]
    B_t = 1.0 / np.maximum(N_f_t, 1e-12)
    ax2.plot(t_grid, B_t, "b-", linewidth=2, label="Inferred")

    if N_f_true is not None:
        B_true = 1.0 / np.maximum(N_f_true, 1e-12)
        ax2.plot(t_grid, B_true, "r--", linewidth=1.5, alpha=0.7, label="True")
        ax2.legend()

    ax2.set_xlabel("Cell age (t)")
    ax2.set_ylabel("B(t) = 1/N_f(t)")
    ax2.set_title("Inverse Free RNAP")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_parameter_recovery(
    theta_inferred: BiophysTheta,
    theta_true: BiophysTheta,
    t_grid: np.ndarray,
    out_path: Path | str | None = None,
) -> dict[str, float]:
    """Plot inferred vs true parameters for validation.

    Creates a multi-panel figure comparing:
    - a_g inferred vs true
    - b_g inferred vs true
    - N_f(t) inferred vs true
    - t_rep inferred vs true

    Args:
        theta_inferred: Inferred parameters.
        theta_true: True parameters.
        t_grid: Time grid.
        out_path: If provided, save figure to this path.

    Returns:
        Dictionary of R-squared values for each parameter.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return {}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    metrics = {}

    # Plot a_g
    ax = axes[0, 0]
    ax.scatter(theta_true.a_g, theta_inferred.a_g, alpha=0.5, s=10)
    ax.plot([theta_true.a_g.min(), theta_true.a_g.max()],
            [theta_true.a_g.min(), theta_true.a_g.max()], "k--")
    ax.set_xlabel("True a_g")
    ax.set_ylabel("Inferred a_g")
    ax.set_title("a_g Recovery")
    r2 = _compute_r2(theta_true.a_g, theta_inferred.a_g)
    metrics["r2_a_g"] = r2
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    # Plot b_g
    ax = axes[0, 1]
    if theta_true.b_g is not None and theta_inferred.b_g is not None:
        ax.scatter(theta_true.b_g, theta_inferred.b_g, alpha=0.5, s=10)
        ax.plot([theta_true.b_g.min(), theta_true.b_g.max()],
                [theta_true.b_g.min(), theta_true.b_g.max()], "k--")
        r2 = _compute_r2(theta_true.b_g, theta_inferred.b_g)
        metrics["r2_b_g"] = r2
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("True b_g")
    ax.set_ylabel("Inferred b_g")
    ax.set_title("b_g Recovery")

    # Plot N_f(t)
    ax = axes[0, 2]
    ax.plot(t_grid, theta_true.N_f_t, "r-", label="True")
    ax.plot(t_grid, theta_inferred.N_f_t, "b--", label="Inferred")
    ax.set_xlabel("Cell age (t)")
    ax.set_ylabel("N_f(t)")
    ax.set_title("N_f(t) Recovery")
    ax.legend()
    r2 = _compute_r2(theta_true.N_f_t, theta_inferred.N_f_t)
    metrics["r2_N_f"] = r2
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    # Plot t_rep
    ax = axes[1, 0]
    t_rep_true = compute_t_rep(theta_true.x_g, theta_true.C, theta_true.D)
    t_rep_inferred = compute_t_rep(theta_inferred.x_g, theta_inferred.C, theta_inferred.D)
    ax.scatter(t_rep_true, t_rep_inferred, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("True t_rep")
    ax.set_ylabel("Inferred t_rep")
    ax.set_title("t_rep Recovery")
    r2 = _compute_r2(t_rep_true, t_rep_inferred)
    metrics["r2_t_rep"] = r2
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    # Plot x_g
    ax = axes[1, 1]
    ax.scatter(theta_true.x_g, theta_inferred.x_g, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("True x_g")
    ax.set_ylabel("Inferred x_g")
    ax.set_title("Position Recovery")
    r2 = _compute_r2(theta_true.x_g, theta_inferred.x_g)
    metrics["r2_x_g"] = r2
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top")

    # Plot regime assignment accuracy
    ax = axes[1, 2]
    regime_match = theta_true.regimes == theta_inferred.regimes
    regime_accuracy = np.mean(regime_match)
    metrics["regime_accuracy"] = float(regime_accuracy)

    # Count regimes
    regime_counts_true = theta_true.count_regimes()
    regime_counts_inferred = theta_inferred.count_regimes()

    bar_width = 0.35
    x = np.arange(3)
    ax.bar(x - bar_width/2, [regime_counts_true.get("regime_I", 0),
                             regime_counts_true.get("regime_II", 0),
                             regime_counts_true.get("full", 0)],
           bar_width, label="True")
    ax.bar(x + bar_width/2, [regime_counts_inferred.get("regime_I", 0),
                             regime_counts_inferred.get("regime_II", 0),
                             regime_counts_inferred.get("full", 0)],
           bar_width, label="Inferred")
    ax.set_xticks(x)
    ax.set_xticklabels(["Regime I", "Regime II", "Full"])
    ax.set_ylabel("Count")
    ax.set_title(f"Regime Assignment (Acc: {regime_accuracy:.1%})")
    ax.legend()

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return metrics


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared between true and predicted values."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def generate_convergence_report(
    theta: BiophysTheta,
    t_grid: np.ndarray,
    t_map: np.ndarray,
    P_ct: np.ndarray,
    G_gt: np.ndarray,
    out_dir: Path | str,
    theta_true: BiophysTheta | None = None,
    G_true: np.ndarray | None = None,
    t_true: np.ndarray | None = None,
    N_f_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate complete convergence report with all plots and JSON.

    This is the main function to call after inference converges.

    Args:
        theta: Fitted biophysical parameters.
        t_grid: Time grid.
        t_map: MAP age estimates.
        P_ct: Posterior distributions.
        G_gt: Inferred gene profiles.
        out_dir: Output directory.
        theta_true: Optional true parameters for validation.
        G_true: Optional true gene profiles.
        t_true: Optional true cell ages.
        N_f_true: Optional true N_f(t).

    Returns:
        Dictionary of metrics and file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"files": {}, "metrics": {}}

    # Save parameters to JSON
    json_path = out_dir / "inferred_parameters.json"
    save_inference_results(theta, t_grid, t_map, P_ct, json_path)
    report["files"]["parameters"] = str(json_path)

    # Plot gene profiles
    profile_path = out_dir / "gene_profiles.png"
    plot_gene_profiles(G_gt, t_grid, theta, out_path=profile_path, G_true=G_true)
    report["files"]["gene_profiles"] = str(profile_path)

    # Plot age distribution
    age_path = out_dir / "age_distribution.png"
    plot_age_distribution(t_map, t_grid, P_ct, t_true=t_true, out_path=age_path)
    report["files"]["age_distribution"] = str(age_path)

    # Plot N_f(t)
    nf_path = out_dir / "nf_trajectory.png"
    plot_nf_trajectory(theta.N_f_t, t_grid, N_f_true=N_f_true, out_path=nf_path)
    report["files"]["nf_trajectory"] = str(nf_path)

    # If true parameters are provided, do parameter recovery analysis
    if theta_true is not None:
        recovery_path = out_dir / "parameter_recovery.png"
        recovery_metrics = plot_parameter_recovery(
            theta, theta_true, t_grid, out_path=recovery_path
        )
        report["files"]["parameter_recovery"] = str(recovery_path)
        report["metrics"]["recovery"] = recovery_metrics

    # Summary metrics
    report["metrics"]["n_genes"] = theta.n_genes
    report["metrics"]["n_cells"] = len(t_map)
    report["metrics"]["n_time"] = theta.n_time
    report["metrics"]["regime_counts"] = theta.count_regimes()
    report["metrics"]["C"] = theta.C
    report["metrics"]["D"] = theta.D

    # Save report summary
    report_path = out_dir / "convergence_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["files"]["report"] = str(report_path)

    return report
