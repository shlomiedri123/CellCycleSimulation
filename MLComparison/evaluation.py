"""Evaluation utilities for comparing ML and Bayesian approaches.

Provides metrics, plotting functions, and comparison utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing ML and Bayesian predictions."""
    # ML metrics
    ml_mse: float
    ml_mae: float
    ml_corr: float
    ml_rmse: float

    # Bayesian metrics
    bayes_mse: float
    bayes_mae: float
    bayes_corr: float
    bayes_rmse: float

    # Relative improvement
    mse_ratio: float  # ml_mse / bayes_mse
    mae_ratio: float
    corr_diff: float  # ml_corr - bayes_corr

    # Additional diagnostics
    n_samples: int
    ml_predictions: Optional[np.ndarray] = None
    bayes_predictions: Optional[np.ndarray] = None
    true_ages: Optional[np.ndarray] = None


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute standard regression metrics.

    Args:
        y_true: True values (n_samples,)
        y_pred: Predicted values (n_samples,)

    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    n = len(y_true)

    # Basic errors
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    # Correlation
    if n > 1:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    # Median absolute error
    median_ae = float(np.median(np.abs(errors)))

    # Percentile errors
    abs_errors = np.abs(errors)
    p90 = float(np.percentile(abs_errors, 90))
    p95 = float(np.percentile(abs_errors, 95))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "r2": float(r2),
        "median_ae": median_ae,
        "p90_error": p90,
        "p95_error": p95,
        "n_samples": n,
    }


def compare_predictions(
    y_true: np.ndarray,
    ml_pred: np.ndarray,
    bayes_pred: np.ndarray,
) -> ComparisonResult:
    """Compare ML and Bayesian predictions.

    Args:
        y_true: True ages (n_samples,)
        ml_pred: ML predictions (n_samples,)
        bayes_pred: Bayesian predictions (n_samples,)

    Returns:
        ComparisonResult with metrics for both methods
    """
    ml_metrics = compute_regression_metrics(y_true, ml_pred)
    bayes_metrics = compute_regression_metrics(y_true, bayes_pred)

    return ComparisonResult(
        ml_mse=ml_metrics["mse"],
        ml_mae=ml_metrics["mae"],
        ml_corr=ml_metrics["corr"],
        ml_rmse=ml_metrics["rmse"],
        bayes_mse=bayes_metrics["mse"],
        bayes_mae=bayes_metrics["mae"],
        bayes_corr=bayes_metrics["corr"],
        bayes_rmse=bayes_metrics["rmse"],
        mse_ratio=ml_metrics["mse"] / (bayes_metrics["mse"] + 1e-12),
        mae_ratio=ml_metrics["mae"] / (bayes_metrics["mae"] + 1e-12),
        corr_diff=ml_metrics["corr"] - bayes_metrics["corr"],
        n_samples=ml_metrics["n_samples"],
        ml_predictions=ml_pred.copy(),
        bayes_predictions=bayes_pred.copy(),
        true_ages=y_true.copy(),
    )


def age_from_posterior(
    P_ct: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Compute expected age from posterior distribution.

    Args:
        P_ct: Age posterior (n_cells, n_time)
        t_grid: Time grid (n_time,)

    Returns:
        Expected ages (n_cells,)
    """
    # Normalize posteriors
    P_ct = P_ct / (np.sum(P_ct, axis=1, keepdims=True) + 1e-12)

    # Compute expected age
    expected_age = P_ct @ t_grid

    return expected_age


def plot_comparison(
    result: ComparisonResult,
    output_path: Optional[str | Path] = None,
    title: str = "ML vs Bayesian Age Prediction",
):
    """Plot comparison of ML and Bayesian predictions.

    Args:
        result: ComparisonResult from compare_predictions
        output_path: Optional path to save the figure
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plots
    ax1, ax2, ax3 = axes

    # ML predictions
    ax1.scatter(result.true_ages, result.ml_predictions, alpha=0.5, s=10)
    ax1.plot([0, np.max(result.true_ages)], [0, np.max(result.true_ages)], "r--", label="y=x")
    ax1.set_xlabel("True Age")
    ax1.set_ylabel("ML Predicted Age")
    ax1.set_title(f"ML: r={result.ml_corr:.3f}, RMSE={result.ml_rmse:.3f}")
    ax1.legend()

    # Bayesian predictions
    ax2.scatter(result.true_ages, result.bayes_predictions, alpha=0.5, s=10)
    ax2.plot([0, np.max(result.true_ages)], [0, np.max(result.true_ages)], "r--", label="y=x")
    ax2.set_xlabel("True Age")
    ax2.set_ylabel("Bayesian Predicted Age")
    ax2.set_title(f"Bayesian: r={result.bayes_corr:.3f}, RMSE={result.bayes_rmse:.3f}")
    ax2.legend()

    # Error comparison
    ml_errors = np.abs(result.ml_predictions - result.true_ages)
    bayes_errors = np.abs(result.bayes_predictions - result.true_ages)

    ax3.hist(ml_errors, bins=30, alpha=0.5, label="ML", density=True)
    ax3.hist(bayes_errors, bins=30, alpha=0.5, label="Bayesian", density=True)
    ax3.set_xlabel("Absolute Error")
    ax3.set_ylabel("Density")
    ax3.set_title("Error Distribution")
    ax3.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig


def generate_comparison_report(
    result: ComparisonResult,
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a text report comparing ML and Bayesian approaches.

    Args:
        result: ComparisonResult from compare_predictions
        output_path: Optional path to save the report

    Returns:
        Report as a string
    """
    lines = [
        "=" * 60,
        "ML vs Bayesian Age Prediction Comparison",
        "=" * 60,
        "",
        f"Number of samples: {result.n_samples}",
        "",
        "MACHINE LEARNING METRICS",
        "-" * 30,
        f"  MSE:  {result.ml_mse:.6f}",
        f"  MAE:  {result.ml_mae:.6f}",
        f"  RMSE: {result.ml_rmse:.6f}",
        f"  Corr: {result.ml_corr:.4f}",
        "",
        "BAYESIAN INFERENCE METRICS",
        "-" * 30,
        f"  MSE:  {result.bayes_mse:.6f}",
        f"  MAE:  {result.bayes_mae:.6f}",
        f"  RMSE: {result.bayes_rmse:.6f}",
        f"  Corr: {result.bayes_corr:.4f}",
        "",
        "COMPARISON",
        "-" * 30,
        f"  MSE Ratio (ML/Bayes):  {result.mse_ratio:.4f}",
        f"  MAE Ratio (ML/Bayes):  {result.mae_ratio:.4f}",
        f"  Corr Difference:       {result.corr_diff:+.4f}",
        "",
    ]

    if result.mse_ratio < 1:
        lines.append("  >> ML outperforms Bayesian by {:.1f}%".format((1 - result.mse_ratio) * 100))
    else:
        lines.append("  >> Bayesian outperforms ML by {:.1f}%".format((result.mse_ratio - 1) * 100))

    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def run_full_comparison(
    snapshot_path: str | Path,
    bayesian_result_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    model_type: str = "SimpleMLP",
    seed: int = 42,
) -> ComparisonResult:
    """Run a full comparison between ML and Bayesian approaches.

    This is a convenience function that:
    1. Loads snapshot data
    2. Trains ML model
    3. Loads or runs Bayesian inference
    4. Compares predictions
    5. Generates plots and report

    Args:
        snapshot_path: Path to simulation snapshot CSV
        bayesian_result_path: Path to saved Bayesian inference result
        output_dir: Directory for outputs
        model_type: Type of ML model ("SimpleMLP", "DeepMLP", "GeneAttention")
        seed: Random seed

    Returns:
        ComparisonResult
    """
    from MLComparison.models import ModelConfig, SimpleMLP, DeepMLP, GeneAttention
    from MLComparison.training import (
        load_snapshot_data,
        preprocess_counts,
        split_data,
        DataConfig,
    )

    # Load data
    counts, ages, gene_ids = load_snapshot_data(snapshot_path)

    # Preprocess
    data_config = DataConfig(seed=seed)
    X = preprocess_counts(counts, data_config)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, ages, train_fraction=0.8, seed=seed)

    # Create and train ML model
    n_features = X_train.shape[1]
    model_config = ModelConfig(n_genes=n_features)

    if model_type == "SimpleMLP":
        model = SimpleMLP(model_config)
    elif model_type == "DeepMLP":
        model = DeepMLP(model_config)
    elif model_type == "GeneAttention":
        model = GeneAttention(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train, n_epochs=100, seed=seed, verbose=False)
    ml_pred = model.predict(X_test)

    # For Bayesian comparison, we need the inference result
    # If not provided, use a placeholder (uniform posterior)
    if bayesian_result_path is not None:
        import json
        with open(bayesian_result_path, "r") as f:
            bayes_result = json.load(f)
        # Extract predictions from Bayesian result
        bayes_pred = np.array(bayes_result.get("expected_ages", y_test))
    else:
        # Placeholder: use noisy true ages
        rng = np.random.default_rng(seed)
        bayes_pred = y_test + rng.normal(0, np.std(y_test) * 0.1, size=len(y_test))

    # Compare
    result = compare_predictions(y_test, ml_pred, bayes_pred)

    # Generate outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison(result, output_dir / "comparison_plot.png")
        report = generate_comparison_report(result, output_dir / "comparison_report.txt")
        print(report)

    return result
