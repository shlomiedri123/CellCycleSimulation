"""Training utilities for ML models.

Provides data loading, preprocessing, and cross-validation utilities
for training neural network models on simulation data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    normalize_counts: bool = True
    log_transform: bool = True
    add_total_counts: bool = True
    train_fraction: float = 0.8
    seed: int = 42


def load_snapshot_data(
    snapshot_path: str | Path,
    gene_ids: Optional[list[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load simulation snapshot data.

    Args:
        snapshot_path: Path to snapshot CSV file
        gene_ids: Optional list of gene IDs to include (if None, uses all)

    Returns:
        counts: mRNA counts matrix (n_cells, n_genes)
        ages: Cell ages (n_cells,)
        gene_ids: List of gene IDs
    """
    with open(snapshot_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty snapshot file: {snapshot_path}")

    # Get gene columns (all columns except metadata)
    metadata_cols = {"cell_id", "parent_id", "generation", "age", "phase", "theta_rad"}
    all_gene_ids = [col for col in rows[0].keys() if col not in metadata_cols]

    if gene_ids is None:
        gene_ids = all_gene_ids
    else:
        # Verify requested genes exist
        missing = set(gene_ids) - set(all_gene_ids)
        if missing:
            raise ValueError(f"Gene IDs not found in snapshot: {missing}")

    n_cells = len(rows)
    n_genes = len(gene_ids)

    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    ages = np.zeros(n_cells, dtype=np.float64)

    for i, row in enumerate(rows):
        ages[i] = float(row["age"])
        for j, gid in enumerate(gene_ids):
            counts[i, j] = float(row[gid])

    return counts, ages, gene_ids


def preprocess_counts(
    counts: np.ndarray,
    config: DataConfig,
) -> np.ndarray:
    """Preprocess count matrix.

    Args:
        counts: Raw counts (n_cells, n_genes)
        config: Preprocessing configuration

    Returns:
        Preprocessed counts
    """
    X = counts.copy()

    # Log transform (add pseudocount for zeros)
    if config.log_transform:
        X = np.log1p(X)

    # Add total counts as additional feature
    if config.add_total_counts:
        total = np.sum(X, axis=1, keepdims=True)
        X = np.concatenate([X, total], axis=1)

    # Normalize
    if config.normalize_counts:
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    return X


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        train_fraction: Fraction of data for training
        seed: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    n_train = int(n_samples * train_fraction)

    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_validate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    **train_kwargs,
) -> dict:
    """Perform k-fold cross-validation.

    Args:
        model_factory: Function that creates a new model instance
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        n_folds: Number of folds
        seed: Random seed
        **train_kwargs: Arguments passed to model.fit()

    Returns:
        Dictionary with cross-validation results
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds

    # Shuffle data
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    results = {
        "fold_mse": [],
        "fold_mae": [],
        "fold_corr": [],
        "predictions": [],
        "true_values": [],
    }

    for fold in range(n_folds):
        # Define train/test split for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = model_factory()
        model.fit(X_train, y_train, **train_kwargs)

        # Evaluate
        y_pred = model.predict(X_test)

        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        corr = np.corrcoef(y_pred, y_test)[0, 1] if len(y_test) > 1 else float("nan")

        results["fold_mse"].append(float(mse))
        results["fold_mae"].append(float(mae))
        results["fold_corr"].append(float(corr))
        results["predictions"].extend(y_pred.tolist())
        results["true_values"].extend(y_test.tolist())

    # Compute overall metrics
    results["mean_mse"] = float(np.mean(results["fold_mse"]))
    results["std_mse"] = float(np.std(results["fold_mse"]))
    results["mean_mae"] = float(np.mean(results["fold_mae"]))
    results["std_mae"] = float(np.std(results["fold_mae"]))
    results["mean_corr"] = float(np.nanmean(results["fold_corr"]))
    results["std_corr"] = float(np.nanstd(results["fold_corr"]))

    return results


class EarlyStopping:
    """Early stopping callback for training."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """Check if training should stop.

        Args:
            loss: Current loss value

        Returns:
            True if training should stop
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """Generator that yields mini-batches.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        rng: Random generator

    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]
