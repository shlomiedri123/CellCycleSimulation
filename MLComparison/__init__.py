"""Machine Learning comparison module for cell age inference.

This module provides ML-based approaches to infer cell ages from single-cell
mRNA count data, for comparison with the Bayesian inference approach.

Main components:
- models: Neural network architectures for age prediction
- training: Training utilities and data loaders
- evaluation: Metrics and comparison with Bayesian inference
"""

__all__ = [
    "models",
    "training",
    "evaluation",
]
