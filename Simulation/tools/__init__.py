"""Utility tools for simulation data generation.

This module contains utilities for generating random gene data,
Nf(t) vectors, and other simulation inputs.
"""

from Simulation.tools.random_gene_data import (
    generate_genes,
    main as run_data_generator,
)

__all__ = [
    "generate_genes",
    "run_data_generator",
]
