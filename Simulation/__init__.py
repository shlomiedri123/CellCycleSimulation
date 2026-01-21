"""Stochastic bacterial cell-cycle simulation package.

This package provides tools for simulating RNAP-limited mRNA dynamics
in growing bacterial populations with cell division.

Main entry points:
- simulation.cli: Command-line interface for running simulations
- simulation.simulator: LineageSimulator class for programmatic use
- simulation.io: I/O utilities for loading configs and saving results
- simulation.gene: Gene parameter dataclasses
- simulation.config: Simulation configuration
- simulation.analysis: Analysis utilities for simulation outputs
"""

from Simulation.cell import Cell
from Simulation.config import SimulationConfig
from Simulation.gene import Gene, GeneConfig, build_gene, compute_t_rep
from Simulation.io import (
    load_simulation_config,
    load_gene_table,
    load_nf_vector,
    save_snapshot_csv,
    load_snapshot_csv,
)
from Simulation.simulator import LineageSimulator, build_genes

__all__ = [
    # Core classes
    "Cell",
    "Gene",
    "GeneConfig",
    "SimulationConfig",
    "LineageSimulator",
    # Functions
    "build_gene",
    "build_genes",
    "compute_t_rep",
    "load_simulation_config",
    "load_gene_table",
    "load_nf_vector",
    "save_snapshot_csv",
    "load_snapshot_csv",
]
