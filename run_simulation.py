from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import numpy as np

from simulation.io.config_io import load_simulation_config
from simulation.io.gene_io import load_gene_table
from simulation.io.nf_io import load_nf_vector
from simulation.io.output_io import (
    build_measured_counts_matrix,
    build_measured_counts_matrix_from_s,
    build_measured_snapshots_from_counts,
    load_lognormal_params,
    load_s_vector,
    save_snapshot_csv,
    save_sparse_measured_matrix,
)
from simulation.lineage.lineage_simulator import LineageSimulator
from simulation.models.replication import build_genes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNAP-limited lineage simulation.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to simulation YAML config (default: config.yaml)",
    )
    return parser.parse_args(argv)


def validate_time_grid(config, nf_vec: np.ndarray) -> int:
    steps_total = config.T_total / config.dt
    steps_total_int = int(round(steps_total))
    if not np.isclose(steps_total, steps_total_int, rtol=0.0, atol=1e-9):
        raise ValueError(f"T_total/dt must be an integer; got {steps_total}")

    steps_cycle = config.T_div / config.dt
    steps_cycle_int = int(round(steps_cycle))
    if not np.isclose(steps_cycle, steps_cycle_int, rtol=0.0, atol=1e-9):
        raise ValueError(f"T_div/dt must be an integer; got {steps_cycle}")
    if nf_vec.size != steps_cycle_int:
        raise ValueError(
            f"Nf vector length mismatch: expected {steps_cycle_int} from T_div/dt, got {nf_vec.size}"
        )
    return steps_total_int


def _resolve_parsed_csv_path(out_path: pathlib.Path, parsed_out_path: str | pathlib.Path | None) -> pathlib.Path:
    if parsed_out_path is not None:
        parsed_path = pathlib.Path(parsed_out_path)
        if parsed_path.suffix != ".npz":
            return parsed_path
    suffix = out_path.suffix or ".csv"
    return out_path.with_name(out_path.stem + "_parsed" + suffix)


def _resolve_sparse_path(out_path: pathlib.Path, parsed_out_path: str | pathlib.Path | None) -> pathlib.Path:
    if parsed_out_path is not None:
        sparse_path = pathlib.Path(parsed_out_path)
    else:
        sparse_path = out_path.with_name(out_path.stem + "_measured.npz")
    if sparse_path.suffix != ".npz":
        sparse_path = sparse_path.with_suffix(".npz")
    return sparse_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sim_config = load_simulation_config(args.config)
    if sim_config.sparse and sim_config.measured_dist_path is None and sim_config.measured_s_vector_path is None:
        raise ValueError("sparse output requires measured_dist_path or measured_s_vector_path in the YAML config")
    if sim_config.measured_dist_path is not None and sim_config.measured_s_vector_path is not None:
        raise ValueError("Provide only one of measured_dist_path or measured_s_vector_path")

    nf_vec = load_nf_vector(sim_config.nf_vector_path)
    validate_time_grid(sim_config, nf_vec)
    gene_configs = load_gene_table(sim_config.genes_path) # Create the config for each gene.
    genes = build_genes(gene_configs, sim_config) # Build Gene objects from the configs.

    simulator = LineageSimulator(sim_config, genes, nf_vec)
    snapshots = simulator.run()
    save_snapshot_csv(snapshots, sim_config.out_path)

    if sim_config.measured_s_vector_path is not None:
        s_vec = load_s_vector(sim_config.measured_s_vector_path)
        gene_ids = [g.gene_id for g in genes]
        measured_counts = build_measured_counts_matrix_from_s(
            snapshots,
            gene_ids,
            s_vec,
            seed=sim_config.random_seed,
        )

        parsed_rows = build_measured_snapshots_from_counts(
            snapshots,
            gene_ids,
            measured_counts,
        )
        parsed_path = _resolve_parsed_csv_path(pathlib.Path(sim_config.out_path), sim_config.parsed_out_path)
        save_snapshot_csv(parsed_rows, parsed_path)
        print(f"Wrote parsed snapshots to {parsed_path}")

        if sim_config.sparse:
            sparse_path = _resolve_sparse_path(pathlib.Path(sim_config.out_path), sim_config.parsed_out_path)
            cell_ids = [str(row["cell_id"]) for row in snapshots]
            matrix_path, cells_path, genes_path = save_sparse_measured_matrix(
                measured_counts,
                cell_ids,
                gene_ids,
                sparse_path,
            )
            print(f"Wrote sparse measured matrix to {matrix_path}")
            print(f"Wrote metadata to {cells_path} and {genes_path}")
    elif sim_config.measured_dist_path is not None:
        params = load_lognormal_params(sim_config.measured_dist_path)
        mu = float(params["mu"])
        sigma = float(params["sigma"])
        gene_ids = [g.gene_id for g in genes]
        measured_counts = build_measured_counts_matrix(
            snapshots,
            gene_ids,
            mu=mu,
            sigma=sigma,
            seed=sim_config.random_seed,
        )

        parsed_rows = build_measured_snapshots_from_counts(
            snapshots,
            gene_ids,
            measured_counts,
        )
        parsed_path = _resolve_parsed_csv_path(pathlib.Path(sim_config.out_path), sim_config.parsed_out_path)
        save_snapshot_csv(parsed_rows, parsed_path)
        print(f"Wrote parsed snapshots to {parsed_path}")

        if sim_config.sparse:
            sparse_path = _resolve_sparse_path(pathlib.Path(sim_config.out_path), sim_config.parsed_out_path)
            cell_ids = [str(row["cell_id"]) for row in snapshots]
            matrix_path, cells_path, genes_path = save_sparse_measured_matrix(
                measured_counts,
                cell_ids,
                gene_ids,
                sparse_path,
            )
            print(f"Wrote sparse measured matrix to {matrix_path}")
            print(f"Wrote metadata to {cells_path} and {genes_path}")

    print(f"Wrote {len(snapshots)} samples to {sim_config.out_path}")


if __name__ == "__main__":
    main()
