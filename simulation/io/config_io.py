from __future__ import annotations

import os
import pathlib
from typing import Any, Mapping

import yaml

from simulation.config.simulation_config import SimulationConfig


def _resolve_path(value: str, base_dir: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _check_readable(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise ValueError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{label} is not readable: {path}")


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config field: {key}")
    return raw[key]


def load_simulation_config(path: str | pathlib.Path) -> SimulationConfig:
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    base_dir = path.resolve().parent

    genes_path = _resolve_path(str(_require(raw, "genes_path")), base_dir)
    nf_vector_path = _resolve_path(str(_require(raw, "nf_vector_path")), base_dir)
    out_path = _resolve_path(str(_require(raw, "out_path")), base_dir)
    measured_dist_path = raw.get("measured_dist_path")
    measured_s_vector_path = raw.get("measured_s_vector_path")
    parsed_out_path = raw.get("parsed_out_path")
    sparse = raw.get("sparse", False)
    if not isinstance(sparse, bool):
        raise ValueError("sparse must be a boolean")

    _check_readable(genes_path, "genes_path")
    _check_readable(nf_vector_path, "nf_vector_path")
    if measured_dist_path is not None:
        measured_dist_path = _resolve_path(str(measured_dist_path), base_dir)
        _check_readable(measured_dist_path, "measured_dist_path")
    if measured_s_vector_path is not None:
        measured_s_vector_path = _resolve_path(str(measured_s_vector_path), base_dir)
        _check_readable(measured_s_vector_path, "measured_s_vector_path")
    if parsed_out_path is not None:
        parsed_out_path = _resolve_path(str(parsed_out_path), base_dir)

    n_target_samples = raw.get("N_target_samples", raw.get("n_cells"))
    if n_target_samples is None:
        raise ValueError("Missing required config field: N_target_samples (or n_cells)")

    cfg = SimulationConfig(
        B_period=float(_require(raw, "B_period")),
        C_period=float(_require(raw, "C_period")),
        D_period=float(_require(raw, "D_period")),
        T_total=float(_require(raw, "T_total")),
        dt=float(_require(raw, "dt")),
        N_target_samples=int(n_target_samples),
        random_seed=int(_require(raw, "random_seed")),
        chromosome_length_bp=float(_require(raw, "chromosome_length_bp")),
        MAX_MRNA_PER_GENE=int(_require(raw, "MAX_MRNA_PER_GENE")),
        genes_path=str(genes_path),
        nf_vector_path=str(nf_vector_path),
        out_path=str(out_path),
        measured_dist_path=str(measured_dist_path) if measured_dist_path is not None else None,
        measured_s_vector_path=str(measured_s_vector_path) if measured_s_vector_path is not None else None,
        parsed_out_path=str(parsed_out_path) if parsed_out_path is not None else None,
        sparse=bool(sparse),
        initial_cell_count=int(raw.get("initial_cell_count", 3)),
        division_time_cv=float(raw.get("division_time_cv", 0.05)),
        division_time_method=str(raw.get("division_time_method", "clip")),
        division_time_min=float(raw["division_time_min"]) if "division_time_min" in raw and raw["division_time_min"] is not None else None,
        division_time_max=float(raw["division_time_max"]) if "division_time_max" in raw and raw["division_time_max"] is not None else None,
        division_time_max_attempts=int(raw.get("division_time_max_attempts", 1000)),
        T_div=float(raw["T_div"]) if "T_div" in raw and raw["T_div"] is not None else None,
        snapshot_min_interval_steps=int(raw.get("snapshot_min_interval_steps", 10)),
        snapshot_jitter_steps=int(raw.get("snapshot_jitter_steps", 10)),
    )
    return cfg
