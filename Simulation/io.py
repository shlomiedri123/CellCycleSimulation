"""I/O utilities for simulation input/output.

Handles loading configuration, gene tables, Nf vectors, and saving snapshots.
Also includes optional measurement post-processing utilities.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
from typing import Any, List, Mapping, Sequence

import numpy as np
import yaml

from Simulation.config import SimulationConfig
from Simulation.gene import GeneConfig


# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------

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
    """Load and validate simulation configuration from YAML."""
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


# -----------------------------------------------------------------------------
# Gene table loading
# -----------------------------------------------------------------------------

def load_gene_table(path: str | pathlib.Path) -> List[GeneConfig]:
    """Read per-gene kinetic parameters from CSV."""
    genes: List[GeneConfig] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"gene_id", "chrom_pos_bp", "k_on_rnap", "k_off_rnap", "Gamma_esc", "gamma_deg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in gene CSV: {missing}")
        for row in reader:
            phase_val = row.get("phase", "") or row.get("regime", "")
            phase_norm = None
            if phase_val:
                phase_key = str(phase_val).strip().upper()
                if phase_key in {"1", "I"}:
                    phase_norm = "I"
                elif phase_key in {"2", "II"}:
                    phase_norm = "II"
                else:
                    raise ValueError(f"phase must be 'I' or 'II' for {row.get('gene_id', '')}")
            genes.append(
                GeneConfig(
                    gene_id=row["gene_id"],
                    chrom_pos_bp=float(row["chrom_pos_bp"]),
                    k_on_rnap=float(row["k_on_rnap"]),
                    k_off_rnap=float(row["k_off_rnap"]),
                    Gamma_esc=float(row["Gamma_esc"]),
                    gamma_deg=float(row["gamma_deg"]),
                    phase=phase_norm,
                )
            )
    return genes


# -----------------------------------------------------------------------------
# Nf vector loading
# -----------------------------------------------------------------------------

def load_nf_vector(path: str | pathlib.Path) -> np.ndarray:
    """Load deterministic Nf(t) vector for RNAP limitation."""
    path = pathlib.Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        data = np.loadtxt(path)

    vec = np.asarray(data, dtype=float).squeeze()
    if vec.ndim != 1:
        raise ValueError(f"Nf vector must be 1D; got shape {vec.shape} from {path}")
    if vec.size == 0:
        raise ValueError(f"Nf vector is empty: {path}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Nf vector contains NaN/inf values: {path}")
    if np.any(vec <= 0):
        raise ValueError(f"Nf vector must be positive: {path}")
    return vec


# -----------------------------------------------------------------------------
# Snapshot I/O
# -----------------------------------------------------------------------------

_BASE_FIELDS = ["cell_id", "parent_id", "generation", "age", "theta_rad", "phase"]


def save_snapshot_csv(rows: Sequence[Mapping[str, object]], path: str | pathlib.Path) -> None:
    """Save simulation snapshots to CSV."""
    if not rows:
        raise ValueError("No snapshot rows to write")
    gene_fields: list[str] = []
    for key in rows[0].keys():
        if key not in _BASE_FIELDS:
            gene_fields.append(str(key))
    fieldnames = _BASE_FIELDS + gene_fields

    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_snapshot_csv(path: str | pathlib.Path) -> list[dict[str, object]]:
    """Load simulation snapshots from CSV."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No snapshot rows found in {path}")
    return rows


# -----------------------------------------------------------------------------
# Measurement post-processing utilities
# -----------------------------------------------------------------------------

def load_lognormal_params(path: str | pathlib.Path) -> dict:
    """Load log-normal parameters (mu, sigma) from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "mu" not in payload or "sigma" not in payload:
        raise ValueError("Lognormal parameter JSON must contain 'mu' and 'sigma'.")
    payload["mu"] = float(payload["mu"])
    payload["sigma"] = float(payload["sigma"])
    return payload


def load_s_vector(path: str | pathlib.Path) -> np.ndarray:
    """Load S vector for measurement sampling."""
    path = pathlib.Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        data = np.loadtxt(path)

    vec = np.asarray(data, dtype=float).squeeze()
    if vec.ndim != 1:
        raise ValueError(f"S vector must be 1D; got shape {vec.shape} from {path}")
    if vec.size == 0:
        raise ValueError(f"S vector is empty: {path}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"S vector contains NaN/inf values: {path}")
    if np.any(vec < 0):
        raise ValueError(f"S vector must be non-negative: {path}")
    return vec


def _counts_matrix_from_rows(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
) -> np.ndarray:
    if not rows:
        raise ValueError("No snapshot rows provided")
    if not gene_ids:
        raise ValueError("gene_ids must be non-empty")
    counts = np.array(
        [[float(row[gid]) for gid in gene_ids] for row in rows],
        dtype=float,
    )
    if np.any(counts < 0):
        raise ValueError("Snapshot counts must be non-negative.")
    counts = np.nan_to_num(counts, nan=0.0)
    if not np.all(np.isclose(counts, np.round(counts))):
        raise ValueError("Snapshot counts must be integers.")
    return counts.astype(int)


def _sample_without_replacement(
    counts: np.ndarray,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample without replacement from per-gene counts (multivariate hypergeometric)."""
    if n_sample < 0:
        raise ValueError("n_sample must be non-negative.")
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    total = int(np.sum(counts))
    if total == 0 or n_sample == 0:
        return np.zeros_like(counts, dtype=int)
    if n_sample > total:
        n_sample = total

    remaining = total
    remaining_sample = n_sample
    out = np.zeros_like(counts, dtype=int)
    for idx, k in enumerate(counts):
        k = int(k)
        if remaining_sample <= 0:
            break
        if k <= 0:
            remaining -= k
            continue
        draw = rng.hypergeometric(ngood=k, nbad=remaining - k, nsample=remaining_sample)
        out[idx] = draw
        remaining_sample -= draw
        remaining -= k
    return out


def _snap_counts(
    counts: np.ndarray,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = counts.astype(int, copy=False)
    total_int = int(np.sum(counts))
    if total_int <= 0:
        return np.zeros_like(counts, dtype=int)

    s_draw = rng.lognormal(mean=mu, sigma=sigma)
    s_int = int(np.floor(s_draw + 0.5))
    if s_int < 0:
        s_int = 0
    if s_int > total_int:
        s_int = total_int
    if s_int == 0:
        return np.zeros_like(counts, dtype=int)

    return _sample_without_replacement(counts, s_int, rng)


def _snap_counts_from_total(
    counts: np.ndarray,
    s_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = counts.astype(int, copy=False)
    total_int = int(np.sum(counts))
    if total_int <= 0:
        return np.zeros_like(counts, dtype=int)

    s_int = int(np.floor(float(s_value) + 0.5))
    if s_int < 0:
        s_int = 0
    if s_int > total_int:
        s_int = total_int
    if s_int == 0:
        return np.zeros_like(counts, dtype=int)

    return _sample_without_replacement(counts, s_int, rng)


def build_measured_counts_matrix(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    mu: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Apply lognormal snapping to snapshot rows and return a counts matrix."""
    counts = _counts_matrix_from_rows(rows, gene_ids)
    rng = np.random.default_rng(seed)
    measured = np.zeros_like(counts, dtype=int)
    for idx, row_counts in enumerate(counts):
        measured[idx] = _snap_counts(row_counts, mu, sigma, rng)
    return measured


def build_measured_counts_matrix_from_s(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    s_vector: Sequence[float],
    seed: int,
) -> np.ndarray:
    """Apply S-vector based sampling to snapshot rows."""
    counts = _counts_matrix_from_rows(rows, gene_ids)
    s_vec = np.asarray(s_vector, dtype=float).squeeze()
    if s_vec.ndim != 1:
        raise ValueError("S vector must be 1D")
    if s_vec.size == 0:
        raise ValueError("S vector must be non-empty")
    if not np.all(np.isfinite(s_vec)):
        raise ValueError("S vector contains NaN/inf values")
    if np.any(s_vec < 0):
        raise ValueError("S vector must be non-negative")
    rng = np.random.default_rng(seed)
    if s_vec.size == counts.shape[0]:
        s_vals = s_vec
    else:
        s_vals = rng.choice(s_vec, size=counts.shape[0], replace=True)
    measured = np.zeros_like(counts, dtype=int)
    for idx, row_counts in enumerate(counts):
        measured[idx] = _snap_counts_from_total(row_counts, s_vals[idx], rng)
    return measured


def _merge_measured_rows(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    measured: np.ndarray,
) -> list[dict]:
    parsed_rows: list[dict] = []
    for row, counts in zip(rows, measured):
        base = {field: row[field] for field in _BASE_FIELDS}
        for gid, val in zip(gene_ids, counts):
            base[gid] = int(val)
        parsed_rows.append(base)
    return parsed_rows


def build_measured_snapshots_from_counts(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    measured: np.ndarray,
) -> list[dict]:
    """Build measured snapshot rows from precomputed counts matrix."""
    measured = np.asarray(measured, dtype=int)
    if measured.ndim != 2:
        raise ValueError("measured must be a 2D array")
    if measured.shape[0] != len(rows):
        raise ValueError("measured row count must match snapshot rows")
    if measured.shape[1] != len(gene_ids):
        raise ValueError("measured column count must match gene_ids")
    return _merge_measured_rows(rows, gene_ids, measured)


def save_sparse_measured_matrix(
    counts: np.ndarray,
    cell_ids: Sequence[str],
    gene_ids: Sequence[str],
    path: str | pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Save a CSR matrix plus cell/gene metadata sidecars."""
    from scipy import sparse

    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D array")
    if counts.shape[0] != len(cell_ids):
        raise ValueError("cell_ids length must match counts rows")
    if counts.shape[1] != len(gene_ids):
        raise ValueError("gene_ids length must match counts columns")

    path_obj = pathlib.Path(path)
    if path_obj.suffix != ".npz":
        path_obj = path_obj.with_suffix(".npz")
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    sp = sparse.csr_matrix(counts)
    sparse.save_npz(path_obj, sp)

    cells_path = path_obj.with_suffix(".cells.txt")
    genes_path = path_obj.with_suffix(".genes.txt")
    with open(cells_path, "w", encoding="utf-8") as f:
        for c in cell_ids:
            f.write(f"{c}\n")
    with open(genes_path, "w", encoding="utf-8") as f:
        for g in gene_ids:
            f.write(f"{g}\n")
    return path_obj, cells_path, genes_path
