from __future__ import annotations

import csv
import json
import pathlib
from typing import Mapping, Sequence

import numpy as np

_BASE_FIELDS = ["cell_id", "parent_id", "generation", "age", "theta_rad", "phase"]


def save_snapshot_csv(rows: Sequence[Mapping[str, object]], path: str | pathlib.Path) -> None:
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
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No snapshot rows found in {path}")
    return rows


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


def build_measured_snapshots(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    mu: float,
    sigma: float,
    seed: int,
) -> list[dict]:
    measured = build_measured_counts_matrix(rows, gene_ids, mu, sigma, seed)
    return _merge_measured_rows(rows, gene_ids, measured)


def build_measured_snapshots_from_s(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    s_vector: Sequence[float],
    seed: int,
) -> list[dict]:
    measured = build_measured_counts_matrix_from_s(rows, gene_ids, s_vector, seed)
    return _merge_measured_rows(rows, gene_ids, measured)


def build_measured_snapshots_from_counts(
    rows: Sequence[Mapping[str, object]],
    gene_ids: Sequence[str],
    measured: np.ndarray,
) -> list[dict]:
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
    _write_lines(cells_path, [str(c) for c in cell_ids])
    _write_lines(genes_path, [str(g) for g in gene_ids])
    return path_obj, cells_path, genes_path


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


def _write_lines(path: pathlib.Path, values: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for val in values:
            f.write(f"{val}\n")
