from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

DEFAULT_METADATA_FIELDS = {
    "cell_id",
    "parent_id",
    "generation",
    "age",
    "theta_rad",
    "phase",
}


@dataclass
class ParamsTruth:
    t_true: np.ndarray | None
    G_true: np.ndarray | None
    gamma_g: np.ndarray | None
    Gamma_g: np.ndarray | None
    kon_g: np.ndarray | None
    koff_g: np.ndarray | None
    b_g: np.ndarray | None
    regime: np.ndarray | None
    nf_t: np.ndarray | None
    t_grid: np.ndarray | None
    mu: float | None
    sigma: float | None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def load_gene_ids(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "gene_id" not in (reader.fieldnames or []):
            raise ValueError("gene_id column is required in gene list CSV")
        gene_ids = [row["gene_id"].strip() for row in reader if row.get("gene_id")]
    return [gid for gid in gene_ids if gid]


def load_gene_positions(
    path: str | Path,
    gene_ids: Sequence[str] | None = None,
    chrom_length_bp: float | None = None,
) -> np.ndarray:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "gene_id" not in fieldnames or "chrom_pos_bp" not in fieldnames:
            raise ValueError("gene metadata CSV must include gene_id and chrom_pos_bp")
        pos_map: dict[str, float] = {}
        for row in reader:
            gid = row.get("gene_id", "").strip()
            if not gid:
                continue
            pos_val = row.get("chrom_pos_bp", "")
            if pos_val in (None, ""):
                continue
            pos_map[gid] = float(pos_val)
    if gene_ids is None:
        gene_ids = list(pos_map.keys())
    if chrom_length_bp is None or chrom_length_bp <= 0:
        raise ValueError("chrom_length_bp must be provided to normalize positions")
    pos = np.zeros(len(gene_ids), dtype=np.float64)
    for idx, gid in enumerate(gene_ids):
        if gid not in pos_map:
            raise ValueError(f"gene_id {gid} missing from gene metadata")
        pos[idx] = pos_map[gid] / float(chrom_length_bp)
    return pos.astype(np.float64)


def load_gene_params(
    path: str | Path,
    gene_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"gene_id", "gamma_deg", "Gamma_esc", "k_on_rnap", "k_off_rnap"}
        if not required.issubset(fieldnames):
            missing = ", ".join(sorted(required - set(fieldnames)))
            raise ValueError(f"gene metadata CSV missing columns: {missing}")
        param_map: dict[str, tuple[float, float, float, float]] = {}
        for row in reader:
            gid = row.get("gene_id", "").strip()
            if not gid:
                continue
            param_map[gid] = (
                float(row["gamma_deg"]),
                float(row["Gamma_esc"]),
                float(row["k_on_rnap"]),
                float(row["k_off_rnap"]),
            )
    if gene_ids is None:
        gene_ids = list(param_map.keys())
    gamma = np.zeros(len(gene_ids), dtype=np.float64)
    Gamma = np.zeros(len(gene_ids), dtype=np.float64)
    kon = np.zeros(len(gene_ids), dtype=np.float64)
    koff = np.zeros(len(gene_ids), dtype=np.float64)
    for idx, gid in enumerate(gene_ids):
        if gid not in param_map:
            raise ValueError(f"gene_id {gid} missing from gene metadata")
        gamma[idx], Gamma[idx], kon[idx], koff[idx] = param_map[gid]
    return gamma, Gamma, kon, koff


def load_sparse_counts(path: str | Path) -> sparse.csr_matrix:
    path = Path(path)
    mat = sparse.load_npz(path)
    if not sparse.isspmatrix_csr(mat):
        mat = mat.tocsr()
    return mat


def save_sparse_counts(path: str | Path, counts: sparse.spmatrix) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    sparse.save_npz(path, counts)
    return path


def load_ids(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_ids(path: str | Path, values: Iterable[str]) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for val in values:
            f.write(f"{val}\n")
    return path


def load_vector(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        data = np.loadtxt(path)
    return np.asarray(data)


def save_vector(path: str | Path, values: Sequence[float]) -> Path:
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, np.asarray(values))
    return path


def load_dataset(
    counts_path: str | Path,
    cells_path: str | Path | None = None,
    genes_path: str | Path | None = None,
) -> tuple[sparse.csr_matrix, list[str] | None, list[str] | None]:
    counts = load_sparse_counts(counts_path)
    cell_ids = load_ids(cells_path) if cells_path else None
    gene_ids = load_ids(genes_path) if genes_path else None
    return counts, cell_ids, gene_ids


def save_dataset(
    out_dir: str | Path,
    counts: sparse.spmatrix,
    cell_ids: Iterable[str] | None = None,
    gene_ids: Iterable[str] | None = None,
    counts_name: str = "counts.npz",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["counts"] = save_sparse_counts(out_dir / counts_name, counts)
    if cell_ids is not None:
        paths["cells"] = save_ids(out_dir / "cells.txt", cell_ids)
    if gene_ids is not None:
        paths["genes"] = save_ids(out_dir / "genes.txt", gene_ids)
    return paths


def load_snapshot_long(
    path: str | Path,
    gene_ids_filter: Sequence[str] | None = None,
    metadata_fields: Sequence[str] | None = None,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    path = Path(path)
    if gene_ids_filter is not None and len(gene_ids_filter) == 0:
        raise ValueError("gene_ids_filter must contain at least one gene id")
    if metadata_fields is None:
        metadata_fields = DEFAULT_METADATA_FIELDS
    gene_ids_list = list(gene_ids_filter) if gene_ids_filter is not None else None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if {"cell_id", "gene_id", "count"}.issubset(fieldnames):
            return _load_snapshot_long(reader, gene_ids_list)
        if "cell_id" in fieldnames:
            return _load_snapshot_wide(reader, fieldnames, gene_ids_list, set(metadata_fields))
        raise ValueError("snapshot.csv must contain either long or wide format columns")


def _load_snapshot_long(
    reader: csv.DictReader,
    gene_ids_filter: Sequence[str] | None,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    cell_map: dict[str, int] = {}
    if gene_ids_filter:
        gene_ids = list(gene_ids_filter)
        gene_map = {gid: idx for idx, gid in enumerate(gene_ids)}
        gene_filter = set(gene_ids)
    else:
        gene_ids = []
        gene_map = {}
        gene_filter = None
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for row in reader:
        cell_id = row.get("cell_id", "").strip()
        gene_id = row.get("gene_id", "").strip()
        count_val = row.get("count", "")
        if not cell_id or not gene_id:
            continue
        count = int(float(count_val)) if count_val not in (None, "") else 0
        if count <= 0:
            continue
        if gene_filter is not None and gene_id not in gene_filter:
            continue
        if cell_id not in cell_map:
            cell_map[cell_id] = len(cell_map)
        if gene_id not in gene_map:
            gene_map[gene_id] = len(gene_map)
            gene_ids.append(gene_id)
        rows.append(cell_map[cell_id])
        cols.append(gene_map[gene_id])
        data.append(count)

    n_cells = len(cell_map)
    n_genes = len(gene_map)
    counts = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=np.int32)
    cell_ids = [None] * n_cells
    for cid, idx in cell_map.items():
        cell_ids[idx] = cid
    return counts, cell_ids, gene_ids


def _load_snapshot_wide(
    reader: csv.DictReader,
    fieldnames: list[str],
    gene_ids_filter: Sequence[str] | None,
    metadata_fields: set[str],
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    if gene_ids_filter is not None:
        missing = [gid for gid in gene_ids_filter if gid not in fieldnames]
        if missing:
            raise ValueError(f"snapshot.csv is missing gene columns: {missing}")
        gene_fields = list(gene_ids_filter)
    else:
        gene_fields = [f for f in fieldnames if f not in metadata_fields]
    if not gene_fields:
        raise ValueError("snapshot.csv wide format must include gene columns")

    cell_ids: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for row_idx, row in enumerate(reader):
        cell_id = row.get("cell_id", "").strip()
        cell_ids.append(cell_id if cell_id else f"cell_{row_idx}")
        for col_idx, gene in enumerate(gene_fields):
            count_val = row.get(gene, "")
            if count_val in (None, ""):
                continue
            count = int(float(count_val))
            if count <= 0:
                continue
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(count)

    counts = sparse.csr_matrix((data, (rows, cols)), shape=(len(cell_ids), len(gene_fields)), dtype=np.int32)
    return counts, cell_ids, gene_fields


def _read_sim_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
    return cfg


def load_hidden_params(
    hidden_path: str | Path,
    sim_config_path: str | Path,
    gene_ids: Sequence[str] | None = None,
) -> ParamsTruth:
    hidden_path = Path(hidden_path)
    sim_config_path = Path(sim_config_path)

    with hidden_path.open("r", encoding="utf-8") as f:
        hidden = json.load(f)
    cfg = _read_sim_config(sim_config_path)

    nf_path = Path(cfg.get("nf_vector_path", ""))
    if not nf_path.is_absolute():
        nf_path = sim_config_path.parent / nf_path
    nf_path = nf_path.resolve()
    if nf_path.suffix == ".npy":
        nf_t = np.load(nf_path)
    else:
        nf_t = np.loadtxt(nf_path)
    nf_t = np.asarray(nf_t, dtype=np.float32).squeeze()
    t_grid = (np.arange(nf_t.size, dtype=np.float32) + 0.5) / float(nf_t.size)

    gene_list = hidden.get("genes", [])
    gene_map = {g["gene_id"]: g for g in gene_list}
    if gene_ids is None:
        gene_ids = list(gene_map.keys())

    gamma_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
    Gamma_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
    kon_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
    koff_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
    b_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
    regime_arr = np.array([""] * len(gene_ids), dtype=object)

    for idx, gid in enumerate(gene_ids):
        g = gene_map.get(gid)
        if g is None:
            continue
        gamma_arr[idx] = float(g.get("gamma_deg", np.nan))
        Gamma_arr[idx] = float(g.get("Gamma_esc", np.nan))
        k_on = float(g.get("k_on_rnap", 0.0))
        k_off = float(g.get("k_off_rnap", 0.0))
        kon_arr[idx] = k_on
        koff_arr[idx] = k_off
        Gamma = float(g.get("Gamma_esc", 0.0))
        if k_on > 0:
            b_arr[idx] = (k_off + Gamma) / k_on
        regime_arr[idx] = g.get("phase", "")

    return ParamsTruth(
        t_true=None,
        G_true=None,
        gamma_g=gamma_arr,
        Gamma_g=Gamma_arr,
        kon_g=kon_arr,
        koff_g=koff_arr,
        b_g=b_arr,
        regime=regime_arr,
        nf_t=nf_t,
        t_grid=t_grid,
        mu=None,
        sigma=None,
    )


def load_params(
    path: str | Path,
    cell_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
) -> ParamsTruth:
    path = Path(path)

    t_true_map: dict[str, float] = {}
    gamma_map: dict[str, float] = {}
    Gamma_map: dict[str, float] = {}
    kon_map: dict[str, float] = {}
    koff_map: dict[str, float] = {}
    b_map: dict[str, float] = {}
    regime_map: dict[str, str] = {}
    nf_map: dict[float, float] = {}
    gtrue_map: dict[tuple[str, float], float] = {}
    mu: float | None = None
    sigma: float | None = None

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = row.get("cell_id", "").strip()
            gene_id = row.get("gene_id", "").strip()
            t_val = _parse_float(row.get("t"))

            mu_val = _parse_float(row.get("mu"))
            sigma_val = _parse_float(row.get("sigma"))
            if mu_val is not None:
                mu = mu_val
            if sigma_val is not None:
                sigma = sigma_val

            t_true_val = _parse_float(row.get("t_true"))
            if cell_id and t_true_val is not None:
                t_true_map[cell_id] = t_true_val

            gamma_val = _parse_float(row.get("gamma_g"))
            if gene_id and gamma_val is not None:
                gamma_map[gene_id] = gamma_val

            Gamma_val = _parse_float(row.get("Gamma_g") or row.get("Gamma_esc"))
            if gene_id and Gamma_val is not None:
                Gamma_map[gene_id] = Gamma_val

            kon_val = _parse_float(row.get("kon_g") or row.get("k_on_rnap"))
            if gene_id and kon_val is not None:
                kon_map[gene_id] = kon_val

            koff_val = _parse_float(row.get("koff_g") or row.get("k_off_rnap"))
            if gene_id and koff_val is not None:
                koff_map[gene_id] = koff_val

            b_val = _parse_float(row.get("b_g"))
            if gene_id and b_val is not None:
                b_map[gene_id] = b_val

            regime_val = row.get("regime", "").strip()
            if gene_id and regime_val:
                regime_map[gene_id] = regime_val

            nf_val = _parse_float(row.get("Nf_t") or row.get("nf_t"))
            if t_val is not None and nf_val is not None:
                nf_map[t_val] = nf_val

            gtrue_val = _parse_float(row.get("G_true") or row.get("g_true"))
            if gene_id and t_val is not None and gtrue_val is not None:
                gtrue_map[(gene_id, t_val)] = gtrue_val

    t_true = None
    if t_true_map:
        if cell_ids is None:
            cell_ids = list(t_true_map.keys())
        t_true_arr = np.full(len(cell_ids), np.nan, dtype=np.float32)
        for idx, cid in enumerate(cell_ids):
            if cid in t_true_map:
                t_true_arr[idx] = float(t_true_map[cid])
        t_true = t_true_arr

    gamma_g = None
    Gamma_g = None
    kon_g = None
    koff_g = None
    b_g = None
    regime = None
    if gene_ids is None and (gamma_map or b_map or regime_map or gtrue_map or Gamma_map or kon_map or koff_map):
        gene_ids = sorted(
            {gid for gid, _ in gtrue_map.keys()}
            | set(gamma_map)
            | set(b_map)
            | set(regime_map)
            | set(Gamma_map)
            | set(kon_map)
            | set(koff_map)
        )

    if gene_ids is not None and gamma_map:
        gamma_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
        for idx, gid in enumerate(gene_ids):
            if gid in gamma_map:
                gamma_arr[idx] = float(gamma_map[gid])
        gamma_g = gamma_arr

    if gene_ids is not None and Gamma_map:
        Gamma_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
        for idx, gid in enumerate(gene_ids):
            if gid in Gamma_map:
                Gamma_arr[idx] = float(Gamma_map[gid])
        Gamma_g = Gamma_arr

    if gene_ids is not None and kon_map:
        kon_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
        for idx, gid in enumerate(gene_ids):
            if gid in kon_map:
                kon_arr[idx] = float(kon_map[gid])
        kon_g = kon_arr

    if gene_ids is not None and koff_map:
        koff_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
        for idx, gid in enumerate(gene_ids):
            if gid in koff_map:
                koff_arr[idx] = float(koff_map[gid])
        koff_g = koff_arr

    if gene_ids is not None and b_map:
        b_arr = np.full(len(gene_ids), np.nan, dtype=np.float32)
        for idx, gid in enumerate(gene_ids):
            if gid in b_map:
                b_arr[idx] = float(b_map[gid])
        b_g = b_arr

    if gene_ids is not None and regime_map:
        reg_arr = np.array([regime_map.get(gid, "") for gid in gene_ids], dtype=object)
        regime = reg_arr

    t_grid = None
    nf_t = None
    if nf_map:
        t_vals = np.array(sorted(nf_map.keys()), dtype=np.float32)
        nf_vals = np.array([nf_map[t] for t in t_vals], dtype=np.float32)
        t_grid = t_vals
        nf_t = nf_vals

    G_true = None
    if gtrue_map:
        t_vals = sorted({t for _, t in gtrue_map.keys()})
        t_grid = np.array(t_vals, dtype=np.float32)
        if gene_ids is None:
            gene_ids = sorted({gid for gid, _ in gtrue_map.keys()})
        G_mat = np.full((len(gene_ids), len(t_vals)), np.nan, dtype=np.float32)
        t_index = {t: i for i, t in enumerate(t_vals)}
        g_index = {g: i for i, g in enumerate(gene_ids)}
        for (gid, t_val), gval in gtrue_map.items():
            if gid in g_index and t_val in t_index:
                G_mat[g_index[gid], t_index[t_val]] = float(gval)
        if np.isnan(G_mat).any():
            raise ValueError("G_true entries are missing for some gene/time combinations")
        G_true = G_mat

    return ParamsTruth(
        t_true=t_true,
        G_true=G_true,
        gamma_g=gamma_g,
        Gamma_g=Gamma_g,
        kon_g=kon_g,
        koff_g=koff_g,
        b_g=b_g,
        regime=regime,
        nf_t=nf_t,
        t_grid=t_grid,
        mu=mu,
        sigma=sigma,
    )
