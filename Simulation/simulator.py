"""Lineage simulator for stochastic single-cell mRNA dynamics.

Implements a birth-death process with time-dependent transcription rates:
    dm/dt = Gamma * g(t) * O(t) - gamma * m
where g(t) is gene copy number (1 before replication, 2 after t_rep) and
O(t) = 1 / (1 + (k_off + Gamma) / (Nf(t) * k_on)).

The stochastic update is performed via Poisson tau-leaping, and division uses
binomial partitioning of mRNA between daughter cells.
"""

from __future__ import annotations

from typing import List

import numpy as np

from Simulation.cell import Cell
from Simulation.config import SimulationConfig
from Simulation.gene import Gene, GeneConfig, build_gene
from Simulation.stochastic import partition_mrna, tau_leap_batch


def build_genes(gene_configs: List[GeneConfig], sim_config: SimulationConfig) -> List[Gene]:
    """Build Gene objects from GeneConfigs with computed replication times."""
    return [
        build_gene(
            cfg,
            sim_config.chromosome_length_bp,
            sim_config.B_period,
            sim_config.C_period,
        )
        for cfg in gene_configs
    ]


class LineageSimulator:
    """Stochastic lineage simulator with tau-leaping dynamics."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        genes: List[Gene],
        nf_vec: np.ndarray,
    ) -> None:
        self.sim_config = sim_config
        self.genes = genes
        self.initial_cell_count = int(sim_config.initial_cell_count)
        self.division_time_cv = float(sim_config.division_time_cv)
        self.division_time_method = sim_config.division_time_method
        self.division_time_min = float(sim_config.division_time_min)
        self.division_time_max = float(sim_config.division_time_max)
        self.division_time_max_attempts = int(sim_config.division_time_max_attempts)
        self.t_rep = np.array([g.t_rep for g in genes], dtype=np.float64)
        self.gamma_deg = np.array([g.gamma_deg for g in genes], dtype=np.float64)
        self.k_on_rnap = np.array([g.k_on_rnap for g in genes], dtype=np.float64)
        self.k_off_rnap = np.array([g.k_off_rnap for g in genes], dtype=np.float64)
        self.Gamma = np.array([g.Gamma_esc for g in genes], dtype=np.float64)
        nf_vec = np.asarray(nf_vec, dtype=np.float64)
        if nf_vec.ndim != 1 or nf_vec.size == 0:
            raise ValueError("nf_vec must be a non-empty 1D array")
        steps_total = self.sim_config.T_total / self.sim_config.dt
        steps_total_int = int(round(steps_total))
        if not np.isclose(steps_total, steps_total_int, rtol=0.0, atol=1e-9):
            raise ValueError(f"T_total/dt must be an integer; got {steps_total}")
        steps_cycle = self.sim_config.T_div / self.sim_config.dt
        steps_cycle_int = int(round(steps_cycle))
        if not np.isclose(steps_cycle, steps_cycle_int, rtol=0.0, atol=1e-9):
            raise ValueError(f"T_div/dt must be an integer; got {steps_cycle}")
        if nf_vec.size != steps_cycle_int:
            raise ValueError(
                f"Nf vector length mismatch: expected {steps_cycle_int} from T_div/dt, got {nf_vec.size}"
            )
        self.nf_vec = nf_vec
        self.max_steps = steps_total_int

    def compute_promoter_occ(self, N_f: float) -> np.ndarray:
        """Compute per-gene promoter occupancy for a given RNAP pool."""
        denominator = 1.0 + (self.k_off_rnap + self.Gamma) / (self.k_on_rnap * N_f)
        return 1 / denominator

    def _transcription_propensity(self, cell: Cell, promoter_occ: np.ndarray) -> np.ndarray:
        """Compute per-gene transcription propensities using precomputed occupancy."""
        gene_copies = np.where(cell.age > self.t_rep, 2, 1)
        return gene_copies * self.Gamma * promoter_occ

    def _degradation_propensity(self, cell: Cell) -> np.ndarray:
        """Compute per-gene degradation propensities."""
        return cell.mrna * self.gamma_deg

    def _compute_steady_state_mRNA(self, age: float, N_f: float) -> np.ndarray:
        """Compute analytical steady-state mRNA for a cell at given age.

        m_ss = g(t) * Gamma * O(t) / gamma
        where g(t) = 1 before replication, 2 after t_rep
        and O = 1 / (1 + (k_off + Gamma) / (k_on * Nf))
        """
        gene_copies = np.where(age > self.t_rep, 2.0, 1.0)
        occupancy = self.compute_promoter_occ(N_f)
        return gene_copies * self.Gamma * occupancy / self.gamma_deg

    def _calculate_cell_phase(self, age: float) -> str:
        """Determine cell phase based on age and division time."""
        if age < self.sim_config.B_period:
            return "B"
        elif age < self.sim_config.B_period + self.sim_config.C_period:
            return "C"
        else:
            return "D"

    def _draw_division_time(self, rng: np.random.Generator) -> float:
        """Draw a cell-specific division time around the mean T_div.

        clip: draw once and clamp to [min, max]
        reject/truncated_normal: resample until within bounds or fail after max_attempts
        """
        base = self.sim_config.T_div
        if self.division_time_cv == 0.0:
            return base
        sigma = base * self.division_time_cv
        t_min = self.division_time_min
        t_max = self.division_time_max
        if self.division_time_method == "clip":
            draw = rng.normal(loc=base, scale=sigma)
            return float(np.clip(draw, t_min, t_max))
        # Rejection sampling approximates a truncated normal when bounds are enforced.
        for _ in range(self.division_time_max_attempts):
            draw = rng.normal(loc=base, scale=sigma)
            if t_min <= draw <= t_max:
                return float(draw)
        raise RuntimeError(
            f"Division time sampling failed after {self.division_time_max_attempts} attempts "
            f"for bounds [{t_min}, {t_max}]"
        )

    def _make_row(self, cell: Cell) -> dict:
        """Create output row dictionary from cell state."""
        cell_phase = self._calculate_cell_phase(cell.age)
        theta_rad = (cell.age / cell.division_time) * (2 * np.pi)
        row = {
            "cell_id": cell.cell_id,
            "parent_id": cell.parent_id,
            "generation": cell.generation,
            "age": cell.age,
            "phase": cell_phase,
            "theta_rad": theta_rad,
        }
        for idx, gene in enumerate(self.genes):
            row[gene.gene_id] = int(cell.mrna[idx])
        return row

    def _make_split_cell(
        self,
        updated_cells: list,
        rng: np.random.Generator,
        cell: Cell,
        next_cell_id: int,
    ) -> tuple[list, int]:
        """Create two daughter cells from parent cell."""
        d1, d2 = partition_mrna(cell.mrna, rng)
        updated_cells.append(
            self._make_cell(
                rng,
                cell_id=next_cell_id,
                parent_id=cell.cell_id,
                generation=cell.generation + 1,
                age=0.0,
                mrna=d1,
            )
        )
        next_cell_id += 1
        updated_cells.append(
            self._make_cell(
                rng,
                cell_id=next_cell_id,
                parent_id=cell.cell_id,
                generation=cell.generation + 1,
                age=0.0,
                mrna=d2,
            )
        )
        next_cell_id += 1
        return updated_cells, next_cell_id

    def _make_cell(
        self,
        rng: np.random.Generator,
        cell_id: int,
        parent_id: int | None,
        generation: int,
        age: float,
        mrna: np.ndarray,
    ) -> Cell:
        """Create a new cell with drawn division time."""
        return Cell(
            cell_id=cell_id,
            parent_id=parent_id,
            generation=generation,
            age=age,
            division_time=self._draw_division_time(rng),
            mrna=mrna,
        )

    def _init_cells(self, rng: np.random.Generator) -> list[Cell]:
        """Initialize starting cell population with exponential age distribution.

        For an exponentially growing population, the age distribution follows:
        P(age) = ln(2)/T_div * 2^(-age/T_div)

        mRNA is initialized around the analytical steady-state with normal noise.
        """
        cells: list[Cell] = []
        T_div = self.sim_config.T_div
        dt = self.sim_config.dt
        n_steps_cycle = int(round(T_div / dt))

        for cid in range(self.initial_cell_count):
            # Draw age from exponential distribution (truncated to cell cycle)
            # For exponentially growing population: P(a) ∝ 2^(-a/T_div)
            # This is equivalent to exponential with rate ln(2)/T_div
            rate = np.log(2) / T_div
            while True:
                age = float(rng.exponential(1.0 / rate))
                if age < T_div:
                    break

            # Get Nf at this age (use step index for nf_vec lookup)
            step_idx = int(age / dt) % n_steps_cycle
            N_f = float(self.nf_vec[step_idx])

            # Compute analytical steady-state mRNA for this cell
            m_ss = self._compute_steady_state_mRNA(age, N_f)

            # Add normal noise around steady-state (CV ~ 0.3, Poisson-like)
            # Ensure non-negative integer counts
            noise_std = np.sqrt(m_ss) * 0.5  # Sub-Poisson noise for smoother init
            mrna = rng.normal(m_ss, noise_std)
            mrna = np.maximum(mrna, 0.0).astype(np.float64)

            cells.append(
                self._make_cell(rng, cell_id=cid, parent_id=None, generation=0, age=age, mrna=mrna)
            )
        return cells

    def run(self) -> list[dict]:
        """Run the stochastic simulation and collect snapshots."""
        rng = np.random.default_rng(self.sim_config.random_seed)
        cells: list[Cell] = self._init_cells(rng)
        next_cell_id = len(cells)
        snapshots: list[dict] = []
        step_idx = 0
        max_steps = self.max_steps
        snapshot_min_steps = self.sim_config.snapshot_min_interval_steps
        snapshot_jitter_steps = self.sim_config.snapshot_jitter_steps
        next_snapshot_step = snapshot_min_steps + int(rng.integers(0, snapshot_jitter_steps + 1))

        while len(snapshots) < self.sim_config.N_target_samples:
            if step_idx >= max_steps:
                raise RuntimeError(
                    "Simulation time exhausted before reaching target samples. "
                    "Increase T_total or reduce N_target_samples."
                )
            N_f = float(self.nf_vec[step_idx % self.nf_vec.size])
            promoter_occ = self.compute_promoter_occ(N_f)
            n_cells = len(cells)
            ages = np.asarray([cell.age for cell in cells], dtype=np.float64)
            mrna_matrix = np.asarray([cell.mrna for cell in cells], dtype=np.float64, order="C")
            gene_copies = np.where(ages[:, None] > self.t_rep[None, :], 2.0, 1.0)
            trans_prop = np.ascontiguousarray(gene_copies * self.Gamma * promoter_occ)
            deg_prop = np.ascontiguousarray(mrna_matrix * self.gamma_deg)
            rng_seeds = rng.integers(
                0,
                np.iinfo(np.int64).max,
                size=n_cells,
                dtype=np.int64,
            ).astype(np.uint64, copy=False)

            new_mrna = tau_leap_batch(
                dt=self.sim_config.dt,
                ages=ages,
                mrna=mrna_matrix,
                t_rep=self.t_rep,
                trans_prop=trans_prop,
                deg_prop=deg_prop,
                gamma_deg=self.gamma_deg,
                max_mrna_per_gene=self.sim_config.MAX_MRNA_PER_GENE,
                rng_seeds=rng_seeds,
            )

            updated_cells: list[Cell] = []
            for idx, cell in enumerate(cells):
                cell.mrna = new_mrna[idx]
                cell.step_age(self.sim_config.dt)

                if cell.age >= cell.division_time:
                    updated_cells, next_cell_id = self._make_split_cell(
                        updated_cells, rng, cell, next_cell_id
                    )
                else:
                    updated_cells.append(cell)
            cells = updated_cells
            step_idx += 1

            if step_idx >= next_snapshot_step:
                remaining = self.sim_config.N_target_samples - len(snapshots)
                if remaining <= 0:
                    break
                if remaining < len(cells):
                    sample_idx = rng.choice(len(cells), size=remaining, replace=False)
                    snapshot_cells = [cells[i] for i in sample_idx]
                else:
                    snapshot_cells = cells
                for cell in snapshot_cells:
                    row = self._make_row(cell)
                    snapshots.append(row)
                if len(snapshots) >= self.sim_config.N_target_samples:
                    break
                next_snapshot_step = step_idx + snapshot_min_steps + int(
                    rng.integers(0, snapshot_jitter_steps + 1)
                )
        return snapshots
