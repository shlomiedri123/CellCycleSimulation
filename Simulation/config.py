"""Simulation configuration for stochastic lineage dynamics.

Defines the global time grid (dt, T_div) and cell-cycle periods (B/C/D) used
by the simulator. The time grid must be consistent with the provided Nf(t)
vector: len(nf_vec) == T_div / dt and T_total / dt must be an integer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the stochastic simulation."""
    B_period: float
    C_period: float
    D_period: float
    T_total: float
    dt: float
    N_target_samples: int
    random_seed: int
    chromosome_length_bp: float
    MAX_MRNA_PER_GENE: int
    genes_path: str
    nf_vector_path: str
    out_path: str
    measured_dist_path: str | None = None
    measured_s_vector_path: str | None = None
    parsed_out_path: str | None = None
    sparse: bool = False
    initial_cell_count: int = 3
    division_time_cv: float = 0.05
    division_time_method: str = "clip"
    division_time_min: float | None = None
    division_time_max: float | None = None
    division_time_max_attempts: int = 1000
    T_div: float | None = field(default=None)
    snapshot_min_interval_steps: int = 10
    snapshot_jitter_steps: int = 10

    def __post_init__(self) -> None:
        computed_t_div = self.B_period + self.C_period + self.D_period
        object.__setattr__(self, "T_div", computed_t_div if self.T_div is None else self.T_div)
        if self.T_div is not None and abs(self.T_div - computed_t_div) > 1e-9:
            raise ValueError("T_div must equal B_period + C_period + D_period")
        if min(self.B_period, self.C_period, self.D_period) <= 0:
            raise ValueError("B_period, C_period, and D_period must be positive")
        if self.T_total <= 0:
            raise ValueError("T_total must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.N_target_samples <= 0:
            raise ValueError("N_target_samples must be positive")
        if self.chromosome_length_bp <= 0:
            raise ValueError("chromosome_length_bp must be positive")
        if self.MAX_MRNA_PER_GENE <= 0:
            raise ValueError("MAX_MRNA_PER_GENE must be positive")
        if self.initial_cell_count <= 0:
            raise ValueError("initial_cell_count must be positive")
        if self.division_time_cv < 0:
            raise ValueError("division_time_cv must be non-negative")
        if self.division_time_method not in ("clip", "reject", "truncated_normal"):
            raise ValueError("division_time_method must be 'clip', 'reject', or 'truncated_normal'")
        if self.division_time_max_attempts <= 0:
            raise ValueError("division_time_max_attempts must be positive")
        if not isinstance(self.sparse, bool):
            raise ValueError("sparse must be boolean")
        if self.snapshot_min_interval_steps < 10:
            raise ValueError("snapshot_min_interval_steps must be at least 10")
        if self.snapshot_jitter_steps < 0:
            raise ValueError("snapshot_jitter_steps must be non-negative")

        t_min = self.division_time_min
        t_max = self.division_time_max
        if t_min is None:
            t_min = 0.9 * float(self.T_div)
        if t_max is None:
            t_max = 1.1 * float(self.T_div)
        if t_min <= 0:
            raise ValueError("division_time_min must be positive")
        if t_max <= t_min:
            raise ValueError("division_time_max must be greater than division_time_min")
        object.__setattr__(self, "division_time_min", float(t_min))
        object.__setattr__(self, "division_time_max", float(t_max))
