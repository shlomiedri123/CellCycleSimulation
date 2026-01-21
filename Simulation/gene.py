"""Gene parameter bundle used by the simulator.

Encapsulates per-gene kinetic rates for RNAP-limited transcription and decay,
plus replication timing along the chromosome.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneConfig:
    """Gene-level parameters for RNAP-limited transcription.

    Stores per-gene kinetic rates that drive promoter occupancy and mRNA decay.
    These parameters feed the stochastic birth-death process in the simulator.
    """
    gene_id: str
    chrom_pos_bp: float
    k_on_rnap: float
    k_off_rnap: float
    Gamma_esc: float
    gamma_deg: float
    phase: str | None = None

    def validate(self) -> None:
        if self.chrom_pos_bp < 0:
            raise ValueError(f"chrom_pos_bp must be non-negative for {self.gene_id}")
        if min(self.k_on_rnap, self.k_off_rnap, self.Gamma_esc, self.gamma_deg) < 0:
            raise ValueError(f"Rates must be positive for {self.gene_id}")
        if self.phase:
            phase_key = str(self.phase).strip().upper()
            if phase_key not in {"I", "II"}:
                raise ValueError(f"phase must be 'I' or 'II' for {self.gene_id}")


@dataclass(frozen=True)
class Gene:
    """Gene with computed replication time for simulator use."""
    gene_id: str
    chrom_pos_bp: float
    phase: str | None
    gamma_deg: float
    Gamma_esc: float
    t_rep: float
    k_on_rnap: float
    k_off_rnap: float


def compute_t_rep(chrom_pos_bp: float, chromosome_length_bp: float, B_period: float, C_period: float) -> float:
    """Compute replication time from chromosomal position using Cooper-Helmstetter model."""
    x_g = chrom_pos_bp / chromosome_length_bp
    if x_g < 0 or x_g > 1:
        raise ValueError(f"chrom_pos_bp out of range: {chrom_pos_bp}")
    return B_period + C_period * x_g


def build_gene(cfg: GeneConfig, chromosome_length_bp: float, B_period: float, C_period: float) -> Gene:
    """Build a Gene object from GeneConfig with computed replication time."""
    cfg.validate()
    t_rep = compute_t_rep(cfg.chrom_pos_bp, chromosome_length_bp, B_period, C_period)
    return Gene(
        gene_id=cfg.gene_id,
        chrom_pos_bp=cfg.chrom_pos_bp,
        phase=cfg.phase,
        gamma_deg=cfg.gamma_deg,
        Gamma_esc=cfg.Gamma_esc,
        t_rep=t_rep,
        k_on_rnap=cfg.k_on_rnap,
        k_off_rnap=cfg.k_off_rnap,
    )
