"""Gene parameter bundle used by the simulator.

Encapsulates per-gene kinetic rates for RNAP-limited transcription and decay,
plus replication timing along the chromosome.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gene:
    gene_id: str
    chrom_pos_bp: float
    phase: str | None
    gamma_deg: float
    Gamma_esc: float
    t_rep: float
    k_on_rnap: float
    k_off_rnap: float
