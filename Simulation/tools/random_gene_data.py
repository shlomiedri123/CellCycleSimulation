"""Utility to generate random gene parameters and configs for simulation.

This module generates biologically realistic gene parameters for the cell cycle
simulation based on the biophysical model from CellSizeNonlinearScaling-3.pdf.

Parameter specifications (from the model requirements):
1. mRNA degradation rate: γ_deg ~ 1/3 min⁻¹, varied within one order of magnitude
2. Average mRNA count: m ~ gene_copy × transcription_rate / degradation × occupancy
3. Gene copy number: ~1-2 (doubles at replication)
4. mRNA counts: ~0.1-100 molecules per gene
5. Transcription initiation rate (Γ_esc): ~(1/3 min⁻¹) × (0.1-100)
6. k_off ≈ 0 (unbinding from promoter is negligible)
7. k_on is constant per gene, chosen such that:
   - Phase I: Γ/(k_on × N_f(t)) >> 1 (expression ∝ N_f(t))
   - Phase II: Γ/(k_on × N_f(t)) << 1 (expression saturated, constant)
8. N_f(t): Global (gene-independent), slowly varying with cell age, range ~[1, 2]

The promoter occupancy is given by (Equation 6):
    O_i = 1 / (1 + (k_off + Γ) / (N_f × k_on))

Phase I (linear scaling): O_i ∝ N_f(t) × k_on
Phase II (saturation): O_i ≈ 1 (high occupancy)

Assumed units: minutes for time, per-minute for rates.

Outputs:
- gene CSV compatible with run_simulation.py
- Nf vector file (nf_vector.npy)
- sim config YAML referencing the gene table and Nf vector
- hidden metadata JSON (true Nf(t) params, gene phases/params)
"""

from __future__ import annotations

import argparse
import json
import dataclasses
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import yaml

from Simulation.gene import GeneConfig
from Simulation.config import SimulationConfig


def _random_nffunc(
    rng: np.random.Generator,
    period: float = 40.0,
    base_range: tuple[float, float] = (1.0, 2.0),
    amp_range: tuple[float, float] = (0.0, 0.3),
) -> tuple[callable, dict]:
    """Return a slowly varying Nf(t) callable and its parameters.

    N_f(t) is the number of free RNAP molecules, which slowly varies
    with cell age in the range [1, 2] as specified in the model.

    Args:
        rng: Random number generator.
        period: Period of the sinusoidal variation (in minutes).
        base_range: Range for the baseline N_f value.
        amp_range: Range for the amplitude of variation.

    Returns:
        Tuple of (callable, dict) with the N_f(t) function and its parameters.
    """
    base = float(rng.uniform(*base_range))
    amp = float(rng.uniform(*amp_range))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    def _nf(t: float) -> float:
        return base + amp * np.sin((2.0 * np.pi * t) / period + phase)

    return _nf, {"base": base, "amp": amp, "phase": phase, "period": period}


def _occupancy(k_on: float, k_off: float, Gamma_esc: float, N_f: float) -> float:
    """Compute promoter occupancy from kinetic parameters.

    Equation 6:
        O_i = 1 / (1 + (k_off + Γ) / (N_f × k_on))

    Args:
        k_on: RNAP binding rate (per minute).
        k_off: RNAP unbinding rate (per minute).
        Gamma_esc: Promoter escape rate (per minute).
        N_f: Free RNAP concentration.

    Returns:
        Promoter occupancy probability [0, 1].
    """
    if N_f <= 0 or k_on <= 0:
        return 0.0
    denom = 1.0 + (k_off + Gamma_esc) / (k_on * N_f)
    return 1.0 / denom


def _compute_regime_ratio(k_on: float, Gamma_esc: float, N_f: float) -> float:
    """Compute the regime-determining ratio Γ/(k_on × N_f).

    This ratio determines the regime:
    - Ratio >> 1: Phase I (linear scaling with N_f)
    - Ratio << 1: Phase II (saturation, constant)

    Args:
        k_on: RNAP binding rate.
        Gamma_esc: Promoter escape rate.
        N_f: Free RNAP concentration.

    Returns:
        The ratio Γ/(k_on × N_f).
    """
    if k_on <= 0 or N_f <= 0:
        return float("inf")
    return Gamma_esc / (k_on * N_f)


def _draw_gene(
    idx: int,
    total: int,
    rng: np.random.Generator,
    chrom_length: float,
    nf_ref: float,
    phase_I_fraction: float = 0.5,
    max_attempts: int = 1000,
) -> tuple[GeneConfig, dict]:
    """Draw random parameters for a single gene.

    Parameters are drawn to satisfy the biological constraints:
    - γ_deg ~ 1/3 min⁻¹ (within one order of magnitude)
    - k_off ≈ 0
    - Γ_esc ~ (1/3) × (0.1-100) min⁻¹
    - k_on chosen to achieve desired phase

    Args:
        idx: Gene index (1-based).
        total: Total number of genes.
        rng: Random number generator.
        chrom_length: Chromosome length in base pairs.
        nf_ref: Reference N_f value (mean of N_f(t)).
        phase_I_fraction: Fraction of genes in Phase I.
        max_attempts: Maximum attempts to find valid parameters.

    Returns:
        Tuple of (GeneConfig, metadata dict).
    """
    gene_id = f"gene_{idx}"

    # Stratify gene positions along the chromosome
    span = chrom_length / float(total)
    base = (idx - 1) * span
    jitter = float(rng.uniform(0.1 * span, 0.9 * span))
    chrom_pos_bp = base + jitter

    # Determine phase (Phase I or Phase II)
    is_phase_I = rng.random() < phase_I_fraction

    # --- Degradation rate ---
    # γ_deg ~ 1/3 min⁻¹, varied within one order of magnitude
    # Log-normal with mean at 1/3 and sigma for ~3x variation
    gamma_deg = float(rng.lognormal(mean=np.log(1.0 / 3.0), sigma=0.5))
    gamma_deg = float(np.clip(gamma_deg, 0.1, 1.0))  # ~0.1 to 1.0 min⁻¹

    # --- k_off ---
    # k_off ≈ 0 (negligible unbinding)
    k_off = 0.0

    # --- Target mRNA count ---
    # mRNA counts should be ~0.1-100 molecules per gene (biological constraint)
    # Draw target steady-state mRNA count, then derive Γ from it
    # m_ss = Γ * O / γ, for Phase II (O ≈ 1): m_ss ≈ Γ / γ
    # So Γ = m_ss * γ
    target_mRNA = float(rng.lognormal(mean=np.log(5.0), sigma=1.0))  # Centered ~5, range ~0.5-50
    target_mRNA = float(np.clip(target_mRNA, 0.1, 100))  # Biological constraint

    # --- Γ_esc (transcription initiation / escape rate) ---
    # Derive Γ from target mRNA count: Γ = m_target * γ
    # This ensures the ratio Γ/γ gives realistic mRNA counts
    Gamma_esc = float(target_mRNA * gamma_deg)
    Gamma_esc = float(np.clip(Gamma_esc, 0.01, 10.0))  # Reasonable rate bounds

    # --- k_on ---
    # Choose k_on such that Γ/(k_on × N_f) achieves desired phase
    # Phase I: ratio > 1 (typically > 5)
    # Phase II: ratio < 1 (typically < 0.2)

    for attempt in range(max_attempts):
        if is_phase_I:
            # Phase I: Want Γ/(k_on × N_f) > 1
            # So k_on < Γ / N_f
            # Choose k_on such that ratio is in [2, 50]
            target_ratio = float(rng.uniform(2.0, 50.0))
            k_on = Gamma_esc / (target_ratio * nf_ref)
            # Ensure k_on is reasonable (not too small)
            k_on = max(k_on, 0.001)
        else:
            # Phase II: Want Γ/(k_on × N_f) < 1
            # So k_on > Γ / N_f
            # Choose k_on such that ratio is in [0.01, 0.5]
            target_ratio = float(rng.uniform(0.01, 0.5))
            k_on = Gamma_esc / (target_ratio * nf_ref)
            # Clamp to reasonable range
            k_on = min(k_on, 100.0)

        # Verify the regime
        actual_ratio = _compute_regime_ratio(k_on, Gamma_esc, nf_ref)
        occ = _occupancy(k_on, k_off, Gamma_esc, nf_ref)

        if is_phase_I:
            # Phase I: ratio > 1, low occupancy
            if actual_ratio > 1.0:
                break
        else:
            # Phase II: ratio < 1, high occupancy
            if actual_ratio < 1.0:
                break
    else:
        # If we couldn't find parameters, use the last attempt
        pass

    phase = "I" if is_phase_I else "II"
    meta = {
        "phase": phase,
        "regime_ratio": float(actual_ratio),
        "occupancy": float(occ),
    }

    return (
        GeneConfig(
            gene_id=gene_id,
            chrom_pos_bp=chrom_pos_bp,
            k_on_rnap=k_on,
            k_off_rnap=k_off,
            Gamma_esc=Gamma_esc,
            gamma_deg=gamma_deg,
            phase=phase,
        ),
        meta,
    )


def generate_genes(
    n_genes: int,
    chrom_length: float,
    nf_ref: float,
    seed: int | None = None,
    phase_I_fraction: float = 0.5,
    max_attempts: int = 1000,
) -> tuple[List[GeneConfig], List[dict]]:
    """Generate random gene parameters.

    Args:
        n_genes: Number of genes to generate.
        chrom_length: Chromosome length in base pairs.
        nf_ref: Reference N_f value (mean of N_f(t)).
        seed: Random seed for reproducibility.
        phase_I_fraction: Fraction of genes in Phase I.
        max_attempts: Maximum attempts per gene to find valid parameters.

    Returns:
        Tuple of (list of GeneConfig, list of metadata dicts).
    """
    rng = np.random.default_rng(seed)
    genes: List[GeneConfig] = []
    meta: List[dict] = []

    for i in range(1, n_genes + 1):
        g, m = _draw_gene(
            i,
            n_genes,
            rng,
            chrom_length,
            nf_ref=nf_ref,
            phase_I_fraction=phase_I_fraction,
            max_attempts=max_attempts,
        )
        g.validate()
        genes.append(g)
        meta.append(m)

    return genes, meta


def _write_gene_csv(path: Path, genes: Iterable[GeneConfig]) -> None:
    """Write gene parameters to CSV file."""
    import csv
    from dataclasses import asdict

    fields = [
        "gene_id",
        "chrom_pos_bp",
        "k_on_rnap",
        "k_off_rnap",
        "Gamma_esc",
        "gamma_deg",
        "phase",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for g in genes:
            writer.writerow(asdict(g))


def _suggest_total_time(n_samples: int, initial_cells: int, t_div: float) -> float:
    """Suggest total simulation time to achieve target sample count."""
    if n_samples <= initial_cells:
        return t_div
    cycles = math.ceil(math.log2(n_samples / float(initial_cells)))
    return (cycles + 1) * t_div


def verify_gene_parameters(
    genes: List[GeneConfig],
    meta: List[dict],
    nf_ref: float,
) -> dict:
    """Verify that gene parameters satisfy model constraints.

    Returns a summary of parameter distributions and violations.
    """
    summary = {
        "n_genes": len(genes),
        "phase_I_count": 0,
        "phase_II_count": 0,
        "gamma_deg": {"min": float("inf"), "max": 0, "mean": 0},
        "Gamma_esc": {"min": float("inf"), "max": 0, "mean": 0},
        "k_on": {"min": float("inf"), "max": 0, "mean": 0},
        "regime_ratios": {"phase_I_min": float("inf"), "phase_I_max": 0,
                          "phase_II_min": float("inf"), "phase_II_max": 0},
        "violations": [],
    }

    gamma_vals = []
    Gamma_vals = []
    k_on_vals = []

    for g, m in zip(genes, meta):
        gamma_vals.append(g.gamma_deg)
        Gamma_vals.append(g.Gamma_esc)
        k_on_vals.append(g.k_on_rnap)

        if m["phase"] == "I":
            summary["phase_I_count"] += 1
            ratio = m.get("regime_ratio", _compute_regime_ratio(g.k_on_rnap, g.Gamma_esc, nf_ref))
            summary["regime_ratios"]["phase_I_min"] = min(summary["regime_ratios"]["phase_I_min"], ratio)
            summary["regime_ratios"]["phase_I_max"] = max(summary["regime_ratios"]["phase_I_max"], ratio)
            if ratio <= 1:
                summary["violations"].append(f"{g.gene_id}: Phase I but ratio={ratio:.2f} <= 1")
        else:
            summary["phase_II_count"] += 1
            ratio = m.get("regime_ratio", _compute_regime_ratio(g.k_on_rnap, g.Gamma_esc, nf_ref))
            summary["regime_ratios"]["phase_II_min"] = min(summary["regime_ratios"]["phase_II_min"], ratio)
            summary["regime_ratios"]["phase_II_max"] = max(summary["regime_ratios"]["phase_II_max"], ratio)
            if ratio >= 1:
                summary["violations"].append(f"{g.gene_id}: Phase II but ratio={ratio:.2f} >= 1")

    summary["gamma_deg"]["min"] = min(gamma_vals)
    summary["gamma_deg"]["max"] = max(gamma_vals)
    summary["gamma_deg"]["mean"] = np.mean(gamma_vals)

    summary["Gamma_esc"]["min"] = min(Gamma_vals)
    summary["Gamma_esc"]["max"] = max(Gamma_vals)
    summary["Gamma_esc"]["mean"] = np.mean(Gamma_vals)

    summary["k_on"]["min"] = min(k_on_vals)
    summary["k_on"]["max"] = max(k_on_vals)
    summary["k_on"]["mean"] = np.mean(k_on_vals)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random genes and configs.")
    parser.add_argument("--n_genes", type=int, required=True, help="Number of genes to generate")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples target (for config)")
    parser.add_argument("--chrom_length", type=float, default=4_600_000, help="Chromosome length (bp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out_dir", type=Path, default=Path("simulation/test_data"), help="Output directory")
    parser.add_argument("--phase_I_fraction", type=float, default=0.5, help="Fraction of genes in Phase I")
    parser.add_argument("--verify", action="store_true", help="Verify and print parameter summary")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Generate N_f(t) function: slowly varying in [1, 2]
    t_div = 40.0  # Division time in minutes
    nf_func, nf_params = _random_nffunc(
        np.random.default_rng(args.seed),
        period=t_div,
        base_range=(1.0, 2.0),
        amp_range=(0.0, 0.3),
    )

    initial_cells = 3
    t_total = _suggest_total_time(args.n_samples, initial_cells, t_div)
    dt = 0.1

    # Create N_f vector for one cell cycle
    n_steps = int(round(t_div / dt))
    time_grid = np.arange(n_steps, dtype=float) * dt
    nf_vec = np.array([nf_func(t) for t in time_grid], dtype=float)
    nf_ref = float(np.mean(nf_vec))

    # Generate genes
    genes, gene_meta = generate_genes(
        args.n_genes,
        args.chrom_length,
        nf_ref=nf_ref,
        seed=args.seed,
        phase_I_fraction=args.phase_I_fraction,
    )

    # Verify if requested
    if args.verify:
        summary = verify_gene_parameters(genes, gene_meta, nf_ref)
        print("\n=== Gene Parameter Summary ===")
        print(f"Total genes: {summary['n_genes']}")
        print(f"Phase I: {summary['phase_I_count']}, Phase II: {summary['phase_II_count']}")
        print(f"γ_deg range: [{summary['gamma_deg']['min']:.3f}, {summary['gamma_deg']['max']:.3f}] (mean: {summary['gamma_deg']['mean']:.3f})")
        print(f"Γ_esc range: [{summary['Gamma_esc']['min']:.3f}, {summary['Gamma_esc']['max']:.3f}] (mean: {summary['Gamma_esc']['mean']:.3f})")
        print(f"k_on range: [{summary['k_on']['min']:.4f}, {summary['k_on']['max']:.3f}] (mean: {summary['k_on']['mean']:.4f})")
        print(f"Phase I regime ratios: [{summary['regime_ratios']['phase_I_min']:.2f}, {summary['regime_ratios']['phase_I_max']:.2f}]")
        print(f"Phase II regime ratios: [{summary['regime_ratios']['phase_II_min']:.4f}, {summary['regime_ratios']['phase_II_max']:.3f}]")
        print(f"N_f reference: {nf_ref:.3f}")
        if summary['violations']:
            print(f"\nViolations ({len(summary['violations'])}):")
            for v in summary['violations'][:10]:  # Show first 10
                print(f"  - {v}")
        else:
            print("\nNo violations detected.")

    # Save N_f vector
    nf_path = args.out_dir / "nf_vector.npy"
    np.save(nf_path, nf_vec)

    # Create simulation config
    sim_config_runtime = SimulationConfig(
        B_period=10.0,
        C_period=20.0,
        D_period=10.0,
        T_total=t_total,
        dt=dt,
        N_target_samples=args.n_samples,
        random_seed=args.seed or 123,
        chromosome_length_bp=args.chrom_length,
        MAX_MRNA_PER_GENE=10_000,
        genes_path=str(args.out_dir / "random_genes.csv"),
        nf_vector_path=str(nf_path),
        out_path=str(args.out_dir / "snapshots.csv"),
        initial_cell_count=initial_cells,
    )

    # Save gene CSV
    _write_gene_csv(args.out_dir / "random_genes.csv", genes)

    # Save YAML config
    cfg_dict = dataclasses.asdict(sim_config_runtime)
    with (args.out_dir / "random_sim_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=True)

    # Save hidden parameters (ground truth for validation)
    hidden = {
        "nf_params": nf_params,
        "nf_ref": nf_ref,
        "genes": [
            {
                "gene_id": g.gene_id,
                "phase": m["phase"],
                "regime_ratio": m.get("regime_ratio", 0),
                "occupancy": m.get("occupancy", 0),
                "k_on_rnap": g.k_on_rnap,
                "k_off_rnap": g.k_off_rnap,
                "Gamma_esc": g.Gamma_esc,
                "gamma_deg": g.gamma_deg,
                "chrom_pos_bp": g.chrom_pos_bp,
            }
            for g, m in zip(genes, gene_meta)
        ],
    }
    with (args.out_dir / "random_hidden_params.json").open("w", encoding="utf-8") as f:
        json.dump(hidden, f, indent=2, sort_keys=True)

    print(f"\nGenerated {len(genes)} genes in {args.out_dir}")
    print(f"  - Genes CSV: {args.out_dir / 'random_genes.csv'}")
    print(f"  - N_f vector: {nf_path}")
    print(f"  - Config: {args.out_dir / 'random_sim_config.yaml'}")
    print(f"  - Hidden params: {args.out_dir / 'random_hidden_params.json'}")


if __name__ == "__main__":
    main()
