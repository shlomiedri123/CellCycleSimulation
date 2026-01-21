"""Gene position and replication parameter inference.

This module infers gene chromosomal positions and Cooper-Helmstetter parameters
(C, D) from gene expression profiles by detecting replication jumps.

The key insight is that when a gene replicates, its dosage doubles from 1 to 2,
causing a characteristic jump in expression. The timing of this jump reveals
the gene's chromosomal position.

Mathematical basis (Cooper-Helmstetter model):
- t_rep(x) = B + C × x = (1 - C - D) + C × x
- Gene at position x replicates at time t_rep(x)
- By detecting t_rep and fitting x ~ t_rep, we infer C and D
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d


@dataclass
class JumpDetectionResult:
    """Result of jump detection for one gene.

    Attributes:
        has_jump: Whether a significant jump was detected.
        t_jump: Time of the jump (NaN if no jump detected).
        jump_ratio: Ratio of expression after/before jump.
        confidence: Confidence in the jump detection.
    """

    has_jump: bool
    t_jump: float
    jump_ratio: float
    confidence: float


@dataclass
class PositionInferenceResult:
    """Result of gene position inference.

    Attributes:
        x_g: Inferred gene positions (n_genes,).
        t_jump_g: Jump times per gene (n_genes,), NaN if no jump.
        has_jump_g: Whether jump was detected per gene (n_genes,).
        C: Inferred C-period.
        D: Inferred D-period.
        regression_r2: R-squared of the position regression.
        n_jumps_detected: Number of genes with detected jumps.
    """

    x_g: np.ndarray
    t_jump_g: np.ndarray
    has_jump_g: np.ndarray
    C: float
    D: float
    regression_r2: float
    n_jumps_detected: int


def smooth_profile(
    profile: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Smooth a gene expression profile using a moving average.

    Args:
        profile: Expression values over time (n_time,).
        window_size: Size of the smoothing window.

    Returns:
        Smoothed profile.
    """
    if window_size <= 1:
        return profile.copy()
    return uniform_filter1d(profile.astype(np.float64), size=window_size, mode="nearest")


def detect_jump_time(
    profile: np.ndarray,
    t_grid: np.ndarray,
    min_ratio: float = 1.3,
    max_jump_time: float = 0.85,
    min_jump_time: float = 0.15,
    smoothing_window: int = 5,
) -> JumpDetectionResult:
    """Detect the replication jump time for a single gene.

    The jump is detected by finding the point where the mean expression
    after the point is significantly higher than before.

    Args:
        profile: Gene expression over time (n_time,).
        t_grid: Time grid (n_time,).
        min_ratio: Minimum ratio to consider as a jump.
        max_jump_time: Maximum time for valid jump.
        min_jump_time: Minimum time for valid jump.
        smoothing_window: Window size for smoothing.

    Returns:
        JumpDetectionResult with jump information.
    """
    profile = np.asarray(profile, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_time = len(profile)

    if n_time < 5:
        return JumpDetectionResult(
            has_jump=False, t_jump=float("nan"), jump_ratio=1.0, confidence=0.0
        )

    # Smooth the profile
    smoothed = smooth_profile(profile, smoothing_window)
    eps = 1e-12

    # Find the best jump point by comparing means before/after each point
    best_idx = -1
    best_ratio = min_ratio
    best_confidence = 0.0

    # Need at least 3 points on each side
    min_points = max(3, n_time // 10)

    for i in range(min_points, n_time - min_points):
        t = t_grid[i]

        # Skip if outside valid time range
        if t < min_jump_time or t > max_jump_time:
            continue

        # Compare mean before vs after this point
        mean_before = np.mean(smoothed[:i])
        mean_after = np.mean(smoothed[i:])
        ratio = mean_after / (mean_before + eps)

        # Check if this is a good jump candidate
        if ratio > best_ratio:
            # Confidence based on how close ratio is to 2 and std of segments
            std_before = np.std(smoothed[:i])
            std_after = np.std(smoothed[i:])
            total_std = (std_before + std_after) / 2

            # Normalized jump size
            jump_size = (mean_after - mean_before) / (total_std + eps)

            # Higher confidence for cleaner jumps (low std, ratio ~2)
            if ratio > 1.3:
                confidence = min(1.0, 0.5 + 0.5 * (1.0 - abs(ratio - 2.0)))
                confidence = max(confidence, 0.1)

                if ratio > best_ratio:
                    best_idx = i
                    best_ratio = ratio
                    best_confidence = confidence

    if best_idx < 0:
        return JumpDetectionResult(
            has_jump=False, t_jump=float("nan"), jump_ratio=1.0, confidence=0.0
        )

    t_jump = t_grid[best_idx]

    return JumpDetectionResult(
        has_jump=True,
        t_jump=float(t_jump),
        jump_ratio=float(best_ratio),
        confidence=float(best_confidence),
    )


def detect_all_jumps(
    G_emp: np.ndarray,
    t_grid: np.ndarray,
    min_ratio: float = 1.3,
    max_jump_time: float = 0.85,
    min_jump_time: float = 0.15,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect replication jumps for all genes.

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        t_grid: Time grid (n_time,).
        min_ratio: Minimum ratio to consider as a jump.
        max_jump_time: Maximum time for valid jump.
        min_jump_time: Minimum time for valid jump.
        smoothing_window: Window size for smoothing.

    Returns:
        t_jump_g: Jump times per gene (n_genes,), NaN if no jump.
        has_jump_g: Whether jump was detected per gene (n_genes,).
        confidence_g: Confidence scores per gene (n_genes,).
    """
    G_emp = np.asarray(G_emp, dtype=np.float64)
    n_genes = G_emp.shape[0]

    t_jump_g = np.full(n_genes, np.nan, dtype=np.float64)
    has_jump_g = np.zeros(n_genes, dtype=bool)
    confidence_g = np.zeros(n_genes, dtype=np.float64)

    for g in range(n_genes):
        result = detect_jump_time(
            G_emp[g],
            t_grid,
            min_ratio=min_ratio,
            max_jump_time=max_jump_time,
            min_jump_time=min_jump_time,
            smoothing_window=smoothing_window,
        )
        if result.has_jump:
            t_jump_g[g] = result.t_jump
            has_jump_g[g] = True
            confidence_g[g] = result.confidence

    return t_jump_g, has_jump_g, confidence_g


def infer_cd_from_jumps(
    t_jump_g: np.ndarray,
    has_jump_g: np.ndarray,
    x_g_prior: np.ndarray | None = None,
    C_prior: float = 0.6,
    D_prior: float = 0.2,
    min_jumps: int = 5,
) -> tuple[float, float, float, np.ndarray]:
    """Infer C and D parameters from detected jump times.

    If prior gene positions are known, use linear regression:
        t_jump = B + C × x = (1 - C - D) + C × x

    If no prior positions, use the distribution of jump times to estimate C, D.

    Args:
        t_jump_g: Jump times per gene (n_genes,).
        has_jump_g: Whether jump was detected per gene (n_genes,).
        x_g_prior: Prior gene positions (if known).
        C_prior: Prior estimate of C (used if regression fails).
        D_prior: Prior estimate of D (used if regression fails).
        min_jumps: Minimum number of jumps required for regression.

    Returns:
        C: Inferred C-period.
        D: Inferred D-period.
        r2: R-squared of the regression (0 if no regression performed).
        x_g_inferred: Inferred gene positions (from jump times and C, D).
    """
    n_genes = len(t_jump_g)
    jump_mask = has_jump_g & np.isfinite(t_jump_g)
    n_jumps = np.sum(jump_mask)

    if n_jumps < min_jumps:
        # Not enough jumps for reliable inference
        # Use prior values and infer positions
        C = C_prior
        D = D_prior
        B = max(1.0 - C - D, 0.0)

        x_g = np.full(n_genes, 0.5, dtype=np.float64)  # Default to middle
        if n_jumps > 0:
            # Infer positions from jumps: x = (t_jump - B) / C
            x_g[jump_mask] = (t_jump_g[jump_mask] - B) / max(C, 0.1)
            x_g = np.clip(x_g, 0.0, 1.0)

        return C, D, 0.0, x_g

    jump_times = t_jump_g[jump_mask]

    if x_g_prior is not None:
        # Use known positions for regression
        x_known = x_g_prior[jump_mask]

        # Linear regression: t_jump = B + C × x
        slope, intercept, r_value, _, _ = stats.linregress(x_known, jump_times)
        r2 = r_value ** 2

        C = max(slope, 0.1)  # C must be positive
        B = max(intercept, 0.0)
        D = max(1.0 - B - C, 0.0)

        # Ensure valid parameters
        if B + C > 1.0:
            # Scale down proportionally
            total = B + C
            B = B / total
            C = C / total
            D = 0.0

        # Infer positions from jump times
        x_g = np.full(n_genes, 0.5, dtype=np.float64)
        x_g[jump_mask] = (t_jump_g[jump_mask] - B) / max(C, 0.1)
        x_g = np.clip(x_g, 0.0, 1.0)

        return C, D, r2, x_g

    # No prior positions: estimate from jump time distribution
    # Assume jumps span the range [B, B+C]
    t_min = np.percentile(jump_times, 5)
    t_max = np.percentile(jump_times, 95)

    B = max(t_min - 0.05, 0.0)  # Small margin
    C = max(t_max - B + 0.05, 0.2)  # Ensure reasonable C
    D = max(1.0 - B - C, 0.0)

    # Ensure valid parameters
    if B + C > 1.0:
        C = 1.0 - B
        D = 0.0

    # Infer positions from jump times
    x_g = np.full(n_genes, 0.5, dtype=np.float64)
    x_g[jump_mask] = (t_jump_g[jump_mask] - B) / max(C, 0.1)
    x_g = np.clip(x_g, 0.0, 1.0)

    # For genes without jumps, assign random positions
    no_jump_mask = ~jump_mask
    if np.any(no_jump_mask):
        rng = np.random.default_rng(42)
        x_g[no_jump_mask] = rng.uniform(0.0, 1.0, size=np.sum(no_jump_mask))

    return C, D, 0.0, x_g


def infer_positions_and_cd(
    G_emp: np.ndarray,
    t_grid: np.ndarray,
    x_g_prior: np.ndarray | None = None,
    C_prior: float = 0.6,
    D_prior: float = 0.2,
    min_ratio: float = 1.3,
    max_jump_time: float = 0.85,
    min_jump_time: float = 0.15,
    smoothing_window: int = 5,
    min_jumps: int = 5,
) -> PositionInferenceResult:
    """Infer gene positions and Cooper-Helmstetter parameters from expression profiles.

    This is the main position inference function that:
    1. Detects replication jumps in all gene profiles
    2. Infers C and D from jump time distribution
    3. Infers gene positions from jump times

    Args:
        G_emp: Empirical gene profiles (n_genes, n_time).
        t_grid: Time grid (n_time,).
        x_g_prior: Prior gene positions (optional, for validation).
        C_prior: Prior C-period estimate.
        D_prior: Prior D-period estimate.
        min_ratio: Minimum ratio to consider as a jump.
        max_jump_time: Maximum time for valid jump.
        min_jump_time: Minimum time for valid jump.
        smoothing_window: Window size for smoothing.
        min_jumps: Minimum jumps required for C, D inference.

    Returns:
        PositionInferenceResult with positions and parameters.
    """
    # Step 1: Detect jumps
    t_jump_g, has_jump_g, _ = detect_all_jumps(
        G_emp,
        t_grid,
        min_ratio=min_ratio,
        max_jump_time=max_jump_time,
        min_jump_time=min_jump_time,
        smoothing_window=smoothing_window,
    )

    n_jumps = int(np.sum(has_jump_g))

    # Step 2: Infer C, D and positions
    C, D, r2, x_g = infer_cd_from_jumps(
        t_jump_g,
        has_jump_g,
        x_g_prior=x_g_prior,
        C_prior=C_prior,
        D_prior=D_prior,
        min_jumps=min_jumps,
    )

    return PositionInferenceResult(
        x_g=x_g,
        t_jump_g=t_jump_g,
        has_jump_g=has_jump_g,
        C=C,
        D=D,
        regression_r2=r2,
        n_jumps_detected=n_jumps,
    )


def refine_positions_with_cd(
    t_jump_g: np.ndarray,
    has_jump_g: np.ndarray,
    C: float,
    D: float,
) -> np.ndarray:
    """Refine gene positions given known C and D values.

    Uses the formula: x = (t_jump - B) / C where B = 1 - C - D.

    Args:
        t_jump_g: Jump times per gene.
        has_jump_g: Whether jump was detected per gene.
        C: C-period value.
        D: D-period value.

    Returns:
        x_g: Refined gene positions.
    """
    n_genes = len(t_jump_g)
    B = max(1.0 - C - D, 0.0)

    x_g = np.full(n_genes, 0.5, dtype=np.float64)

    jump_mask = has_jump_g & np.isfinite(t_jump_g)
    if np.any(jump_mask):
        x_g[jump_mask] = (t_jump_g[jump_mask] - B) / max(C, 0.1)
        x_g = np.clip(x_g, 0.0, 1.0)

    return x_g
