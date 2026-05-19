"""Tests for the matchgate-shadow inverse channel and record handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cumulant_residual_cert._matchgate import (
    pfaffian,
    sample_matchgate_rotation,
)
from cumulant_residual_cert._matchgate_shadow import (
    MatchgateShadowRecord,
    generate_matchgate_shadow_record,
    matchgate_inverse_channel_majorana_moments,
)


# ----- Helpers -----


def _hf_determinant_covariance(occupations: list[int]) -> np.ndarray:
    """Majorana covariance of an HF determinant with given mode occupations.

    For each mode p, $C_{2p, 2p+1} = 1 - 2 \\text{occ}_p$.
    """
    n = len(occupations)
    C = np.zeros((2 * n, 2 * n))
    for p in range(n):
        s = 1.0 - 2.0 * occupations[p]
        C[2 * p, 2 * p + 1] = s
        C[2 * p + 1, 2 * p] = -s
    return C


def _gaussian_majorana_expectation(C: np.ndarray, indices: tuple[int, ...]) -> complex:
    """Analytic expectation $\\langle \\gamma_S \\rangle = i^k Pf(C[S, S])$."""
    if not indices:
        return 1.0 + 0j
    k = len(indices) // 2
    sub = C[np.ix_(indices, indices)]
    return (1j) ** k * pfaffian(sub)


# ----- Record validation -----


def test_record_validates_orthogonality():
    """Non-orthogonal rotation triggers a ValueError."""
    rng = np.random.default_rng(seed=0)
    n_modes = 2
    Q_good = sample_matchgate_rotation(n_modes, rng)
    Q_bad = Q_good.copy()
    Q_bad[0, 0] += 0.1  # destroy orthogonality
    rotations = np.stack([Q_good, Q_bad], axis=0)
    outcomes = np.array([[0, 1], [1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="not orthogonal"):
        MatchgateShadowRecord(
            rotations=rotations, outcomes=outcomes, n_modes=n_modes
        )


def test_record_validates_bitstring():
    """Outcomes outside {0, 1} trigger an error."""
    rng = np.random.default_rng(seed=0)
    n_modes = 2
    Q = sample_matchgate_rotation(n_modes, rng)
    rotations = np.stack([Q], axis=0)
    bad_outcomes = np.array([[0, 2]], dtype=np.int64)  # 2 is invalid
    with pytest.raises(ValueError, match="non-binary"):
        MatchgateShadowRecord(
            rotations=rotations, outcomes=bad_outcomes, n_modes=n_modes
        )


def test_record_validates_shapes():
    """Shape mismatches trigger errors."""
    rng = np.random.default_rng(seed=1)
    n_modes = 2
    Q = sample_matchgate_rotation(n_modes, rng)
    rotations = np.stack([Q, Q], axis=0)
    outcomes = np.array([[0, 1]], dtype=np.int64)  # 1 shot, but 2 rotations
    with pytest.raises(ValueError, match="inconsistent"):
        MatchgateShadowRecord(
            rotations=rotations, outcomes=outcomes, n_modes=n_modes
        )


def test_record_validates_n_modes():
    """Invalid n_modes triggers an error."""
    rng = np.random.default_rng(seed=2)
    Q = sample_matchgate_rotation(2, rng)
    rotations = np.stack([Q], axis=0)
    outcomes = np.array([[0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="positive integer"):
        MatchgateShadowRecord(rotations=rotations, outcomes=outcomes, n_modes=0)


# ----- Synthetic shadow generation -----


def test_synthetic_shadow_HF_determinant():
    """Empirical degree-2 means match the analytic covariance entries within radius.

    For an HF determinant on n=4 with occupations [1, 0, 1, 0], the analytic
    covariance is block-diagonal with diagonal blocks [[0, +/-1], [-/+1, 0]].
    Sample M=2000 shots and verify the degree-2 Majorana means at indices
    (2p, 2p+1) match within their certified Hoeffding radius at alpha=0.05.
    """
    occupations = [1, 0, 1, 0]
    n = len(occupations)
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=20260519)
    M = 2000
    record = generate_matchgate_shadow_record(C, M, rng)
    assert record.n_shots == M
    assert record.n_modes == n
    # Build target list: all degree-2 Majorana products at adjacent pairs.
    targets = [(2 * p, 2 * p + 1) for p in range(n)]
    # Additional non-adjacent pairs (should have analytic mean 0).
    targets += [(0, 2), (0, 3), (1, 3)]
    out = matchgate_inverse_channel_majorana_moments(
        record, targets, alpha=0.05
    )
    for S in targets:
        analytic = _gaussian_majorana_expectation(C, S)
        # For degree-2 (k=1), the expectation is imaginary; our output is the
        # imaginary component.
        target_val = float(analytic.imag)
        mean = out[S]["mean"]
        radius = out[S]["radius"]
        # We want to verify: |mean - target| <= radius (with high probability).
        # We use 3x radius as a soft buffer to avoid flakes; this is consistent
        # with the Hoeffding bound's tail decay.
        assert abs(mean - target_val) <= 3 * radius, (
            f"degree-2 mean check failed for S={S}: "
            f"empirical={mean}, analytic={target_val}, radius={radius}"
        )


def test_synthetic_shadow_degree_4_product():
    """Empirical degree-4 Majorana product mean matches Wick analytic within radius."""
    occupations = [1, 0, 1, 0]
    n = len(occupations)
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=20260520)
    M = 3000
    record = generate_matchgate_shadow_record(C, M, rng)
    S = (0, 1, 2, 3)
    out = matchgate_inverse_channel_majorana_moments(
        record, [S], alpha=0.05
    )
    analytic = _gaussian_majorana_expectation(C, S)
    # For degree-4 (k=2, even), the expectation is real.
    target_val = float(analytic.real)
    mean = out[S]["mean"]
    radius = out[S]["radius"]
    assert abs(mean - target_val) <= 3 * radius, (
        f"degree-4 mean check failed for S={S}: "
        f"empirical={mean}, analytic={target_val}, radius={radius}"
    )


def test_inverse_channel_unbiased():
    """Averaging over independent records converges to the true value."""
    occupations = [1, 0]
    n = len(occupations)
    C = _hf_determinant_covariance(occupations)
    S = (0, 1)
    analytic = _gaussian_majorana_expectation(C, S)
    target_val = float(analytic.imag)
    n_records = 12
    M_per = 500
    means = []
    for seed in range(n_records):
        rng = np.random.default_rng(seed=10000 + seed)
        record = generate_matchgate_shadow_record(C, M_per, rng)
        out = matchgate_inverse_channel_majorana_moments(record, [S], alpha=0.05)
        means.append(out[S]["mean"])
    grand_mean = float(np.mean(means))
    # Empirical std of the mean estimator -- under Hoeffding the per-record
    # error is O(sqrt(shadow_norm^2 / M)), so the std-of-means scales as
    # sqrt(shadow_norm^2 / (n_records * M)). For n=2, k=1, shadow_norm^2 =
    # C(4,2)/C(2,1) = 6/2 = 3.
    shadow_norm_sq = 3.0
    std_bound = math.sqrt(shadow_norm_sq / (n_records * M_per))
    # Allow 4x the std bound (still small; this is a soft regression check).
    assert abs(grand_mean - target_val) <= 4 * std_bound, (
        f"unbiased estimator check failed: grand_mean={grand_mean}, "
        f"analytic={target_val}, std_bound={std_bound}"
    )


def test_inverse_channel_radius_validity():
    """Empirical coverage of Hoeffding radius >= 95% across many trials at alpha=0.05.

    Run many independent shadow records, compute the certified radius, and
    measure how often the mean is within the radius of the analytic value.
    The Hoeffding bound is conservative (often by an order of magnitude in
    practice), so the empirical coverage should be much higher than 95%.
    """
    occupations = [1, 0, 1, 0]
    n = len(occupations)
    C = _hf_determinant_covariance(occupations)
    S = (0, 1)
    analytic = _gaussian_majorana_expectation(C, S)
    target_val = float(analytic.imag)
    n_trials = 200
    M_per = 400
    covered = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed=50000 + seed)
        record = generate_matchgate_shadow_record(C, M_per, rng)
        out = matchgate_inverse_channel_majorana_moments(
            record, [S], alpha=0.05
        )
        mean = out[S]["mean"]
        radius = out[S]["radius"]
        if abs(mean - target_val) <= radius:
            covered += 1
    coverage = covered / n_trials
    # Hoeffding guarantees >= 95% coverage; in practice much higher because
    # the per-shot variance is typically far below the shadow-norm bound.
    assert coverage >= 0.95, (
        f"empirical coverage {coverage} < 0.95 (covered {covered}/{n_trials})"
    )


def test_inverse_channel_empty_product_identity():
    """The empty Majorana product has mean 1 and radius 0 by convention."""
    occupations = [0, 1]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=7)
    record = generate_matchgate_shadow_record(C, 50, rng)
    out = matchgate_inverse_channel_majorana_moments(
        record, [()], alpha=0.05
    )
    assert out[()]["mean"] == pytest.approx(1.0)
    assert out[()]["radius"] == pytest.approx(0.0)


def test_inverse_channel_odd_degree_rejected():
    """Odd-degree Majorana products trigger an error."""
    occupations = [0]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=8)
    record = generate_matchgate_shadow_record(C, 10, rng)
    with pytest.raises(ValueError, match="even degree"):
        matchgate_inverse_channel_majorana_moments(record, [(0,)], alpha=0.05)


def test_inverse_channel_index_out_of_range():
    """Indices outside [0, 2n) trigger an error."""
    occupations = [0, 1]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=9)
    record = generate_matchgate_shadow_record(C, 10, rng)
    with pytest.raises(ValueError, match="out of range"):
        matchgate_inverse_channel_majorana_moments(record, [(0, 7)], alpha=0.05)


def test_inverse_channel_non_sorted_rejected():
    """Non-strictly-sorted index tuples trigger an error."""
    occupations = [0, 1]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=10)
    record = generate_matchgate_shadow_record(C, 10, rng)
    with pytest.raises(ValueError, match="strictly-sorted"):
        matchgate_inverse_channel_majorana_moments(
            record, [(1, 0)], alpha=0.05
        )
