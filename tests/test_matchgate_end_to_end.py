"""End-to-end matchgate-shadow UCB pipeline tests.

Covers ``delta_ucb_matchgate_shadows`` (added in the Pass 3 wiring) plus
its integration with:

- ``MatchgateShadowRecord`` and ``generate_matchgate_shadow_record``
  (synthetic shadow generation).
- ``matchgate_inverse_channel_majorana_moments`` (inverse channel).
- ``delta_ucb_from_majorana_moments`` (downstream protocol-agnostic UCB).

Both provenance labels (``ucb_matchgate_shadows`` and
``ucb_matchgate_shadows_u1_assumed``) are exercised, as is the
``radius="empirical_bernstein"`` opt-in canonical sample split.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from cumulant_residual_cert import (
    Catalog,
    MatchgateShadowRecord,
    delta_ucb_from_majorana_moments,
    delta_ucb_matchgate_shadows,
    generate_matchgate_shadow_record,
)
from cumulant_residual_cert._majorana import word_majorana_decomposition
from cumulant_residual_cert._matchgate import (
    pfaffian,
    sample_matchgate_rotation,
)
from cumulant_residual_cert._matchgate_shadow import (
    matchgate_inverse_channel_majorana_moments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_determinant_covariance(occupations: list[int]) -> np.ndarray:
    """Majorana covariance of an HF determinant in the matchgate-shadow
    convention (0-based Majorana indices; (C_b)[2p, 2p+1] = 1 - 2 b_p)."""
    n = len(occupations)
    C = np.zeros((2 * n, 2 * n))
    for p in range(n):
        s = 1.0 - 2.0 * occupations[p]
        C[2 * p, 2 * p + 1] = s
        C[2 * p + 1, 2 * p] = -s
    return C


def _gaussian_majorana_expectation(C: np.ndarray, indices_0based: tuple[int, ...]) -> complex:
    """Wick expectation $\\langle \\gamma_S \\rangle = i^k Pf(C[S, S])$ on a
    Gaussian state with Majorana covariance C (0-based indices)."""
    if not indices_0based:
        return 1.0 + 0j
    if len(indices_0based) % 2 != 0:
        return 0.0 + 0j
    k = len(indices_0based) // 2
    sub = C[np.ix_(indices_0based, indices_0based)]
    return (1j) ** k * pfaffian(sub)


def _exact_majorana_moments_from_covariance(
    C: np.ndarray,
    catalog: Catalog,
    sites_per_word: list[tuple[int, ...]],
) -> dict[tuple[int, ...], tuple[complex, float]]:
    """Exact (no shot noise) Majorana moments for every term in any catalog subword.

    Returned in the format expected by ``delta_ucb_from_majorana_moments``:
    keys are strictly-sorted 1-based Majorana index tuples; values are
    ``(complex_mean, radius=0)``.
    """
    moments: dict[tuple[int, ...], tuple[complex, float]] = {(): (1.0 + 0j, 0.0)}
    for w, sites in zip(catalog, sites_per_word, strict=False):
        m = w.length
        for k_sub in range(1, m + 1):
            for B in combinations(range(1, m + 1), k_sub):
                key = tuple(sorted(B))
                sub_letters = tuple(w.letters[i - 1] for i in key)
                sub_sites = tuple(sites[i - 1] for i in key)
                decomp = word_majorana_decomposition(sub_letters, sub_sites)
                for indices_1based in decomp:
                    if not indices_1based or indices_1based in moments:
                        continue
                    indices_0based = tuple(j - 1 for j in indices_1based)
                    val = _gaussian_majorana_expectation(C, indices_0based)
                    moments[indices_1based] = (val, 0.0)
    return moments


def _default_chemistry_sites() -> list[tuple[int, ...]]:
    """Site assignments matching the implicit default of
    ``delta_ucb_matchgate_shadows`` (leading consecutive sites)."""
    cat = Catalog.chemistry_r4()
    return [tuple(range(1, w.length + 1)) for w in cat]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_hf_determinant_covers_zero():
    """HF determinant: cumulants are zero, so any UCB >= 0 (trivial wiring check)."""
    cat = Catalog.chemistry_r4()
    occupations = [1, 0, 1, 0]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=20260519)
    record = generate_matchgate_shadow_record(C, n_shots=500, rng=rng)
    result = delta_ucb_matchgate_shadows(cat, record, alpha=0.05)
    assert result.delta_ucb >= 0.0
    # Sanity: the protocol-term count is the number of distinct Majorana
    # products in the union of the chemistry-r4 catalog subwords.
    assert result.n_paulis > 0
    # Every catalog word should produce a per-word breakdown.
    for w in cat:
        assert w.name in result.per_word
        for field in ("kappa_hat", "radius", "ucb"):
            assert field in result.per_word[w.name]


def test_e2e_correlated_state_covers_true_delta():
    """Correlated Gaussian state: UCB >= analytically computed true Delta.

    Constructs a small Bogoliubov-rotated HF determinant on n=4 (Majorana
    covariance C = Q C0 Q^T for a fixed Q in O(8)), computes the exact
    catalog envelope by evaluating each per-word cumulant on C via Wick's
    theorem (zero-radius path through delta_ucb_from_majorana_moments),
    and verifies the shadow-based UCB upper-bounds that envelope.
    """
    cat = Catalog.chemistry_r4()
    sites = _default_chemistry_sites()
    # Fixed seed for both reproducibility and a non-trivial Q.
    rng_fixed = np.random.default_rng(seed=20260519)
    Q = sample_matchgate_rotation(4, rng_fixed)
    occupations = [1, 0, 1, 0]
    C0 = _hf_determinant_covariance(occupations)
    C = Q @ C0 @ Q.T
    C = 0.5 * (C - C.T)  # enforce skew-symmetry against round-off

    # Compute the true Delta via zero-radius routing of the exact Majorana
    # moments through the same Mobius assembly used by the UCB pipeline.
    exact_moments = _exact_majorana_moments_from_covariance(C, cat, sites)
    exact_result = delta_ucb_from_majorana_moments(
        majorana_moments=exact_moments,
        catalog=cat,
        sites_per_word=sites,
        confidence=0.95,
        n_protocol_terms=1,
        require_all_terms=True,
    )
    true_delta = exact_result.delta_ucb

    # Now run the matchgate-shadow UCB and verify it bounds the true Delta.
    rng_shots = np.random.default_rng(seed=99)
    M = 10000
    record = generate_matchgate_shadow_record(C, n_shots=M, rng=rng_shots)
    ucb_result = delta_ucb_matchgate_shadows(cat, record, alpha=0.05)
    assert ucb_result.delta_ucb >= true_delta, (
        f"matchgate-shadow UCB {ucb_result.delta_ucb} does not cover true "
        f"Delta {true_delta} (gap {true_delta - ucb_result.delta_ucb})"
    )


def test_e2e_provenance_unverified_default():
    """Default u1_certified=False -> delta_provenance flags the U(1) assumption."""
    cat = Catalog.chemistry_r4()
    C = _hf_determinant_covariance([0, 1, 0, 1])
    rng = np.random.default_rng(seed=1)
    record = generate_matchgate_shadow_record(C, n_shots=200, rng=rng)
    result = delta_ucb_matchgate_shadows(cat, record, alpha=0.05)
    assert result.delta_provenance == "ucb_matchgate_shadows_u1_assumed"


def test_e2e_provenance_certified():
    """Explicit u1_certified=True -> delta_provenance is the unconditional label."""
    cat = Catalog.chemistry_r4()
    C = _hf_determinant_covariance([0, 1, 0, 1])
    rng = np.random.default_rng(seed=2)
    record = generate_matchgate_shadow_record(C, n_shots=200, rng=rng)
    result = delta_ucb_matchgate_shadows(
        cat, record, alpha=0.05, u1_certified=True
    )
    assert result.delta_provenance == "ucb_matchgate_shadows"


def test_e2e_empirical_bernstein_opt_in():
    """``radius='empirical_bernstein'`` enforces a 50/50 canonical split.

    The diagnostic half feeds the UCB; the holdout is reserved per the
    Maurer-Pontil canonical split. The per-word radii should match what
    Hoeffding produces on a half-sized record, which is strictly larger
    than the Hoeffding radii on the full record. We use the per-word
    radius monotonicity as the observable proxy for "the split happened".
    """
    cat = Catalog.chemistry_r4()
    C = _hf_determinant_covariance([1, 0, 1, 0])
    rng = np.random.default_rng(seed=42)
    record_full = generate_matchgate_shadow_record(C, n_shots=400, rng=rng)
    hoeff_result = delta_ucb_matchgate_shadows(
        cat, record_full, alpha=0.05, radius="hoeffding"
    )
    eb_result = delta_ucb_matchgate_shadows(
        cat, record_full, alpha=0.05, radius="empirical_bernstein"
    )
    # The split version uses M/2 shots; Hoeffding radius scales as
    # 1/sqrt(M), so the empirical-Bernstein-path radius should be larger
    # by ~sqrt(2). Check per-word radii are strictly larger.
    for w in cat:
        r_full = hoeff_result.per_word[w.name]["radius"]
        r_split = eb_result.per_word[w.name]["radius"]
        # Strictly larger; small slack for floating-point ties on the
        # rare per-word degenerate case where the radius collapses.
        assert r_split > r_full * 1.2, (
            f"word {w.name!r}: empirical-Bernstein radius {r_split} not "
            f"meaningfully larger than full-Hoeffding {r_full}; the split "
            "may not have engaged"
        )


def test_e2e_matches_majorana_moments_path():
    """Wrapper matches the manual two-step composition exactly.

    Calls ``delta_ucb_matchgate_shadows`` and the equivalent manual chain
    ``matchgate_inverse_channel_majorana_moments`` + ``delta_ucb_from_majorana_moments``
    with the same Bonferroni splitting and verifies they produce the same
    UCB (numerical equality to high precision; this is wiring, not statistics).
    """
    cat = Catalog.chemistry_r4()
    sites = _default_chemistry_sites()
    occupations = [1, 0, 1, 0]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=12345)
    record = generate_matchgate_shadow_record(C, n_shots=300, rng=rng)
    alpha = 0.05

    wrapper_result = delta_ucb_matchgate_shadows(cat, record, alpha=alpha)

    # Manual replication of the wrapper.
    target_set_1based: set[tuple[int, ...]] = set()
    for w, st in zip(cat, sites, strict=False):
        m = w.length
        for k_sub in range(1, m + 1):
            for B in combinations(range(1, m + 1), k_sub):
                key = tuple(sorted(B))
                sub_letters = tuple(w.letters[i - 1] for i in key)
                sub_sites = tuple(st[i - 1] for i in key)
                decomp = word_majorana_decomposition(sub_letters, sub_sites)
                for indices in decomp:
                    if indices:
                        target_set_1based.add(indices)
    T = len(target_set_1based)
    alpha_per = alpha / T

    even_1 = sorted(S for S in target_set_1based if len(S) % 2 == 0)
    odd_1 = {S for S in target_set_1based if len(S) % 2 == 1}
    even_0 = [tuple(j - 1 for j in S) for S in even_1]
    inv_out = matchgate_inverse_channel_majorana_moments(
        record, even_0, alpha=alpha_per
    )
    manual_moments: dict[tuple[int, ...], tuple[complex, float]] = {
        (): (1.0 + 0j, 0.0)
    }
    for S1, S0 in zip(even_1, even_0, strict=True):
        k = len(S1) // 2
        m = inv_out[S0]["mean"]
        r = inv_out[S0]["radius"]
        if k % 2 == 0:
            manual_moments[S1] = (complex(m, 0.0), r)
        else:
            manual_moments[S1] = (complex(0.0, m), r)
    for S in odd_1:
        manual_moments[S] = (0.0 + 0j, 0.0)
    manual_result = delta_ucb_from_majorana_moments(
        majorana_moments=manual_moments,
        catalog=cat,
        sites_per_word=sites,
        confidence=1.0 - alpha,
        n_protocol_terms=T,
        require_all_terms=True,
    )

    assert wrapper_result.delta_ucb == pytest.approx(
        manual_result.delta_ucb, rel=1e-12, abs=1e-12
    )
    assert wrapper_result.confidence == pytest.approx(manual_result.confidence)
    assert wrapper_result.n_paulis == manual_result.n_paulis
    for w in cat:
        for field in ("kappa_hat", "radius", "ucb"):
            assert wrapper_result.per_word[w.name][field] == pytest.approx(
                manual_result.per_word[w.name][field], rel=1e-12, abs=1e-12
            )


def test_e2e_rejects_non_record():
    """A non-record input raises a clean TypeError, not a deep traceback."""
    cat = Catalog.chemistry_r4()
    with pytest.raises(TypeError, match="MatchgateShadowRecord"):
        delta_ucb_matchgate_shadows(cat, "not a record", alpha=0.05)


def test_e2e_rejects_bad_alpha():
    """alpha outside (0, 1) raises a ValueError."""
    cat = Catalog.chemistry_r4()
    C = _hf_determinant_covariance([0, 1])
    rng = np.random.default_rng(seed=3)
    record = generate_matchgate_shadow_record(C, n_shots=10, rng=rng)
    with pytest.raises(ValueError, match="alpha"):
        delta_ucb_matchgate_shadows(cat, record, alpha=1.5)


def test_e2e_validates_sites_length():
    """A wrong-length sites_per_word raises a ValueError."""
    cat = Catalog.chemistry_r4()
    C = _hf_determinant_covariance([0, 1, 0, 1])
    rng = np.random.default_rng(seed=4)
    record = generate_matchgate_shadow_record(C, n_shots=20, rng=rng)
    with pytest.raises(ValueError, match="sites_per_word"):
        delta_ucb_matchgate_shadows(
            cat, record, alpha=0.05, sites_per_word=[(1, 2, 3)]
        )
