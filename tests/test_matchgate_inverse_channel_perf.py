"""Parity and performance tests for the vectorized matchgate inverse channel.

These tests exercise the vectorized
:func:`matchgate_inverse_channel_majorana_moments` against the reference
:func:`_matchgate_inverse_channel_unvectorized` implementation to confirm
that the speed-up (closed-form Pfaffian batch in place of per-shot
canonicalized expansion) produces numerically identical results. The
performance check is opt-in via the ``perf`` marker so the suite stays
fast in the default configuration.
"""

from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np
import pytest

from cumulant_residual_cert import Catalog
from cumulant_residual_cert._majorana import word_majorana_decomposition
from cumulant_residual_cert._matchgate import sample_matchgate_rotation
from cumulant_residual_cert._matchgate_shadow import (
    _matchgate_inverse_channel_unvectorized,
    generate_matchgate_shadow_record,
    matchgate_inverse_channel_majorana_moments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_determinant_covariance(occupations: list[int]) -> np.ndarray:
    n = len(occupations)
    C = np.zeros((2 * n, 2 * n))
    for p in range(n):
        s = 1.0 - 2.0 * occupations[p]
        C[2 * p, 2 * p + 1] = s
        C[2 * p + 1, 2 * p] = -s
    return C


def _chemistry_r4_even_products_0based() -> list[tuple[int, ...]]:
    """The 0-based even-degree Majorana products spanning the chemistry-r4
    catalog at default site assignments, mirroring the wiring inside
    ``delta_ucb_matchgate_shadows``."""
    cat = Catalog.chemistry_r4()
    target_set: set[tuple[int, ...]] = set()
    for w in cat:
        m = w.length
        sites = tuple(range(1, m + 1))
        for k_sub in range(1, m + 1):
            for B in combinations(range(1, m + 1), k_sub):
                key = tuple(sorted(B))
                sub_letters = tuple(w.letters[i - 1] for i in key)
                sub_sites = tuple(sites[i - 1] for i in key)
                decomp = word_majorana_decomposition(sub_letters, sub_sites)
                for indices in decomp:
                    if indices:
                        target_set.add(indices)
    even_1 = sorted(S for S in target_set if len(S) % 2 == 0)
    return [tuple(j - 1 for j in S) for S in even_1]


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def test_vectorized_matches_unvectorized_small():
    """At n=4, M=100, vectorized and unvectorized paths agree to 1e-10."""
    occupations = [1, 0, 1, 0]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=20260519)
    record = generate_matchgate_shadow_record(C, n_shots=100, rng=rng)

    # Mix of degree-2, degree-4, degree-6, degree-8 targets.
    targets = [
        (),  # empty product / identity
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 4),
        (0, 7),
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (2, 3, 5, 6),
        (0, 1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 5, 6, 7),
    ]
    out_v = matchgate_inverse_channel_majorana_moments(record, targets, alpha=0.05)
    out_u = _matchgate_inverse_channel_unvectorized(record, targets, alpha=0.05)
    for S in targets:
        mv = out_v[S]["mean"]
        mu = out_u[S]["mean"]
        rv = out_v[S]["radius"]
        ru = out_u[S]["radius"]
        # Mean: exact equality is achievable up to round-off; allow 1e-10.
        # Use abs tolerance because empirical means can be near zero.
        assert abs(mv - mu) <= 1e-10, (
            f"mean mismatch at S={S}: vectorized={mv}, unvectorized={mu}, "
            f"diff={abs(mv - mu):.3e}"
        )
        # Radius is closed-form and should match bit-for-bit.
        assert rv == pytest.approx(ru, rel=1e-12, abs=1e-15)


def test_vectorized_matches_unvectorized_correlated_state():
    """On a non-trivial Bogoliubov-rotated Gaussian state, the two paths agree."""
    occupations = [1, 0, 1, 0]
    n = len(occupations)
    C0 = _hf_determinant_covariance(occupations)
    # Fixed seed for a non-trivial orthogonal rotation.
    rng_q = np.random.default_rng(seed=20260520)
    Q = sample_matchgate_rotation(n, rng_q)
    C = Q @ C0 @ Q.T
    C = 0.5 * (C - C.T)

    rng = np.random.default_rng(seed=20260521)
    record = generate_matchgate_shadow_record(C, n_shots=120, rng=rng)

    # All distinct even-degree products from the chemistry-r4 catalog.
    targets = _chemistry_r4_even_products_0based()
    # Truncate to keep the unvectorized comparison fast.
    targets = targets[:40]

    out_v = matchgate_inverse_channel_majorana_moments(record, targets, alpha=0.05)
    out_u = _matchgate_inverse_channel_unvectorized(record, targets, alpha=0.05)
    for S in targets:
        mv = out_v[S]["mean"]
        mu = out_u[S]["mean"]
        rv = out_v[S]["radius"]
        ru = out_u[S]["radius"]
        assert abs(mv - mu) <= 1e-10, (
            f"mean mismatch at S={S}: vectorized={mv}, unvectorized={mu}, "
            f"diff={abs(mv - mu):.3e}"
        )
        assert rv == pytest.approx(ru, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# Performance test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_vectorized_speedup():
    """On a realistic chemistry-r4 workload, vectorized is >= 5x faster.

    Compares wall-clock time for one call of the vectorized vs unvectorized
    inverse channel at n=4, M=5000 on the full chemistry-r4 even-degree
    Majorana-product catalog. Prints both timings and the achieved speedup.
    """
    occupations = [1, 0, 1, 0]
    C = _hf_determinant_covariance(occupations)
    rng = np.random.default_rng(seed=20260522)
    record = generate_matchgate_shadow_record(C, n_shots=5000, rng=rng)
    targets = _chemistry_r4_even_products_0based()

    # Warm-up call on a tiny subset to avoid first-call import / JIT overhead.
    matchgate_inverse_channel_majorana_moments(record, targets[:1], alpha=0.05)
    _matchgate_inverse_channel_unvectorized(record, targets[:1], alpha=0.05)

    t0 = time.perf_counter()
    out_v = matchgate_inverse_channel_majorana_moments(record, targets, alpha=0.05)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_u = _matchgate_inverse_channel_unvectorized(record, targets, alpha=0.05)
    t_unv = time.perf_counter() - t0

    speedup = t_unv / t_vec if t_vec > 0 else math.inf
    print(
        f"\n[perf] chemistry-r4 n=4 M=5000 targets={len(targets)}: "
        f"vectorized={t_vec:.3f}s unvectorized={t_unv:.3f}s "
        f"speedup={speedup:.1f}x"
    )

    # Parity sanity check on a random subset to confirm the speedup is not
    # being bought at the cost of correctness.
    for S in targets[::7]:
        assert abs(out_v[S]["mean"] - out_u[S]["mean"]) <= 1e-10

    assert speedup >= 5.0, (
        f"vectorized inverse channel only {speedup:.1f}x faster than "
        f"unvectorized (target >= 5x); vec={t_vec:.3f}s unv={t_unv:.3f}s"
    )
