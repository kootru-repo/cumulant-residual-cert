"""Tests for the random-Pauli shadow UCB diagnostic.

Heavy linear-algebra tests are isolated here so they can be skipped on CI when
numpy is not available (it always is, but the marker makes the dependency
explicit). These tests run in seconds on a single core.
"""

from __future__ import annotations

import numpy as np
import pytest

from cumulant_residual_cert import Catalog, delta_ucb
from cumulant_residual_cert.diagnostic import collect_shadows


def _vacuum_state(n: int) -> np.ndarray:
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _two_particle_basis_state(n: int) -> np.ndarray:
    """Slater determinant with sites 1 and 2 occupied. Charge-neutral Bernoulli case."""
    bits = [1, 1] + [0] * (n - 2)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return np.outer(psi, psi.conj())


def test_ucb_returns_positive_finite_value():
    n = 4
    rho = _two_particle_basis_state(n)
    cat = Catalog.chemistry_r4()
    # All catalog words need 3 or 4 distinct sites; pick a fixed assignment.
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    shadows = collect_shadows(rho, n=n, M=500, seed=42)
    result = delta_ucb(
        shadow_samples=shadows,
        catalog=cat,
        sites_per_word=sites_per_word,
        n_qubits=n,
        confidence=0.95,
    )
    assert result.delta_ucb > 0
    assert np.isfinite(result.delta_ucb)
    assert result.n_paulis > 0


def test_ucb_rejects_invalid_confidence():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        delta_ucb(
            shadow_samples=[(("X",) * 4, (1,) * 4)],
            catalog=cat,
            sites_per_word=[(1, 2, 3)] * 2 + [(1, 2, 3, 4)] * 3,
            n_qubits=4,
            confidence=1.5,
        )


def test_ucb_rejects_mismatched_sites():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        delta_ucb(
            shadow_samples=[(("X",) * 4, (1,) * 4)],
            catalog=cat,
            sites_per_word=[(1, 2, 3)],  # too few entries
            n_qubits=4,
            confidence=0.95,
        )


def test_ucb_per_word_breakdown_is_populated():
    n = 4
    rho = _two_particle_basis_state(n)
    cat = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    shadows = collect_shadows(rho, n=n, M=200, seed=7)
    result = delta_ucb(
        shadow_samples=shadows,
        catalog=cat,
        sites_per_word=sites_per_word,
        n_qubits=n,
    )
    for w in cat:
        assert w.name in result.per_word
        assert {"kappa_hat", "radius", "ucb"} <= set(result.per_word[w.name].keys())


def _valid_sites():
    return [(1, 2, 3), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]


def _valid_shot(n_qubits: int):
    return (("X",) * n_qubits, (1,) * n_qubits)


def test_ucb_rejects_empty_shadow_samples():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError, match="empty"):
        delta_ucb(
            shadow_samples=[],
            catalog=cat,
            sites_per_word=_valid_sites(),
            n_qubits=4,
        )


def test_ucb_rejects_iterable_consumed_after_pass():
    """An iterable (generator) shadow source must work; len() is materialized internally."""
    cat = Catalog.chemistry_r4()
    n = 4
    rho = _two_particle_basis_state(n)
    shadows_list = collect_shadows(rho, n=n, M=80, seed=11)
    shadow_iter = (shot for shot in shadows_list)
    result = delta_ucb(
        shadow_samples=shadow_iter,
        catalog=cat,
        sites_per_word=_valid_sites(),
        n_qubits=n,
    )
    assert result.delta_ucb > 0


def test_ucb_rejects_bad_basis_label():
    cat = Catalog.chemistry_r4()
    bad = (("Q", "X", "X", "X"), (1, 1, 1, 1))
    with pytest.raises(ValueError, match="basis must be X/Y/Z"):
        delta_ucb(
            shadow_samples=[bad],
            catalog=cat,
            sites_per_word=_valid_sites(),
            n_qubits=4,
        )


def test_ucb_rejects_outcomes_outside_pm1():
    cat = Catalog.chemistry_r4()
    bad = (("X", "X", "X", "X"), (1, 0, 1, 1))
    with pytest.raises(ValueError, match=r"outcomes must be in \{-1, \+1\}"):
        delta_ucb(
            shadow_samples=[bad],
            catalog=cat,
            sites_per_word=_valid_sites(),
            n_qubits=4,
        )


def test_ucb_rejects_duplicate_site_indices_in_word():
    cat = Catalog.chemistry_r4()
    bad_sites = [(1, 1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
    with pytest.raises(ValueError, match="duplicate site"):
        delta_ucb(
            shadow_samples=[_valid_shot(4)],
            catalog=cat,
            sites_per_word=bad_sites,
            n_qubits=4,
        )


def test_ucb_rejects_site_out_of_range():
    cat = Catalog.chemistry_r4()
    bad_sites = [(1, 2, 99), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
    with pytest.raises(ValueError, match="outside 1"):
        delta_ucb(
            shadow_samples=[_valid_shot(4)],
            catalog=cat,
            sites_per_word=bad_sites,
            n_qubits=4,
        )


def test_ucb_dense_path_rejects_n_qubits_above_10():
    cat = Catalog.chemistry_r4()
    n = 11
    with pytest.raises(ValueError, match=r"refusing to enumerate"):
        delta_ucb(
            shadow_samples=[_valid_shot(n)],
            catalog=cat,
            sites_per_word=_valid_sites(),
            n_qubits=n,
        )
