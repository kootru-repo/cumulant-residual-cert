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
