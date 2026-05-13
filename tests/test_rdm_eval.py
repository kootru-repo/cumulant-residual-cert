"""End-to-end tests for the RDM-based cumulant evaluator.

These tests build dense fermionic states on small registers, compute their
RDMs in this library's convention via the dense letter operators in
``_fermion``, and verify that the RDM-based cumulant evaluator reproduces
the cumulants computed directly from the full state.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from cumulant_residual_cert import Catalog
from cumulant_residual_cert._fermion import letter_op
from cumulant_residual_cert._rdm_eval import (
    evaluate_normal_ordered_term,
    evaluate_subword_moment,
    evaluate_word_cumulant,
)


def _slater_state(occupied: tuple[int, ...], n: int) -> np.ndarray:
    """Slater determinant with the given 1-based occupied sites."""
    bits = [1 if (i + 1) in occupied else 0 for i in range(n)]
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = 1.0
    return np.outer(psi, psi.conj())


def _build_rdm(rho: np.ndarray, k: int, n: int) -> np.ndarray:
    """Build the rank-$k$ RDM in this library's convention from a dense state.

    Convention:
        D^(k)[p1, ..., pk, q1, ..., qk]
            = <a^dag_{p1} ... a^dag_{pk} a_{qk} ... a_{q1}>.

    Indices are 0-based.
    """
    shape = (n,) * (2 * k)
    rdm = np.zeros(shape, dtype=complex)
    for indices in product(range(n), repeat=2 * k):
        p = indices[:k]
        q = indices[k:]
        # Build a^dag_{p1} ... a^dag_{pk} a_{qk} ... a_{q1}.
        op = letter_op("I", 1, n)
        for site_0based in p:
            op = op @ letter_op("a_dag", site_0based + 1, n)
        for site_0based in reversed(q):
            op = op @ letter_op("a", site_0based + 1, n)
        rdm[indices] = np.trace(rho @ op)
    return rdm


# ---------- Building-block correctness -----------------------------------


def test_evaluate_normal_ordered_term_identity():
    """Identity element <I> = 1 for any normalized state."""
    n = 2
    rdm1 = np.eye(n, dtype=complex)
    val = evaluate_normal_ordered_term((), (), rdm1)
    assert val == pytest.approx(1.0)


def test_evaluate_subword_moment_n_p_matches_1rdm_diagonal():
    """<n_p> = D^(1)[p, p]."""
    n = 3
    rho = _slater_state(occupied=(1, 3), n=n)
    rdm1 = _build_rdm(rho, 1, n)
    for p in (1, 2, 3):
        val = evaluate_subword_moment(("n",), (p,), rdm1)
        expected = rdm1[p - 1, p - 1]
        assert val == pytest.approx(expected)


def test_evaluate_subword_moment_a_dag_a_matches_1rdm():
    """<a^dag_p a_q> = D^(1)[p, q] (the 1-RDM element)."""
    n = 3
    # Pick a non-trivial state: equal superposition of two Slaters with
    # different occupations to give nontrivial off-diagonal 1-RDM.
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0b011] = 1.0 / np.sqrt(2)
    psi[0b101] = 1.0 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rdm1 = _build_rdm(rho, 1, n)
    for p, q in product((1, 2, 3), repeat=2):
        val = evaluate_subword_moment(("a_dag", "a"), (p, q), rdm1)
        expected = rdm1[p - 1, q - 1]
        assert val == pytest.approx(expected), (p, q, val, expected)


# ---------- Bernoulli-class theorem: cumulants vanish -------------------


def test_slater_determinant_gives_zero_catalog_cumulants():
    """The worked example: occupation-basis Slater -> Delta_cat = 0."""
    n = 4
    rho = _slater_state(occupied=(1, 2), n=n)
    rdm1 = _build_rdm(rho, 1, n)
    rdm2 = _build_rdm(rho, 2, n)
    rdm3 = _build_rdm(rho, 3, n)
    rdm4 = _build_rdm(rho, 4, n)

    cat = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    for w, sites in zip(cat, sites_per_word):
        kappa = evaluate_word_cumulant(w, sites, rdm1, rdm2, rdm3, rdm4)
        assert abs(kappa) < 1e-10, (w.name, kappa)


def test_random_product_state_gives_zero_catalog_cumulants():
    """A randomly-chosen *product* Bernoulli state also gives Delta_cat = 0.

    The Bernoulli class is the product family rho = prod_i (q_i |0><0| +
    p_i |1><1|), not an arbitrary diagonal density matrix.
    """
    rng = np.random.default_rng(42)
    n = 4
    occupation_probs = rng.uniform(0.1, 0.9, size=n)
    weights = np.zeros(2 ** n, dtype=complex)
    for idx in range(2 ** n):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        prob = 1.0
        for p_i, b in zip(occupation_probs, bits):
            prob *= p_i if b == 1 else (1 - p_i)
        weights[idx] = prob
    rho = np.diag(weights)

    rdm1 = _build_rdm(rho, 1, n)
    rdm2 = _build_rdm(rho, 2, n)
    rdm3 = _build_rdm(rho, 3, n)
    rdm4 = _build_rdm(rho, 4, n)

    cat = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    for w, sites in zip(cat, sites_per_word):
        kappa = evaluate_word_cumulant(w, sites, rdm1, rdm2, rdm3, rdm4)
        assert abs(kappa) < 1e-10, (w.name, kappa)


# ---------- Correlated state: nonzero cumulants pass dense cross-check ----


def _connected_cumulant_dense(rho: np.ndarray, letters, sites, n: int) -> complex:
    """Compute kappa_W(rho) directly via dense subword-moment evaluation."""
    from itertools import combinations
    from math import factorial

    from cumulant_residual_cert._fermion import subword_op
    from cumulant_residual_cert._partition import set_partitions

    m = len(letters)
    moments: dict[tuple[int, ...], complex] = {}
    for k in range(1, m + 1):
        for B in combinations(range(1, m + 1), k):
            key = tuple(sorted(B))
            A_B = subword_op(letters, sites, key, n)
            moments[key] = complex(np.trace(rho @ A_B))

    kappa = 0.0 + 0j
    for pi in set_partitions(list(range(1, m + 1))):
        prod = 1.0 + 0j
        for B in pi:
            prod *= moments[tuple(sorted(B))]
        k_pi = len(pi)
        kappa += (-1) ** (k_pi - 1) * factorial(k_pi - 1) * prod
    return kappa


def test_correlated_state_rdm_cumulant_matches_dense():
    """On a non-Slater $U(1)$-invariant state, RDM cumulants match dense ones.

    The evaluator assumes $U(1)$-invariance of $\\rho$; a generic random
    Hermitian PSD state would have nonzero off-particle-number sectors and
    is outside the library's scope. We construct a block-random $U(1)$-
    invariant state by drawing an independent random density matrix in
    each fixed-particle-number sector.
    """
    rng = np.random.default_rng(7)
    n = 4
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    # Group basis indices by particle number.
    sectors: dict[int, list[int]] = {}
    for idx in range(dim):
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        sectors.setdefault(sum(bits), []).append(idx)
    sector_weights = rng.dirichlet(np.ones(len(sectors)))
    for w_sec, (_N, basis_idxs) in zip(sector_weights, sorted(sectors.items())):
        k = len(basis_idxs)
        M = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
        M = M @ M.conj().T
        M /= np.trace(M)
        for i, gi in enumerate(basis_idxs):
            for j, gj in enumerate(basis_idxs):
                rho[gi, gj] = w_sec * M[i, j]

    rdm1 = _build_rdm(rho, 1, n)
    rdm2 = _build_rdm(rho, 2, n)
    rdm3 = _build_rdm(rho, 3, n)
    rdm4 = _build_rdm(rho, 4, n)

    cat = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    for w, sites in zip(cat, sites_per_word):
        kappa_rdm = evaluate_word_cumulant(w, sites, rdm1, rdm2, rdm3, rdm4)
        kappa_dense = _connected_cumulant_dense(rho, w.letters, sites, n)
        assert kappa_rdm == pytest.approx(kappa_dense, abs=1e-9, rel=1e-6), (
            w.name, kappa_rdm, kappa_dense
        )
