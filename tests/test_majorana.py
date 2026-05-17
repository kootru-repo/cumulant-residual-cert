"""Tests for the letter-to-Majorana decomposition."""

from __future__ import annotations

import pytest
from cumulant_residual_cert._majorana import (
    _canonicalize_majorana_product,
    letter_majorana_decomposition,
    multiply_majorana_terms,
    word_majorana_decomposition,
)

# ----- letter-level decomposition -----


def test_identity_letter():
    assert letter_majorana_decomposition("I", 1) == {(): 1.0 + 0j}


def test_number_operator_at_site_1():
    """n_1 = 1/2 + (i/2) gamma_1 gamma_2."""
    assert letter_majorana_decomposition("n", 1) == {(): 0.5 + 0j, (1, 2): 0.5j}


def test_number_operator_at_site_5():
    """n_5 maps to Majorana indices (9, 10)."""
    decomp = letter_majorana_decomposition("n", 5)
    assert decomp == {(): 0.5 + 0j, (9, 10): 0.5j}


def test_a_operator_at_site_3():
    """a_3 = (gamma_5 + i gamma_6) / 2."""
    decomp = letter_majorana_decomposition("a", 3)
    assert decomp == {(5,): 0.5 + 0j, (6,): 0.5j}


def test_a_dag_operator_at_site_3():
    """a_dag_3 = (gamma_5 - i gamma_6) / 2."""
    decomp = letter_majorana_decomposition("a_dag", 3)
    assert decomp == {(5,): 0.5 + 0j, (6,): -0.5j}


def test_unknown_letter_rejected():
    with pytest.raises(ValueError, match="unknown"):
        letter_majorana_decomposition("Q", 1)


def test_nonpositive_site_rejected():
    with pytest.raises(ValueError, match="positive"):
        letter_majorana_decomposition("n", 0)


# ----- canonicalization -----


def test_canonicalize_empty():
    assert _canonicalize_majorana_product([]) == (1, ())


def test_canonicalize_single_index():
    assert _canonicalize_majorana_product([5]) == (1, (5,))


def test_canonicalize_already_sorted_pair():
    assert _canonicalize_majorana_product([1, 3]) == (1, (1, 3))


def test_canonicalize_inverted_pair():
    """gamma_3 gamma_1 = -gamma_1 gamma_3."""
    assert _canonicalize_majorana_product([3, 1]) == (-1, (1, 3))


def test_canonicalize_squared_pair():
    """gamma_5 gamma_5 = 1, both removed."""
    assert _canonicalize_majorana_product([5, 5]) == (1, ())


def test_canonicalize_three_index_cycle():
    """gamma_3 gamma_1 gamma_2 = -gamma_1 gamma_3 gamma_2 = +gamma_1 gamma_2 gamma_3."""
    assert _canonicalize_majorana_product([3, 1, 2]) == (1, (1, 2, 3))


def test_canonicalize_sandwich():
    """gamma_2 gamma_1 gamma_2 = -gamma_1 gamma_2 gamma_2 = -gamma_1 * 1 = -gamma_1."""
    assert _canonicalize_majorana_product([2, 1, 2]) == (-1, (1,))


def test_canonicalize_self_inverse_via_swap():
    """gamma_1 gamma_2 gamma_2 gamma_1 = gamma_1 * 1 * gamma_1 = gamma_1^2 = 1."""
    assert _canonicalize_majorana_product([1, 2, 2, 1]) == (1, ())


# ----- multiplication -----


def test_multiply_terms_with_complex_coefficients():
    a = ((1,), 0.5 + 0j)
    b = ((2,), 0.5j)
    indices, coeff = multiply_majorana_terms(a, b)
    assert indices == (1, 2)
    assert coeff == pytest.approx(0.25j)


def test_multiply_picks_up_minus_on_swap():
    """gamma_3 * gamma_1 = -gamma_1 gamma_3 (one swap)."""
    a = ((3,), 1.0 + 0j)
    b = ((1,), 1.0 + 0j)
    indices, coeff = multiply_majorana_terms(a, b)
    assert indices == (1, 3)
    assert coeff == pytest.approx(-1.0 + 0j)


# ----- word-level decomposition -----


def test_word_a_dag_a_at_same_site_recovers_n():
    """a_dag_p a_p = n_p in fermionic algebra."""
    site = 2
    decomp = word_majorana_decomposition(["a_dag", "a"], [site, site])
    expected = letter_majorana_decomposition("n", site)
    # Compare key by key, allowing complex tolerance.
    assert set(decomp) == set(expected)
    for k in expected:
        assert decomp[k] == pytest.approx(expected[k])


def test_word_n_squared_collapses_to_n():
    """n_p n_p = n_p."""
    site = 3
    decomp = word_majorana_decomposition(["n", "n"], [site, site])
    expected = letter_majorana_decomposition("n", site)
    assert set(decomp) == set(expected)
    for k in expected:
        assert decomp[k] == pytest.approx(expected[k])


def test_word_n_p_n_q_distinct_sites():
    """n_1 n_2: product of two distinct number operators.

    n_p n_q = (1/2 + i/2 gamma_{2p-1} gamma_{2p}) (1/2 + i/2 gamma_{2q-1} gamma_{2q})
            = 1/4 + i/4 gamma_{2p-1} gamma_{2p} + i/4 gamma_{2q-1} gamma_{2q}
              + (i/2)^2 gamma_{2p-1} gamma_{2p} gamma_{2q-1} gamma_{2q}
            = 1/4 + i/4 g1 g2 + i/4 g3 g4 - 1/4 g1 g2 g3 g4    (p=1, q=2 -> indices 1,2,3,4)
    """
    decomp = word_majorana_decomposition(["n", "n"], [1, 2])
    expected = {
        (): 0.25 + 0j,
        (1, 2): 0.25j,
        (3, 4): 0.25j,
        (1, 2, 3, 4): -0.25 + 0j,
    }
    assert set(decomp) == set(expected)
    for k, v in expected.items():
        assert decomp[k] == pytest.approx(v)


def test_word_charged_word_has_only_odd_majorana_products():
    """a_dag_1 a_2 (charge 0) decomposition has only even-degree Majorana products.

    More generally: total charge of a word equals 0 mod 2 iff its Majorana
    decomposition lives entirely in even-degree products. The chemistry
    catalog has charge-neutral words, so each catalog word decomposes into
    pure even-degree Majorana products (which is what matchgate shadows
    estimate).
    """
    decomp = word_majorana_decomposition(["a_dag", "a"], [1, 2])
    for indices in decomp:
        assert len(indices) % 2 == 0, indices


def test_word_chemistry_catalog_word_charge_neutral_to_even_majorana():
    """a_dag a_dag a a at sites 1, 2, 3, 4 decomposes into even-degree products."""
    decomp = word_majorana_decomposition(["a_dag", "a_dag", "a", "a"], [1, 2, 3, 4])
    for indices, coeff in decomp.items():
        assert len(indices) % 2 == 0, indices
        assert abs(coeff) > 1e-15


def test_word_length_mismatch_rejected():
    with pytest.raises(ValueError, match="equal length"):
        word_majorana_decomposition(["n", "n"], [1])


# ----- cross-check against dense fermion algebra -----


def test_word_decomposition_matches_dense_operator():
    """The Majorana decomposition reproduces the dense fermion-operator matrix.

    For each Majorana product gamma_S that appears, we build it as a dense
    matrix on n=4 modes and combine with the decomposition coefficients.
    The result must equal the dense letter-product operator from _fermion.
    """
    import numpy as np
    from cumulant_residual_cert._fermion import I2, _kron, letter_op

    n = 4
    dim = 2**n
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I_mat = I2

    def majorana_dense(j: int) -> np.ndarray:
        """gamma_j in JW form. 1-based j; site p has gamma_{2p-1}, gamma_{2p}."""
        site = (j + 1) // 2  # 1-based site
        is_odd = j % 2 == 1
        blocks = []
        for s in range(1, n + 1):
            if s < site:
                blocks.append(Z)
            elif s == site:
                blocks.append(X if is_odd else Y)
            else:
                blocks.append(I_mat)
        return _kron(blocks)

    def majorana_product_dense(indices) -> np.ndarray:
        op = np.eye(dim, dtype=complex)
        for j in indices:
            op = op @ majorana_dense(j)
        return op

    # Test on "a_dag a n" at sites (1, 2, 3).
    letters = ("a_dag", "a", "n")
    sites = (1, 2, 3)
    decomp = word_majorana_decomposition(letters, sites)
    # Build from decomposition.
    reconstructed = np.zeros((dim, dim), dtype=complex)
    for indices, coeff in decomp.items():
        reconstructed += coeff * majorana_product_dense(indices)
    # Direct word operator.
    direct = letter_op(letters[0], sites[0], n)
    for L, s in zip(letters[1:], sites[1:], strict=False):
        direct = direct @ letter_op(L, s, n)
    assert np.allclose(
        reconstructed, direct, atol=1e-10
    ), f"max diff: {np.max(np.abs(reconstructed - direct))}"
