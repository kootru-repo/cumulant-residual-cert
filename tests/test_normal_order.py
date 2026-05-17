"""Tests for the in-house fermionic normal-ordering routine."""

from __future__ import annotations

import pytest
from cumulant_residual_cert._normal_order import (
    NormalOrderedTerm,
    combine_terms,
    letter_primitives,
    normal_order,
    word_primitives,
)


def test_letter_primitives():
    assert letter_primitives("I", 1) == []
    assert letter_primitives("n", 3) == [("+", 3), ("-", 3)]
    assert letter_primitives("a", 5) == [("-", 5)]
    assert letter_primitives("a_dag", 7) == [("+", 7)]


def test_letter_primitives_rejects_unknown():
    with pytest.raises(ValueError):
        letter_primitives("Q", 1)


def test_word_primitives_concatenates():
    prims = word_primitives(["a_dag", "a", "n"], [1, 2, 3])
    assert prims == [("+", 1), ("-", 2), ("+", 3), ("-", 3)]


def test_normal_order_already_ordered():
    """a^dag_1 a_2 is already normal-ordered."""
    terms = normal_order([("+", 1), ("-", 2)])
    assert terms == [NormalOrderedTerm(sign=1, creations=(1,), annihilations=(2,))]


def test_normal_order_single_inversion_distinct_sites():
    """a_1 a^dag_2 = -a^dag_2 a_1 (no contraction since sites differ)."""
    terms = normal_order([("-", 1), ("+", 2)])
    assert terms == [NormalOrderedTerm(sign=-1, creations=(2,), annihilations=(1,))]


def test_normal_order_single_inversion_same_site_contracts():
    """a_1 a^dag_1 = 1 - a^dag_1 a_1."""
    terms = normal_order([("-", 1), ("+", 1)])
    # Combine: identity term (empty creations, empty annihilations) with +1,
    # plus a^dag_1 a_1 with -1.
    combined = combine_terms(terms)
    assert combined == {((), ()): 1, ((1,), (1,)): -1}


def test_normal_order_number_operator():
    """n_p = a^dag_p a_p is already normal-ordered."""
    terms = normal_order(letter_primitives("n", 7))
    assert terms == [NormalOrderedTerm(sign=1, creations=(7,), annihilations=(7,))]


def test_normal_order_n_p_n_q_distinct_sites():
    """n_p n_q (p != q) normal-orders to a^dag_p a^dag_q a_q a_p with sign +1.

    Derivation:
      n_p n_q = a^dag_p a_p a^dag_q a_q
              = a^dag_p (delta_{p,q} - a^dag_q a_p) a_q
              = (p != q) -a^dag_p a^dag_q a_p a_q
              = +a^dag_p a^dag_q a_q a_p     (a_p a_q = -a_q a_p)

    With canonical sorting (creations ascending, annihilations ascending),
    the result is a^dag_min a^dag_max a_min a_max with a sign that depends
    on the permutation parity. For p < q, sorting gives the same order with
    parity +1 on annihilations (already a_q a_p becomes a_p a_q with one swap,
    sign flip), giving final coefficient +1 * -1 = -1 on a^dag_p a^dag_q a_p a_q.

    Let me re-derive with the convention in this module (annihilations
    sorted ascending):
      Final operator: a^dag_p a^dag_q a_q a_p (p < q).
      Annihilation list [q, p] needs one swap to become [p, q]; sign -1.
      Creation list [p, q] is already sorted; sign +1.
      Net: sign = +1 * (+1) * (-1) = -1 on the term
        s * a^dag_p a^dag_q a_p a_q.
    """
    p, q = 1, 2
    prims = letter_primitives("n", p) + letter_primitives("n", q)
    terms = normal_order(prims)
    combined = combine_terms(terms)
    # Only the canonical-form term should remain after combination.
    assert combined == {((p, q), (p, q)): -1}


def test_normal_order_number_squared_collapses_to_number():
    """n_p^2 = n_p for a fermion (Pauli exclusion).

    Concretely: n_p n_p = a^dag_p a_p a^dag_p a_p
                       = a^dag_p (1 - a^dag_p a_p) a_p
                       = a^dag_p a_p - a^dag_p a^dag_p a_p a_p
                       = a^dag_p a_p              (Pauli: a^dag_p a^dag_p = 0).
    """
    prims = letter_primitives("n", 4) + letter_primitives("n", 4)
    terms = normal_order(prims)
    combined = combine_terms(terms)
    assert combined == {((4,), (4,)): 1}


def test_normal_order_a_dag_a_n():
    """a^dag_i a_j n_k for distinct i, j, k.

    Derivation:
      a^dag_i a_j a^dag_k a_k
        = a^dag_i (delta_{j,k} - a^dag_k a_j) a_k
        = delta_{j,k} a^dag_i a_k - a^dag_i a^dag_k a_j a_k.
      For distinct j, k: -a^dag_i a^dag_k a_j a_k.
      Canonical sort: creations [i, k] -> (i, k) parity +1 (assuming i < k).
      Annihilations [j, k] -> sorted (min(j,k), max(j,k)). Sign depends on j vs k.

    We just check the operator structure: should be one term with creations
    (min(i,k), max(i,k)) and annihilations (min(j,k), max(j,k)) and overall
    coefficient determined by the permutation.
    """
    i, j, k = 1, 2, 3
    prims = letter_primitives("a_dag", i) + letter_primitives("a", j) + letter_primitives("n", k)
    combined = combine_terms(normal_order(prims))
    # Expect a single term: -a^dag_1 a^dag_3 a_2 a_3.
    # Canonical creations (1, 3) +1 parity. Canonical annihilations [2, 3]
    # already sorted, parity +1. Overall sign: -1.
    assert combined == {((i, k), (j, k)): -1}


def test_combine_terms_drops_zero_total():
    t1 = NormalOrderedTerm(sign=+1, creations=(1,), annihilations=(2,))
    t2 = NormalOrderedTerm(sign=-1, creations=(1,), annihilations=(2,))
    assert combine_terms([t1, t2]) == {}


def test_combine_terms_aggregates_signs():
    t1 = NormalOrderedTerm(sign=+1, creations=(1,), annihilations=(2,))
    t2 = NormalOrderedTerm(sign=+1, creations=(1,), annihilations=(2,))
    t3 = NormalOrderedTerm(sign=-1, creations=(3,), annihilations=(4,))
    assert combine_terms([t1, t2, t3]) == {((1,), (2,)): 2, ((3,), (4,)): -1}
