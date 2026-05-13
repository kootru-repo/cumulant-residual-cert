"""Normal-ordering of fermionic letter products in the dictionary alphabet.

The dictionary alphabet is :math:`\\mathcal L = \\{I, n, a, a^\\dagger\\}` over a
set of fermionic modes indexed by integers. This module:

- Expands a letter product (catalog word with site assignments) into a list
  of "primitive" creation and annihilation operators.
- Normal-orders the product, applying anti-commutation rules to push every
  creation operator to the left of every annihilation operator and to remove
  duplicate creations or annihilations (which vanish under fermion algebra).
- Returns a list of normal-ordered terms, each carrying a sign and a tuple
  of distinct creation indices followed by a tuple of distinct annihilation
  indices.

A "normal-ordered term" has the canonical form

.. math::
    s \\cdot a^\\dagger_{p_1} a^\\dagger_{p_2} \\cdots a^\\dagger_{p_k}
    a_{q_1} a_{q_2} \\cdots a_{q_k}

with creations sorted ascending in :math:`p_i` and annihilations sorted
ascending in :math:`q_i`. The sign :math:`s \\in \\{+1, -1\\}` absorbs the
permutation parity from sorting.

The expectation of a normal-ordered term in a $U(1)$-invariant state is a
specific element of the spin-orbital :math:`k`-RDM under a clearly-defined
convention; see :mod:`cumulant_residual_cert._rdm_eval`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# A primitive operator is ("+", site) for a creation or ("-", site) for an
# annihilation. Sites are 1-based, matching the rest of the library.
Primitive = tuple[str, int]


@dataclass(frozen=True)
class NormalOrderedTerm:
    """A canonical normal-ordered fermionic term.

    Attributes
    ----------
    sign : int
        ``+1`` or ``-1``, absorbing all anti-commutation sign flips.
    creations : tuple[int, ...]
        Strictly increasing tuple of distinct creation-operator site indices.
    annihilations : tuple[int, ...]
        Strictly increasing tuple of distinct annihilation-operator site indices.
    """

    sign: int
    creations: tuple[int, ...]
    annihilations: tuple[int, ...]


# ---------- Letter expansion ----------------------------------------------


def letter_primitives(letter: str, site: int) -> list[Primitive]:
    """Expand a dictionary letter at ``site`` into a list of primitive operators."""
    if letter == "I":
        return []
    if letter == "n":
        return [("+", site), ("-", site)]
    if letter == "a":
        return [("-", site)]
    if letter == "a_dag":
        return [("+", site)]
    raise ValueError(f"unknown dictionary letter {letter!r}")


def word_primitives(letters: Iterable[str], sites: Iterable[int]) -> list[Primitive]:
    """Expand a letter sequence at site assignments into a primitive-operator list."""
    out: list[Primitive] = []
    letters = list(letters)
    sites = list(sites)
    if len(letters) != len(sites):
        raise ValueError("letters and sites must have equal length")
    for L, s in zip(letters, sites):
        out.extend(letter_primitives(L, s))
    return out


# ---------- Normal-ordering -----------------------------------------------


def _sign_sort_creations(ops: list[int]) -> tuple[int, tuple[int, ...]]:
    """Sort creation indices ascending, returning the permutation parity and result.

    Returns ``(0, ())`` if any duplicate is found (the term vanishes).
    """
    if len(set(ops)) < len(ops):
        return 0, ()
    sign = 1
    arr = list(ops)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sign = -sign
    return sign, tuple(arr)


def _sign_sort_annihilations(ops: list[int]) -> tuple[int, tuple[int, ...]]:
    """Sort annihilation indices ascending, returning the permutation parity and result.

    Annihilation operators anti-commute among themselves, so the parity rule
    is identical to that of creations.
    """
    return _sign_sort_creations(ops)


def normal_order(primitives: list[Primitive]) -> list[NormalOrderedTerm]:
    """Normal-order a primitive-operator product.

    Returns a list of :class:`NormalOrderedTerm` instances such that the
    original product equals the signed sum of those terms (acting on any
    state). Terms that vanish by the Pauli exclusion rule
    (duplicate creations or annihilations) are dropped.
    """
    return _normal_order_with_sign(primitives, sign=1)


def _normal_order_with_sign(primitives: list[Primitive], sign: int) -> list[NormalOrderedTerm]:
    # Find the leftmost inversion: an annihilation immediately followed by a
    # creation, i.e. ("-", p) at index i and ("+", q) at index i+1.
    for i in range(len(primitives) - 1):
        kind_left, site_left = primitives[i]
        kind_right, site_right = primitives[i + 1]
        if kind_left == "-" and kind_right == "+":
            # Anti-commutation: a_p a^dag_q = delta_{p,q} - a^dag_q a_p.
            head = primitives[:i]
            tail = primitives[i + 2:]
            results: list[NormalOrderedTerm] = []
            # Contracted term (only if p == q).
            if site_left == site_right:
                contracted = head + tail
                results.extend(_normal_order_with_sign(contracted, sign))
            # Swapped term with sign flip.
            swapped = head + [("+", site_right), ("-", site_left)] + tail
            results.extend(_normal_order_with_sign(swapped, -sign))
            return results
    # No inversions: all creations are to the left of all annihilations.
    # Split, sign-sort each half, drop on duplicate index.
    creations = [s for k, s in primitives if k == "+"]
    annihilations = [s for k, s in primitives if k == "-"]
    sign_c, sorted_c = _sign_sort_creations(creations)
    if sign_c == 0:
        return []
    sign_a, sorted_a = _sign_sort_annihilations(annihilations)
    if sign_a == 0:
        return []
    return [NormalOrderedTerm(sign=sign * sign_c * sign_a,
                              creations=sorted_c,
                              annihilations=sorted_a)]


def combine_terms(terms: Iterable[NormalOrderedTerm]) -> dict[tuple[tuple[int, ...], tuple[int, ...]], int]:
    """Group normal-ordered terms by their (creations, annihilations) key.

    Returns a mapping ``(creations, annihilations) -> integer coefficient``,
    with zero entries removed. This lets the RDM-lookup pass evaluate each
    distinct operator only once.
    """
    combined: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for t in terms:
        key = (t.creations, t.annihilations)
        combined[key] = combined.get(key, 0) + t.sign
    return {key: coeff for key, coeff in combined.items() if coeff != 0}
