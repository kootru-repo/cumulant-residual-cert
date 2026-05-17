"""Letter-to-Majorana decomposition for dictionary fermionic words.

The dictionary alphabet $\\{I, n, a, a^\\dagger\\}$ decomposes into Majorana
operators $\\{\\gamma_j\\}_{j=1}^{2n}$ on $n$ fermionic modes via

.. math::
    \\gamma_{2p-1} = a_p + a^\\dagger_p, \\qquad
    \\gamma_{2p}   = i (a_p - a^\\dagger_p),

which gives

.. math::
    a_p          &= (\\gamma_{2p-1} + i\\, \\gamma_{2p}) / 2, \\\\
    a^\\dagger_p &= (\\gamma_{2p-1} - i\\, \\gamma_{2p}) / 2, \\\\
    n_p          &= \\tfrac12 + \\tfrac{i}{2}\\, \\gamma_{2p-1}\\, \\gamma_{2p}.

A catalog word at site assignments expands into a sum of canonical Majorana
products via the Cartesian product of the per-letter decompositions. Each
ordered product is canonicalized by anti-commutation
$\\gamma_a \\gamma_b = -\\gamma_b \\gamma_a$ (for $a \\ne b$) and the
relation $\\gamma_a^2 = 1$ (for $a = b$).

The output of :func:`word_majorana_decomposition` is a dict mapping a
strictly-sorted tuple of distinct Majorana indices to a complex coefficient.

This module is the bridge from dictionary words to matchgate-shadow
estimators: each Majorana product can be evaluated by a matchgate-shadow
snapshot estimator, and the per-word moment then follows by linearity.
"""

from __future__ import annotations

from collections.abc import Iterable

MajoranaTerm = tuple[tuple[int, ...], complex]
MajoranaDecomposition = dict[tuple[int, ...], complex]


def letter_majorana_decomposition(letter: str, site: int) -> MajoranaDecomposition:
    """Decomposition of a single dictionary letter at ``site`` (1-based)."""
    if site < 1:
        raise ValueError(f"site must be a positive 1-based index; got {site}")
    j_odd = 2 * site - 1
    j_even = 2 * site
    if letter == "I":
        return {(): 1.0 + 0j}
    if letter == "n":
        return {(): 0.5 + 0j, (j_odd, j_even): 0.5j}
    if letter == "a":
        return {(j_odd,): 0.5 + 0j, (j_even,): 0.5j}
    if letter == "a_dag":
        return {(j_odd,): 0.5 + 0j, (j_even,): -0.5j}
    raise ValueError(f"unknown dictionary letter {letter!r}")


def _canonicalize_majorana_product(indices: list[int]) -> tuple[int, tuple[int, ...]]:
    """Canonicalize an ordered Majorana product to (sign, sorted distinct indices).

    Applies $\\gamma_a \\gamma_b = -\\gamma_b \\gamma_a$ ($a \\ne b$) via bubble
    sort with sign tracking, and $\\gamma_a^2 = 1$ via pair removal.
    """
    sign = 1
    arr = list(indices)
    i = 0
    while i < len(arr) - 1:
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            sign = -sign
            if i > 0:
                i -= 1
        elif arr[i] == arr[i + 1]:
            del arr[i : i + 2]
            if i > 0:
                i -= 1
        else:
            i += 1
    return sign, tuple(arr)


def multiply_majorana_terms(a: MajoranaTerm, b: MajoranaTerm) -> MajoranaTerm:
    """Product of two canonical Majorana terms, returned as a canonical term."""
    merged = list(a[0]) + list(b[0])
    sign, canon = _canonicalize_majorana_product(merged)
    return canon, sign * a[1] * b[1]


def word_majorana_decomposition(
    letters: Iterable[str],
    sites: Iterable[int],
) -> MajoranaDecomposition:
    """Decompose a fermionic word into a sum of canonical Majorana products."""
    letters_list = list(letters)
    sites_list = list(sites)
    if len(letters_list) != len(sites_list):
        raise ValueError("letters and sites must have equal length")

    result: MajoranaDecomposition = {(): 1.0 + 0j}
    for L, s in zip(letters_list, sites_list, strict=False):
        letter_terms = letter_majorana_decomposition(L, s)
        new_result: MajoranaDecomposition = {}
        for prev_idx, prev_coeff in result.items():
            for letter_idx, letter_coeff in letter_terms.items():
                idx, coeff = multiply_majorana_terms(
                    (prev_idx, prev_coeff),
                    (letter_idx, letter_coeff),
                )
                new_result[idx] = new_result.get(idx, 0.0 + 0j) + coeff
        # Drop numerical zeros.
        result = {k: v for k, v in new_result.items() if abs(v) > 1e-15}
    return result
