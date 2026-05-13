"""Evaluate catalog-word connected cumulants from user-supplied RDMs.

Uses the in-house normal-ordering routine in :mod:`._normal_order` plus the
Möbius formula from :mod:`._partition` to compute

.. math::
    \\kappa_W(\\rho) = \\sum_{\\pi \\in \\Pi_m} (-1)^{|\\pi| - 1} (|\\pi| - 1)!
    \\prod_{B \\in \\pi} \\mu_B(W; \\rho)

for every catalog word $W$ of length $m$. Each subword moment $\\mu_B$ is a
charge-conserving expectation value, evaluated by normal-ordering the
subword operator and looking up the resulting fermionic-RDM element under
this module's convention.

**RDM convention** (matches OpenFermion's canonical fermion RDM):

.. math::
    D^{(k)}[p_1, \\ldots, p_k, q_1, \\ldots, q_k]
    = \\langle a^\\dagger_{p_1} a^\\dagger_{p_2} \\cdots a^\\dagger_{p_k}
              a_{q_k} a_{q_{k-1}} \\cdots a_{q_1} \\rangle .

Site indices are 0-based on the RDM tensor; the catalog's 1-based site
indices are converted internally.
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from ._normal_order import (
    NormalOrderedTerm,
    combine_terms,
    normal_order,
    word_primitives,
)
from ._partition import set_partitions
from .catalog import FermionicWord

if TYPE_CHECKING:
    import numpy as np


def evaluate_normal_ordered_term(
    creations: tuple[int, ...],
    annihilations: tuple[int, ...],
    rdm1: "np.ndarray",
    rdm2: "np.ndarray | None" = None,
    rdm3: "np.ndarray | None" = None,
    rdm4: "np.ndarray | None" = None,
) -> complex:
    """Look up a normal-ordered fermion-operator expectation in user RDMs.

    Site indices are 1-based on input; converted to 0-based for tensor lookup.

    Charge non-conserving operators (``len(creations) != len(annihilations)``)
    have zero expectation in any $U(1)$-invariant state; the library assumes
    $U(1)$-invariance of $\\rho$ throughout, so such terms return 0 here.
    """
    k_c = len(creations)
    k_a = len(annihilations)
    if k_c != k_a:
        # Non-particle-number-conserving; vanishes under U(1) invariance.
        return 0.0 + 0j
    k = k_c
    if k == 0:
        # Identity: <I> = 1.
        return 1.0 + 0j

    # 0-based indices. Annihilations are reversed per the convention.
    c_idx = tuple(p - 1 for p in creations)
    a_idx_reversed = tuple(q - 1 for q in reversed(annihilations))

    if k == 1:
        return complex(rdm1[c_idx[0], a_idx_reversed[0]])
    if k == 2:
        if rdm2 is None:
            raise ValueError("2-RDM is required to evaluate a 2-body term")
        idx = c_idx + a_idx_reversed
        return complex(rdm2[idx])
    if k == 3:
        if rdm3 is None:
            raise ValueError("3-RDM is required to evaluate a 3-body term")
        idx = c_idx + a_idx_reversed
        return complex(rdm3[idx])
    if k == 4:
        if rdm4 is None:
            raise ValueError("4-RDM is required to evaluate a 4-body term")
        idx = c_idx + a_idx_reversed
        return complex(rdm4[idx])
    raise NotImplementedError(
        f"k = {k} body operators are not yet supported; catalog word too long"
    )


def evaluate_subword_moment(
    sub_letters: tuple[str, ...],
    sub_sites: tuple[int, ...],
    rdm1: "np.ndarray",
    rdm2: "np.ndarray | None" = None,
    rdm3: "np.ndarray | None" = None,
    rdm4: "np.ndarray | None" = None,
) -> complex:
    """Compute $\\langle A_B(W) \\rangle$ for a subword by normal-ordering."""
    prims = word_primitives(sub_letters, sub_sites)
    terms = normal_order(prims)
    combined = combine_terms(terms)
    if not combined:
        return 0.0 + 0j
    total = 0.0 + 0j
    for (creations, annihilations), coeff in combined.items():
        val = evaluate_normal_ordered_term(
            creations, annihilations, rdm1, rdm2, rdm3, rdm4,
        )
        total += coeff * val
    return total


def evaluate_word_cumulant(
    word: FermionicWord,
    sites: tuple[int, ...],
    rdm1: "np.ndarray",
    rdm2: "np.ndarray | None" = None,
    rdm3: "np.ndarray | None" = None,
    rdm4: "np.ndarray | None" = None,
) -> complex:
    """Connected cumulant $\\kappa_W(\\rho)$ from RDMs via the Möbius formula."""
    if len(sites) != word.length:
        raise ValueError(
            f"word {word.name!r} has length {word.length} but {len(sites)} sites supplied"
        )
    m = word.length
    sub_moments: dict[tuple[int, ...], complex] = {}
    for pi in set_partitions(list(range(1, m + 1))):
        for B in pi:
            key = tuple(sorted(B))
            if key not in sub_moments:
                sub_letters = tuple(word.letters[i - 1] for i in key)
                sub_sites = tuple(sites[i - 1] for i in key)
                sub_moments[key] = evaluate_subword_moment(
                    sub_letters, sub_sites, rdm1, rdm2, rdm3, rdm4,
                )

    kappa: complex = 0.0 + 0j
    for pi in set_partitions(list(range(1, m + 1))):
        prod: complex = 1.0 + 0j
        for B in pi:
            prod *= sub_moments[tuple(sorted(B))]
        k_pi = len(pi)
        kappa += (-1) ** (k_pi - 1) * factorial(k_pi - 1) * prod
    return kappa
