"""Many-body fermionic letter / word operators in Jordan-Wigner form.

Internal helper for :mod:`cumulant_residual_cert.diagnostic`. The user-facing
adapters in :mod:`cumulant_residual_cert.adapters` should not depend on this
module directly; they consume framework-native state representations and
estimate the envelope without explicit dense matrices.

Site indices are 1-based. Operators on site $i$ carry the Jordan-Wigner
$Z$-string on sites $1, \\ldots, i - 1$.

This module is dense and $O(2^n)$ in memory. Use it only for small validation
states (typically $n \\le 8$); shadow-based estimation does not require it.
"""

from __future__ import annotations

from functools import cache

import numpy as np

I2 = np.eye(2, dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
N2 = np.array([[0, 0], [0, 1]], dtype=complex)
A2 = np.array([[0, 1], [0, 0]], dtype=complex)
ADAG2 = np.array([[0, 0], [1, 0]], dtype=complex)

LOCAL_MAT: dict[str, np.ndarray] = {
    "I": I2,
    "n": N2,
    "a": A2,
    "a_dag": ADAG2,
}


def _kron(mats: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


@cache
def letter_op(letter: str, site: int, n: int) -> np.ndarray:
    """Single-letter operator on site ``site`` (1-based) in an ``n``-site space."""
    if letter == "I":
        return _kron([I2] * n)
    if letter not in LOCAL_MAT:
        raise KeyError(letter)
    needs_jw = letter in ("a", "a_dag")
    blocks: list[np.ndarray] = []
    for s in range(1, n + 1):
        if s < site:
            blocks.append(Z2 if needs_jw else I2)
        elif s == site:
            blocks.append(LOCAL_MAT[letter])
        else:
            blocks.append(I2)
    return _kron(blocks)


def word_op(letters: tuple[str, ...], sites: tuple[int, ...], n: int) -> np.ndarray:
    """$A_W = L_1(s_1) \\cdot L_2(s_2) \\cdots L_m(s_m)$."""
    if len(letters) != len(sites):
        raise ValueError("letters and sites must have the same length")
    out = letter_op(letters[0], sites[0], n)
    for L, s in zip(letters[1:], sites[1:], strict=False):
        out = out @ letter_op(L, s, n)
    return out


def subword_op(
    letters: tuple[str, ...],
    sites: tuple[int, ...],
    block: tuple[int, ...],
    n: int,
) -> np.ndarray:
    """$A_B(W)$ for a subword index set $B$ (1-based)."""
    block = tuple(sorted(block))
    sub_letters = tuple(letters[i - 1] for i in block)
    sub_sites = tuple(sites[i - 1] for i in block)
    if not sub_letters:
        return _kron([I2] * n)
    return word_op(sub_letters, sub_sites, n)
