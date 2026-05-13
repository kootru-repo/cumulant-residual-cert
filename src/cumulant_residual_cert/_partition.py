"""Partition-lattice arithmetic for the sector-resolved residual bounds.

This is the authoritative numerical kernel. Every quantity is computed by
direct enumeration from the definition; no formulas, no shortcuts.

Notation:

    Pi_m              set partitions of [m] = {1, ..., m}
    M_r               max_{1 <= m <= r} sum_{pi in Pi_m} (|pi| - 1)!
    B_r               max_{3 <= m <= r} sum_{pi in Pi_m, exists B in pi: |B| > 2}
                          M_r ** (|pi| - 1)
    Pi^nl_m(W)        neutral-block partitions of [m] with at least one |B| > 2
    B^charge_r(W)     sum_{pi in Pi^nl_m(W)} M_r ** (|pi| - 1)
    Bhat^charge_r(W)  sum_{pi in Pi^nl_m(W)} min_{B*, |B*| > 2}
                          prod_{B != B*} M_{|B|}

This module is intentionally side-effect free and depends only on the standard
library, so the golden constants table can be regenerated in isolation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from math import factorial


def set_partitions(items: Sequence[int]) -> Iterator[list[list[int]]]:
    """Yield every set partition of ``items`` as a list of blocks."""
    items = list(items)
    if not items:
        yield []
        return
    if len(items) == 1:
        yield [[items[0]]]
        return
    first = items[0]
    rest = items[1:]
    for sub in set_partitions(rest):
        for i in range(len(sub)):
            new_part = [list(b) for b in sub]
            new_part[i].append(first)
            yield new_part
        yield [[first]] + [list(b) for b in sub]


def partitions_of_m(m: int) -> list[list[list[int]]]:
    """All set partitions of ``[1..m]``."""
    return list(set_partitions(list(range(1, m + 1))))


@lru_cache(maxsize=None)
def M_r(r: int) -> int:
    """Mobius-bound constant $M_r = \\max_m \\sum_{\\pi \\in \\Pi_m} (|\\pi| - 1)!$."""
    if r < 1:
        raise ValueError("r must be >= 1")
    best = 0
    for m in range(1, r + 1):
        total = sum(factorial(len(pi) - 1) for pi in partitions_of_m(m))
        best = max(best, total)
    return best


def _is_size_gt_2_partition(pi: list[list[int]]) -> bool:
    return any(len(b) > 2 for b in pi)


def _is_blockwise_neutral(pi: list[list[int]], charges: Sequence[int]) -> bool:
    for block in pi:
        if sum(charges[i - 1] for i in block) != 0:
            return False
    return True


@lru_cache(maxsize=None)
def B_r(r: int) -> int:
    """Universal partition-lattice constant $B_r$ for residual bounds of order $r$."""
    if r < 3:
        raise ValueError("B_r is defined for r >= 3")
    M = M_r(r)
    best = 0
    for m in range(3, r + 1):
        total = 0
        for pi in partitions_of_m(m):
            if _is_size_gt_2_partition(pi):
                total += M ** (len(pi) - 1)
        best = max(best, total)
    return best


def neutral_large_partitions(charges: Sequence[int]) -> list[list[list[int]]]:
    """Partitions in $\\Pi^{nl}_{|W|}(W)$: blockwise charge-neutral with a block of size > 2."""
    m = len(charges)
    out: list[list[list[int]]] = []
    for pi in partitions_of_m(m):
        if not _is_blockwise_neutral(pi, charges):
            continue
        if not _is_size_gt_2_partition(pi):
            continue
        out.append(pi)
    return out


def B_charge_r(r: int, charges: Sequence[int]) -> int:
    """Charge-filtered constant $B^{\\mathrm{charge}}_r(W)$ for word with given letter charges."""
    M = M_r(r)
    return sum(M ** (len(pi) - 1) for pi in neutral_large_partitions(charges))


def Bhat_charge_r(r: int, charges: Sequence[int]) -> int:
    """Block-refined constant $\\widehat B^{\\mathrm{charge}}_r(W)$.

    For each contributing partition pi, the contribution is the minimum over
    the choice of distinguished large block $B^*$ of $\\prod_{B \\ne B^*} M_{|B|}$.
    For the chemistry catalog with $|W| \\le 4$ each contributing partition has
    a unique block of size > 2, so the min is vacuous; the implementation is
    general.
    """
    total = 0
    for pi in neutral_large_partitions(charges):
        candidates: list[int] = []
        large_blocks = [b for b in pi if len(b) > 2]
        for B_star in large_blocks:
            prod = 1
            for B in pi:
                if B is B_star:
                    continue
                prod *= M_r(len(B))
            candidates.append(prod)
        total += min(candidates)
    return total
