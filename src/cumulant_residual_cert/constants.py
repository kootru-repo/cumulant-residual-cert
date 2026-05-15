"""Public access to the catalog constants $B_r$, $B^{\\mathrm{charge}}_r$, $\\widehat B^{\\mathrm{charge}}_r$.

These are integer-valued partition-lattice sums; the actual enumeration lives
in :mod:`cumulant_residual_cert._partition`. The functions here wrap the
enumeration with a high-level API that takes :class:`~.catalog.Catalog` and
:class:`~.catalog.FermionicWord` instances directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from . import _partition
from .catalog import Catalog, FermionicWord

Level = Literal["universal", "charge_filtered", "block_refined"]


@dataclass(frozen=True)
class ConstantsTable:
    """All three constants for every word in a catalog, at a given $r$."""

    r: int
    universal: int
    per_word: dict[str, WordConstants]


@dataclass(frozen=True)
class WordConstants:
    """The triple $(B_r, B^{\\mathrm{charge}}_r(W), \\widehat B^{\\mathrm{charge}}_r(W))$ for a word."""

    word_name: str
    universal: int
    charge_filtered: int
    block_refined: int


def universal(r: int) -> int:
    """Universal partition-lattice constant $B_r$."""
    return _partition.B_r(r)


def charge_filtered(r: int, word: FermionicWord) -> int:
    """Charge-filtered constant $B^{\\mathrm{charge}}_r(W)$ for ``word``."""
    return _partition.B_charge_r(r, word.charges)


def block_refined(r: int, word: FermionicWord) -> int:
    """Block-refined constant $\\widehat B^{\\mathrm{charge}}_r(W)$ for ``word``."""
    return _partition.Bhat_charge_r(r, word.charges)


def compute(catalog: Catalog) -> ConstantsTable:
    """Compute the full constants table for every word in ``catalog``."""
    r = catalog.r
    B_r_universal = universal(r)
    per_word: dict[str, WordConstants] = {}
    for w in catalog:
        per_word[w.name] = WordConstants(
            word_name=w.name,
            universal=B_r_universal,
            charge_filtered=charge_filtered(r, w),
            block_refined=block_refined(r, w),
        )
    return ConstantsTable(r=r, universal=B_r_universal, per_word=per_word)


def get(level: Level, r: int, word: FermionicWord) -> int:
    """Look up a single constant by level name."""
    if level == "universal":
        return universal(r)
    if level == "charge_filtered":
        return charge_filtered(r, word)
    if level == "block_refined":
        return block_refined(r, word)
    raise ValueError(f"unknown level {level!r}; expected one of {{universal, charge_filtered, block_refined}}")


def chemistry_r4_table() -> ConstantsTable:
    """Pre-built constants table for the $r = 4$ chemistry catalog."""
    return compute(Catalog.chemistry_r4())


def load_golden() -> dict:
    """Load the shipped golden JSON constants table.

    Returns the on-disk table verbatim. CI cross-checks it against a freshly
    computed table and against the audit repository's pinned JSON.
    """
    data_files = resources.files("cumulant_residual_cert.data")
    return json.loads(data_files.joinpath("chemistry_catalog_r4.json").read_text(encoding="utf-8"))
