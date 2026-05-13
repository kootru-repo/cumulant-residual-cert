"""Deterministic bias certificates for charge-neutral fermionic-word observables.

Public surface:

- :class:`Catalog` and :class:`FermionicWord`: build or import the chemistry catalog.
- :func:`certify`: turn $(\\text{catalog}, \\Delta)$ into a per-word bias bar.
- :func:`delta_ucb`: estimate $\\Delta$ from shadow data with a confidence guarantee.
- :mod:`constants`: direct access to $B_r$, $B^{\\mathrm{charge}}_r$, $\\widehat B^{\\mathrm{charge}}_r$.
"""

from __future__ import annotations

from . import constants
from .catalog import Catalog, FermionicWord, word
from .certify import CertifiedBound, certify
from .diagnostic import (
    UCBResult,
    UCBSplitResult,
    delta_ucb,
    delta_ucb_from_majorana_moments,
    delta_ucb_from_subword_moments,
    delta_ucb_split,
)

__all__ = [
    "Catalog",
    "CertifiedBound",
    "FermionicWord",
    "UCBResult",
    "UCBSplitResult",
    "certify",
    "constants",
    "delta_ucb",
    "delta_ucb_from_majorana_moments",
    "delta_ucb_from_subword_moments",
    "delta_ucb_split",
    "word",
]

__version__ = "0.4.0"
