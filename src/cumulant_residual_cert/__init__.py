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
from .diagnostic import UCBResult, delta_ucb

__all__ = [
    "Catalog",
    "CertifiedBound",
    "FermionicWord",
    "UCBResult",
    "certify",
    "constants",
    "delta_ucb",
    "word",
]

__version__ = "0.1.0"
