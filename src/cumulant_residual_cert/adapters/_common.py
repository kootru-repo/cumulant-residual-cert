"""Common helpers shared across chemistry-framework adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import Catalog
from ..certify import CertifiedBound, certify
from ..constants import Level


@dataclass(frozen=True)
class AdapterEstimate:
    """A framework-adapter result.

    Attributes
    ----------
    delta : float
        The exact value of, or upper bound on, $\\Delta_{r,U(1)}^{\\mathrm{cat}}(\\rho)$
        used to produce the bound.
    delta_is_exact : bool
        True if ``delta`` is the exact envelope (e.g. from a closed-form
        Bernoulli-class state), False if it is an upper bound (e.g. from a
        shadow UCB or a finite-precision RDM evaluation).
    framework : str
        Identifier of the producing adapter (``"pyscf"``, ``"openfermion"``,
        ``"qiskit_nature"``).
    bound : CertifiedBound
        Certified bias bars for every word in the catalog.
    notes : tuple[str, ...]
        Free-text annotations describing the route the adapter took
        (e.g. ``"Hartree-Fock determinant in canonical basis -> Bernoulli class -> Delta = 0 exactly"``).
    """

    delta: float
    delta_is_exact: bool
    framework: str
    bound: CertifiedBound
    notes: tuple[str, ...] = ()


def package_estimate(
    catalog: Catalog,
    delta: float,
    *,
    delta_is_exact: bool,
    framework: str,
    level: Level = "block_refined",
    notes: tuple[str, ...] = (),
) -> AdapterEstimate:
    """Wrap a delta value into a certified estimate."""
    bound = certify(catalog, delta=delta, level=level)
    return AdapterEstimate(
        delta=delta,
        delta_is_exact=delta_is_exact,
        framework=framework,
        bound=bound,
        notes=notes,
    )
