"""PySCF adapter.

The chemistry workflow this adapter is built for: a user has solved an
electronic-structure problem in `PySCF <https://pyscf.org>`_ and wants to
attach a deterministic bias bar to a cumulant-truncated estimate of some
fermionic-word observable.

One route ships in v0.2:

- :func:`from_mean_field`: a closed-form $\\Delta = 0$ for the Bernoulli class
  (HF / DFT determinant in its canonical orbital basis, single-Slater pure
  state with dictionary-aligned occupations, or a grand-canonical Gibbs state
  of a number-conserving free Hamiltonian diagonal in the same orbital basis).
  This state class is covered by the underlying worked-example theorem.

A general :func:`from_rdms` route that evaluates $\\Delta$ from supplied
RDMs is on the v0.3 roadmap. The signature is reserved here for forward
compatibility; calling it raises ``NotImplementedError`` in v0.2.

Install with::

    pip install "cumulant-residual-cert[pyscf]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..catalog import Catalog
from ..constants import Level
from ._common import AdapterEstimate, package_estimate

if TYPE_CHECKING:
    import numpy as np

try:
    import pyscf  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PySCF is required for cumulant_residual_cert.adapters.pyscf. "
        "Install with: pip install 'cumulant-residual-cert[pyscf]'"
    ) from e


def from_mean_field(
    mf: Any,
    catalog: Catalog,
    *,
    basis: str = "canonical",
    level: Level = "block_refined",
) -> AdapterEstimate:
    """Certify the truncation bias on a PySCF mean-field state.

    For a PySCF ``MeanField`` (HF or KS-DFT) converged in the canonical molecular
    orbital basis, the resulting Slater determinant is occupation-basis diagonal
    in that basis. The worked-example theorem then gives
    $\\Delta^{\\mathrm{cat}}_{4, U(1)}(\\rho) = 0$ identically, so every bound
    in the chemistry catalog is exactly zero.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        Converged PySCF mean-field object (RHF, UHF, ROHF, RKS, UKS, ...).
    catalog : Catalog
        Charge-neutral catalog. The Bernoulli-class result holds for any
        catalog whose words are in the dictionary letter alphabet.
    basis : {"canonical", "other"}, default "canonical"
        Which orbital basis the mean-field solution is expressed in. Only
        ``"canonical"`` triggers the closed-form $\\Delta = 0$; any other
        value raises ``NotImplementedError`` (route the caller to
        :func:`from_rdms`).
    level : {"universal", "charge_filtered", "block_refined"}
        Which constant family to apply.

    Returns
    -------
    AdapterEstimate
        With ``delta = 0`` and ``delta_is_exact = True`` for the canonical case.
    """
    if not getattr(mf, "converged", False):
        raise ValueError(
            "PySCF mean-field object is not converged; converge it first or "
            "supply RDMs to from_rdms()"
        )

    if basis != "canonical":
        raise NotImplementedError(
            "The closed-form Bernoulli-class result requires the dictionary "
            "basis to coincide with the canonical molecular orbital basis. "
            "For other bases, supply RDMs via from_rdms()."
        )

    notes = (
        f"PySCF {type(mf).__name__} mean-field state in canonical molecular orbital basis.",
        "Falls in the Bernoulli (occupation-basis diagonal product) class.",
        "Worked-example theorem: Delta = 0 identically on this class.",
    )
    return package_estimate(
        catalog,
        delta=0.0,
        delta_is_exact=True,
        framework="pyscf",
        level=level,
        notes=notes,
    )


def from_rdms(
    rdm1: "np.ndarray",
    rdm2: "np.ndarray",
    catalog: Catalog,
    *,
    level: Level = "block_refined",
    rdm3: "np.ndarray | None" = None,
    rdm4: "np.ndarray | None" = None,
) -> AdapterEstimate:
    """Certify the truncation bias from supplied RDMs.

    Computes $\\Delta_{r, U(1)}^{\\mathrm{cat}}(\\rho)$ by evaluating each
    catalog word's connected cumulant directly from the supplied RDMs.

    .. note::
        This route is preliminary in v0.2. The full evaluation requires RDMs
        up to order $r$ to compute the length-$r$ catalog cumulant exactly.
        For words of length $\\le 2$, ``rdm1`` and ``rdm2`` suffice. For
        length-3 words, ``rdm3`` is required. For length-4 words, ``rdm4``
        is required. If a required RDM is missing the adapter raises
        ``NotImplementedError`` with the catalog-word name; combine the
        adapter with shadow estimates from
        :func:`~cumulant_residual_cert.delta_ucb` for those words.

    Parameters
    ----------
    rdm1 : ndarray of shape (n_orb, n_orb)
        Spin-summed 1-RDM in the working orbital basis.
    rdm2 : ndarray of shape (n_orb, n_orb, n_orb, n_orb)
        Spin-summed 2-RDM in the working orbital basis.
    catalog : Catalog
    level : {"universal", "charge_filtered", "block_refined"}
    rdm3, rdm4 : ndarray, optional
        Higher-order RDMs required for length-3 and length-4 catalog words.

    Returns
    -------
    AdapterEstimate
        With ``delta_is_exact = True`` only if every required RDM is supplied
        at full precision; otherwise the result is an upper bound (currently
        raised as ``NotImplementedError`` pending v0.3 closure).
    """
    raise NotImplementedError(
        "from_rdms() ships in v0.3 once the catalog-word cumulant evaluator "
        "is wired up. For v0.2 use from_mean_field() for Bernoulli-class "
        "states, or supply Delta directly via certify()."
    )
