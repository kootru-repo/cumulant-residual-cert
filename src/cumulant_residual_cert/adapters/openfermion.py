"""OpenFermion adapter.

The chemistry workflow this adapter is built for: a user has chosen
`OpenFermion <https://quantumai.google/openfermion>`_ as the bridge between
their chemistry problem and a qubit-shadow measurement protocol.

Currently shipped:

- Catalog-to-``FermionOperator`` conversion via :func:`word_to_fermion_operator`
  and :func:`catalog_to_fermion_operators`, so catalog words can be fed to
  OpenFermion's measurement-grouping or shadow-protocol utilities.

A wrapper that consumes matchgate / fermionic-Gaussian shadow records and
runs the UCB diagnostic without the $3^{|P|}$ random-Pauli range penalty
is reserved at :func:`delta_ucb_from_matchgate_shadows` and will ship in
a later release; current invocation raises ``NotImplementedError``.

Install with::

    pip install "cumulant-residual-cert[openfermion]"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..catalog import Catalog, FermionicWord

if TYPE_CHECKING:
    from openfermion import FermionOperator

_OPENFERMION_MISSING_MSG = (
    "OpenFermion is required for cumulant_residual_cert.adapters.openfermion. "
    "Install with: pip install 'cumulant-residual-cert[openfermion]'"
)


def _require_openfermion():
    """Import and return OpenFermion's ``FermionOperator`` or raise a helpful error.

    Kept lazy so that ``import cumulant_residual_cert.adapters.openfermion``
    does not fail when only docs are being built or when an introspection tool
    walks the module without intending to call any of its functions.
    """
    try:
        from openfermion import FermionOperator as _FermionOperator
    except ImportError as e:  # pragma: no cover
        raise ImportError(_OPENFERMION_MISSING_MSG) from e
    return _FermionOperator


def word_to_fermion_operator(
    word: FermionicWord,
    sites: Sequence[int],
) -> "FermionOperator":
    """Convert a :class:`FermionicWord` with site assignments to an OpenFermion ``FermionOperator``.

    OpenFermion uses 0-based site indices; we accept 1-based as elsewhere in
    this library and shift internally.

    Parameters
    ----------
    word : FermionicWord
        A word in the dictionary letter alphabet ``{I, n, a, a_dag}``.
    sites : sequence of int
        1-based site assignments, same length as ``word.letters``.

    Returns
    -------
    FermionOperator
        Product of the letter operators in the order they appear in ``word``.
    """
    FermionOperator = _require_openfermion()
    if len(sites) != word.length:
        raise ValueError(
            f"word {word.name!r} has length {word.length} but {len(sites)} sites given"
        )
    for s in sites:
        if not isinstance(s, int) or s < 1:
            raise ValueError(
                f"site {s} must be a positive 1-based integer index"
            )
    if len(set(sites)) != len(sites):
        raise ValueError(f"word {word.name!r} has duplicate sites: {sites}")

    op = FermionOperator("")  # identity
    for L, s in zip(word.letters, sites):
        site0 = s - 1
        if L == "I":
            continue
        if L == "n":
            op = op * FermionOperator(f"{site0}^ {site0}")
        elif L == "a":
            op = op * FermionOperator(f"{site0}")
        elif L == "a_dag":
            op = op * FermionOperator(f"{site0}^")
        else:
            raise ValueError(f"unknown letter {L!r}")
    return op


def catalog_to_fermion_operators(
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
) -> dict[str, "FermionOperator"]:
    """Convert every word in ``catalog`` into an ``OpenFermion`` operator."""
    if len(sites_per_word) != len(catalog):
        raise ValueError(
            f"sites_per_word has {len(sites_per_word)} entries; catalog has {len(catalog)}"
        )
    return {
        w.name: word_to_fermion_operator(w, sites)
        for w, sites in zip(catalog, sites_per_word)
    }


def delta_ucb_from_matchgate_shadows(
    shadows: Any,
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    n_qubits: int,
    confidence: float = 0.95,
) -> Any:
    """Run the UCB diagnostic on matchgate / fermionic-Gaussian shadow data.

    .. note::
        This wrapper will ship in a later release once the matchgate-shadow
        estimator wiring from OpenFermion is firmed up. The mathematical
        pipeline is the same as :func:`~cumulant_residual_cert.delta_ucb` but
        each per-Pauli range factor is replaced by the matchgate range,
        removing the $3^{|P|}$ penalty.
    """
    _require_openfermion()
    raise NotImplementedError(
        "delta_ucb_from_matchgate_shadows() will ship in a later release. In "
        "the meantime, use cumulant_residual_cert.delta_ucb() with random-Pauli "
        "shadows, or compute Delta from a closed-form expression and call "
        "certify() directly."
    )
