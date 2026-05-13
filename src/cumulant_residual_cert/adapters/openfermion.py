"""OpenFermion adapter.

The chemistry workflow this adapter is built for: a user has chosen
`OpenFermion <https://quantumai.google/openfermion>`_ as the bridge between
their chemistry problem and a qubit-shadow measurement protocol.

v0.2 ships:

- Catalog-to-``FermionOperator`` conversion via :func:`word_to_fermion_operator`
  and :func:`catalog_to_fermion_operators`, so catalog words can be fed to
  OpenFermion's measurement-grouping or shadow-protocol utilities.

A wrapper that consumes matchgate / fermionic-Gaussian shadow records and
runs the UCB diagnostic without the $3^{|P|}$ random-Pauli range penalty
is reserved at :func:`delta_ucb_from_matchgate_shadows` and will ship in
v0.3; v0.2 invocation raises ``NotImplementedError``.

Install with::

    pip install "cumulant-residual-cert[openfermion]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..catalog import Catalog, FermionicWord

if TYPE_CHECKING:
    pass

try:
    from openfermion import FermionOperator  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OpenFermion is required for cumulant_residual_cert.adapters.openfermion. "
        "Install with: pip install 'cumulant-residual-cert[openfermion]'"
    ) from e


def word_to_fermion_operator(
    word: FermionicWord,
    sites: tuple[int, ...],
) -> "FermionOperator":
    """Convert a :class:`FermionicWord` with site assignments to an OpenFermion ``FermionOperator``.

    OpenFermion uses 0-based site indices; we accept 1-based as elsewhere in
    this library and shift internally.

    Parameters
    ----------
    word : FermionicWord
        A word in the dictionary letter alphabet ``{I, n, a, a_dag}``.
    sites : tuple of int
        1-based site assignments, same length as ``word.letters``.

    Returns
    -------
    FermionOperator
        Normal-ordered FermionOperator product of the letter operators.
    """
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
    sites_per_word: list[tuple[int, ...]],
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
    sites_per_word: list[tuple[int, ...]],
    *,
    n_qubits: int,
    confidence: float = 0.95,
) -> Any:
    """Run the UCB diagnostic on matchgate / fermionic-Gaussian shadow data.

    .. note::
        This wrapper ships in v0.3 once the matchgate-shadow estimator wiring
        from OpenFermion is firmed up. The mathematical pipeline is the same
        as :func:`~cumulant_residual_cert.delta_ucb` but each per-Pauli range
        factor is replaced by the matchgate range, removing the $3^{|P|}$
        penalty. Use random-Pauli shadows via
        :func:`~cumulant_residual_cert.delta_ucb` in the meantime.
    """
    raise NotImplementedError(
        "delta_ucb_from_matchgate_shadows() ships in v0.3. For v0.2 use "
        "cumulant_residual_cert.delta_ucb() with random-Pauli shadows, or "
        "compute Delta from a closed-form expression and call certify() directly."
    )
