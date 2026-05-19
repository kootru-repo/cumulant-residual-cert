"""OpenFermion adapter.

The chemistry workflow this adapter is built for: a user has chosen
`OpenFermion <https://quantumai.google/openfermion>`_ as the bridge between
their chemistry problem and a qubit-shadow measurement protocol.

Currently shipped:

- Catalog-to-``FermionOperator`` conversion via :func:`word_to_fermion_operator`
  and :func:`catalog_to_fermion_operators`, so catalog words can be fed to
  OpenFermion's measurement-grouping or shadow-protocol utilities.

- :func:`delta_ucb_from_matchgate_shadows`: thin wrapper that accepts
  user-supplied per-Majorana-product ``(mean, radius)`` estimates from a
  matchgate / fermionic-Gaussian shadow protocol and routes them through
  :func:`~cumulant_residual_cert.delta_ucb_from_majorana_moments`. The
  matchgate route avoids the random-Pauli $3^{|P|}$ Jordan-Wigner range
  penalty. A built-in matchgate-snapshot estimator (Pfaffian +
  orthogonal-matrix algebra) is planned for a later release; the current
  wrapper expects the caller to perform the snapshot estimation.

Install with::

    uv add "cumulant-residual-cert[openfermion]"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..catalog import Catalog, FermionicWord

if TYPE_CHECKING:
    from openfermion import FermionOperator

_OPENFERMION_MISSING_MSG = (
    "OpenFermion is required for cumulant_residual_cert.adapters.openfermion. "
    "Install with: uv add 'cumulant-residual-cert[openfermion]'"
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
) -> FermionOperator:
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
            raise ValueError(f"site {s} must be a positive 1-based integer index")
    if len(set(sites)) != len(sites):
        raise ValueError(f"word {word.name!r} has duplicate sites: {sites}")

    op = FermionOperator("")  # identity
    for L, s in zip(word.letters, sites, strict=False):
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
) -> dict[str, FermionOperator]:
    """Convert every word in ``catalog`` into an ``OpenFermion`` operator."""
    if len(sites_per_word) != len(catalog):
        raise ValueError(
            f"sites_per_word has {len(sites_per_word)} entries; catalog has {len(catalog)}"
        )
    return {
        w.name: word_to_fermion_operator(w, sites)
        for w, sites in zip(catalog, sites_per_word, strict=False)
    }


def delta_ucb_from_matchgate_shadows(
    majorana_moments: dict[tuple[int, ...], tuple[complex, float]],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    confidence: float = 0.95,
    n_protocol_terms: int,
    require_all_terms: bool = True,
) -> Any:
    """UCB diagnostic on matchgate / fermionic-Gaussian shadow output.

    Thin wrapper that routes user-supplied per-Majorana-product
    ``(mean, radius)`` estimates through
    :func:`~cumulant_residual_cert.delta_ucb_from_majorana_moments`.
    Matchgate-shadow protocols (e.g.
    :doi:`Wan-Hadfield-Cleve-Babbush 2022 <10.1103/PRXQuantum.4.030337>`,
    :doi:`Zhao-Rubin-Miyake-Babbush 2021 <10.1103/PhysRevLett.127.110504>`)
    produce a single-shot estimator
    $\\hat{\\langle \\gamma_S \\rangle}_t$ for every degree-$|S|$ Majorana
    product, with a range factor that scales polynomially in the qubit
    count instead of the random-Pauli $3^{|P|}$ Jordan-Wigner penalty.

    Caller responsibility: take the matchgate-shadow record, compute the
    empirical mean and Hoeffding-style radius for every Majorana product
    that appears in the catalog's letter-to-Majorana decomposition, and
    pass them in via ``majorana_moments``. The Bonferroni correction on
    the radii is also the caller's responsibility.

    A built-in snapshot estimator (with the Pfaffian + matchgate-orthogonal
    matrix algebra) is planned for a future release; until then, this
    wrapper exists to make the moment-to-UCB pipeline immediately usable
    by anyone who already has matchgate-shadow output.

    Parameters
    ----------
    majorana_moments : dict[tuple[int, ...], (complex, float)]
        See :func:`~cumulant_residual_cert.delta_ucb_from_majorana_moments`
        for the convention.
    catalog : Catalog
    sites_per_word : sequence of sequences of int
    confidence : float, default 0.95
    n_protocol_terms : int
        Number of distinct Majorana products in the Bonferroni union.
    require_all_terms : bool, default True
        If True (default), raise on any missing Majorana product entry
        that appears in a catalog subword decomposition. This is the safe
        default for a certification API. If False, missing entries are
        treated as $(0, 0)$; this is exact for odd-degree products on
        $U(1)$-invariant states but unsafe for missing even-degree
        entries. Opt in to ``False`` only after verifying that all
        missing terms are odd-degree.

    Returns
    -------
    UCBResult
    """
    # No actual OpenFermion call is made here yet; the wrapper is pure
    # routing. The optional-dependency import is kept lazy so docs builds
    # work without OpenFermion installed.
    from ..diagnostic import delta_ucb_from_majorana_moments

    return delta_ucb_from_majorana_moments(
        majorana_moments=majorana_moments,
        catalog=catalog,
        sites_per_word=sites_per_word,
        confidence=confidence,
        n_protocol_terms=n_protocol_terms,
        require_all_terms=require_all_terms,
    )
