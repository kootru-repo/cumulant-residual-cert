"""OpenFermion adapter.

The chemistry workflow this adapter is built for: a user has chosen
`OpenFermion <https://quantumai.google/openfermion>`_ as the bridge between
their chemistry problem and a qubit-shadow measurement protocol.

Currently shipped:

- Catalog-to-``FermionOperator`` conversion via :func:`word_to_fermion_operator`
  and :func:`catalog_to_fermion_operators`, so catalog words can be fed to
  OpenFermion's measurement-grouping or shadow-protocol utilities.

- :func:`delta_ucb_from_matchgate_shadows`: dispatch wrapper that accepts
  either a :class:`~cumulant_residual_cert._matchgate_shadow.MatchgateShadowRecord`
  (end-to-end pipeline, routes to
  :func:`~cumulant_residual_cert.delta_ucb_matchgate_shadows`) or a
  pre-computed per-Majorana-product ``(mean, radius)`` dictionary (routes to
  :func:`~cumulant_residual_cert.delta_ucb_from_majorana_moments`). The
  matchgate route avoids the random-Pauli $3^{|P|}$ Jordan-Wigner range
  penalty. The built-in matchgate-snapshot estimator (Pfaffian +
  orthogonal-matrix algebra) is now supplied by the
  ``_matchgate_shadow`` module.

Install with::

    uv add "cumulant-residual-cert[openfermion]"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

from ..catalog import Catalog, FermionicWord

if TYPE_CHECKING:
    from openfermion import FermionOperator

    from .._matchgate_shadow import MatchgateShadowRecord

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
    moments_or_record: Union[
        dict[tuple[int, ...], tuple[complex, float]], "MatchgateShadowRecord"
    ],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]] | None = None,
    *,
    confidence: float = 0.95,
    n_protocol_terms: int | None = None,
    require_all_terms: bool = True,
    u1_certified: bool = False,
    radius: str = "hoeffding",
) -> Any:
    """UCB diagnostic on matchgate / fermionic-Gaussian shadow output.

    Two input modes are supported:

    1. **Shadow-record path (recommended):** pass a
       :class:`~cumulant_residual_cert._matchgate_shadow.MatchgateShadowRecord`
       as ``moments_or_record``. The library handles the entire
       inverse-channel + Bonferroni + Möbius assembly pipeline via
       :func:`~cumulant_residual_cert.delta_ucb_matchgate_shadows`. This is
       the supported end-to-end matchgate-shadow workflow.

    2. **Pre-computed Majorana-moments path:** pass a dictionary
       ``{majorana_indices: (mean, radius)}`` keyed by 1-based Majorana
       index tuples; the wrapper routes to
       :func:`~cumulant_residual_cert.delta_ucb_from_majorana_moments`.
       Use this if you ran the inverse channel separately (or used a
       different snapshot estimator) and already have moment estimates.

    Matchgate-shadow protocols (e.g.
    :doi:`Zhao-Rubin-Miyake-Babbush 2021 <10.1103/PhysRevLett.127.110504>`,
    :doi:`Wan-Hadfield-Cleve-Babbush 2022 <10.1103/PRXQuantum.4.030337>`)
    produce a single-shot estimator
    $\\hat{\\langle \\gamma_S \\rangle}_t$ for every degree-$|S|$ Majorana
    product, with a range factor that scales polynomially in the qubit
    count instead of the random-Pauli $3^{|P|}$ Jordan-Wigner penalty.

    Parameters
    ----------
    moments_or_record : MatchgateShadowRecord or dict
        Either a shadow record (route 1) or a pre-computed Majorana-moments
        dictionary (route 2). See module-level docs.
    catalog : Catalog
    sites_per_word : sequence of sequences of int, optional
        1-based site assignments per catalog word. Required for the
        pre-computed-moments path; optional for the shadow-record path
        (defaults to ``(1, 2, ..., w.length)`` per word).
    confidence : float, default 0.95
        Target confidence $1 - \\alpha$ of the simultaneous bound.
    n_protocol_terms : int, optional
        Number of distinct Majorana products in the Bonferroni union.
        Required for the pre-computed-moments path; ignored on the
        shadow-record path (counted automatically).
    require_all_terms : bool, default True
        Pre-computed-moments path only. If True (default), raise on any
        missing Majorana product entry that appears in a catalog subword
        decomposition. If False, missing entries are treated as $(0, 0)$;
        this is exact for odd-degree products on $U(1)$-invariant states
        but unsafe for missing even-degree entries.
    u1_certified : bool, default False
        Shadow-record path only. Controls the result's
        ``delta_provenance`` (see
        :func:`~cumulant_residual_cert.delta_ucb_matchgate_shadows`).
    radius : {"hoeffding", "empirical_bernstein"}, default "hoeffding"
        Shadow-record path only. Selects the per-term radius rule.

    Returns
    -------
    UCBResult
    """
    # No actual OpenFermion call is made here yet; the wrapper is pure
    # routing. The optional-dependency import is kept lazy so docs builds
    # work without OpenFermion installed.
    from .._matchgate_shadow import MatchgateShadowRecord
    from ..diagnostic import (
        delta_ucb_from_majorana_moments,
        delta_ucb_matchgate_shadows,
    )

    if isinstance(moments_or_record, MatchgateShadowRecord):
        alpha = 1.0 - confidence
        return delta_ucb_matchgate_shadows(
            catalog=catalog,
            record=moments_or_record,
            alpha=alpha,
            sites_per_word=sites_per_word,
            u1_certified=u1_certified,
            radius=radius,
        )

    if sites_per_word is None:
        raise ValueError(
            "sites_per_word is required on the pre-computed Majorana-moments path"
        )
    if n_protocol_terms is None:
        raise ValueError(
            "n_protocol_terms is required on the pre-computed Majorana-moments path"
        )
    return delta_ucb_from_majorana_moments(
        majorana_moments=moments_or_record,
        catalog=catalog,
        sites_per_word=sites_per_word,
        confidence=confidence,
        n_protocol_terms=n_protocol_terms,
        require_all_terms=require_all_terms,
    )
