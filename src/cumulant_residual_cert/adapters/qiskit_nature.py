"""Qiskit-Nature adapter.

The chemistry workflow this adapter is built for: a user is running a VQE or
QPE pipeline through `Qiskit-Nature <https://qiskit.org/ecosystem/nature/>`_
and wants to certify the bias of a cumulant-truncated post-processing step on
a chemistry-catalog observable.

Two routes are supported currently:

- :func:`from_problem`: take an ``ElectronicStructureProblem`` whose ground
  state is computed via mean-field, returning $\\Delta = 0$ when the state
  falls in the Bernoulli class.

- :func:`word_to_fermionic_op`: convert :class:`FermionicWord` instances to
  Qiskit-Nature ``FermionicOp`` so users can plug catalog observables into
  Qiskit estimators directly.

Install with::

    pip install "cumulant-residual-cert[qiskit-nature]"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..catalog import Catalog, FermionicWord
from ..constants import Level
from ._common import AdapterEstimate, package_estimate

if TYPE_CHECKING:
    from qiskit_nature.second_q.operators import FermionicOp

_QISKIT_NATURE_MISSING_MSG = (
    "qiskit-nature is required for cumulant_residual_cert.adapters.qiskit_nature. "
    "Install with: pip install 'cumulant-residual-cert[qiskit-nature]'"
)


def _require_qiskit_nature():
    """Import and return Qiskit-Nature's ``FermionicOp`` or raise a helpful error.

    Kept lazy so that ``import cumulant_residual_cert.adapters.qiskit_nature``
    does not fail when only docs are being built or when an introspection tool
    walks the module without intending to call any of its functions.
    """
    try:
        from qiskit_nature.second_q.operators import FermionicOp as _FermionicOp
    except ImportError as e:  # pragma: no cover
        raise ImportError(_QISKIT_NATURE_MISSING_MSG) from e
    return _FermionicOp


def word_to_fermionic_op(
    word: FermionicWord,
    sites: Sequence[int],
) -> "FermionicOp":
    """Convert a :class:`FermionicWord` to a Qiskit-Nature ``FermionicOp``.

    Site indices are 1-based on input and converted to 0-based for Qiskit-Nature.
    """
    FermionicOp = _require_qiskit_nature()
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

    factors: list[str] = []
    for L, s in zip(word.letters, sites):
        site0 = s - 1
        if L == "I":
            continue
        if L == "n":
            factors.extend([f"+_{site0}", f"-_{site0}"])
        elif L == "a":
            factors.append(f"-_{site0}")
        elif L == "a_dag":
            factors.append(f"+_{site0}")
        else:
            raise ValueError(f"unknown letter {L!r}")

    if not factors:
        return FermionicOp({"": 1.0})
    return FermionicOp({" ".join(factors): 1.0})


def from_problem(
    problem: Any,
    catalog: Catalog,
    *,
    basis: str = "canonical",
    level: Level = "block_refined",
    user_asserts_bernoulli_class: bool = False,
) -> AdapterEstimate:
    """Certify the truncation bias on a Qiskit-Nature electronic-structure problem.

    .. warning::
        An ``ElectronicStructureProblem`` is a *problem specification*, not a
        prepared quantum state. The adapter cannot tell from the problem
        object alone whether the state actually executed on hardware is the
        canonical-basis mean-field Slater determinant required for the
        Bernoulli-class theorem. Callers must affirmatively pass
        ``user_asserts_bernoulli_class=True`` to acknowledge this.

    Parameters
    ----------
    problem : ElectronicStructureProblem
    catalog : Catalog
    basis : {"canonical"}
        Only ``"canonical"`` is supported; other bases would require RDM
        evaluation (planned for a later release).
    level : {"universal", "charge_filtered", "block_refined"}
    user_asserts_bernoulli_class : bool
        Required to be True. Asserts that the prepared state is a canonical-
        basis mean-field Slater determinant in the chosen dictionary basis.

    Returns
    -------
    AdapterEstimate
    """
    if basis != "canonical":
        raise NotImplementedError(
            "The closed-form Bernoulli-class result requires the dictionary "
            "basis to coincide with the canonical molecular orbital basis."
        )
    if not user_asserts_bernoulli_class:
        raise ValueError(
            "from_problem() needs an explicit user assertion that the "
            "prepared state is a canonical-basis mean-field Slater "
            "determinant. Pass user_asserts_bernoulli_class=True to confirm, "
            "or use word_to_fermionic_op() to plug catalog observables into "
            "a Qiskit estimator directly."
        )

    notes = (
        f"Qiskit-Nature {type(problem).__name__} with user-asserted Bernoulli-class state.",
        "Falls in the Bernoulli (occupation-basis diagonal product) class.",
        "Worked-example theorem: Delta = 0 identically on this class.",
    )
    return package_estimate(
        catalog,
        delta=0.0,
        delta_is_exact=True,
        framework="qiskit_nature",
        level=level,
        notes=notes,
        delta_provenance="closed_form_bernoulli",
    )
