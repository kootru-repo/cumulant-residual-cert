"""One-call API: turn a catalog and a $\\Delta$ bound into a certified bias bar."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from . import constants
from .catalog import Catalog
from .constants import Level


def _lookup_library_version() -> str:
    """Best-effort lookup of the installed package version.

    Uses ``importlib.metadata.version`` so the value is correct under
    standard installs (``uv add ...``) and editable installs
    (``uv sync`` on the repo itself). Falls back to ``"unknown"`` if
    the package is not installed at all (running directly from source
    without install), which keeps the certificate machinery functional
    in development.
    """
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("cumulant-residual-cert")
    except Exception:  # pragma: no cover - defensive
        return "unknown"


# Strings the library uses to label the provenance of the supplied $\Delta$.
# Add new entries here as new pipelines land; the field itself is
# free-form ``str`` so callers can also use their own labels when
# integrating with custom estimators.
DeltaProvenance = Literal[
    "user_supplied",
    "closed_form_bernoulli",
    "from_rdms",
    "ucb_random_pauli",
    "ucb_subword",
    "ucb_majorana",
    "ucb_matchgate_shadows",
]


@dataclass(frozen=True)
class CertifiedBound:
    """Result of a single certification call.

    The dataclass is JSON-serialisable directly via
    ``json.dumps(dataclasses.asdict(result))``. Persistence is the caller's
    responsibility; the library's job is to make sure every field needed to
    re-evaluate the certificate is present and honestly labelled.

    Attributes
    ----------
    bounds : dict[str, float]
        Per-word certified bias bars: ``bounds[word_name] = constant * delta``.
    delta : float
        The $\\Delta$ value (exact or upper bound) used as input.
    level : Level
        Which constants table was used (``"universal"``, ``"charge_filtered"``,
        or ``"block_refined"``).
    constants_used : dict[str, int]
        The integer constants applied to each word.
    library_version : str
        Version string of the producing ``cumulant-residual-cert`` install,
        looked up at construction time via ``importlib.metadata``. Useful
        as part of the persisted certificate so future audits can
        identify which library version produced the bound.
    delta_provenance : str
        How the supplied $\\Delta$ was obtained. Library-known values are
        listed in :data:`DeltaProvenance`; callers integrating with custom
        estimators may use their own strings. Default is
        ``"user_supplied"``, which is the honest label when the user
        passes a number with no further library help.
    catalog_name : str
        The catalog's ``name`` attribute, recorded so the persisted
        certificate identifies which catalog the constants apply to.
    """

    bounds: dict[str, float]
    delta: float
    level: Level
    constants_used: dict[str, int]
    library_version: str = field(default_factory=_lookup_library_version)
    delta_provenance: str = "user_supplied"
    catalog_name: str = ""


def certify(
    catalog: Catalog,
    delta: float,
    *,
    level: Level = "block_refined",
    delta_provenance: str = "user_supplied",
) -> CertifiedBound:
    """Certify the deterministic bias of an order-$\\le 2$ cumulant closure.

    For each word $W$ in ``catalog``, returns the bound

    .. math::
        |\\tau_W(\\rho)| \\;\\le\\; C_W \\cdot \\Delta,

    where $C_W$ is the integer constant at the requested ``level``.

    Parameters
    ----------
    catalog : Catalog
        Charge-neutral fermionic-word catalog. All words must satisfy
        ``word.length <= catalog.r``.
    delta : float
        An exact value of, or rigorous upper bound on, the high-cumulant
        envelope $\\Delta_{r, U(1)}^{\\mathrm{cat}}(\\rho)$. Must be
        non-negative. Compute it from state knowledge, or estimate it from
        shadow data via :func:`cumulant_residual_cert.delta_ucb`.
    level : {"universal", "charge_filtered", "block_refined"}, default "block_refined"
        Which constant family to apply. ``"block_refined"`` is tightest and
        is the recommended default.
    delta_provenance : str, default ``"user_supplied"``
        Free-form label describing how ``delta`` was obtained. Recorded on
        the returned :class:`CertifiedBound` for auditability. Library-
        recognised values appear in :data:`DeltaProvenance`; callers
        integrating with custom estimators may pass their own labels.

    Returns
    -------
    CertifiedBound
        Structured result; see attributes.

    Raises
    ------
    ValueError
        If ``delta`` is negative or non-finite.
    """
    if not math.isfinite(delta):
        raise ValueError(f"delta must be a finite real number; got {delta!r}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0; got {delta!r}")

    bounds: dict[str, float] = {}
    constants_used: dict[str, int] = {}
    for w in catalog:
        c = constants.get(level, catalog.r, w)
        constants_used[w.name] = c
        bounds[w.name] = c * delta

    return CertifiedBound(
        bounds=bounds,
        delta=delta,
        level=level,
        constants_used=constants_used,
        delta_provenance=delta_provenance,
        catalog_name=catalog.name,
    )
