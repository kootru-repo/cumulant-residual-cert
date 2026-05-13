"""One-call API: turn a catalog and a $\\Delta$ bound into a certified bias bar."""

from __future__ import annotations

from dataclasses import dataclass

from . import constants
from .catalog import Catalog
from .constants import Level


@dataclass(frozen=True)
class CertifiedBound:
    """Result of a single certification call.

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
    """

    bounds: dict[str, float]
    delta: float
    level: Level
    constants_used: dict[str, int]


def certify(
    catalog: Catalog,
    delta: float,
    *,
    level: Level = "block_refined",
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

    Returns
    -------
    CertifiedBound
        Structured result; see attributes.

    Raises
    ------
    ValueError
        If ``delta`` is negative.
    """
    import math

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
    )
