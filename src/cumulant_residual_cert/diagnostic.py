"""Random-Pauli shadow upper-confidence-bound diagnostic on $\\Delta_{r, U(1)}^{\\mathrm{cat}}$.

Given a random-Pauli shadow record on a $U(1)$-invariant state and a catalog
of charge-neutral fermionic words, ``delta_ucb`` returns an upper bound on the
catalog envelope $\\Delta^{\\mathrm{cat}}_{r, U(1)}(\\rho)$ that holds with
probability $\\ge 1 - \\alpha$ simultaneously over the catalog.

The implementation Bonferroni-corrects over every Pauli string appearing in
any subword expansion, then propagates Hoeffding radii through the Mobius
transform. Two public entry points consume shadow data:

- :func:`delta_ucb` reuses the entire shadow record for every Pauli mean
  (a valid union bound). This is the recommended default.
- :func:`delta_ucb_split` implements the manuscript's sample-split form:
  the shadow record is split into a diagnostic half (drives the bound)
  and a holdout half (returned to the caller for downstream estimation or
  empirical-Bernstein refinement).

Both feed through :func:`_ucb_from_subword_moments`. Protocol-agnostic
entry points (:func:`delta_ucb_from_subword_moments`,
:func:`delta_ucb_from_majorana_moments`) accept user-supplied
``(mean, radius)`` estimates directly, bypassing the dense Pauli expansion.

> **Range note.** Random Pauli shadows incur a $3^{|P|}$ range factor per Pauli
> $P$, which makes the bound data-hungry for word lengths $r \\ge 3$. For
> chemistry-relevant problems prefer matchgate / fermionic-Gaussian shadows
> via the OpenFermion adapter (planned). The diagnostic itself is
> measurement-protocol-agnostic; you supply the per-Pauli mean estimates and
> Hoeffding (or tighter) per-Pauli radii.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from math import factorial, log, sqrt

import numpy as np

from ._fermion import _kron, subword_op
from ._partition import set_partitions
from .catalog import Catalog

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LOCAL: dict[str, np.ndarray] = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def _pauli_op(label: tuple[str, ...]) -> np.ndarray:
    return _kron([PAULI_LOCAL[L] for L in label])


def _pauli_weight(label: tuple[str, ...]) -> int:
    return sum(1 for L in label if L != "I")


def _pauli_expand(M: np.ndarray, n: int, tol: float = 1e-12) -> dict[tuple[str, ...], complex]:
    """Decompose an $n$-qubit operator into Pauli-string coefficients.

    The implementation enumerates all $4^n$ Pauli strings, so it is dense and
    intended only for small validation states (typically $n \\le 8$). For
    chemistry-scale registers, replace this with a matchgate-shadow or
    fermionic-Gaussian protocol via the OpenFermion adapter (planned).
    """
    if n > 10:
        raise ValueError(
            f"_pauli_expand is dense (O(4^n)); refusing to enumerate for n_qubits={n}. "
            "Use a matchgate or fermionic-Gaussian protocol for chemistry-scale registers."
        )
    dim = 2 ** n
    out: dict[tuple[str, ...], complex] = {}
    for label in product("IXYZ", repeat=n):
        P = _pauli_op(label)
        c = np.trace(P @ M) / dim
        if abs(c) > tol:
            out[label] = complex(c)
    return out


ShadowShot = tuple[tuple[str, ...], tuple[int, ...]]


def _one_shot_estimator(label: tuple[str, ...], basis: tuple[str, ...], outcomes: tuple[int, ...]) -> float:
    """Random-Pauli shadow snapshot estimator $\\hat{\\langle P \\rangle}_t$.

    Returns $3^{|P|} \\prod_{i:\\,P_i \\ne I} b_{t,i}$ if $B_{t,i} = P_i$ on the
    support of $P$, else $0$.

    The $3^{|P|}$ factor is required for the snapshot to be an unbiased
    estimator of $\\langle P \\rangle$ under the textbook random-Pauli shadow
    protocol. Without it the empirical mean converges to $3^{-|P|} \\langle P \\rangle$
    and the propagated UCB no longer bounds the catalog envelope.
    """
    val = 1.0
    weight = 0
    for i, P_i in enumerate(label):
        if P_i == "I":
            continue
        weight += 1
        if basis[i] != P_i:
            return 0.0
        val *= outcomes[i]
    return (3 ** weight) * val


def _hoeffding_radius(weight: int, M: int, alpha_per: float) -> float:
    """Per-Pauli Hoeffding half-width with the textbook random-Pauli range factor."""
    return (3 ** weight) * sqrt(2 * log(2 / alpha_per) / M)


def _partition_radius_contribution(
    block_hat_mu_mag: Sequence[float],
    block_rads: Sequence[float],
    operator_norm_bound: float = 1.0,
) -> float:
    """Telescoping radius bound on $|\\prod_B \\hat\\mu_B - \\prod_B \\mu_B|$.

    Each factor satisfies $|\\mu_B| \\le \\min(\\mathrm{op\\_norm\\_bound},
    |\\hat\\mu_B| + \\mathrm{rad}_B)$ simultaneously (the first by the
    operator-norm bound on normal-ordered letter products of catalog
    operators; the second by Hoeffding, w.h.p. across the catalog under the
    Bonferroni correction). The tightest valid upper bound takes the min of
    the two and sums the telescoping single-factor errors:

    $$
    \\sum_j \\mathrm{rad}_j \\cdot \\prod_{j' \\ne j} \\min(c, |\\hat\\mu_{j'}|
    + \\mathrm{rad}_{j'}).
    $$

    Splitting this out as a helper makes the tightening directly regression-testable.
    """
    n = len(block_hat_mu_mag)
    if len(block_rads) != n:
        raise ValueError("block_hat_mu_mag and block_rads must have equal length")
    if n == 0:
        return 0.0
    factor_upper = [
        min(operator_norm_bound, block_hat_mu_mag[i] + block_rads[i]) for i in range(n)
    ]
    total = 0.0
    for j in range(n):
        others = 1.0
        for jj in range(n):
            if jj == j:
                continue
            others *= factor_upper[jj]
        total += block_rads[j] * others
    return total


@dataclass(frozen=True)
class UCBResult:
    """Result of a UCB diagnostic call.

    Attributes
    ----------
    delta_ucb : float
        The one-sided upper bound on $\\Delta^{\\mathrm{cat}}_{r, U(1)}(\\rho)$.
    confidence : float
        $1 - \\alpha$, the probability with which the bound holds simultaneously
        across all catalog words.
    n_paulis : int
        Total number of distinct Pauli strings that entered the Bonferroni
        correction. Reported because the diagnostic's data efficiency is
        governed by this count.
    per_word : dict[str, dict[str, float]]
        Per-word breakdown: ``{word_name: {"kappa_hat": ..., "radius": ..., "ucb": ...}}``.
    """

    delta_ucb: float
    confidence: float
    n_paulis: int
    per_word: dict[str, dict[str, float]]


def _ucb_from_subword_moments(
    per_subword: dict[str, dict[tuple[int, ...], tuple[complex, float]]],
    catalog: Catalog,
    confidence: float,
    n_protocol_terms: int,
) -> UCBResult:
    """Protocol-agnostic UCB pipeline.

    Given per-subword empirical moment and Hoeffding-style radius for every
    catalog word, compute the per-word connected cumulant via the Mobius
    formula and propagate the radius through a telescoping product bound.
    The Bonferroni union is assumed to have already been applied to the
    radii (so they hold simultaneously with probability $\\ge 1 - \\alpha$).
    """
    per_word: dict[str, dict[str, float]] = {}
    ucb_values: list[float] = []

    for w in catalog:
        m = w.length
        subword_data = per_subword[w.name]
        partitions = list(set_partitions(list(range(1, m + 1))))

        hat_kappa: complex = 0.0
        for pi in partitions:
            prod: complex = 1.0
            for C in pi:
                key = tuple(sorted(C))
                prod *= subword_data[key][0]
            k = len(pi)
            hat_kappa += (-1) ** (k - 1) * factorial(k - 1) * prod

        # Telescoping product-error bound; see _partition_radius_contribution.
        rad_kappa: float = 0.0
        for pi in partitions:
            k = len(pi)
            block_keys = [tuple(sorted(C)) for C in pi]
            block_hat = [abs(subword_data[key][0]) for key in block_keys]
            block_rads = [subword_data[key][1] for key in block_keys]
            block_contribution = _partition_radius_contribution(block_hat, block_rads)
            rad_kappa += factorial(k - 1) * block_contribution

        ucb_w = abs(hat_kappa) + rad_kappa
        ucb_values.append(ucb_w)
        per_word[w.name] = {
            "kappa_hat": float(abs(hat_kappa)),
            "radius": float(rad_kappa),
            "ucb": float(ucb_w),
        }

    delta_bar = max(ucb_values) if ucb_values else 0.0
    return UCBResult(
        delta_ucb=float(delta_bar),
        confidence=confidence,
        n_paulis=n_protocol_terms,
        per_word=per_word,
    )


def delta_ucb_from_subword_moments(
    per_subword: dict[str, dict[tuple[int, ...], tuple[complex, float]]],
    catalog: Catalog,
    *,
    confidence: float = 0.95,
    n_protocol_terms: int,
) -> UCBResult:
    """Protocol-agnostic UCB on the catalog envelope from per-subword moments.

    Use this when you have already estimated empirical moments and radii for
    every subword of every catalog word from some measurement protocol of
    your choice (random-Pauli shadows, matchgate / fermionic-Gaussian
    shadows, optimized derandomized shadows, Pauli grouping, direct RDM
    measurement, simulation, etc.). The library handles the Mobius assembly
    and product-error propagation; the protocol-specific snapshot
    estimation is your responsibility.

    Parameters
    ----------
    per_subword : dict[str, dict[tuple[int, ...], (complex, float)]]
        Outer key is the catalog word's name. Inner key is the block index
        tuple (1-based, sorted ascending). Inner value is
        ``(empirical_moment, hoeffding_radius)``. Every non-empty subword of
        every catalog word must appear. Radii must already incorporate any
        Bonferroni / union-bound correction so they hold simultaneously at
        the stated confidence.
    catalog : Catalog
    confidence : float, default 0.95
    n_protocol_terms : int
        Number of distinct protocol primitives (Pauli strings, Majorana
        products, etc.) that entered the Bonferroni union. Reported on the
        result as ``n_paulis`` for backward compatibility; the field is
        protocol-agnostic.

    Returns
    -------
    UCBResult
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    if n_protocol_terms < 0:
        raise ValueError(f"n_protocol_terms must be >= 0; got {n_protocol_terms!r}")

    # Validate per_subword shape: every word, every non-empty subword.
    for w in catalog:
        if w.name not in per_subword:
            raise ValueError(f"per_subword missing entry for word {w.name!r}")
        subword_data = per_subword[w.name]
        m = w.length
        expected_blocks = {
            tuple(sorted(B))
            for k in range(1, m + 1)
            for B in combinations(range(1, m + 1), k)
        }
        missing = expected_blocks - set(subword_data.keys())
        if missing:
            raise ValueError(
                f"per_subword[{w.name!r}] missing entries for blocks: {sorted(missing)}"
            )
        for key, (mean_val, rad_val) in subword_data.items():
            import math
            if not math.isfinite(rad_val):
                raise ValueError(
                    f"per_subword[{w.name!r}][{key}]: radius must be finite; got {rad_val}"
                )
            if rad_val < 0:
                raise ValueError(
                    f"per_subword[{w.name!r}][{key}]: radius must be >= 0; got {rad_val}"
                )
            mean_complex = complex(mean_val)
            if not (math.isfinite(mean_complex.real) and math.isfinite(mean_complex.imag)):
                raise ValueError(
                    f"per_subword[{w.name!r}][{key}]: mean must be finite; got {mean_val}"
                )

    return _ucb_from_subword_moments(
        per_subword, catalog, confidence, n_protocol_terms,
    )


def delta_ucb(
    shadow_samples: Iterable[ShadowShot],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    n_qubits: int,
    confidence: float = 0.95,
) -> UCBResult:
    """Compute a Bonferroni-corrected UCB on the catalog envelope.

    The built-in random-Pauli expansion is dense in $n_{\\mathrm{qubits}}$ and
    refuses to run beyond $n_{\\mathrm{qubits}} = 10$. For larger registers,
    bypass this function: supply your own per-Pauli mean estimates and
    Hoeffding-style radii to the propagation pipeline (matchgate-shadow
    adapter planned for a later release).

    Parameters
    ----------
    shadow_samples : iterable of ``(basis, outcomes)`` tuples
        Random-Pauli shadow record. ``basis`` is a tuple of ``"X"``/``"Y"``/``"Z"``
        on each of ``n_qubits`` sites; ``outcomes`` is a tuple of $\\pm 1$.
        A general iterable is accepted; it is materialized once at entry.
    catalog : Catalog
        Charge-neutral fermionic-word catalog.
    sites_per_word : sequence of sequences of int
        Site assignments for each word in ``catalog``, in the same order.
        Each inner sequence is 1-based and has the same length as the
        corresponding word.
    n_qubits : int
        Total system size used to draw the shadows. Must be at least
        ``max(max(sites) for sites in sites_per_word)``. Capped at 10 due to
        the dense Pauli expansion.
    confidence : float, default 0.95
        Target confidence level $1 - \\alpha$ of the simultaneous bound.

    Returns
    -------
    UCBResult
        Structured result; see attributes.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    if len(sites_per_word) != len(catalog):
        raise ValueError(
            f"sites_per_word has {len(sites_per_word)} entries but catalog has "
            f"{len(catalog)} words"
        )

    # Materialize the shadow iterable once; downstream code iterates twice and
    # also needs len().
    shadow_samples = tuple(shadow_samples)
    M = len(shadow_samples)
    if M == 0:
        raise ValueError("shadow_samples is empty")

    # Validate sites and shadow shots up front so a bad index produces a
    # readable error rather than failing deep inside the Mobius transform.
    for w, sites in zip(catalog, sites_per_word, strict=False):
        if len(sites) != w.length:
            raise ValueError(
                f"word {w.name!r} has length {w.length} but {len(sites)} sites given"
            )
        if len(set(sites)) != len(sites):
            raise ValueError(f"word {w.name!r} has duplicate site indices: {sites}")
        for s in sites:
            if not (1 <= s <= n_qubits):
                raise ValueError(
                    f"word {w.name!r} site {s} outside 1..n_qubits={n_qubits}"
                )

    for idx, shot in enumerate(shadow_samples):
        if not (isinstance(shot, tuple) and len(shot) == 2):
            raise ValueError(f"shadow_samples[{idx}] must be a (basis, outcomes) tuple")
        basis, outcomes = shot
        if len(basis) != n_qubits or len(outcomes) != n_qubits:
            raise ValueError(
                f"shadow_samples[{idx}]: basis/outcomes length must equal n_qubits={n_qubits}"
            )
        if any(b not in ("X", "Y", "Z") for b in basis):
            raise ValueError(f"shadow_samples[{idx}]: basis must be X/Y/Z only")
        if any(o not in (-1, 1) for o in outcomes):
            raise ValueError(f"shadow_samples[{idx}]: outcomes must be in {{-1, +1}}")

    alpha = 1 - confidence

    # 1. Build subword Pauli expansions for every catalog word.
    word_sub_paulis: dict[str, dict[tuple[int, ...], dict]] = {}
    all_paulis: set[tuple[str, ...]] = set()
    for w, sites in zip(catalog, sites_per_word, strict=False):
        sites_t = tuple(sites)
        m = w.length
        sub_exp: dict[tuple[int, ...], dict] = {}
        for k in range(1, m + 1):
            for B in combinations(range(1, m + 1), k):
                key = tuple(sorted(B))
                A_B = subword_op(w.letters, sites_t, key, n_qubits)
                expansion = _pauli_expand(A_B, n_qubits)
                sub_exp[key] = expansion
                all_paulis.update(expansion.keys())
        word_sub_paulis[w.name] = sub_exp

    pauli_list = sorted(all_paulis)
    T = len(pauli_list)
    alpha_per = alpha / max(T, 1)

    # 2. Empirical Pauli means and Hoeffding radii.
    pauli_means: dict[tuple[str, ...], float] = {}
    pauli_rads: dict[tuple[str, ...], float] = {}
    for P in pauli_list:
        s = 0.0
        for basis, outcomes in shadow_samples:
            s += _one_shot_estimator(P, basis, outcomes)
        pauli_means[P] = s / M
        pauli_rads[P] = _hoeffding_radius(_pauli_weight(P), M, alpha_per)

    # 3. Per-subword empirical (mean, radius), built from the Pauli decomposition.
    per_subword: dict[str, dict[tuple[int, ...], tuple[complex, float]]] = {}
    for w in catalog:
        sub_exp = word_sub_paulis[w.name]
        per_subword[w.name] = {
            B: (
                sum(c * pauli_means[P] for P, c in expansion.items()),
                sum(abs(c) * pauli_rads[P] for P, c in expansion.items()),
            )
            for B, expansion in sub_exp.items()
        }

    # 4. Delegate the protocol-agnostic Mobius + propagation step.
    return _ucb_from_subword_moments(
        per_subword, catalog, confidence, n_protocol_terms=T,
    )


def delta_ucb_from_majorana_moments(
    majorana_moments: dict[tuple[int, ...], tuple[complex, float]],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    confidence: float = 0.95,
    n_protocol_terms: int,
    require_all_terms: bool = True,
) -> UCBResult:
    """UCB pipeline from per-Majorana-product (mean, radius) estimates.

    This is the protocol-agnostic entry point for matchgate /
    fermionic-Gaussian shadow protocols, classical-shadow variants based on
    Majorana sampling, or any custom estimator that produces (mean, radius)
    per Majorana product. The library handles the dictionary-letter to
    Majorana decomposition, the subword propagation, and the Mobius
    assembly; the snapshot estimation is up to the caller.

    Parameters
    ----------
    majorana_moments : dict[tuple[int, ...], (complex, float)]
        Keyed by strictly-sorted tuple of distinct Majorana indices
        $S = (j_1 < j_2 < \\ldots < j_{2k})$. The value is
        ``(empirical_moment, hoeffding_radius)`` for the Majorana product
        $\\gamma_S = \\gamma_{j_1} \\gamma_{j_2} \\cdots \\gamma_{j_{2k}}$.
        Site $p$ (1-based) maps to Majorana indices $(2p-1, 2p)$. The empty
        tuple ``()`` represents the identity; if absent the mean is taken
        as $1$ (exact, zero radius). Radii must already incorporate any
        Bonferroni / union-bound correction.
    catalog : Catalog
    sites_per_word : sequence of sequences of int
        1-based site assignments per catalog word, in catalog order.
    confidence : float, default 0.95
    n_protocol_terms : int
        Number of distinct Majorana products counted in the Bonferroni
        union. Reported on the result as ``n_paulis``.
    require_all_terms : bool, default True
        When True (default), every Majorana product that appears in a
        catalog subword decomposition must be present in
        ``majorana_moments``; missing entries raise ``ValueError``. This
        is the safe default for a certification API.

        When False, missing entries are taken as $(0, 0)$. This is exact
        for odd-degree Majorana products on $U(1)$-invariant states (their
        expectation is zero by symmetry), but missing **even-degree**
        entries are also silently treated as zero radius, which
        under-estimates the bound and can invalidate the certificate.
        Opt in to ``False`` only when you have independently verified
        that all missing terms are odd-degree.

    Returns
    -------
    UCBResult
    """
    from ._majorana import word_majorana_decomposition

    if len(sites_per_word) != len(catalog):
        raise ValueError(
            f"sites_per_word has {len(sites_per_word)} entries but catalog has "
            f"{len(catalog)} words"
        )

    def _lookup(idx_tuple: tuple[int, ...]) -> tuple[complex, float]:
        if idx_tuple == ():
            return majorana_moments.get((), (1.0 + 0j, 0.0))
        if idx_tuple not in majorana_moments:
            if require_all_terms:
                raise ValueError(
                    f"majorana_moments missing required entry for Majorana indices {idx_tuple}"
                )
            return (0.0 + 0j, 0.0)
        return majorana_moments[idx_tuple]

    per_subword: dict[str, dict[tuple[int, ...], tuple[complex, float]]] = {}
    for w, sites in zip(catalog, sites_per_word, strict=False):
        sites_t = tuple(int(s) for s in sites)
        if len(sites_t) != w.length:
            raise ValueError(
                f"word {w.name!r} has length {w.length} but {len(sites_t)} sites supplied"
            )
        sub_data: dict[tuple[int, ...], tuple[complex, float]] = {}
        m = w.length
        for k in range(1, m + 1):
            for B in combinations(range(1, m + 1), k):
                key = tuple(sorted(B))
                sub_letters = tuple(w.letters[i - 1] for i in key)
                sub_sites = tuple(sites_t[i - 1] for i in key)
                decomp = word_majorana_decomposition(sub_letters, sub_sites)
                hat_mu: complex = 0.0 + 0j
                rad_mu: float = 0.0
                for indices, coeff in decomp.items():
                    mean_val, rad_val = _lookup(indices)
                    hat_mu += coeff * mean_val
                    rad_mu += abs(coeff) * rad_val
                sub_data[key] = (hat_mu, rad_mu)
        per_subword[w.name] = sub_data

    return delta_ucb_from_subword_moments(
        per_subword,
        catalog,
        confidence=confidence,
        n_protocol_terms=n_protocol_terms,
    )


@dataclass(frozen=True)
class UCBSplitResult:
    """Result of a sample-split UCB diagnostic call.

    Attributes
    ----------
    ucb : UCBResult
        Bound computed on the diagnostic half $\\mathcal S^{(1)}$.
    diagnostic_indices : tuple[int, ...]
        Indices in the original ``shadow_samples`` that were routed into the
        diagnostic half.
    holdout_indices : tuple[int, ...]
        Indices in the original ``shadow_samples`` that were reserved as the
        independent holdout half $\\mathcal S^{(2)}$. Caller may use them for
        downstream estimation, calibration, or empirical-Bernstein variance
        bounds; the diagnostic certificate is independent of them.
    n_diagnostic : int
    n_holdout : int
    """

    ucb: UCBResult
    diagnostic_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    n_diagnostic: int
    n_holdout: int


def delta_ucb_split(
    shadow_samples: Iterable[ShadowShot],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    n_qubits: int,
    confidence: float = 0.95,
    fraction_diagnostic: float = 0.5,
    seed: int | None = None,
) -> UCBSplitResult:
    """Sample-split UCB diagnostic.

    Splits ``shadow_samples`` into two disjoint halves with a public seed:

    - $\\mathcal S^{(1)}$, the **diagnostic half**, drives the UCB computation
      via :func:`delta_ucb`.
    - $\\mathcal S^{(2)}$, the **holdout half**, is returned untouched as
      ``holdout_indices`` so the caller can use it for downstream estimation
      or calibration. The certificate is then independent of any subsequent
      accept/reject or prediction step that uses $\\mathcal S^{(2)}$.

    Parameters
    ----------
    shadow_samples : iterable of ``(basis, outcomes)`` tuples
        Random-Pauli shadow record. Materialized once.
    catalog : Catalog
    sites_per_word : sequence of sequences of int
    n_qubits : int
    confidence : float, default 0.95
    fraction_diagnostic : float, default 0.5
        Fraction of shots routed to the diagnostic half; the rest is the
        holdout. Must be in $(0, 1)$.
    seed : int, optional
        Public seed for the split. When ``None``, an even/odd split by index
        is used so the split is reproducible without an RNG.

    Returns
    -------
    UCBSplitResult
        Structured result; see attributes.
    """
    if not 0 < fraction_diagnostic < 1:
        raise ValueError(
            f"fraction_diagnostic must be in (0, 1); got {fraction_diagnostic!r}"
        )
    shots = list(shadow_samples)
    M = len(shots)
    if M < 2:
        raise ValueError(
            f"sample-split needs at least 2 shadow shots; got {M}"
        )

    if seed is None:
        # Deterministic even/odd split. With fraction_diagnostic = 0.5 this
        # gives an exact 50/50 split (M even) or 50/50 +/-1 (M odd).
        if fraction_diagnostic == 0.5:
            diag_indices = tuple(i for i in range(M) if i % 2 == 0)
            hold_indices = tuple(i for i in range(M) if i % 2 == 1)
        else:
            n_diag = int(round(fraction_diagnostic * M))
            diag_indices = tuple(range(n_diag))
            hold_indices = tuple(range(n_diag, M))
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(M)
        n_diag = int(round(fraction_diagnostic * M))
        diag_indices = tuple(int(i) for i in order[:n_diag])
        hold_indices = tuple(int(i) for i in order[n_diag:])

    if len(diag_indices) == 0 or len(hold_indices) == 0:
        raise ValueError(
            f"fraction_diagnostic={fraction_diagnostic!r} on M={M} shots "
            f"produces an empty half (diag={len(diag_indices)}, "
            f"hold={len(hold_indices)}); pick a fraction that leaves at "
            "least 1 shot on each side"
        )

    diag = [shots[i] for i in diag_indices]
    ucb = delta_ucb(
        shadow_samples=diag,
        catalog=catalog,
        sites_per_word=sites_per_word,
        n_qubits=n_qubits,
        confidence=confidence,
    )
    return UCBSplitResult(
        ucb=ucb,
        diagnostic_indices=diag_indices,
        holdout_indices=hold_indices,
        n_diagnostic=len(diag_indices),
        n_holdout=len(hold_indices),
    )


def collect_shadows(rho: np.ndarray, n: int, M: int, seed: int = 0) -> list[ShadowShot]:
    """Convenience: draw $M$ random-Pauli shadow shots from a dense state.

    For real workflows the shots come from hardware or another simulator; this
    helper exists only to make the worked-example notebooks self-contained.
    """
    rng = np.random.default_rng(seed)
    shots: list[ShadowShot] = []
    for _ in range(M):
        basis = tuple(rng.choice(["X", "Y", "Z"]) for _ in range(n))
        U_blocks: list[np.ndarray] = []
        for c in basis:
            if c == "X":
                U_blocks.append(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))
            elif c == "Y":
                U_blocks.append(np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2))
            else:
                U_blocks.append(I2)
        U = _kron(U_blocks)
        rho_rot = U @ rho @ U.conj().T
        probs = np.clip(np.real(np.diag(rho_rot)), 0.0, None)
        probs /= probs.sum()
        idx = int(rng.choice(2 ** n, p=probs))
        bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
        outcomes = tuple(1 - 2 * b for b in bits)
        shots.append((basis, outcomes))
    return shots
