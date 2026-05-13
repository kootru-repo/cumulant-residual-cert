"""Random-Pauli shadow upper-confidence-bound diagnostic on $\\Delta_{r, U(1)}^{\\mathrm{cat}}$.

Given a random-Pauli shadow record on a $U(1)$-invariant state and a catalog
of charge-neutral fermionic words, ``delta_ucb`` returns an upper bound on the
catalog envelope $\\Delta^{\\mathrm{cat}}_{r, U(1)}(\\rho)$ that holds with
probability $\\ge 1 - \\alpha$ simultaneously over the catalog.

The implementation Bonferroni-corrects over every Pauli string appearing in
any subword expansion, then propagates Hoeffding radii through the Mobius
transform. A full sample-splitting variant is on the v0.3 roadmap; the v0.2
estimator re-uses the entire shadow record for every Pauli mean, which gives
a valid union bound and is the right starting point for v0.1.

> **Range note.** Random Pauli shadows incur a $3^{|P|}$ range factor per Pauli
> $P$, which makes the bound data-hungry for word lengths $r \\ge 3$. For
> chemistry-relevant problems prefer matchgate / fermionic-Gaussian shadows
> via the OpenFermion adapter (added in v0.3). The diagnostic itself is
> measurement-protocol-agnostic; you supply the per-Pauli mean estimates and
> Hoeffding (or tighter) per-Pauli radii.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    fermionic-Gaussian protocol via the OpenFermion adapter (v0.3).
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


def delta_ucb(
    shadow_samples: Sequence[ShadowShot],
    catalog: Catalog,
    sites_per_word: Sequence[Sequence[int]],
    *,
    n_qubits: int,
    confidence: float = 0.95,
) -> UCBResult:
    """Compute a Bonferroni-corrected UCB on the catalog envelope.

    Parameters
    ----------
    shadow_samples : sequence of ``(basis, outcomes)`` tuples
        Random-Pauli shadow record. ``basis`` is a tuple of ``"X"``/``"Y"``/``"Z"``
        on each of ``n_qubits`` sites; ``outcomes`` is a tuple of $\\pm 1$.
    catalog : Catalog
        Charge-neutral fermionic-word catalog.
    sites_per_word : sequence of sequences of int
        Site assignments for each word in ``catalog``, in the same order.
        Each inner sequence is 1-based and has the same length as the
        corresponding word.
    n_qubits : int
        Total system size used to draw the shadows. Must be at least
        ``max(max(sites) for sites in sites_per_word)``.
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

    M = len(shadow_samples)
    if M == 0:
        raise ValueError("shadow_samples is empty")

    # Validate sites and shadow shots up front so a bad index produces a
    # readable error rather than failing deep inside the Mobius transform.
    for w, sites in zip(catalog, sites_per_word):
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
    word_sub_paulis: dict[tuple[str, ...], dict[tuple[int, ...], dict]] = {}
    all_paulis: set[tuple[str, ...]] = set()
    for w, sites in zip(catalog, sites_per_word):
        sites_t = tuple(sites)
        if len(sites_t) != w.length:
            raise ValueError(
                f"word {w.name!r} has length {w.length} but {len(sites_t)} sites supplied"
            )
        m = w.length
        sub_exp: dict[tuple[int, ...], dict] = {}
        for k in range(1, m + 1):
            for B in combinations(range(1, m + 1), k):
                key = tuple(sorted(B))
                A_B = subword_op(w.letters, sites_t, key, n_qubits)
                expansion = _pauli_expand(A_B, n_qubits)
                sub_exp[key] = expansion
                all_paulis.update(expansion.keys())
        word_sub_paulis[(w.letters, sites_t)] = sub_exp

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

    # 3. Per-word kappa-hat and propagated radius.
    per_word: dict[str, dict[str, float]] = {}
    ucb_values: list[float] = []
    for w, sites in zip(catalog, sites_per_word):
        sites_t = tuple(sites)
        m = w.length
        sub_exp = word_sub_paulis[(w.letters, sites_t)]

        hat_mu: dict[tuple[int, ...], complex] = {}
        rad_mu: dict[tuple[int, ...], float] = {}
        for B, expansion in sub_exp.items():
            hat_mu[B] = sum(c * pauli_means[P] for P, c in expansion.items())
            rad_mu[B] = sum(abs(c) * pauli_rads[P] for P, c in expansion.items())

        hat_kappa: complex = 0.0
        partitions = list(set_partitions(list(range(1, m + 1))))
        for pi in partitions:
            prod: complex = 1.0
            for C in pi:
                prod *= hat_mu[tuple(sorted(C))]
            k = len(pi)
            hat_kappa += (-1) ** (k - 1) * factorial(k - 1) * prod

        rad_kappa: float = 0.0
        for pi in partitions:
            k = len(pi)
            block_rads = [rad_mu[tuple(sorted(C))] for C in pi]
            block_contribution = 0.0
            for j, _ in enumerate(pi):
                others_prod = 1.0
                for jj in range(len(pi)):
                    if jj == j:
                        continue
                    others_prod *= max(1.0, 1.0 + block_rads[jj])
                block_contribution += block_rads[j] * others_prod
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
        n_paulis=T,
        per_word=per_word,
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
