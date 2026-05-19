"""Matchgate-shadow record generation and inverse channel (Zhao-Rubin-Miyake, 2021).

This module builds on :mod:`_matchgate` and provides:

- :class:`MatchgateShadowRecord` -- dataclass storing per-shot rotations and
  computational-basis outcomes.
- :func:`generate_matchgate_shadow_record` -- synthetic shadow-record sampler
  for a fermionic-Gaussian state with given Majorana covariance.
- :func:`matchgate_inverse_channel_majorana_moments` -- the closed-form
  Zhao-Rubin-Miyake (2021) inverse channel applied to a list of Majorana
  products, returning empirical means and certified Hoeffding radii. The
  implementation is **vectorized**: see the function docstring for the
  Pfaffian batching strategy.

Vectorized inverse-channel pipeline
-----------------------------------
The per-shot ZRM (2021) estimator
$\\hat{\\langle\\gamma_S\\rangle}_t = \\binom{2n}{2k}/\\binom{n}{k} \\cdot
\\langle b_t | U_{Q_t}\\gamma_S U_{Q_t}^\\dagger | b_t\\rangle$ collapses to a
**single Pfaffian** of a $2k\\times 2k$ submatrix once the rotated
Schrödinger-side covariance $\\tilde C_t := Q_t^T C_{b_t} Q_t$ is in hand:

.. math::
    \\langle b_t | U_{Q_t}\\gamma_S U_{Q_t}^\\dagger | b_t\\rangle
    = i^k \\, \\operatorname{Pf}(\\tilde C_t[S, S]).

This identity follows from the Heisenberg-row Majorana convention
$U_Q\\gamma_a U_Q^\\dagger = \\sum_j Q_{ja}\\gamma_j$ combined with the
Schrödinger-side covariance transform $C \\mapsto Q^T C Q$ and the
Gaussian Wick rule $\\langle\\gamma_S\\rangle_C = i^k\\operatorname{Pf}(C[S,S])$.
It is **exactly equivalent** to the prior $(2n)^{2k}$ Cartesian-product
expansion implemented in :func:`_matchgate.apply_matchgate_rotation_to_majorana_product`,
but reduces the per-(shot, product) cost from $O((2n)^{2k})$ Python-bookkept
canonicalizations to a single batched LU-style Pfaffian step on a tiny
$2k\\times 2k$ matrix.

The vectorization layout is:

- Compute the per-shot rotated bitstring covariance tensor
  $\\tilde C \\in \\mathbb{R}^{M\\times 2n\\times 2n}$ once per call.
- For each Majorana product $S$ of degree $2k$, extract the rank-2 slice
  $\\tilde C[:, S, S] \\in \\mathbb{R}^{M\\times 2k\\times 2k}$ and apply the
  batched Pfaffian :func:`_pfaffian_batch` along the leading $M$ axis.
- Aggregate the per-shot estimator $\\binom{2n}{2k}/\\binom{n}{k} \\cdot i^k
  \\operatorname{Pf}(\\tilde C_t[S,S])$ into an empirical mean.

The peak memory footprint is dominated by the rotated covariance tensor:
$M \\cdot (2n)^2 \\cdot 8$ bytes for ``float64``. At $M=10^6$ and $n=4$ this
is $64\\,\\text{MB}$; at $n=20$ it is $1.6\\,\\text{GB}$. The library is
intended for $n \\le 16$; the $M=10^6$ working set fits comfortably in RAM
for the chemistry-r4 regime.

The unvectorized reference path is retained as
:func:`_matchgate_inverse_channel_unvectorized` for parity testing.

Convention summary
------------------
- Majorana indices are 0-based; mode $p \\in \\{0, \\ldots, n - 1\\}$ has
  Majoranas at indices $2p$ and $2p + 1$.
- The Majorana covariance is the real skew-symmetric matrix
  $C_{ab} = -i \\, \\langle \\gamma_a \\gamma_b \\rangle_\\rho$ for $a \\ne b$
  (with $C_{aa} = 0$ by skew-symmetry).
- Computational-basis bitstring $b \\in \\{0, 1\\}^n$ corresponds to the
  Gaussian state with covariance $C_b$, where $(C_b)_{2p, 2p+1} = 1 - 2 b_p$
  and all other entries are zero (block-diagonal $\\pm 1$ "skew" form). With
  the convention $n_p = \\tfrac12 + \\tfrac{i}{2} \\gamma_{2p} \\gamma_{2p+1}$
  from :mod:`_majorana`, $b_p = 0$ means $\\langle n_p \\rangle = 0$ and
  $\\langle \\gamma_{2p} \\gamma_{2p+1} \\rangle = i$, giving
  $C_b[2p, 2p+1] = +1$. $b_p = 1$ gives $-1$.
- Rotations $Q \\in O(2n)$ act on Majoranas in the Heisenberg picture
  $\\gamma_i \\mapsto \\sum_j Q_{ij} \\gamma_j$; the Majorana covariance of the
  rotated state transforms as $C \\mapsto Q C Q^T$ (see SCOPE.md line 18 and
  the docstring of :func:`_matchgate.apply_matchgate_rotation_to_majorana_product`).

ZRM 2021 inverse channel
------------------------
For each shot $t$, sample $Q_t$ Haar-uniformly on $O(2n)$, apply the Gaussian
Clifford rotation $U_{Q_t}$ on the qubit Hilbert space, and measure in the
computational basis to get outcome $b_t \\in \\{0, 1\\}^n$. For a degree-$2k$
Majorana product $\\gamma_S = \\gamma_{j_1} \\cdots \\gamma_{j_{2k}}$, the
unbiased per-shot estimator (ZRM 2021 Eq. 5) is

.. math::
    \\hat{\\langle \\gamma_S \\rangle}_t
    = \\frac{\\binom{2n}{2k}}{\\binom{n}{k}} \\,
      \\langle b_t | U_{Q_t} \\gamma_S U_{Q_t}^\\dagger | b_t \\rangle.

Schrödinger evolution of operators by $U_Q$ is
$U_Q \\gamma_a U_Q^\\dagger = \\sum_j Q_{ja} \\gamma_j$ (the column form), which
is the **transpose** of the Heisenberg row form. So we compute the rotated
product via :func:`apply_matchgate_rotation_to_majorana_product` applied to
$Q^T$. The result is a dictionary $\\{S': c_{S'}\\}$ of canonical sorted
Majorana index tuples and complex coefficients.

The matrix element $\\langle b_t | \\gamma_{S'} | b_t \\rangle$ is evaluated
classically via Wick's theorem on the diagonal Gaussian state $|b_t\\rangle$.
For a Gaussian state with Majorana covariance $C$, an even-degree product
expectation is

.. math::
    \\langle \\gamma_{S'} \\rangle_C = i^{|S'| / 2} \\operatorname{Pf}(C[S', S'])

(see :mod:`_matchgate` for the Pfaffian-of-restricted-covariance identity).
Odd-degree products have expectation zero in any Gaussian state. The full
per-shot estimator is

.. math::
    \\hat{\\langle \\gamma_S \\rangle}_t
    = \\frac{\\binom{2n}{2k}}{\\binom{n}{k}} \\sum_{S'} c_{S'} \\cdot i^{|S'|/2}
      \\operatorname{Pf}(C_{b_t}[S', S']),

where the sum is over the canonical sorted tuples appearing in the rotated
expansion. For ideally orthogonal $Q_t$ the rotated product preserves degree
exactly, so only $|S'| = 2k$ terms contribute (degree-shrinking cross-terms
vanish algebraically); numerically this is enforced up to floating-point
noise.

The per-shot estimator is bounded in magnitude by the shadow norm

.. math::
    \\lVert \\gamma_S \\rVert_{\\mathrm{FG}}^2 = \\binom{2n}{2k} / \\binom{n}{k}

(see :func:`_matchgate.fermionic_shadow_norm_pfaffian`). The Hoeffding radius
at confidence $1 - \\alpha$ is

.. math::
    r_{\\alpha}(\\gamma_S, M)
    = \\sqrt{\\frac{2 \\lVert \\gamma_S \\rVert_{\\mathrm{FG}}^2 \\log(2/\\alpha)}{M}}.

References
----------
Zhao, A., Rubin, N. C., Miyake, A. (2021). *Fermionic Partial Tomography via
Classical Shadows.* Phys. Rev. Lett. 127, 110504. Equations (4)-(6).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log, sqrt
from typing import Iterable

import numpy as np

from ._matchgate import (
    apply_matchgate_rotation_to_majorana_product,
    fermionic_shadow_norm_pfaffian,
    pfaffian,
    sample_matchgate_rotation,
)


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass
class MatchgateShadowRecord:
    """Per-shot matchgate-shadow data for $M$ shots on $n$ fermionic modes.

    Attributes
    ----------
    rotations : np.ndarray
        Real-valued array of shape ``(M, 2*n_modes, 2*n_modes)``. Each
        ``rotations[t]`` is an orthogonal matrix $Q_t \\in O(2n)$ sampled at
        shot $t$. Orthogonality is verified in :meth:`__post_init__` to a
        tolerance of ``1e-9`` on $\\lVert Q Q^T - I \\rVert_F$.
    outcomes : np.ndarray
        Integer array of shape ``(M, n_modes)``. Each ``outcomes[t]`` is a
        computational-basis bitstring $b_t \\in \\{0, 1\\}^n$.
    n_modes : int
        Number of fermionic modes $n$. Under Jordan-Wigner this equals the
        number of qubits.

    Notes
    -----
    The record is the minimal raw data needed by the inverse channel; it
    contains no derived moment estimates. Downstream protocols consume it via
    :func:`matchgate_inverse_channel_majorana_moments`.
    """

    rotations: np.ndarray
    outcomes: np.ndarray
    n_modes: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_modes, (int, np.integer)) or self.n_modes < 1:
            raise ValueError(
                f"n_modes must be a positive integer; got {self.n_modes!r}"
            )
        self.n_modes = int(self.n_modes)
        rot = np.asarray(self.rotations, dtype=np.float64)
        if rot.ndim != 3:
            raise ValueError(
                f"rotations must be a 3D array (M, 2n, 2n); got shape {rot.shape}"
            )
        m, d1, d2 = rot.shape
        dim = 2 * self.n_modes
        if d1 != dim or d2 != dim:
            raise ValueError(
                f"rotations shape ({m}, {d1}, {d2}) inconsistent with n_modes="
                f"{self.n_modes} (expected {dim}x{dim} per shot)"
            )
        # Verify orthogonality of each rotation.
        eye = np.eye(dim)
        for t in range(m):
            q = rot[t]
            err = np.linalg.norm(q @ q.T - eye)
            if err > 1e-9:
                raise ValueError(
                    f"rotation at shot {t} is not orthogonal: "
                    f"||Q Q^T - I||_F = {err:.3e}"
                )
        self.rotations = rot

        out = np.asarray(self.outcomes)
        if out.ndim != 2:
            raise ValueError(
                f"outcomes must be a 2D array (M, n); got shape {out.shape}"
            )
        if out.shape[0] != m or out.shape[1] != self.n_modes:
            raise ValueError(
                f"outcomes shape {out.shape} inconsistent with M={m}, "
                f"n_modes={self.n_modes}"
            )
        # Validate binary entries.
        if not np.issubdtype(out.dtype, np.integer):
            # Allow conversion from numeric arrays that contain {0, 1}.
            if not np.all(np.isin(out, [0, 1])):
                raise ValueError(
                    "outcomes contains non-binary entries; must be in {0, 1}"
                )
            out = out.astype(np.int64)
        if not np.all(np.isin(out, [0, 1])):
            raise ValueError(
                "outcomes contains non-binary entries; must be in {0, 1}"
            )
        self.outcomes = out

    @property
    def n_shots(self) -> int:
        return int(self.rotations.shape[0])


# ---------------------------------------------------------------------------
# Synthetic shadow generation
# ---------------------------------------------------------------------------


def _bitstring_covariance(bitstring: np.ndarray) -> np.ndarray:
    """Diagonal-block Majorana covariance for a computational-basis state.

    Parameters
    ----------
    bitstring : np.ndarray
        Integer array of shape ``(n,)`` with entries in $\\{0, 1\\}$.

    Returns
    -------
    np.ndarray
        Real skew-symmetric $(2n) \\times (2n)$ matrix with $(C_b)_{2p, 2p+1}
        = 1 - 2 b_p$ and $(C_b)_{2p+1, 2p} = -(1 - 2 b_p)$.
    """
    n = bitstring.shape[0]
    C = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for p in range(n):
        s = 1.0 - 2.0 * float(bitstring[p])
        C[2 * p, 2 * p + 1] = s
        C[2 * p + 1, 2 * p] = -s
    return C


def _sample_bitstring_from_covariance(
    C: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sequential conditional sampler from a Gaussian-state covariance.

    Implements the textbook Wick-conditioning recursion: at step $p$ the
    marginal probability of $b_p = 0$ given prior bits $b_0, \\ldots, b_{p-1}$
    is $(1 + C^{(p)}_{0, 1}) / 2$, where $C^{(p)}$ is the conditional Majorana
    covariance on modes $p, p+1, \\ldots, n - 1$ after measuring modes
    $0, \\ldots, p - 1$. The conditional covariance update uses the Schur-
    complement-like formula derived from Wick's theorem on a Gaussian state:

    .. math::
        C'_{ab} = C_{ab} + r (C_{0,a} C_{1,b} - C_{0,b} C_{1,a}),

    with $r = (2 b_p - 1) / (1 - (2 b_p - 1) C_{01})$, and then dropping the
    first two rows/columns to descend to the remaining $(n - p - 1)$ modes.

    Parameters
    ----------
    C : np.ndarray
        Real skew-symmetric $(2n) \\times (2n)$ matrix; the input Gaussian
        state's Majorana covariance.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Integer bitstring of shape ``(n,)``.
    """
    cov = np.array(C, dtype=np.float64, copy=True)
    dim = cov.shape[0]
    n = dim // 2
    bits = np.zeros(n, dtype=np.int64)
    for p in range(n):
        c01 = cov[0, 1]
        # Clip for numerical safety; |C_{01}| <= 1 holds in exact arithmetic.
        c01 = float(np.clip(c01, -1.0, 1.0))
        p0 = 0.5 * (1.0 + c01)
        p0 = float(np.clip(p0, 0.0, 1.0))
        b = 0 if rng.random() < p0 else 1
        bits[p] = b
        # Condition on outcome and drop the measured mode.
        if p < n - 1:
            s = 2 * b - 1  # s = -1 for b=0, +1 for b=1
            denom = 1.0 - s * c01
            if abs(denom) < 1e-14:
                # The outcome we sampled has near-zero probability; just drop
                # the mode (the conditional covariance is degenerate but the
                # remaining marginals are well-defined via continuity).
                cov = cov[2:, 2:]
                continue
            r = s / denom
            # Outer-product update on rows 2..end, cols 2..end.
            row0 = cov[0, 2:]
            row1 = cov[1, 2:]
            update = r * (np.outer(row0, row1) - np.outer(row1, row0))
            new_cov = cov[2:, 2:] + update
            # Enforce exact skew-symmetry against round-off.
            new_cov = 0.5 * (new_cov - new_cov.T)
            cov = new_cov
    return bits


def generate_matchgate_shadow_record(
    state_majorana_covariance: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> MatchgateShadowRecord:
    """Generate a synthetic matchgate-shadow record from a Gaussian state.

    For each of ``n_shots`` independent shots, samples a Haar-uniform
    $Q_t \\in O(2n)$, rotates the input Majorana covariance as
    $C_t = Q_t C Q_t^T$, and samples a computational-basis outcome
    $b_t \\in \\{0, 1\\}^n$ from the marginal distribution of the rotated
    Gaussian state.

    Parameters
    ----------
    state_majorana_covariance : np.ndarray
        Real skew-symmetric $(2n) \\times (2n)$ matrix $C$. Must satisfy
        $C^T C \\preceq I$ for a valid Gaussian state (not strictly
        verified beyond skew-symmetry to within tolerance).
    n_shots : int
        Number of shots $M$ in the synthetic record. Must be positive.
    rng : np.random.Generator

    Returns
    -------
    MatchgateShadowRecord
        Record with ``rotations.shape == (M, 2n, 2n)`` and
        ``outcomes.shape == (M, n)``.
    """
    if n_shots < 1:
        raise ValueError(f"n_shots must be positive; got {n_shots}")
    C = np.asarray(state_majorana_covariance, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(
            f"state_majorana_covariance must be square; got shape {C.shape}"
        )
    dim = C.shape[0]
    if dim % 2 != 0:
        raise ValueError(
            f"state_majorana_covariance dim {dim} must be even (= 2 * n_modes)"
        )
    if np.linalg.norm(C + C.T) > 1e-9:
        raise ValueError("state_majorana_covariance is not skew-symmetric")
    n_modes = dim // 2

    rotations = np.empty((n_shots, dim, dim), dtype=np.float64)
    outcomes = np.empty((n_shots, n_modes), dtype=np.int64)
    for t in range(n_shots):
        Q = sample_matchgate_rotation(n_modes, rng)
        # Rotated state covariance under Heisenberg-row Majorana convention.
        C_t = Q @ C @ Q.T
        # Enforce skew-symmetry against round-off.
        C_t = 0.5 * (C_t - C_t.T)
        b = _sample_bitstring_from_covariance(C_t, rng)
        rotations[t] = Q
        outcomes[t] = b
    return MatchgateShadowRecord(
        rotations=rotations, outcomes=outcomes, n_modes=n_modes
    )


# ---------------------------------------------------------------------------
# Inverse channel
# ---------------------------------------------------------------------------


def _gaussian_majorana_product_expectation(
    cov_b: np.ndarray, indices: tuple[int, ...]
) -> complex:
    """Wick expectation of $\\gamma_S$ on a Gaussian state with covariance ``cov_b``.

    For an even-degree product $\\gamma_S = \\gamma_{j_1} \\cdots \\gamma_{j_{2k}}$
    with indices in strictly-sorted order,

    .. math::
        \\langle \\gamma_S \\rangle = i^k \\operatorname{Pf}(C[S, S]).

    Odd-degree (including length-zero, treated separately) returns 0; the
    empty product returns 1 by convention.
    """
    deg = len(indices)
    if deg == 0:
        return 1.0 + 0j
    if deg % 2 != 0:
        return 0.0 + 0j
    k = deg // 2
    sub = cov_b[np.ix_(indices, indices)]
    pf = pfaffian(sub)
    return (1j) ** k * pf


def _pfaffian_batch(A: np.ndarray) -> np.ndarray:
    """Batched Pfaffian of skew-symmetric matrices.

    Computes $\\operatorname{Pf}(A_b)$ for every batch element of an array
    ``A`` with shape ``(..., d, d)`` and ``d`` even. Skew-symmetry
    $A_b^T = -A_b$ is assumed (not verified).

    Algorithmically identical to the scalar :func:`pfaffian` -- Parlett-Reid
    skew-LTL decomposition with partial pivoting (Wimmer 2012, ACM TOMS 38,
    30, Algorithm 2) -- but every step is broadcast over the leading batch
    axis using NumPy vectorized indexing rather than a Python loop. The
    per-iteration cost is $O(B \\cdot d^2)$ array work on the trailing
    $d \\times d$ block; for the matchgate-shadow application
    $d = 2k \\le 8$ so the inner cost is essentially constant and the
    dominant runtime is the per-call NumPy overhead amortized across $B$.

    Parameters
    ----------
    A : np.ndarray
        Real-valued skew-symmetric matrix batch, shape ``(..., d, d)``.
        ``d`` must be even.

    Returns
    -------
    np.ndarray
        Pfaffian values, shape ``(...)``. Returns ``1.0`` for the empty
        $0\\times 0$ block (consistent with the scalar Pfaffian convention).

    Notes
    -----
    The function makes a single defensive ``copy=True`` on input so the
    elimination can write in place. For an $M \\times d \\times d$ batch
    this allocates $M d^2 \\cdot 8$ bytes; at $M=10^6$, $d=8$ that is
    $0.5\\,\\text{GB}$ of scratch, which is comfortable for the intended
    chemistry-catalog regime.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    shape = A.shape
    if A.ndim < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            f"_pfaffian_batch: trailing dims must be square; got shape {shape}"
        )
    n = shape[-1]
    batch_shape = shape[:-2]
    if n % 2 == 1:
        return np.zeros(batch_shape, dtype=np.float64)
    if n == 0:
        return np.ones(batch_shape, dtype=np.float64)
    # Flatten leading dims so we always operate on (B, n, n).
    A = A.reshape(-1, n, n)
    B = A.shape[0]
    if B == 0:
        return np.zeros(batch_shape, dtype=np.float64)
    pf_sign = np.ones(B, dtype=np.float64)
    pf_value = np.ones(B, dtype=np.float64)
    batch_idx = np.arange(B)
    row_idx = np.arange(n)[None, :]

    for k in range(0, n - 2, 2):
        # ---- Partial pivot on column k below row k ----
        # Find row q in (k+1, n) maximizing |A[b, q, k]| per batch element.
        col_abs = np.abs(A[:, k + 1 :, k])  # (B, n-k-1)
        q_local = np.argmax(col_abs, axis=-1)
        q = k + 1 + q_local  # (B,)

        # Vectorized row swap (k+1 <-> q) per batch element.
        row_kp1 = A[batch_idx, k + 1, :].copy()
        row_q = A[batch_idx, q, :].copy()
        A[batch_idx, k + 1, :] = row_q
        A[batch_idx, q, :] = row_kp1
        # Vectorized column swap (k+1 <-> q) per batch element.
        col_kp1 = A[:, :, k + 1].copy()
        col_q = A[batch_idx[:, None], row_idx, q[:, None]]
        A[:, :, k + 1] = col_q
        A[batch_idx[:, None], row_idx, q[:, None]] = col_kp1
        # Sign flip if a swap happened.
        pf_sign = pf_sign * np.where(q != k + 1, -1.0, 1.0)

        # ---- Zero-pivot detection ----
        pivot = A[:, k + 1, k]
        zero_pivot = np.abs(pivot) < 1e-300
        # Mark Pf = 0 for any batch element that has a zero pivot, but keep
        # arithmetic safe by faking a pivot of 1.
        safe_pivot = np.where(zero_pivot, 1.0, pivot)
        pf_value = pf_value * np.where(zero_pivot, 0.0, 1.0)

        # ---- Skew-symmetric rank-2 elimination ----
        if k + 2 < n:
            tau = A[:, k + 2 :, k] / safe_pivot[:, None]  # (B, n-k-2)
            row_kplus1 = A[:, k + 1, k + 2 :].copy()  # (B, n-k-2)
            sub = A[:, k + 2 :, k + 2 :]
            outer1 = tau[:, :, None] * row_kplus1[:, None, :]
            outer2 = row_kplus1[:, :, None] * tau[:, None, :]
            A[:, k + 2 :, k + 2 :] = sub - (outer1 - outer2)
            A[:, k + 2 :, k] = 0.0
            A[:, k, k + 2 :] = 0.0

    # Pfaffian = product of super-diagonal pivots, signed.
    for i in range(0, n, 2):
        pf_value = pf_value * A[:, i, i + 1]
    return (pf_sign * pf_value).reshape(batch_shape)


def _bitstring_covariance_batch(bitstrings: np.ndarray) -> np.ndarray:
    """Vectorized block-diagonal Majorana covariance for a batch of bitstrings.

    Builds the same block-diagonal $\\pm 1$ covariance as
    :func:`_bitstring_covariance` for an array of shape ``(M, n)``,
    returning shape ``(M, 2n, 2n)``.
    """
    bits = np.asarray(bitstrings, dtype=np.int64)
    if bits.ndim != 2:
        raise ValueError(
            f"_bitstring_covariance_batch: expected 2D (M, n); got shape {bits.shape}"
        )
    M, n = bits.shape
    s = 1.0 - 2.0 * bits.astype(np.float64)  # (M, n)
    C = np.zeros((M, 2 * n, 2 * n), dtype=np.float64)
    diag_idx = np.arange(n)
    C[:, 2 * diag_idx, 2 * diag_idx + 1] = s
    C[:, 2 * diag_idx + 1, 2 * diag_idx] = -s
    return C


def matchgate_inverse_channel_majorana_moments(
    record: MatchgateShadowRecord,
    target_majorana_products: Iterable[tuple[int, ...]],
    *,
    alpha: float = 0.05,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Inverse-channel moment estimates from a matchgate-shadow record.

    Applies the Zhao-Rubin-Miyake (2021) closed-form inverse channel to each
    Majorana product in ``target_majorana_products`` and returns its
    empirical mean (averaged over all ``M`` shots) together with a certified
    Hoeffding radius at confidence $1 - \\alpha$.

    Parameters
    ----------
    record : MatchgateShadowRecord
        The synthetic or experimental shadow record.
    target_majorana_products : iterable of tuple of int
        Each element is a strictly-sorted tuple of distinct Majorana indices
        in ``[0, 2 * n_modes)``. The empty tuple represents the identity and
        is handled exactly (mean = 1, radius = 0). Odd-degree tuples are
        rejected; their expectation in any U(1)-invariant Gaussian state is
        zero but the inverse-channel normalization is undefined.
    alpha : float, default 0.05
        Failure probability per term for the Hoeffding radius. The library
        does **not** apply a Bonferroni correction over multiple terms; the
        caller is responsible for adjusting ``alpha`` to match the union
        bound they need downstream.

    Returns
    -------
    dict[tuple[int, ...], dict[str, float]]
        Mapping from each Majorana index tuple to
        ``{"mean": float, "radius": float}``. The mean is a real float
        (imaginary part is dropped after a numerical-sanity check; for any
        Majorana product $\\gamma_S$ with even $|S|$, $i^{|S|/2} \\gamma_S$
        is Hermitian and its expectation is real).

    Notes
    -----
    The returned mean is the **expectation of $\\gamma_S$ itself** (a pure
    Majorana product), not of any Hermitization. For a Majorana product
    $\\gamma_S$ with $|S| = 2k$, $\\langle \\gamma_S \\rangle$ is purely
    imaginary if $k$ is odd and purely real if $k$ is even. We return the
    real component when $k$ is even, and the imaginary component when $k$
    is odd, packaged as a float -- the caller is expected to know the
    expected reality structure of the Majorana product they queried.

    The radius is the Hoeffding bound

    .. math::
        r = \\sqrt{\\frac{2 \\lVert \\gamma_S \\rVert_{\\mathrm{FG}}^2
                          \\log(2 / \\alpha)}{M}}.

    Implementation
    --------------
    For each shot $t$, the per-shot estimator
    $\\hat{\\langle \\gamma_S \\rangle}_t = \\binom{2n}{2k} / \\binom{n}{k}
    \\cdot \\langle b_t | U_{Q_t} \\gamma_S U_{Q_t}^\\dagger | b_t \\rangle$
    is computed by:

    1. Forming the rotated product
       $U_{Q_t} \\gamma_S U_{Q_t}^\\dagger
       = \\sum_{S'} c_{S'}(Q_t^T) \\gamma_{S'}$ via
       :func:`apply_matchgate_rotation_to_majorana_product` applied to $Q_t^T$
       (Schrödinger column form).
    2. Summing $c_{S'} \\langle b_t | \\gamma_{S'} | b_t \\rangle$ over the
       expansion, with $\\langle b_t | \\gamma_{S'} | b_t \\rangle$ evaluated
       by Wick's theorem on the diagonal covariance $C_{b_t}$.
    3. Multiplying by the inverse-channel normalization
       $\\binom{2n}{2k} / \\binom{n}{k}$.

    Aggregation across shots is a plain arithmetic mean.

    See Also
    --------
    :func:`diagnostic.delta_ucb_from_majorana_moments` -- consumes a
    Majorana-moment dictionary in a different value format (``(complex,
    float)`` tuple per key). Wiring code to adapt the output of this
    function into that format is deferred to the next implementation pass
    (see the SCOPE.md entry on the ``ucb_matchgate_shadows`` wrapper).
    """
    # Speed-up mechanism (for future maintainers)
    # ------------------------------------------
    # The prior implementation iterated `M * len(targets)` Python-level
    # passes through :func:`apply_matchgate_rotation_to_majorana_product`,
    # each of which built up to `(2n)^{2k}` canonicalized Majorana terms in
    # a Python dict. At the chemistry-r4 catalog (~67 products, max degree
    # 2k = 8) with M = 10^4 shots that is ~6.7 x 10^5 dict-building loops,
    # each touching up to (2n)^{2k} = 8^8 = 1.7e7 raw terms -- the dominant
    # cost of the legacy pipeline.
    #
    # The vectorized path replaces that inner loop with the closed-form
    # identity
    #     <b_t | U_{Q_t} gamma_S U_{Q_t}^dagger | b_t>
    #         = i^k * Pf((Q_t^T C_{b_t} Q_t)[S, S]),
    # which is provably exactly equal to the dict-expansion answer (the
    # Pfaffian collects the same Gaussian Wick contractions, with the
    # rotation factored in through the Schrödinger-side covariance
    # transform C -> Q^T C Q). The new computational pattern is:
    #     1. Build the (M, 2n, 2n) rotated-covariance tensor in one batched
    #        matmul: Cb_tilde = Q^T @ Cb @ Q (per-shot rotation, batched
    #        with np.einsum).
    #     2. For each Majorana product S of degree 2k, slice the
    #        (M, 2k, 2k) sub-tensor Cb_tilde[:, S, S] and call the batched
    #        Pfaffian :func:`_pfaffian_batch`.
    #     3. Aggregate the per-shot estimator (norm * i^k * Pf) into a
    #        complex mean and project to the appropriate real component.
    # The per-shot estimator bound, shadow norm, and Hoeffding radius are
    # *unchanged*: this is a faithful reformulation of the same ZRM 2021
    # estimator, only the inner arithmetic differs.
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    M = record.n_shots
    if M < 1:
        raise ValueError("record has zero shots; cannot estimate moments")
    n = record.n_modes
    two_n = 2 * n

    targets = list(target_majorana_products)
    # Validate each target.
    for S in targets:
        if len(S) > 0:
            if not all(isinstance(j, (int, np.integer)) for j in S):
                raise ValueError(
                    f"target_majorana_products entries must be tuples of int; got {S!r}"
                )
            S_sorted = tuple(sorted(int(j) for j in S))
            if S_sorted != tuple(int(j) for j in S):
                raise ValueError(
                    f"target_majorana_products entries must be strictly-sorted; got {S!r}"
                )
            if len(set(S_sorted)) != len(S_sorted):
                raise ValueError(
                    f"target_majorana_products entries must have distinct indices; got {S!r}"
                )
            if any(j < 0 or j >= two_n for j in S_sorted):
                raise ValueError(
                    f"target index out of range [0, {two_n}); got {S!r}"
                )
            if len(S_sorted) % 2 != 0:
                raise ValueError(
                    f"target Majorana products must have even degree; got {S!r} "
                    "(odd-degree products have zero expectation but undefined "
                    "inverse-channel normalization)"
                )

    # --- Stage 1: batched rotated bitstring covariance ---
    # Build the (M, 2n, 2n) batch of computational-basis covariances in one
    # vectorized pass, then apply the Schrödinger-side rotation
    # C_b -> Q^T C_b Q per shot via einsum (no Python loop).
    bit_covs = _bitstring_covariance_batch(record.outcomes)  # (M, 2n, 2n)
    Q = record.rotations  # (M, 2n, 2n)
    # rotated_covs[t] = Q[t]^T @ bit_covs[t] @ Q[t].
    # einsum: 'tji,tjk,tkl->til' i.e. (Q.T C)_il = sum_j Q_ji C_jl, then
    # ((Q.T C) Q)_im = sum_l (Q.T C)_il Q_lm.
    rotated_covs = np.einsum("tji,tjk,tkl->til", Q, bit_covs, Q, optimize=True)
    # Enforce skew-symmetry against round-off (single broadcast op).
    rotated_covs = 0.5 * (rotated_covs - np.transpose(rotated_covs, (0, 2, 1)))

    results: dict[tuple[int, ...], dict[str, float]] = {}
    for S in targets:
        S = tuple(int(j) for j in S)
        if len(S) == 0:
            results[S] = {"mean": 1.0, "radius": 0.0}
            continue
        deg = len(S)
        k = deg // 2
        # Inverse-channel normalization C(2n, 2k) / C(n, k).
        # Use math.comb (exact int) and convert to float at the end; for the
        # ranges we care about (n <= 20, k <= 4) this avoids any overflow.
        norm = comb(two_n, deg) / comb(n, k)
        # Shadow-norm squared = same ratio; the Hoeffding range is shadow_norm.
        shadow_norm_sq = norm

        # --- Stage 2: batched Pfaffian on the (M, 2k, 2k) submatrix slice ---
        S_arr = np.asarray(S, dtype=np.intp)
        sub = rotated_covs[:, S_arr[:, None], S_arr[None, :]]  # (M, 2k, 2k)
        pf_vals = _pfaffian_batch(sub)  # (M,) real

        # Per-shot estimator <b_t|U_Q gamma_S U_Q^dag|b_t> = i^k * Pf((Q^T C_b Q)[S,S]).
        # Multiplying by the inverse-channel normalization yields the ZRM 2021
        # per-shot estimator. The aggregate is the empirical mean over shots.
        i_k = (1j) ** k  # complex scalar
        mean_pf = float(np.mean(pf_vals))
        mean_c = complex(norm * i_k * mean_pf)
        # Project to the expected reality: real if k even, imaginary if k odd.
        if k % 2 == 0:
            mean_val = float(mean_c.real)
        else:
            mean_val = float(mean_c.imag)

        # Hoeffding radius: r = sqrt(2 * shadow_norm^2 * log(2/alpha) / M).
        radius = sqrt(2.0 * shadow_norm_sq * log(2.0 / alpha) / M)
        # Sanity check: validates indices via the existing scalar helper.
        _ = fermionic_shadow_norm_pfaffian(S, n_modes=n)
        results[S] = {"mean": mean_val, "radius": radius}

    return results


def _matchgate_inverse_channel_unvectorized(
    record: MatchgateShadowRecord,
    target_majorana_products: Iterable[tuple[int, ...]],
    *,
    alpha: float = 0.05,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Reference (slow) inverse-channel implementation retained for parity tests.

    This is the legacy per-shot, per-product implementation that expands the
    rotated Majorana product into the canonical basis via
    :func:`apply_matchgate_rotation_to_majorana_product` and then evaluates
    each canonical term as $i^{|S'|/2}\\operatorname{Pf}(C_{b_t}[S', S'])$.
    It is **not** used by the public API; see
    :func:`matchgate_inverse_channel_majorana_moments` for the vectorized
    fast path. The two paths produce numerically identical results to within
    floating-point round-off.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    M = record.n_shots
    if M < 1:
        raise ValueError("record has zero shots; cannot estimate moments")
    n = record.n_modes
    two_n = 2 * n

    targets = list(target_majorana_products)
    for S in targets:
        if len(S) > 0:
            if not all(isinstance(j, (int, np.integer)) for j in S):
                raise ValueError(
                    f"target_majorana_products entries must be tuples of int; got {S!r}"
                )
            S_sorted = tuple(sorted(int(j) for j in S))
            if S_sorted != tuple(int(j) for j in S):
                raise ValueError(
                    f"target_majorana_products entries must be strictly-sorted; got {S!r}"
                )
            if len(set(S_sorted)) != len(S_sorted):
                raise ValueError(
                    f"target_majorana_products entries must have distinct indices; got {S!r}"
                )
            if any(j < 0 or j >= two_n for j in S_sorted):
                raise ValueError(
                    f"target index out of range [0, {two_n}); got {S!r}"
                )
            if len(S_sorted) % 2 != 0:
                raise ValueError(
                    f"target Majorana products must have even degree; got {S!r}"
                )

    bit_covs = np.empty((M, two_n, two_n), dtype=np.float64)
    for t in range(M):
        bit_covs[t] = _bitstring_covariance(record.outcomes[t])

    results: dict[tuple[int, ...], dict[str, float]] = {}
    for S in targets:
        S = tuple(int(j) for j in S)
        if len(S) == 0:
            results[S] = {"mean": 1.0, "radius": 0.0}
            continue
        deg = len(S)
        k = deg // 2
        norm = comb(two_n, deg) / comb(n, k)
        shadow_norm_sq = norm

        per_shot = np.empty(M, dtype=np.complex128)
        for t in range(M):
            Q = record.rotations[t]
            rotated = apply_matchgate_rotation_to_majorana_product(Q.T, S)
            acc = 0.0 + 0j
            cov_b = bit_covs[t]
            for S_prime, coeff in rotated.items():
                if not S_prime:
                    acc += coeff
                    continue
                exp_val = _gaussian_majorana_product_expectation(cov_b, S_prime)
                acc += coeff * exp_val
            per_shot[t] = norm * acc

        mean_c = complex(np.mean(per_shot))
        if k % 2 == 0:
            mean_val = float(mean_c.real)
        else:
            mean_val = float(mean_c.imag)

        radius = sqrt(2.0 * shadow_norm_sq * log(2.0 / alpha) / M)
        _ = fermionic_shadow_norm_pfaffian(S, n_modes=n)
        results[S] = {"mean": mean_val, "radius": radius}

    return results


__all__ = [
    "MatchgateShadowRecord",
    "generate_matchgate_shadow_record",
    "matchgate_inverse_channel_majorana_moments",
]
