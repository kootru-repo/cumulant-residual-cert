"""Matchgate-shadow primitive routines (Zhao-Rubin-Miyake, 2021).

This module provides the foundational numerical primitives for fermionic-
Gaussian (a.k.a. "strict matchgate") classical shadows in the framework of
Zhao, Rubin, and Miyake (2021). The four public entry points are:

- :func:`pfaffian` -- numerically stable Pfaffian via Householder
  tridiagonalization of a skew-symmetric matrix.
- :func:`sample_matchgate_rotation` -- Haar-uniform sampling of orthogonal
  rotations $Q \\in O(2n)$ acting on the $2n$ Majorana modes.
- :func:`apply_matchgate_rotation_to_majorana_product` -- propagation of a
  Majorana product $\\gamma_{i_1} \\cdots \\gamma_{i_{2k}}$ through a
  rotation $Q$, expanded into the canonical Majorana basis.
- :func:`fermionic_shadow_norm_pfaffian` -- the per-shot shadow-norm
  $\\lVert \\gamma_S \\rVert_{\\mathrm{FG}}$ for a degree-$2k$ Majorana
  product in the strict-matchgate ensemble.

Conventions
-----------
- Majorana indices in this module are **0-based** in the range
  ``[0, 2 * n_modes)`` to align with the rotation matrix row/column indices.
  This differs from the 1-based convention used elsewhere in the package
  (``_majorana.py``); the canonicalization routine in ``_majorana.py`` is
  index-agnostic and is reused here.
- The rotation acts on Majorana operators via the **Heisenberg** convention
  $\\gamma_i \\mapsto \\sum_j Q_{ij} \\gamma_j$, i.e. row-wise contraction of
  $Q$ with the $\\gamma$ vector. Equivalently, with $\\vec{\\gamma}$ a column
  vector, $\\vec{\\gamma}' = Q \\vec{\\gamma}$, and the Majorana covariance
  transforms as $\\Gamma \\mapsto Q \\Gamma Q^T$. This is the convention used
  in Zhao-Rubin-Miyake (2021) and the SCOPE.md declaration of the matchgate-
  shadow-diagnostic companion paper.
- Sampled rotations cover the **full** $O(2n)$, i.e. both connected
  components ($\\det = +1$ and $\\det = -1$). Restricting to the connected
  component $SO(2n)$ would force a parity choice that the matchgate-shadow
  framework does not require.

References
----------
Zhao, A., Rubin, N. C., Miyake, A. (2021).
*Fermionic Partial Tomography via Classical Shadows.*
Phys. Rev. Lett. 127, 110504. Equations (4)-(6) of that paper give the
per-shot variance scaling that underlies :func:`fermionic_shadow_norm_pfaffian`.
"""

from __future__ import annotations

from math import comb, sqrt

import numpy as np

from ._majorana import MajoranaDecomposition, _canonicalize_majorana_product


def pfaffian(M: np.ndarray) -> float:
    """Numerically stable Pfaffian of a skew-symmetric matrix.

    Uses Householder tridiagonalization to bring $M$ to skew-symmetric
    block-bidiagonal form, then reads off the Pfaffian as the product of
    the super-diagonal entries (with the appropriate Householder-determinant
    sign correction). This is more stable than $\\pm\\sqrt{\\det M}$ because
    it preserves the sign of the Pfaffian, which $\\det$ throws away.

    The implementation follows the algorithm of Wimmer (2012), *ACM Trans.
    Math. Softw.* 38, 30, Algorithm 1 (the "PfaffianH" routine).

    Parameters
    ----------
    M : np.ndarray
        Real or complex skew-symmetric square matrix of even dimension
        $2k$. The skew-symmetry $M^T = -M$ is assumed; it is not verified.

    Returns
    -------
    float
        $\\operatorname{Pf}(M)$. The return type is ``float`` if the input
        is real-valued; otherwise it is ``complex``. The function returns
        ``0.0`` if $M$ has odd dimension (since the Pfaffian of any
        odd-dimensional skew-symmetric matrix is zero).

    Identities
    ----------
    For a 2x2 skew-symmetric matrix $\\begin{pmatrix} 0 & a \\\\ -a & 0
    \\end{pmatrix}$, $\\operatorname{Pf} = a$.

    For any skew-symmetric $M$, $\\operatorname{Pf}(M)^2 = \\det(M)$.

    For any (real or complex) orthogonal $B$,
    $\\operatorname{Pf}(B M B^T) = \\det(B) \\cdot \\operatorname{Pf}(M)$.
    """
    A = np.array(M, dtype=np.complex128 if np.iscomplexobj(M) else np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"pfaffian: input must be a square matrix, got shape {A.shape}")
    n = A.shape[0]
    if n % 2 == 1:
        return 0.0
    if n == 0:
        return 1.0

    # Parlett-Reid skew-LTL decomposition with partial pivoting
    # (Wimmer 2012, ACM TOMS 38, 30, Algorithm 2). The matrix is reduced
    # to skew-symmetric block-tridiagonal form by a sequence of row/column
    # swaps and Gauss-style eliminations, all applied as A <- P A P^T so
    # skew-symmetry is preserved at every step. The Pfaffian of the final
    # block-tridiagonal form is the product of the super-diagonal entries
    # A[2i, 2i + 1]; the overall sign tracks the pivot transpositions.
    pf_sign = 1.0

    for k in range(0, n - 2, 2):
        # ---- Pivot step ----
        # Find row q in (k, n) maximizing |A[q, k]|; swap it with k + 1.
        col = A[k + 1 :, k]
        q_local = int(np.argmax(np.abs(col)))
        q = k + 1 + q_local
        if abs(A[q, k]) < 1e-300:
            # Entire remaining column k is zero -> Pfaffian is zero.
            return 0.0 if not np.iscomplexobj(M) else 0.0 + 0j
        if q != k + 1:
            # Swap rows k + 1 and q; swap columns k + 1 and q.
            # A single row+column swap on a skew matrix flips Pf sign.
            A[[k + 1, q], :] = A[[q, k + 1], :]
            A[:, [k + 1, q]] = A[:, [q, k + 1]]
            pf_sign = -pf_sign

        # ---- Elimination step ----
        # Pivot: A[k + 1, k]. Build the multipliers tau[r] = A[r, k]/pivot
        # for r in [k + 2, n). Apply the transform P A P^T where P is the
        # identity except P[r, k + 1] = -tau[r] for each r >= k + 2.
        # Effect on A: new A[r, c] = A[r, c] - tau[r] * A[k + 1, c]
        #                          - tau[c] * A[r, k + 1]  (only for r, c >= k + 2)
        # Because A is skew, A[r, k + 1] = -A[k + 1, r], so the formula
        # simplifies to a skew-symmetric rank-2 update.
        if k + 2 < n:
            pivot = A[k + 1, k]
            tau = A[k + 2 :, k] / pivot  # shape (n - k - 2,)
            # Row k + 1 contribution: vector of length (n - k - 2).
            row_kplus1 = A[k + 1, k + 2 :].copy()
            sub = A[k + 2 :, k + 2 :]
            # Update: sub[r, c] -= tau[r] * row_kplus1[c] - row_kplus1[r] * tau[c]
            sub -= np.outer(tau, row_kplus1) - np.outer(row_kplus1, tau)
            # Zero the eliminated entries.
            A[k + 2 :, k] = 0.0
            A[k, k + 2 :] = 0.0
            # Also update column k + 1: A[r, k + 1] -= tau[r] * A[k + 1, k + 1]
            # but diagonal is zero, so the (r, k + 1) entry is unchanged in
            # value -- however, the transform P A P^T also modifies entries
            # A[r, k + 1] for r >= k + 2 via the column transformation:
            #   new A[r, k + 1] = old A[r, k + 1] - tau[r] * A[k + 1, k + 1] = old A[r, k + 1]
            # since A[k + 1, k + 1] = 0. So no further update needed.

    # The matrix is now block-tridiagonal (skew). Pfaffian = product of
    # super-diagonal pivots, modulated by the accumulated swap sign.
    pf_value = A.dtype.type(1.0)
    for i in range(0, n, 2):
        pf_value = pf_value * A[i, i + 1]

    result = pf_sign * pf_value
    if np.iscomplexobj(M):
        return complex(result)
    return float(np.real(result))


def sample_matchgate_rotation(n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-uniform orthogonal matrix $Q \\in O(2n)$.

    Uses the QR-decomposition algorithm of Mezzadri (2007), *Notices Amer.
    Math. Soc.* 54, 592: draw a real Gaussian matrix $G$ with iid
    $\\mathcal{N}(0, 1)$ entries, factor $G = Q R$, and normalize the
    diagonal of $R$ to be positive by absorbing its sign into the columns
    of $Q$. The resulting $Q$ is Haar-uniform on $O(2n)$.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes $n$. The returned rotation acts on the
        $2n$ Majorana operators.
    rng : np.random.Generator
        Source of randomness (e.g. ``np.random.default_rng(seed)``).

    Returns
    -------
    np.ndarray
        Real-valued $(2n) \\times (2n)$ orthogonal matrix, Haar-uniformly
        distributed on the **full** orthogonal group $O(2n)$. Both
        connected components ($\\det Q = +1$ and $\\det Q = -1$) are
        sampled with equal probability, because the QR sign-normalization
        does not enforce a determinant sign.

    Notes
    -----
    To restrict to $SO(2n)$ (only $\\det = +1$), the caller would flip the
    sign of the last row of $Q$ whenever $\\det Q = -1$. This is **not**
    done here because the strict-matchgate shadow ensemble of Zhao-Rubin-
    Miyake (2021) is defined as Haar on the full $O(2n)$.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be a positive integer; got {n_modes}")
    dim = 2 * n_modes
    G = rng.standard_normal(size=(dim, dim))
    Q, R = np.linalg.qr(G)
    # Sign-normalize so the distribution is exactly Haar.
    d = np.diagonal(R)
    sign = np.sign(d)
    # np.sign returns 0 for exact zero; treat as +1 to avoid annihilating a column.
    sign = np.where(sign == 0, 1.0, sign)
    Q = Q * sign  # broadcast: multiply each column by the sign of its R-pivot
    return Q


def apply_matchgate_rotation_to_majorana_product(
    rotation: np.ndarray,
    majorana_indices: tuple[int, ...],
) -> MajoranaDecomposition:
    """Apply a matchgate rotation $Q$ to a Majorana product.

    The rotation acts as
    $\\gamma_i \\mapsto \\sum_j Q_{ij} \\gamma_j$
    (Heisenberg / row convention), so the rotated product is

    .. math::
        Q[\\gamma_{i_1} \\cdots \\gamma_{i_{2k}}]
        = \\sum_{j_1, \\ldots, j_{2k}}
          Q_{i_1, j_1} \\cdots Q_{i_{2k}, j_{2k}}
          \\gamma_{j_1} \\cdots \\gamma_{j_{2k}}.

    Each unordered Majorana product on the right-hand side is then
    canonicalized via the anti-commutation relations
    $\\gamma_a \\gamma_b = -\\gamma_b \\gamma_a$ (for $a \\ne b$) and
    $\\gamma_a^2 = 1$, using the canonicalizer from :mod:`_majorana`. The
    resulting decomposition has strictly-sorted index tuples as keys.

    Parameters
    ----------
    rotation : np.ndarray
        $(2n) \\times (2n)$ matrix acting on Majorana modes. Need not be
        orthogonal (e.g. for testing); but for shadow sampling it will be
        Haar-orthogonal from :func:`sample_matchgate_rotation`.
    majorana_indices : tuple[int, ...]
        Strictly-sorted tuple of distinct Majorana indices (0-based in
        ``[0, 2n)``) specifying the input product
        $\\gamma_{i_1} \\cdots \\gamma_{i_{2k}}$. The input is **not**
        required to have even length, but for charge-neutral / matchgate-
        evaluable products it will always be even.

    Returns
    -------
    dict[tuple[int, ...], complex]
        Mapping from each strictly-sorted Majorana-index tuple appearing
        in the rotated product to its complex coefficient. Numerically-
        zero entries (``abs < 1e-15``) are pruned.

    Notes
    -----
    The full expansion has $(2n)^{2k}$ raw terms. For small $k$ and
    moderate $n$ this is tractable; for shadow-shot processing it is
    typically called with $k \\le 4$ and $n \\le 16$. No attempt is made
    to exploit sparsity of the rotation here; callers that need a faster
    path can use Pfaffian-based estimators downstream.
    """
    Q = np.asarray(rotation)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"rotation must be a square matrix, got shape {Q.shape}")
    dim = Q.shape[0]
    for idx in majorana_indices:
        if not 0 <= idx < dim:
            raise ValueError(
                f"majorana index {idx} out of range for rotation of dim {dim}"
            )

    # Empty product: identity.
    if not majorana_indices:
        return {(): 1.0 + 0j}

    # Iterate over the Cartesian product of new-index choices for each
    # input gamma. Maintain a running decomposition keyed by canonical
    # sorted tuples.
    result: MajoranaDecomposition = {(): 1.0 + 0j}
    for i in majorana_indices:
        row = Q[i, :]  # Q_{i, j} for j in range(dim)
        new_result: MajoranaDecomposition = {}
        for prev_indices, prev_coeff in result.items():
            for j in range(dim):
                qij = row[j]
                if qij == 0:
                    continue
                merged = [*prev_indices, j]
                sign, canon = _canonicalize_majorana_product(merged)
                contribution = prev_coeff * complex(qij) * sign
                if canon in new_result:
                    new_result[canon] += contribution
                else:
                    new_result[canon] = contribution
        # Prune numerical zeros.
        result = {k: v for k, v in new_result.items() if abs(v) > 1e-15}
    return result


def fermionic_shadow_norm_pfaffian(
    majorana_indices: tuple[int, ...],
    n_modes: int,
) -> float:
    """Strict-matchgate shadow norm $\\lVert \\gamma_S \\rVert_{\\mathrm{FG}}$.

    For a degree-$2k$ Majorana product
    $\\gamma_S = \\gamma_{i_1} \\cdots \\gamma_{i_{2k}}$ with
    $i_1 < i_2 < \\cdots < i_{2k}$ and indices in ``[0, 2 * n_modes)``,
    the fermionic-Gaussian shadow norm in the Zhao-Rubin-Miyake (2021)
    ensemble is

    .. math::
        \\lVert \\gamma_S \\rVert_{\\mathrm{FG}}
        = \\sqrt{\\binom{2n}{2k} \\big/ \\binom{n}{k}}.

    This is the square-root of the per-shot estimator variance bound for
    a degree-$2k$ Majorana product under the strict-matchgate ensemble.
    The formula follows directly from Zhao-Rubin-Miyake (2021),
    equations (4)-(6) (closed-form inverse channel for the fermionic-
    Gaussian ensemble, evaluated on a degree-$2k$ Majorana monomial).
    The result depends only on the degree $2k$ and the number of modes
    $n$; it is independent of which specific indices appear in $S$
    (Haar-invariance under $O(2n)$).

    Parameters
    ----------
    majorana_indices : tuple[int, ...]
        Strictly-sorted tuple of distinct Majorana indices (0-based) in
        ``[0, 2 * n_modes)``. Length must be even and at most $2n$.
    n_modes : int
        Number of fermionic modes $n$. Must satisfy $n \\ge 1$.

    Returns
    -------
    float
        $\\lVert \\gamma_S \\rVert_{\\mathrm{FG}} \\ge 0$. Returns ``1.0``
        for the empty product (identity).

    Notes
    -----
    For degree $2$ (single Majorana bilinear), the shadow norm reduces to
    $\\sqrt{(2n)(2n - 1) / n} = \\sqrt{2(2n - 1)}$... no, more carefully:
    $\\binom{2n}{2} = n(2n - 1)$ and $\\binom{n}{1} = n$, so the ratio is
    $2n - 1$ and the norm is $\\sqrt{2n - 1}$.

    The formula extends to **all** even degrees $2k \\le 2n$. Beyond
    $2k = 2n$ the binomial $\\binom{n}{k}$ vanishes; the shadow norm is
    formally infinite, reflecting that degree-$2n$ Majorana monomials
    cannot be estimated by a finite-shot fermionic-Gaussian shadow.

    References
    ----------
    Zhao, Rubin, Miyake (2021), Eq. (4) defines the inverse channel; Eq.
    (5) gives the per-shot estimator; the shadow-norm bound follows from
    the variance computation immediately after Eq. (6).
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be a positive integer; got {n_modes}")
    two_n = 2 * n_modes
    for idx in majorana_indices:
        if not 0 <= idx < two_n:
            raise ValueError(
                f"majorana index {idx} out of range [0, {two_n}) for n_modes={n_modes}"
            )
    deg = len(majorana_indices)
    if deg == 0:
        return 1.0
    if deg % 2 != 0:
        raise ValueError(
            f"matchgate shadows estimate even-degree Majorana products only; got degree {deg}"
        )
    k = deg // 2
    if k > n_modes:
        raise ValueError(
            f"degree {deg} exceeds 2 * n_modes = {two_n}; shadow norm undefined"
        )
    numerator = comb(two_n, deg)
    denominator = comb(n_modes, k)
    if denominator == 0:
        raise ValueError(
            f"binomial denominator C({n_modes}, {k}) = 0; shadow norm undefined"
        )
    return sqrt(numerator / denominator)
