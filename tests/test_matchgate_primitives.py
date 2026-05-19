"""Tests for the matchgate-shadow primitive routines.

Covers:
- Pfaffian numerical identities (2x2 closed form, Pf^2 = det, orthogonal
  conjugation rule Pf(B M B^T) = det(B) Pf(M)).
- Haar-uniform sampling of $O(2n)$ rotations (orthogonality, both
  determinant sectors appear).
- Linear propagation of Majorana products through a rotation, including
  canonicalization correctness.
- Fermionic-Gaussian shadow norm formula at degree 2 and 4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cumulant_residual_cert._matchgate import (
    apply_matchgate_rotation_to_majorana_product,
    fermionic_shadow_norm_pfaffian,
    pfaffian,
    sample_matchgate_rotation,
)

# ----- Pfaffian -----


def test_pfaffian_2x2():
    """Pfaffian of [[0, a], [-a, 0]] = a."""
    for a in [1.0, -3.5, 0.0, 7.25]:
        M = np.array([[0.0, a], [-a, 0.0]])
        assert pfaffian(M) == pytest.approx(a)


def test_pfaffian_4x4_block_diagonal():
    """Pf(diag(block_a, block_b)) = a * b for 2x2 blocks [[0, a], [-a, 0]]."""
    a, b = 2.0, -3.0
    M = np.array(
        [
            [0.0, a, 0.0, 0.0],
            [-a, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, b],
            [0.0, 0.0, -b, 0.0],
        ]
    )
    assert pfaffian(M) == pytest.approx(a * b)


def test_pfaffian_odd_dim_returns_zero():
    """Skew-symmetric matrices of odd dimension have Pfaffian zero."""
    M = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]])
    assert pfaffian(M) == 0.0


def test_pfaffian_zero_dim_returns_one():
    """The empty product convention: Pf of the 0x0 matrix is 1."""
    M = np.zeros((0, 0))
    assert pfaffian(M) == pytest.approx(1.0)


@pytest.mark.parametrize("dim", [2, 4, 6, 8, 10])
def test_pfaffian_squared_equals_det(dim):
    """For random skew M of even dim, pfaffian(M)^2 ~ det(M)."""
    rng = np.random.default_rng(seed=20260519 + dim)
    A = rng.standard_normal(size=(dim, dim))
    M = A - A.T  # skew-symmetrize
    pf = pfaffian(M)
    det = float(np.linalg.det(M))
    assert pf * pf == pytest.approx(det, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dim", [2, 4, 6])
def test_pfaffian_orthogonal_invariance(dim):
    """pfaffian(B M B^T) = det(B) * pfaffian(M) for orthogonal B."""
    rng = np.random.default_rng(seed=42 + dim)
    A = rng.standard_normal(size=(dim, dim))
    M = A - A.T
    G = rng.standard_normal(size=(dim, dim))
    B, _ = np.linalg.qr(G)
    pf_M = pfaffian(M)
    pf_BMBt = pfaffian(B @ M @ B.T)
    det_B = float(np.linalg.det(B))
    assert pf_BMBt == pytest.approx(det_B * pf_M, rel=1e-9, abs=1e-9)


def test_pfaffian_orthogonal_invariance_with_sign_flip():
    """Explicitly construct B with det(B) = -1 and verify the sign flip."""
    rng = np.random.default_rng(seed=314)
    dim = 4
    A = rng.standard_normal(size=(dim, dim))
    M = A - A.T
    # Construct an orthogonal B with det = -1 by flipping a row of a Haar sample.
    G = rng.standard_normal(size=(dim, dim))
    B, _ = np.linalg.qr(G)
    if np.linalg.det(B) > 0:
        B = B.copy()
        B[-1, :] *= -1
    pf_M = pfaffian(M)
    pf_BMBt = pfaffian(B @ M @ B.T)
    assert pf_BMBt == pytest.approx(-pf_M, rel=1e-9, abs=1e-9)


def test_pfaffian_non_square_rejected():
    with pytest.raises(ValueError, match="square"):
        pfaffian(np.zeros((3, 4)))


# ----- Haar-uniform O(2n) sampling -----


@pytest.mark.parametrize("n_modes", [1, 2, 3, 4, 6])
def test_sample_matchgate_rotation_orthogonal(n_modes):
    """Q Q^T = I, dim is 2*n_modes."""
    rng = np.random.default_rng(seed=2026)
    Q = sample_matchgate_rotation(n_modes, rng)
    assert Q.shape == (2 * n_modes, 2 * n_modes)
    assert np.allclose(Q @ Q.T, np.eye(2 * n_modes), atol=1e-12)
    assert np.allclose(Q.T @ Q, np.eye(2 * n_modes), atol=1e-12)


def test_sample_matchgate_rotation_both_dets():
    """Over many samples, both det = +1 and det = -1 appear (full O(2n))."""
    rng = np.random.default_rng(seed=2025)
    n_modes = 3
    n_samples = 200
    pos = 0
    neg = 0
    for _ in range(n_samples):
        Q = sample_matchgate_rotation(n_modes, rng)
        d = np.linalg.det(Q)
        # Sanity: orthogonal => |det| = 1.
        assert abs(abs(d) - 1.0) < 1e-10
        if d > 0:
            pos += 1
        else:
            neg += 1
    # Each component should hold roughly half; require at least 30% of either.
    assert pos > n_samples * 0.3
    assert neg > n_samples * 0.3


def test_sample_matchgate_rotation_invalid_n():
    with pytest.raises(ValueError, match="positive"):
        sample_matchgate_rotation(0, np.random.default_rng(0))


# ----- Rotation application -----


def test_apply_matchgate_rotation_identity():
    """The identity rotation maps gamma_S to itself with coefficient 1."""
    dim = 8  # n_modes = 4
    Q = np.eye(dim)
    indices = (1, 3, 5, 6)
    out = apply_matchgate_rotation_to_majorana_product(Q, indices)
    # The canonical form of (1, 3, 5, 6) is itself (sorted distinct).
    assert set(out.keys()) == {indices}
    assert out[indices] == pytest.approx(1.0 + 0j)


def test_apply_matchgate_rotation_empty_product():
    """The empty Majorana product (identity operator) is fixed by every Q."""
    rng = np.random.default_rng(seed=7)
    Q = sample_matchgate_rotation(3, rng)
    out = apply_matchgate_rotation_to_majorana_product(Q, ())
    assert out == {(): 1.0 + 0j}


def test_apply_matchgate_rotation_single_gamma():
    """gamma_i -> sum_j Q_{i,j} gamma_j (Heisenberg row convention)."""
    rng = np.random.default_rng(seed=11)
    Q = sample_matchgate_rotation(2, rng)  # 4x4
    dim = 4
    i = 1
    out = apply_matchgate_rotation_to_majorana_product(Q, (i,))
    for j in range(dim):
        key = (j,)
        if abs(Q[i, j]) > 1e-15:
            assert out[key] == pytest.approx(Q[i, j] + 0j)
        else:
            assert key not in out or abs(out[key]) < 1e-15


def test_apply_matchgate_rotation_canonicalization():
    """A rotation that swaps two Majoranas picks up the anti-commutation sign.

    Take Q to be the permutation that swaps indices 0 and 1. Then
    gamma_0 gamma_1 -> gamma_1 gamma_0 = -gamma_0 gamma_1. After
    canonicalization the output should be {(0, 1): -1}, not {(1, 0): +1}.
    """
    dim = 4
    Q = np.eye(dim)
    # Swap rows 0 and 1 of Q. Since gamma_i -> sum_j Q_{i,j} gamma_j,
    # row 0 (now e_1) sends gamma_0 -> gamma_1, and row 1 (now e_0) sends
    # gamma_1 -> gamma_0.
    Q[[0, 1]] = Q[[1, 0]]
    out = apply_matchgate_rotation_to_majorana_product(Q, (0, 1))
    # Expected: gamma_0 gamma_1 -> gamma_1 gamma_0 = -gamma_0 gamma_1
    assert set(out.keys()) == {(0, 1)}
    assert out[(0, 1)] == pytest.approx(-1.0 + 0j)


def test_apply_matchgate_rotation_squared_indices_collapse():
    """A rotation that sends gamma_i and gamma_j to the same gamma collapses via gamma^2 = 1.

    With the row convention gamma_i -> sum_j Q_{i,j} gamma_j, rows 0 and 1 are
    both set to e_0 so gamma_0 -> gamma_0 and gamma_1 -> gamma_0. This is not
    orthogonal but it tests canonicalization for repeated indices. Then
    gamma_0 gamma_1 -> gamma_0 gamma_0 = 1 (empty product).
    """
    dim = 4
    Q = np.zeros((dim, dim))
    Q[0, 0] = 1.0
    Q[1, 0] = 1.0
    Q[2, 1] = 1.0
    Q[3, 2] = 1.0
    out = apply_matchgate_rotation_to_majorana_product(Q, (0, 1))
    assert set(out.keys()) == {()}
    assert out[()] == pytest.approx(1.0 + 0j)


def test_apply_matchgate_rotation_preserves_degree_parity():
    """Even-degree Majorana products map to even-degree products under any rotation."""
    rng = np.random.default_rng(seed=99)
    Q = sample_matchgate_rotation(3, rng)
    out = apply_matchgate_rotation_to_majorana_product(Q, (0, 2, 3, 5))
    for indices in out:
        assert len(indices) % 2 == 0


def test_apply_matchgate_rotation_index_out_of_range():
    Q = np.eye(4)
    with pytest.raises(ValueError, match="out of range"):
        apply_matchgate_rotation_to_majorana_product(Q, (0, 5))


def test_apply_matchgate_rotation_composition_consistency():
    """Applying Q1 then Q2 equals applying (Q1 @ Q2) to the same gamma.

    With the row convention gamma_i -> sum_j Q[i, j] gamma_j, applying Q1
    then Q2 sends gamma_i to sum_{j, k} Q1[i, j] Q2[j, k] gamma_k, with
    coefficient (Q1 @ Q2)[i, k] on gamma_k.
    """
    rng = np.random.default_rng(seed=2026519)
    n_modes = 2
    Q1 = sample_matchgate_rotation(n_modes, rng)
    Q2 = sample_matchgate_rotation(n_modes, rng)
    dim = 2 * n_modes
    i = 1
    # Apply Q1 first.
    step1 = apply_matchgate_rotation_to_majorana_product(Q1, (i,))
    # Apply Q2 to each result (linearly).
    final: dict[tuple[int, ...], complex] = {}
    for indices, coeff in step1.items():
        rotated = apply_matchgate_rotation_to_majorana_product(Q2, indices)
        for new_idx, new_coeff in rotated.items():
            final[new_idx] = final.get(new_idx, 0.0 + 0j) + coeff * new_coeff
    # Composed rotation Q1 @ Q2 applied directly.
    Qcomp = Q1 @ Q2
    direct = apply_matchgate_rotation_to_majorana_product(Qcomp, (i,))
    for j in range(dim):
        key = (j,)
        a = final.get(key, 0.0 + 0j)
        b = direct.get(key, 0.0 + 0j)
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9)


# ----- Shadow norm -----


def test_shadow_norm_empty_product():
    """The identity (degree 0) has shadow norm 1."""
    assert fermionic_shadow_norm_pfaffian((), n_modes=4) == pytest.approx(1.0)


def test_shadow_norm_degree_2():
    """||gamma_i gamma_j||_FG = sqrt(binom(2n, 2) / binom(n, 1)) = sqrt(2n - 1)."""
    for n_modes in [1, 2, 3, 5, 10]:
        # Use a representative degree-2 product.
        norm = fermionic_shadow_norm_pfaffian((0, 1), n_modes=n_modes)
        expected = math.sqrt(2 * n_modes - 1)
        assert norm == pytest.approx(expected)


def test_shadow_norm_degree_4_at_n4():
    """At n_modes=4, degree-4 product: sqrt(binom(8, 4) / binom(4, 2)) = sqrt(70 / 6)."""
    norm = fermionic_shadow_norm_pfaffian((0, 1, 2, 3), n_modes=4)
    expected = math.sqrt(70.0 / 6.0)
    assert norm == pytest.approx(expected)


def test_shadow_norm_degree_4_at_n6():
    """At n_modes=6, degree-4 product: sqrt(binom(12, 4) / binom(6, 2)) = sqrt(495 / 15) = sqrt(33)."""
    norm = fermionic_shadow_norm_pfaffian((0, 1, 4, 7), n_modes=6)
    expected = math.sqrt(495.0 / 15.0)
    assert norm == pytest.approx(expected)
    assert expected == pytest.approx(math.sqrt(33.0))


def test_shadow_norm_index_independent():
    """The shadow norm depends only on the degree, not on which indices appear."""
    n_modes = 5
    norm_a = fermionic_shadow_norm_pfaffian((0, 1, 2, 3), n_modes=n_modes)
    norm_b = fermionic_shadow_norm_pfaffian((1, 4, 6, 9), n_modes=n_modes)
    assert norm_a == pytest.approx(norm_b)


def test_shadow_norm_odd_degree_rejected():
    with pytest.raises(ValueError, match="even-degree"):
        fermionic_shadow_norm_pfaffian((0, 1, 2), n_modes=3)


def test_shadow_norm_index_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        fermionic_shadow_norm_pfaffian((0, 99), n_modes=2)


def test_shadow_norm_full_degree():
    """Top-degree product gamma_1 ... gamma_{2n}: norm = sqrt(binom(2n, 2n) / binom(n, n)) = 1."""
    n_modes = 3
    indices = tuple(range(2 * n_modes))
    norm = fermionic_shadow_norm_pfaffian(indices, n_modes=n_modes)
    assert norm == pytest.approx(1.0)


def test_shadow_norm_degree_exceeds_modes_rejected():
    """Cannot have a degree exceeding 2 * n_modes."""
    # Construct a product with degree > 2n by allowing repeated indices? No -- indices
    # must be distinct and in range. The only way to exceed is by trying a too-long
    # tuple that fails the in-range check first.
    with pytest.raises(ValueError, match="out of range"):
        fermionic_shadow_norm_pfaffian(tuple(range(10)), n_modes=2)
