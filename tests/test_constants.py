"""Integer-exact tests of the catalog constants against closed-form values."""

from __future__ import annotations

import pytest
from cumulant_residual_cert import Catalog, constants
from cumulant_residual_cert._partition import (
    B_charge_r,
    B_r,
    Bhat_charge_r,
    M_r,
    partitions_of_m,
)


def test_B_r_headline_value():
    """The headline universal constant for r = 4: B_4 = 105."""
    assert B_r(4) == 105


def test_B_r_undefined_below_3():
    with pytest.raises(ValueError):
        B_r(2)


def test_M_r_matches_recomputation_from_definition():
    """M_r matches a direct recomputation from the definition for r = 1..5."""
    from math import factorial

    expected = []
    for r in range(1, 6):
        best = 0
        for m in range(1, r + 1):
            total = sum(factorial(len(pi) - 1) for pi in partitions_of_m(m))
            best = max(best, total)
        expected.append(best)
    actual = [M_r(r) for r in range(1, 6)]
    assert actual == expected


def test_M_r_monotone_nondecreasing():
    from itertools import pairwise

    values = [M_r(r) for r in range(1, 7)]
    for a, b in pairwise(values):
        assert a <= b, values


def test_chemistry_catalog_charge_filtered_set():
    """B^charge_4(W) takes the set {1, 53, 105} over the chemistry catalog."""
    cat = Catalog.chemistry_r4()
    values = sorted({constants.charge_filtered(cat.r, w) for w in cat})
    assert values == [1, 53, 105]


def test_chemistry_catalog_block_refined_set():
    """B^hat^charge_4(W) takes the set {1, 3, 5} over the chemistry catalog."""
    cat = Catalog.chemistry_r4()
    values = sorted({constants.block_refined(cat.r, w) for w in cat})
    assert values == [1, 3, 5]


def test_block_refined_no_larger_than_charge_filtered():
    """For every catalog word, the block-refined constant tightens the charge-filtered one."""
    cat = Catalog.chemistry_r4()
    for w in cat:
        bc = constants.charge_filtered(cat.r, w)
        bh = constants.block_refined(cat.r, w)
        assert bh <= bc, f"{w.name}: block-refined {bh} > charge-filtered {bc}"


def test_charge_filtered_no_larger_than_universal():
    """Charge-filtered is no larger than universal for every catalog word."""
    cat = Catalog.chemistry_r4()
    Bu = constants.universal(cat.r)
    for w in cat:
        bc = constants.charge_filtered(cat.r, w)
        assert bc <= Bu, f"{w.name}: charge-filtered {bc} > universal {Bu}"


def test_per_word_values():
    """Exact per-word constants in canonical catalog order."""
    cat = Catalog.chemistry_r4()
    table = constants.compute(cat)
    expected = {
        "n_i n_j n_k":              (105, 1, 1),
        "a_dag_i a_j n_k":          (105, 1, 1),
        "a_dag_i a_j n_k n_ell":    (105, 53, 3),
        "a_dag_i a_dag_j a_k a_ell": (105, 1, 1),
        "n_i n_j n_k n_ell":        (105, 105, 5),
    }
    for name, (Bu, Bc, Bh) in expected.items():
        wc = table.per_word[name]
        assert (wc.universal, wc.charge_filtered, wc.block_refined) == (Bu, Bc, Bh), name


def test_compute_universal_consistent():
    """compute() reports the same universal constant as the standalone function."""
    cat = Catalog.chemistry_r4()
    table = constants.compute(cat)
    assert table.universal == constants.universal(cat.r)


def test_hierarchy_holds_for_synthetic_charges():
    """The bound hierarchy B_r >= B^charge_r >= B^hat^charge_r holds for ad-hoc charge patterns."""
    cases = [
        (3, (0, 0, 0)),
        (4, (1, 1, -1, -1)),
        (4, (1, -1, 0, 0)),
        (5, (1, -1, 1, -1, 0)),
    ]
    for r, ch in cases:
        Bu = B_r(r)
        Bc = B_charge_r(r, ch)
        Bh = Bhat_charge_r(r, ch)
        # Charge-filtered <= universal at the same r.
        assert Bc <= Bu, (r, ch, Bc, Bu)
        # Block-refined <= charge-filtered.
        assert Bh <= Bc, (r, ch, Bh, Bc)
