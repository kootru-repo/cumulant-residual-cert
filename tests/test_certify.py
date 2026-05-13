"""Tests for the one-call certify() API."""

from __future__ import annotations

import pytest

from cumulant_residual_cert import Catalog, certify


def test_certify_with_zero_delta_returns_zero_bounds():
    """Bernoulli class: $\\Delta = 0$ proves every bound is exactly zero."""
    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=0.0)
    assert result.delta == 0.0
    assert all(bar == 0.0 for bar in result.bounds.values())


def test_certify_block_refined_default():
    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=1.0)
    assert result.level == "block_refined"
    # The block-refined set on this catalog is {1, 1, 3, 1, 5}.
    assert sorted(result.constants_used.values()) == [1, 1, 1, 3, 5]


def test_certify_levels_strictly_widen():
    """Going from block-refined -> charge-filtered -> universal never tightens."""
    cat = Catalog.chemistry_r4()
    bh = certify(cat, delta=1.0, level="block_refined")
    bc = certify(cat, delta=1.0, level="charge_filtered")
    bu = certify(cat, delta=1.0, level="universal")
    for w in cat:
        assert bh.bounds[w.name] <= bc.bounds[w.name]
        assert bc.bounds[w.name] <= bu.bounds[w.name]


def test_certify_scales_linearly_in_delta():
    cat = Catalog.chemistry_r4()
    r1 = certify(cat, delta=0.01)
    r2 = certify(cat, delta=0.05)
    for w in cat:
        assert r2.bounds[w.name] == pytest.approx(5 * r1.bounds[w.name])


def test_certify_rejects_negative_delta():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        certify(cat, delta=-0.1)


def test_certify_universal_matches_b_r():
    """At level='universal' every word's constant is B_r."""
    from cumulant_residual_cert import constants

    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=1.0, level="universal")
    Bu = constants.universal(cat.r)
    assert all(c == Bu for c in result.constants_used.values())
