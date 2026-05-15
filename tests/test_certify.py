"""Tests for the one-call certify() API."""

from __future__ import annotations

import math

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


def test_certify_rejects_nan_delta():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        certify(cat, delta=float("nan"))


def test_certify_rejects_inf_delta():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        certify(cat, delta=math.inf)


def test_certify_rejects_unknown_level():
    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError):
        certify(cat, delta=0.01, level="not_a_level")  # type: ignore[arg-type]


def test_certify_universal_matches_b_r():
    """At level='universal' every word's constant is B_r."""
    from cumulant_residual_cert import constants

    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=1.0, level="universal")
    Bu = constants.universal(cat.r)
    assert all(c == Bu for c in result.constants_used.values())


# ----- provenance + persisted-certificate fields (v0.5) -----


def test_certify_defaults_provenance_to_user_supplied():
    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=0.01)
    assert result.delta_provenance == "user_supplied"


def test_certify_passes_through_caller_provenance():
    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=0.01, delta_provenance="ucb_random_pauli")
    assert result.delta_provenance == "ucb_random_pauli"


def test_certify_captures_catalog_name():
    cat = Catalog.chemistry_r4()
    assert cat.name == "chemistry_r4"
    result = certify(cat, delta=0.01)
    assert result.catalog_name == "chemistry_r4"


def test_certify_records_library_version():
    """library_version is populated from importlib.metadata at construction time."""
    cat = Catalog.chemistry_r4()
    result = certify(cat, delta=0.01)
    # In an installed environment the version matches the package metadata;
    # in a not-installed-from-source environment it falls back to "unknown".
    assert result.library_version, "library_version must be a non-empty string"


def test_certified_bound_round_trips_through_asdict_and_json():
    """The certificate dataclass serializes losslessly via stdlib only."""
    import json
    from dataclasses import asdict

    cat = Catalog.chemistry_r4()
    result = certify(
        cat,
        delta=0.012,
        level="block_refined",
        delta_provenance="from_rdms",
    )
    payload = asdict(result)
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    # Spot-check the structural fields a downstream auditor would read.
    assert decoded["delta"] == 0.012
    assert decoded["level"] == "block_refined"
    assert decoded["delta_provenance"] == "from_rdms"
    assert decoded["catalog_name"] == "chemistry_r4"
    assert set(decoded["bounds"]) == {w.name for w in cat}
    assert isinstance(decoded["library_version"], str) and decoded["library_version"]


def test_pyscf_from_mean_field_records_closed_form_bernoulli_provenance():
    """The PySCF Bernoulli helper labels its certificate honestly."""
    pytest.importorskip("pyscf", reason="PySCF not installed")
    from cumulant_residual_cert.adapters.pyscf import from_mean_field
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    cat = Catalog.chemistry_r4()
    est = from_mean_field(mf, cat, user_asserts_bernoulli_class=True)
    assert est.bound.delta_provenance == "closed_form_bernoulli"
    assert est.bound.catalog_name == "chemistry_r4"
