"""Golden-table sync: regenerated constants match the shipped JSON and the audit repo.

The shipped ``chemistry_catalog_r4.json`` is treated as a frozen artefact within
each release. This test verifies:

1. A fresh computation from :func:`cumulant_residual_cert.constants.compute`
   reproduces the shipped JSON exactly.
2. If the environment variable ``AUDIT_REPO_PATH`` points at a checkout of
   ``charge-filtered-cumulant-residuals``, the values also match that
   repository's own computation. CI sets ``AUDIT_REPO_PATH`` to enforce drift
   detection; locally, the audit cross-check is skipped if the repo is not
   present.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from cumulant_residual_cert import Catalog, constants


def test_golden_json_matches_freshly_computed():
    golden = constants.load_golden()
    cat = Catalog.chemistry_r4()
    table = constants.compute(cat)

    assert golden["r"] == table.r
    assert golden["universal"] == table.universal
    assert golden["catalog"] == cat.name

    by_name = {entry["name"]: entry for entry in golden["words"]}
    # Word set must match exactly (no missing or extra entries).
    assert set(by_name) == {w.name for w in cat}
    assert len(golden["words"]) == len(cat)
    # Ordering must match too, so downstream readers that iterate by index
    # see the same word at the same position.
    assert [entry["name"] for entry in golden["words"]] == [w.name for w in cat]

    for w in cat:
        entry = by_name[w.name]
        wc = table.per_word[w.name]
        assert list(w.letters) == entry["letters"]
        assert list(w.charges) == entry["charges"]
        assert entry["B_universal"] == wc.universal
        assert entry["B_charge_filtered"] == wc.charge_filtered
        assert entry["B_block_refined"] == wc.block_refined


def test_golden_json_universal_is_consistent_across_words():
    """All per-word B_universal entries equal the top-level universal constant."""
    golden = constants.load_golden()
    assert all(entry["B_universal"] == golden["universal"] for entry in golden["words"])


def test_golden_json_b_charge_set_is_canonical():
    """The set of charge-filtered constants on this catalog is {1, 53, 105}."""
    golden = constants.load_golden()
    bc = sorted({entry["B_charge_filtered"] for entry in golden["words"]})
    assert bc == [1, 53, 105]


def test_golden_json_bhat_charge_set_is_canonical():
    """The set of block-refined constants on this catalog is {1, 3, 5}."""
    golden = constants.load_golden()
    bh = sorted({entry["B_block_refined"] for entry in golden["words"]})
    assert bh == [1, 3, 5]


def test_audit_repo_cross_check():
    """Cross-check the constants against the audit repo's independent enumeration."""
    audit_path_env = os.environ.get("AUDIT_REPO_PATH")
    if audit_path_env is None:
        pytest.skip("AUDIT_REPO_PATH not set; skipping audit cross-check")
    audit_path = Path(audit_path_env)
    if not (audit_path / "verification" / "partition_lattice.py").exists():
        pytest.skip(
            f"AUDIT_REPO_PATH={audit_path} does not look like the audit checkout"
        )

    sys.path.insert(0, str(audit_path))
    try:
        from verification.partition_lattice import (  # type: ignore[import-not-found]
            B_charge_r as audit_B_charge_r,
        )
        from verification.partition_lattice import (
            B_r as audit_B_r,
        )
        from verification.partition_lattice import (
            Bhat_charge_r as audit_Bhat_charge_r,
        )
    finally:
        sys.path.pop(0)

    cat = Catalog.chemistry_r4()
    assert audit_B_r(cat.r) == constants.universal(cat.r)
    for w in cat:
        assert audit_B_charge_r(cat.r, w.charges) == constants.charge_filtered(cat.r, w)
        assert audit_Bhat_charge_r(cat.r, w.charges) == constants.block_refined(cat.r, w)
