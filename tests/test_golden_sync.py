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
    """Cross-check the constants against the audit repo's independent enumeration.

    The audit repo's published layout (``src/connected_layer_sector/``) uses a
    closed-form polynomial parameterisation keyed by ``(h, z)`` (number of
    paired charged letters and number of zero-charge letters), whereas this
    library uses direct enumeration keyed by an explicit ``charges`` tuple.
    Both routes compute the same theorem; the cross-check verifies the two
    independent implementations agree numerically on every catalog word.

    Cross-checks all three constants: universal $B_r$, charge-filtered
    $B^{\\mathrm{charge}}_r(W)$, and block-refined
    $\\widehat B^{\\mathrm{charge}}_r(W)$.
    """
    audit_path_env = os.environ.get("AUDIT_REPO_PATH")
    if audit_path_env is None:
        pytest.skip("AUDIT_REPO_PATH not set; skipping audit cross-check")
    audit_path = Path(audit_path_env)
    audit_module_path = audit_path / "src" / "connected_layer_sector" / "constants.py"
    if not audit_module_path.exists():
        pytest.skip(
            f"AUDIT_REPO_PATH={audit_path} does not look like the audit checkout "
            f"(no src/connected_layer_sector/constants.py)"
        )

    sys.path.insert(0, str(audit_path / "src"))
    try:
        from connected_layer_sector.constants import (  # type: ignore[import-not-found]
            B_charge_r as audit_B_charge_r,
        )
        from connected_layer_sector.constants import (
            B_r_const as audit_B_r,
        )
        from connected_layer_sector.constants import (
            Bhat_charge_r as audit_Bhat_charge_r,
        )
    finally:
        sys.path.pop(0)

    cat = Catalog.chemistry_r4()
    # B_r: audit returns float; library returns int. Compare as integers.
    assert int(audit_B_r(cat.r)) == constants.universal(cat.r)

    # B^charge_r: audit takes (h, z, r) where
    #   h = number of paired +1/-1 letters; z = number of zero-charge letters.
    # Translate each catalog word's charges to (h, z).
    for w in cat:
        positives = sum(1 for c in w.charges if c == +1)
        negatives = sum(1 for c in w.charges if c == -1)
        zeros = sum(1 for c in w.charges if c == 0)
        assert positives == negatives, f"word {w.name!r} should be charge-neutral; got {w.charges}"
        h, z = positives, zeros
        audit_charge = int(round(audit_B_charge_r(h, z, cat.r)))
        library_charge = constants.charge_filtered(cat.r, w)
        assert audit_charge == library_charge, (
            f"word {w.name!r} (h={h}, z={z}): "
            f"B_charge audit={audit_charge}, library={library_charge}"
        )
        audit_block_refined = int(round(audit_Bhat_charge_r(h, z, cat.r)))
        library_block_refined = constants.block_refined(cat.r, w)
        assert audit_block_refined == library_block_refined, (
            f"word {w.name!r} (h={h}, z={z}): "
            f"Bhat audit={audit_block_refined}, library={library_block_refined}"
        )


def test_audit_catalog_covers_all_five_chemistry_word_types():
    """Cross-check that the audit-side enumerate_chemistry_catalog emits all
    five Corollary 1 word types.

    This catches the failure mode where the constants functions know about
    a word type but the catalog enumerator that feeds the live audit
    pipeline does not produce instances of it. (Historically the audit's
    early enumerator omitted the a^dag a n n word type while the constants
    module had complete coverage; this test guards against that drift
    coming back.)
    """
    audit_path_env = os.environ.get("AUDIT_REPO_PATH")
    if audit_path_env is None:
        pytest.skip("AUDIT_REPO_PATH not set; skipping audit cross-check")
    audit_path = Path(audit_path_env)
    audit_catalog_path = audit_path / "src" / "connected_layer_sector" / "catalog.py"
    if not audit_catalog_path.exists():
        pytest.skip(
            f"AUDIT_REPO_PATH={audit_path} does not have " f"src/connected_layer_sector/catalog.py"
        )

    sys.path.insert(0, str(audit_path / "src"))
    try:
        from connected_layer_sector import (  # type: ignore[import-not-found]
            enumerate_chemistry_catalog as audit_enumerate,
        )
    finally:
        sys.path.pop(0)

    catalog_entries = audit_enumerate(n=5)
    labels_present = {entry[0] for entry in catalog_entries}
    expected_labels = {"nnn", "nnnn", "hopn", "doublex", "hopnn"}
    missing = expected_labels - labels_present
    assert not missing, (
        f"audit catalog enumerator missing word-type labels: {missing}; "
        f"present: {labels_present}"
    )
