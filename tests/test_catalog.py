"""Tests for catalog construction and invariants."""

from __future__ import annotations

import pytest

from cumulant_residual_cert import Catalog, FermionicWord, word
from cumulant_residual_cert.catalog import LETTER_CHARGE


def test_chemistry_catalog_has_five_words():
    cat = Catalog.chemistry_r4()
    assert len(cat) == 5


def test_chemistry_catalog_all_charge_neutral():
    cat = Catalog.chemistry_r4()
    for w in cat:
        assert w.is_charge_neutral, w.name


def test_word_charges_match_letter_charges():
    w = word("a_dag a n")
    assert w.charges == (1, -1, 0)
    assert w.total_charge == 0
    assert w.is_charge_neutral


def test_unknown_letter_rejected():
    with pytest.raises(ValueError):
        FermionicWord(letters=("a_dag", "Q"))


def test_catalog_rejects_overlong_word():
    long_word = word("a_dag a_dag a_dag a a a", name="too long for r=4")
    with pytest.raises(ValueError):
        Catalog(words=(long_word,), r=4)


def test_catalog_rejects_charged_word():
    charged = word("a_dag a_dag a", name="net charge +1")
    with pytest.raises(ValueError):
        Catalog(words=(charged,), r=4)


def test_catalog_iter_and_contains():
    cat = Catalog.chemistry_r4()
    names = {w.name for w in cat}
    assert "n_i n_j n_k" in names


def test_letter_charge_table():
    assert LETTER_CHARGE["I"] == 0
    assert LETTER_CHARGE["n"] == 0
    assert LETTER_CHARGE["a"] == -1
    assert LETTER_CHARGE["a_dag"] == +1


def test_empty_fermionic_word_rejected():
    with pytest.raises(ValueError):
        FermionicWord(letters=())


def test_empty_catalog_rejected():
    with pytest.raises(ValueError):
        Catalog(words=(), r=4)


def test_catalog_rejects_duplicate_word_names():
    w_a = word("n n n", name="dup")
    w_b = word("n n n n", name="dup")
    with pytest.raises(ValueError):
        Catalog(words=(w_a, w_b), r=4)
