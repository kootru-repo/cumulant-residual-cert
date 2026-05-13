"""Adapter smoke tests.

Each adapter test individually skips if its chemistry stack is not installed.
This is intentional: in a developer environment that has only one of the three
chemistry libraries, the other two tests should not block the run.
"""

from __future__ import annotations

import pytest

from cumulant_residual_cert import Catalog


# ---------------- PySCF ----------------------------------------------------


def test_pyscf_bernoulli_class_returns_zero_delta():
    """Canonical HF -> Bernoulli class -> Delta = 0 exactly."""
    pytest.importorskip("pyscf", reason="PySCF not installed")
    from pyscf import gto, scf

    from cumulant_residual_cert.adapters.pyscf import from_mean_field

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    assert mf.converged

    cat = Catalog.chemistry_r4()
    est = from_mean_field(mf, cat)
    assert est.delta == 0.0
    assert est.delta_is_exact is True
    assert est.framework == "pyscf"
    assert all(bar == 0.0 for bar in est.bound.bounds.values())


def test_pyscf_non_canonical_basis_raises_not_implemented():
    pytest.importorskip("pyscf", reason="PySCF not installed")
    from pyscf import gto, scf

    from cumulant_residual_cert.adapters.pyscf import from_mean_field

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)

    cat = Catalog.chemistry_r4()
    with pytest.raises(NotImplementedError):
        from_mean_field(mf, cat, basis="other")


# ---------------- OpenFermion ---------------------------------------------


def test_openfermion_word_to_fermion_operator():
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from cumulant_residual_cert.adapters.openfermion import word_to_fermion_operator
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("a_dag", "a", "n"), name="ad a n")
    op = word_to_fermion_operator(w, sites=(1, 2, 3))
    assert any(abs(c) > 0 for _, c in op.terms.items())


def test_openfermion_catalog_to_fermion_operators_yields_five_entries():
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from cumulant_residual_cert.adapters.openfermion import catalog_to_fermion_operators

    cat = Catalog.chemistry_r4()
    sites = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    ops = catalog_to_fermion_operators(cat, sites)
    assert len(ops) == 5
    for w in cat:
        assert w.name in ops


# ---------------- Qiskit-Nature -------------------------------------------


def test_qiskit_nature_word_to_fermionic_op():
    pytest.importorskip("qiskit_nature", reason="qiskit-nature not installed")
    from cumulant_residual_cert.adapters.qiskit_nature import word_to_fermionic_op
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("n", "n", "n"), name="n n n")
    op = word_to_fermionic_op(w, sites=(1, 2, 3))
    assert op is not None
