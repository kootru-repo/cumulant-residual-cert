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
    """Canonical HF + explicit user assertion -> Bernoulli class -> Delta = 0 exactly."""
    pytest.importorskip("pyscf", reason="PySCF not installed")
    from pyscf import gto, scf

    from cumulant_residual_cert.adapters.pyscf import from_mean_field

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    assert mf.converged

    cat = Catalog.chemistry_r4()
    est = from_mean_field(mf, cat, user_asserts_bernoulli_class=True)
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
        from_mean_field(mf, cat, basis="other", user_asserts_bernoulli_class=True)


def test_pyscf_requires_explicit_bernoulli_assertion():
    pytest.importorskip("pyscf", reason="PySCF not installed")
    from pyscf import gto, scf

    from cumulant_residual_cert.adapters.pyscf import from_mean_field

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)

    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError, match="user_asserts_bernoulli_class"):
        from_mean_field(mf, cat)


def test_pyscf_unconverged_meanfield_raises():
    pytest.importorskip("pyscf", reason="PySCF not installed")

    from cumulant_residual_cert.adapters.pyscf import from_mean_field

    class _Stub:
        converged = False

    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError, match="not converged"):
        from_mean_field(_Stub(), cat, user_asserts_bernoulli_class=True)


# ---------------- OpenFermion ---------------------------------------------


def test_openfermion_word_to_fermion_operator_terms():
    """The produced FermionOperator has exactly the expected normal-ordered string."""
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from openfermion import FermionOperator

    from cumulant_residual_cert.adapters.openfermion import word_to_fermion_operator
    from cumulant_residual_cert.catalog import FermionicWord

    # n_1 n_2 n_3 in 1-based site indexing -> n_0 n_1 n_2 in 0-based.
    w = FermionicWord(("n", "n", "n"), name="n n n")
    op = word_to_fermion_operator(w, sites=(1, 2, 3))
    expected = FermionOperator("0^ 0") * FermionOperator("1^ 1") * FermionOperator("2^ 2")
    assert op == expected


def test_openfermion_word_to_fermion_operator_a_dag_a_n():
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from openfermion import FermionOperator

    from cumulant_residual_cert.adapters.openfermion import word_to_fermion_operator
    from cumulant_residual_cert.catalog import FermionicWord

    # a_dag_1 a_2 n_3 -> a_dag_0 a_1 (a_dag_2 a_2).
    w = FermionicWord(("a_dag", "a", "n"), name="ad a n")
    op = word_to_fermion_operator(w, sites=(1, 2, 3))
    expected = FermionOperator("0^") * FermionOperator("1") * FermionOperator("2^ 2")
    assert op == expected


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


def test_openfermion_rejects_zero_site_index():
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from cumulant_residual_cert.adapters.openfermion import word_to_fermion_operator
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("n", "n", "n"), name="n n n")
    with pytest.raises(ValueError, match="positive 1-based"):
        word_to_fermion_operator(w, sites=(0, 1, 2))


def test_openfermion_rejects_duplicate_sites():
    pytest.importorskip("openfermion", reason="OpenFermion not installed")
    from cumulant_residual_cert.adapters.openfermion import word_to_fermion_operator
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("n", "n", "n"), name="n n n")
    with pytest.raises(ValueError, match="duplicate sites"):
        word_to_fermion_operator(w, sites=(1, 1, 2))


# ---------------- Qiskit-Nature -------------------------------------------


def test_qiskit_nature_word_to_fermionic_op_exact_label():
    """The produced FermionicOp has the exact expected label string."""
    pytest.importorskip("qiskit_nature", reason="qiskit-nature not installed")
    from cumulant_residual_cert.adapters.qiskit_nature import word_to_fermionic_op
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("n", "n", "n"), name="n n n")
    op = word_to_fermionic_op(w, sites=(1, 2, 3))
    # n_i is +_i -_i, so n n n -> "+_0 -_0 +_1 -_1 +_2 -_2".
    expected_terms = dict(op.terms())
    expected_label = "+_0 -_0 +_1 -_1 +_2 -_2"
    assert expected_label in {label for label, _ in op.terms()}, dict(op.terms())
    # And coefficient is 1.0 on that label.
    coeffs_by_label = {label: coeff for label, coeff in op.terms()}
    assert coeffs_by_label[expected_label] == pytest.approx(1.0)


def test_qiskit_nature_word_to_fermionic_op_a_dag_a_n():
    pytest.importorskip("qiskit_nature", reason="qiskit-nature not installed")
    from cumulant_residual_cert.adapters.qiskit_nature import word_to_fermionic_op
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("a_dag", "a", "n"), name="ad a n")
    op = word_to_fermionic_op(w, sites=(1, 2, 3))
    expected_label = "+_0 -_1 +_2 -_2"
    assert expected_label in {label for label, _ in op.terms()}


def test_qiskit_nature_from_problem_requires_bernoulli_assertion():
    pytest.importorskip("qiskit_nature", reason="qiskit-nature not installed")
    from cumulant_residual_cert.adapters.qiskit_nature import from_problem

    cat = Catalog.chemistry_r4()
    with pytest.raises(ValueError, match="user_asserts_bernoulli_class"):
        from_problem(problem=object(), catalog=cat)


def test_qiskit_nature_rejects_zero_site_index():
    pytest.importorskip("qiskit_nature", reason="qiskit-nature not installed")
    from cumulant_residual_cert.adapters.qiskit_nature import word_to_fermionic_op
    from cumulant_residual_cert.catalog import FermionicWord

    w = FermionicWord(("n", "n", "n"), name="n n n")
    with pytest.raises(ValueError, match="positive 1-based"):
        word_to_fermionic_op(w, sites=(0, 1, 2))
