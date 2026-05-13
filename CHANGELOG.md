# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-13

Initial publicly-shipped release. Combines the originally-planned 0.1.0 core
API with the 0.2.0 adapter layer; 0.1.0 was never tagged or published.

### Added
- Core API: `Catalog`, `FermionicWord`, `certify`, `delta_ucb`, `constants`.
- Chemistry catalog at $r = 4$ with block-refined constants $\{1, 3, 5\}$.
- Partition-lattice enumeration for arbitrary catalogs and $r$.
- Random-Pauli shadow UCB diagnostic with Bonferroni correction. Built-in
  expansion is dense and refuses `n_qubits > 10`; bring your own per-Pauli
  estimates for larger registers.
- PySCF adapter (`from_mean_field`) with explicit Bernoulli-class
  assertion. `from_rdms` reserved for a later release.
- OpenFermion adapter (`word_to_fermion_operator`, `catalog_to_fermion_operators`).
  Matchgate-shadow UCB wrapper reserved for a later release.
- Qiskit-Nature adapter (`word_to_fermionic_op`, `from_problem` with explicit
  Bernoulli-class assertion).
- Quickstart, Bernoulli worked-example, and PySCF Hartree-Fock notebooks.
- MkDocs documentation site.
- Apache 2.0 license.
- CI matrix over Python 3.10/3.11/3.12 on Linux and macOS, plus a
  docs-build-and-adapter-extras job and a pinned-SHA golden-sync job against
  the external audit repository.

### Fixed
- Random-Pauli snapshot estimator now correctly carries the $3^{|P|}$ unbiased
  factor (previously biased low; caught by an external code review).
