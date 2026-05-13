# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-XX-XX

### Added
- Core API: `Catalog`, `FermionicWord`, `certify`, `delta_ucb`, `constants`.
- Chemistry catalog at $r=4$ with block-refined constants $\{1, 3, 5\}$.
- Partition-lattice enumeration for arbitrary catalogs and $r$.
- Random-Pauli shadow UCB diagnostic with Bonferroni correction.
- Quickstart and Bernoulli-worked-example notebooks.
- Apache 2.0 license.
- CI matrix over Python 3.10/3.11/3.12 on Linux and macOS.
- Golden-sync test against the audit repo's pinned constants table.

## [0.2.0] - planned

### Added
- PySCF adapter with Hartree-Fock worked example.
- OpenFermion adapter with matchgate-shadow recipe.
- Qiskit-Nature adapter.
- MkDocs documentation site.
