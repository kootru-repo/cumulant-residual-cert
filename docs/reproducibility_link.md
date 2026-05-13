# Audit cross-check

The constants table shipped in this library
(`cumulant_residual_cert/data/chemistry_catalog_r4.json`) is verified by two
complementary tests on every CI run.

## Local consistency

`tests/test_golden_sync.py::test_golden_json_matches_freshly_computed`
regenerates the table from this library's own partition-lattice enumeration
in `_partition.py` and compares it field-by-field against the shipped JSON.
This catches drift between code and data within this repository.

## External audit cross-check

`tests/test_golden_sync.py::test_audit_repo_cross_check` imports the
partition-lattice enumeration **functions** from an external audit
repository checkout and re-runs them on the same catalog, asserting that
the resulting integer constants match this library's own values exactly.

The cross-check is opt-in: it activates when the environment variable
`AUDIT_REPO_PATH` points at a working tree of a compatible audit repository
that exposes `verification/partition_lattice.py`. CI sets `AUDIT_REPO_PATH`
to a tree checked out at the SHA pinned in this repository's
`AUDIT_COMMIT` file (a 40-character lowercase hex digest, validated by the
CI workflow before checkout). Locally the test skips cleanly when the
environment variable is absent.

> **Note.** The audit cross-check compares against the audit repository's
> independent enumeration **functions**, not against a JSON artefact in the
> audit repository. The two enumerations are independent implementations of
> the same definition; any divergence would indicate a mistake in one of
> them and fails CI.

If you find a discrepancy in your local environment, that is interesting:
please open an issue with the output of `pytest tests/test_golden_sync.py -v`
and the audit-repository commit you compared against.
