# Audit cross-check

The constants table shipped in this library
(`cumulant_residual_cert/data/chemistry_catalog_r4.json`) is regenerated on
every CI run from this library's own partition-lattice enumeration code, then
optionally diffed against an external audit repository's pinned JSON. Drift
between the two fails CI.

The audit cross-check is opt-in: if the `AUDIT_REPO_PATH` environment variable
points at a checkout of a compatible audit repository, the test
`tests/test_golden_sync.py::test_audit_repo_cross_check` runs and compares
values against the audit repository's independent enumeration. Otherwise the
test skips cleanly.

If you find a discrepancy in your local environment, that is interesting:
please open an issue with the output of `pytest tests/test_golden_sync.py -v`
and the audit repository's commit hash you compared against.
