# Contributing

This is a small library with a deliberately narrow scope. Please read the
**Non-goals** section of the project README before opening a feature request.

## Development setup

```bash
git clone https://github.com/kootru-repo/cumulant-residual-cert
cd cumulant-residual-cert
uv sync --extra dev
uv run pytest
```

If you need every adapter extra and the docs toolchain:

```bash
uv sync --extra dev --extra all --extra docs
```

## Code style

- `ruff format` for formatting.
- `ruff check` for linting.
- `pyright` for type checking.
- All three must pass in CI.

## Tests

- Unit tests live in `tests/`.
- Anything that touches the partition-lattice arithmetic must include an
  integer-exact test against a closed-form expected value.
- Any new constants must be added to the golden JSON and exercised by the
  `test_golden_sync.py` cross-check.

## Issues

When reporting a bug, please include:
- The function signature you called and the inputs.
- The version of `cumulant-residual-cert` (`uv tree --depth 0` shows it).
- The version of any chemistry adapter (`pyscf`, `openfermion`, `qiskit-nature`).
- A minimal reproducing example.

## What we will not accept

- New adapter for a chemistry framework with fewer than 100 active monthly users
  (keeps the maintenance burden bounded).
- A re-implementation of shadow tomography. Wire into existing libraries.
- Optional dependencies that pull in heavyweight chemistry stacks into the
  default install.
