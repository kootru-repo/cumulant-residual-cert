#!/usr/bin/env bash
#
# Post-create setup for the GitHub Codespaces / VS Code Dev Container.
# Installs uv to a known location, syncs dev + docs extras, and runs an
# import smoke test so the user's first prompt confirms the library is
# importable on this image.
#
# Idempotent: safe to re-run.

set -euo pipefail

echo "==> installing uv"
if ! command -v uv >/dev/null 2>&1; then
    # Pin UV_INSTALL_DIR so PATH inheritance is deterministic (same lesson as
    # the Colab bootstrap; see notebook bootstrap cells for the writeup).
    export UV_INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> syncing dev + docs extras"
uv sync --extra dev --extra docs

echo "==> running import smoke test"
uv run python3 -c "import cumulant_residual_cert as c; print(f'cumulant_residual_cert {c.__version__} importable')"

echo
echo "================================================================"
echo "  cumulant-residual-cert devcontainer ready (Ubuntu)"
echo "================================================================"
echo
echo "Try one of:"
echo "  uv run pytest --cov=cumulant_residual_cert         # unit suite + coverage"
echo "  uv run python tools/benchmark.py                    # perf numbers"
echo "  uv run python tools/benchmark.py --json > b.json    # JSON for plots"
echo "  uv run jupyter lab --ip=0.0.0.0 --no-browser        # notebooks on :8888"
echo "  uv run mkdocs serve --dev-addr=0.0.0.0:8000         # docs preview on :8000"
echo
