# cumulant-residual-cert

Deterministic bias certificates for charge-neutral fermionic-word observables on
$U(1)$-invariant states.

> Given a cumulant-truncated estimate of a chemistry-relevant fermionic-word
> observable, this library produces a rigorous upper bound on the truncation
> bias, in one function call.

## Where to start

Local (recommended for reproducible work):

```bash
git clone https://github.com/kootru-repo/cumulant-residual-cert
cd cumulant-residual-cert
uv sync --extra dev
uv run jupyter lab notebooks/
```

Or open any notebook directly in Colab via the badge at the top of each. Each notebook detects on first run whether the library is already installed; if not, it bootstraps `uv` via `astral.sh/uv/install.sh` and installs the library into the kernel.

| If you are... | Open | Colab | Time |
| --- | --- | --- | --- |
| New to the library; don't yet know which path applies to your problem | [`00_tutorial.ipynb`](notebooks/00_tutorial.ipynb) | <a href="https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/00_tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="200"/></a> | ~10 min |
| Want to see the theorem work on a concrete example before trusting it | [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | <a href="https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/01_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="200"/></a> | ~5 min |
| Have a specific workflow in mind; need the code, not the theory | [`05_cookbook.ipynb`](notebooks/05_cookbook.ipynb) | <a href="https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/05_cookbook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="200"/></a> | scan |

> *Peer reviewer or referee of the underlying paper?* The canonical reproducibility artifact is [`charge-filtered-cumulant-residuals`](https://github.com/kootru-repo/charge-filtered-cumulant-residuals) (Zenodo concept DOI [10.5281/zenodo.20129664](https://doi.org/10.5281/zenodo.20129664), auto-tracks the latest version): claim-indexed audit notebooks, pytest suite with explicit assertions on the manuscript headlines, lightweight mutation sanity harness, SHA256-pinned deposited data, and a headless-CI notebook workflow that reruns every notebook end-to-end on every push. The present library is the user-facing companion; its catalog constants are CI-checked against that repo's independent implementation on every push.

### What you get from each notebook

- **[`00_tutorial.ipynb`](notebooks/00_tutorial.ipynb)** — the front door. You'll leave knowing the one inequality the whole library is about, which of four paths fits your situation, and what the same physical state's certificate looks like through every path side by side. Closes with the go/no-go decision rule against your tolerance. *For users who don't know which adapter or diagnostic to use yet.*

- **[`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb)** — the theorem made concrete. A small two-electron correlated state, the actual bias number $|⟨W⟩_\text{true} - ⟨W⟩_\text{trunc}|$, the certified bound $C_W·\Delta$, and a visible $21\times$ tightening from universal to block-refined. *For users who want to see the math work before relying on it.*

- **[`02_bernoulli_worked.ipynb`](notebooks/02_bernoulli_worked.ipynb)** — proof at machine precision. A random Bernoulli product state at $n=6$, every catalog cumulant evaluated directly, and $\Delta = 0$ confirmed to float-64 zero. *For users who want to see the worked-example theorem hold up numerically before they trust the closed-form $\Delta = 0$ claim for Hartree-Fock baselines.*

- **[`04_real_shadow_data.ipynb`](notebooks/04_real_shadow_data.ipynb)** — pluggable shadow-data routing. Simulator branch runs out of the box; IBM Quantum Runtime, Rigetti Forest, IonQ, and IQM Resonance stubs are documented and ready to wire. *For users who have shadow records from a real measurement and need to plug them in.*

- **[`05_cookbook.ipynb`](notebooks/05_cookbook.ipynb)** — nine direct recipes: HF baseline (end-to-end PySCF on H₂ STO-3G), post-HF from RDMs, random-Pauli shadows, matchgate shadows, custom catalog, JSON persistence, go/no-go decision rule, level selection, OpenFermion / Qiskit-Nature operator conversion. *For users who know what they want and need a copy-paste solution.*

## Requirements

- Python `>=3.10` (CI runs 3.10, 3.11, 3.12 on Linux + macOS).
- [uv](https://docs.astral.sh/uv/) (one-line install: `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux/macOS; `irm https://astral.sh/uv/install.ps1 | iex` on Windows).
- Optional chemistry adapters (PySCF, OpenFermion, Qiskit Nature) are off by default; opt in via the extras matrix below.

Core dep is `numpy >= 1.24`. Adapter and dev extras live in [`pyproject.toml`](pyproject.toml); the locked, durable resolution is in [`uv.lock`](uv.lock).

## Install

This package is distributed via this GitHub repository; no PyPI publication is
required. Install directly from the repo:

```bash
GIT=git+https://github.com/kootru-repo/cumulant-residual-cert.git

# Add to a uv-managed project
uv add "cumulant_residual_cert@${GIT}"

# With chemistry-stack adapters (choose what you need)
uv add "cumulant_residual_cert[pyscf]@${GIT}"
uv add "cumulant_residual_cert[openfermion]@${GIT}"
uv add "cumulant_residual_cert[qiskit-nature]@${GIT}"
uv add "cumulant_residual_cert[all]@${GIT}"
```

If your environment is not yet a uv-managed project, initialize one first:

```bash
uv init my-workflow
cd my-workflow
uv add "cumulant_residual_cert@${GIT}"
```

Pin to a specific release by appending `@v0.5.0` (or another tag) to the
git URL.

## 30-second example

```python
from cumulant_residual_cert import Catalog, certify

catalog = Catalog.chemistry_r4()       # 5-word chemistry catalog at r = 4
delta = 0.01                           # an envelope upper bound you trust

result = certify(catalog, delta, level="block_refined")
for word, bar in result.bounds.items():
    print(f"|tau({word})|  <=  {bar:.4g}")
```

Output:

```
|tau(n_i n_j n_k)|                   <=  0.01
|tau(a_dag_i a_j n_k)|               <=  0.01
|tau(a_dag_i a_j n_k n_ell)|         <=  0.03
|tau(a_dag_i a_dag_j a_k a_ell)|     <=  0.01
|tau(n_i n_j n_k n_ell)|             <=  0.05
```

The block-refined constants $\widehat B^{\mathrm{charge}}_4(W) \in \{1,3,5\}$ are
the per-word multipliers; the universal $B_4 = 105$ baseline is also available
via `level="universal"`.

## Estimating $\Delta$ from shadow data

```python
from cumulant_residual_cert import delta_ucb

ucb = delta_ucb(
    shadow_samples=shots,           # iterable of (basis, outcomes) tuples
    catalog=catalog,
    sites_per_word=[(1, 2, 3), (1, 2, 3),
                    (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)],
    n_qubits=n_qubits,
    confidence=0.95,
)
result = certify(catalog, ucb.delta_ucb)
```

The diagnostic is Bonferroni-corrected over every Pauli string in the catalog's
subword expansions, so the upper bound holds with probability at least
$1 - \alpha$ simultaneously across all words.

> **Note on shadow type.** `delta_ucb` uses random Pauli shadows with the
> textbook $3^{|P|}$ Jordan-Wigner range factor. For chemistry-scale
> $r \ge 3$, supply per-Majorana-product `(mean, radius)` estimates from
> a matchgate / fermionic-Gaussian shadow protocol of your choice and call
> [`delta_ucb_from_majorana_moments`](docs/api.md) or the OpenFermion
> wrapper `adapters.openfermion.delta_ucb_from_matchgate_shadows`. A
> built-in matchgate-snapshot estimator is planned for a later release.
> See [docs/jw_range_caveat.md](docs/jw_range_caveat.md) for the details
> and mitigations.

## What this does and does not do

- Yes: certifies the deterministic bias of an order-$\le 2$ cumulant closure of
  a chemistry-catalog observable, given a (state-dependent) envelope $\Delta$.
- Yes: provides a one-sided certified upper bound on $\Delta$ from random-Pauli
  shadow data (matchgate-shadow support is planned for a later release).
- No: not a shadow-tomography implementation. Bring your own shadows from
  OpenFermion, PennyLane, Qiskit, or your own pipeline. The built-in
  random-Pauli expansion is dense and refuses `n_qubits > 10`; for larger
  registers, supply per-Pauli estimates directly to the diagnostic.
- No: not a measurement-advantage tool. Variance and shot count are governed by
  whatever shadow protocol you use; this library is orthogonal to that.
- No: does not perform error mitigation or improve QPU fidelity.

## Persisting a certificate

`certify()` returns a frozen dataclass; there is no `.save()` method by
design. Standard-library `dataclasses.asdict` plus `json.dump` produces a
useable scientific artefact:

```python
import json
from dataclasses import asdict
from cumulant_residual_cert import Catalog, certify

cat = Catalog.chemistry_r4()
result = certify(cat, delta=0.012, delta_provenance="from_rdms")
with open("my_workflow_certificate.json", "w") as f:
    json.dump(asdict(result), f, indent=2)
```

The persisted dict carries the bound values, the integer constants used,
the catalog name, the library version (auto-populated via
`importlib.metadata`), and the caller-declared `delta_provenance` label
(one of `"closed_form_bernoulli"`, `"from_rdms"`, `"ucb_random_pauli"`,
`"ucb_subword"`, `"ucb_majorana"`, `"ucb_matchgate_shadows"`, or
`"user_supplied"`; custom labels are also accepted). That is enough for a
downstream auditor or a journal supplement to identify exactly what
produced the bound.

## About

Maintained by **Kootru Labs**, Burlington, USA. Website: [labs.kootru.com](https://labs.kootru.com).

Author: **Andrew Craton** ([ORCID 0009-0001-2269-8599](https://orcid.org/0009-0001-2269-8599), [acraton@kootru.com](mailto:acraton@kootru.com)).

The library is the user-facing companion to the reproducibility envelope at [`charge-filtered-cumulant-residuals`](https://github.com/kootru-repo/charge-filtered-cumulant-residuals), Zenodo concept DOI [10.5281/zenodo.20129664](https://doi.org/10.5281/zenodo.20129664) (auto-tracks the latest version). Constants computed here are continuously cross-checked against that repository's independent implementation in CI.

## How to cite

Please cite both the library and the underlying reproducibility deposit. A machine-readable [`CITATION.cff`](CITATION.cff) is provided; the BibTeX below is the equivalent.

**Software (this library):**

```bibtex
@software{cumulant_residual_cert,
  author       = {Craton, Andrew},
  title        = {{cumulant-residual-cert}: deterministic bias certificates for
                  charge-neutral fermionic-word observables},
  organization = {Kootru Labs},
  url          = {https://github.com/kootru-repo/cumulant-residual-cert},
  version      = {0.5.0},
  year         = {2026}
}
```

**Reproducibility deposit:**

```bibtex
@dataset{charge_filtered_cumulant_residuals,
  author       = {Craton, Andrew},
  title        = {{charge-filtered-cumulant-residuals}: reproducibility envelope},
  organization = {Kootru Labs},
  doi          = {10.5281/zenodo.20129665},
  url          = {https://doi.org/10.5281/zenodo.20129665},
  year         = {2026}
}
```

**Manuscript:**

```bibtex
@unpublished{craton_charge_filtered_cumulant_residuals_manuscript,
  author       = {Craton, Andrew},
  title        = {Charge- and block-refined bias bounds for second-order
                  cumulant truncation on {$U(1)$}-invariant fermionic states},
  organization = {Kootru Labs},
  year         = {2026},
  note         = {Manuscript in preparation}
}
```

## License

Apache 2.0 (see [`LICENSE`](LICENSE)). Copyright held by Kootru Labs (a DBA of Kootru LLC).
