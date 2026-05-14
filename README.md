# cumulant-residual-cert

Deterministic bias certificates for charge-neutral fermionic-word observables on
$U(1)$-invariant states.

> Given a cumulant-truncated estimate of a chemistry-relevant fermionic-word
> observable, this library produces a rigorous upper bound on the truncation
> bias, in one function call.

[![Open tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/00_tutorial.ipynb)
[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/01_quickstart.ipynb)
[![Open cookbook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/05_cookbook.ipynb)

- [`00_tutorial.ipynb`](notebooks/00_tutorial.ipynb): the front door. The inequality + a decision tree mapping your situation to one of four paths + one concrete state walked through all four side by side + the go/no-go rule against your tolerance.
- [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb): the theorem in action on a concrete worked example, with expected vs actual numbers at every step.
- [`05_cookbook.ipynb`](notebooks/05_cookbook.ipynb): nine recipes for applying the certificate to your own state, RDM data, shadow data, custom catalog, or workflow-decision context.

## Install

This package is distributed via this GitHub repository; no PyPI publication is
required. Install directly from the repo with `pip`:

```bash
GIT=git+https://github.com/kootru-repo/cumulant-residual-cert.git
pip install "cumulant_residual_cert@${GIT}"
# with chemistry-stack adapters
pip install "cumulant_residual_cert[pyscf]@${GIT}"
pip install "cumulant_residual_cert[openfermion]@${GIT}"
pip install "cumulant_residual_cert[qiskit-nature]@${GIT}"
pip install "cumulant_residual_cert[all]@${GIT}"
```

Pin to a specific release by appending `@v0.4.0` (or another tag) to the
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

## License

Apache 2.0.
