# cumulant-residual-cert

Deterministic bias certificates for charge-neutral fermionic-word observables on
$U(1)$-invariant states.

> Given a cumulant-truncated estimate of a chemistry-relevant fermionic-word
> observable, this library produces a rigorous upper bound on the truncation
> bias, in one function call.

[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/01_quickstart.ipynb)

## Install

```bash
pip install cumulant-residual-cert
# with chemistry-stack adapters
pip install "cumulant-residual-cert[pyscf]"           # PySCF adapter
pip install "cumulant-residual-cert[openfermion]"     # OpenFermion adapter
pip install "cumulant-residual-cert[qiskit-nature]"   # Qiskit-Nature adapter
pip install "cumulant-residual-cert[all]"             # all three
```

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

> **Note on shadow type.** The current implementation uses random Pauli shadows
> with the textbook $3^{|P|}$ Jordan-Wigner range factor. For chemistry-scale
> $r \ge 3$, a matchgate / fermionic-Gaussian shadow adapter is planned for
> v0.3; the v0.2 OpenFermion adapter currently ships only operator-conversion
> utilities. See [docs/jw_range_caveat.md](docs/jw_range_caveat.md) for the
> details and mitigations.

## What this does and does not do

- Yes: certifies the deterministic bias of an order-$\le 2$ cumulant closure of
  a chemistry-catalog observable, given a (state-dependent) envelope $\Delta$.
- Yes: provides a one-sided certified upper bound on $\Delta$ from random-Pauli
  or matchgate shadow data.
- No: not a shadow-tomography implementation. Bring your own shadows from
  OpenFermion, PennyLane, Qiskit, or your own pipeline.
- No: not a measurement-advantage tool. Variance and shot count are governed by
  whatever shadow protocol you use; this library is orthogonal to that.
- No: does not perform error mitigation or improve QPU fidelity.

## License

Apache 2.0.
