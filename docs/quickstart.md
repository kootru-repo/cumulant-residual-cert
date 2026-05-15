# Quickstart

## Install

The package is installed directly from the GitHub repository via [uv](https://docs.astral.sh/uv/):

```bash
GIT=git+https://github.com/kootru-repo/cumulant-residual-cert.git

# Add to a uv-managed project
uv add "cumulant_residual_cert@${GIT}"

# Or install into the current venv (one-off scripts, notebooks)
uv pip install "cumulant_residual_cert@${GIT}"
```

With one or more chemistry-stack adapters (same pattern, use whichever
form fits your workflow):

```bash
uv add "cumulant_residual_cert[pyscf]@${GIT}"
uv add "cumulant_residual_cert[openfermion]@${GIT}"
uv add "cumulant_residual_cert[qiskit-nature]@${GIT}"
uv add "cumulant_residual_cert[all]@${GIT}"
```

Pin to a release by appending `@v0.5.0` (or another tag) to the git URL.

## 30 seconds

```python
from cumulant_residual_cert import Catalog, certify

cat = Catalog.chemistry_r4()
result = certify(cat, delta=0.01, level="block_refined")

for word, bar in result.bounds.items():
    print(f"|tau({word})| <= {bar}")
```

The result is a :class:`~cumulant_residual_cert.CertifiedBound`; the bounds
themselves live on `result.bounds` and the integer constants applied to each
word are on `result.constants_used`.

Output:

```
|tau(n_i n_j n_k)|                   <= 0.01
|tau(a_dag_i a_j n_k)|               <= 0.01
|tau(a_dag_i a_j n_k n_ell)|         <= 0.03
|tau(a_dag_i a_dag_j a_k a_ell)|     <= 0.01
|tau(n_i n_j n_k n_ell)|             <= 0.05
```

## Where does $\Delta$ come from?

Three honest options, in order of cost:

1. **Closed form**, when the state is in the Bernoulli class
   (occupation-basis diagonal product). The worked-example theorem gives
   $\Delta = 0$ on this class, and the
   [PySCF adapter](api.md) returns this exactly for canonical Hartree-Fock.

2. **Direct evaluation** from supplied RDMs via
   `cumulant_residual_cert.adapters.pyscf.from_rdms`. Pass 1-, 2-, 3-, 4-RDMs
   in spin-orbital convention; the Mobius formula evaluates $\Delta$ exactly
   under $U(1)$-invariance.

3. **Shadow estimation** via the UCB diagnostic
   `cumulant_residual_cert.delta_ucb`. Returns an upper bound on
   $\Delta$ that holds with probability $\ge 1 - \alpha$ simultaneously
   over the catalog. Be aware of the
   [Jordan-Wigner range caveat](jw_range_caveat.md) for random-Pauli
   shadows.

## Choosing a `level`

The three constant families ship in the same dict-shaped result:

| level | description | chemistry-catalog set |
| --- | --- | --- |
| `"universal"` | Same constant for every word: $B_4 = 105$. | `{105}` |
| `"charge_filtered"` | Word-dependent, exploits charge structure. | `{1, 53, 105}` |
| `"block_refined"` | Tightest. Exploits per-block charge structure. **Default.** | `{1, 3, 5}` |

The bound at each tighter level is no larger than the next looser level, word
by word.
