# Constants table for the chemistry catalog at $r = 4$

The chemistry catalog has five words. The three constants are:

| word | charges | $B_4$ | $B^{\mathrm{charge}}_4(W)$ | $\widehat B^{\mathrm{charge}}_4(W)$ |
| --- | --- | ---: | ---: | ---: |
| $n_i n_j n_k$ | $(0, 0, 0)$ | $105$ | $1$ | $1$ |
| $a^\dagger_i a_j n_k$ | $(+1, -1, 0)$ | $105$ | $1$ | $1$ |
| $a^\dagger_i a_j n_k n_\ell$ | $(+1, -1, 0, 0)$ | $105$ | $53$ | $3$ |
| $a^\dagger_i a^\dagger_j a_k a_\ell$ | $(+1, +1, -1, -1)$ | $105$ | $1$ | $1$ |
| $n_i n_j n_k n_\ell$ | $(0, 0, 0, 0)$ | $105$ | $105$ | $5$ |

The block-refined column $\{1, 1, 3, 1, 5\}$ has unique-value set
$\widehat B^{\mathrm{charge}}_4(W) \in \{1, 3, 5\}$.

## Universal vs. block-refined per-word tightening

| word | $B_4 / \widehat B^{\mathrm{charge}}_4$ |
| --- | ---: |
| $n_i n_j n_k$ | $105 \times$ |
| $a^\dagger_i a_j n_k$ | $105 \times$ |
| $a^\dagger_i a_j n_k n_\ell$ | $35 \times$ |
| $a^\dagger_i a^\dagger_j a_k a_\ell$ | $105 \times$ |
| $n_i n_j n_k n_\ell$ | $21 \times$ |

This is the 21 to 105 per-word reduction range, depending on the word's charge pattern.

## Regenerating the table

```python
from cumulant_residual_cert import Catalog, constants

cat = Catalog.chemistry_r4()
table = constants.compute(cat)

for w in cat:
    wc = table.per_word[w.name]
    print(f"{w.name:32s}  B={wc.universal}  B^c={wc.charge_filtered}  Bhat={wc.block_refined}")
```

The shipped golden values live in `cumulant_residual_cert/data/chemistry_catalog_r4.json`
and are cross-checked in CI against an independent enumeration in the audit
repository.
