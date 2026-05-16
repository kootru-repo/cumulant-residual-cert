# Performance characteristics

How long do the entry points take, and how do they scale?
The short answer:

| Entry point | Asymptotic cost | Practical cap |
| --- | --- | --- |
| [`certify`](api.md) | catalog size $\\times$ partition lattice; sub-millisecond on `chemistry_r4` | $r \\le 5$ in the manuscript; the partition lattice grows as $\\mathrm{Bell}(r)$ |
| [`delta_ucb`](api.md) (random-Pauli shadows) | $3^n$ per subword in the catalog (dense Bonferroni union); the implementation rejects `n_qubits > 10` by design | use `n_qubits \\le 4` for demos; route chemistry-scale work through the matchgate path below |
| [`delta_ucb_from_majorana_moments`](api.md) (matchgate / fermionic-Gaussian shadows) | linear in the number of Majorana products in the catalog's subword expansion; effectively $n$-independent at fixed catalog | scales to the manuscript-relevant chemistry register sizes |

See `tools/benchmark.py` for the exact harness; run it with
`uv run python tools/benchmark.py` to reproduce on your own machine.

## Measured wall-clock medians

Single-machine baseline, taken from `tools/benchmark.py` on a developer laptop
(AMD Ryzen-class CPU, Python 3.13, NumPy 2.4, Windows 11). Medians are over 3-20
repeats per row; minimums track medians within a few percent.

### `certify()`

| Call | Median |
| --- | ---: |
| `certify(chemistry_r4, delta=0.01)` | 0.5 ms |

`certify` is pure partition-lattice constant lookup plus dictionary
arithmetic; the cost is independent of any state, shadow data, or moment
estimate. Treat it as free relative to any of the diagnostic paths.

### `delta_ucb()` (random-Pauli shadows)

| `n_qubits` | `n_shots` | Median |
| ---: | ---: | ---: |
| 4 | 500 | 633 ms |
| 4 | 1000 | 717 ms |
| 4 | 2000 | 719 ms |
| 6 | 500 | **46.4 s** |

Two observations:

1. **Shots are not the bottleneck** at this register size: the cost is
   dominated by the Bonferroni-union construction over Pauli strings
   ($3^{|P|}$ per subword) and the Mobius assembly, not the shot loop.
   Going from 500 to 2000 shots at $n = 4$ moves the wall clock from
   633 ms to 719 ms.
2. **$n_\\text{qubits}$ is the bottleneck.** Six qubits is already
   ~46 seconds. The implementation refuses past $n_\\text{qubits} = 10$,
   but in practice you should treat $n \\le 4$ as the comfort zone for
   the random-Pauli path. For any chemistry-scale register, use
   `delta_ucb_from_majorana_moments` instead.

If you must use the random-Pauli path at chemistry scale, supply
caller-side per-subword $(\\hat\\mu, \\mathrm{rad})$ estimates and call
`delta_ucb_from_subword_moments` directly. That bypasses the dense
Pauli enumeration and lets you bring whatever measurement protocol
your hardware supports.

### `delta_ucb_from_majorana_moments()` (matchgate / fermionic-Gaussian shadows)

| `n_qubits` | Catalog | Majorana products | Median |
| ---: | --- | ---: | ---: |
| 4 | `chemistry_r4` | 120 | 0.9 ms |
| 6 | `chemistry_r4` | 120 | 1.3 ms |
| 8 | `chemistry_r4` | 120 | 1.3 ms |

The matchgate path is essentially $n$-independent at fixed catalog: the
work is proportional to the number of degree-$2k$ Majorana products the
catalog's subword expansion touches (here 120 for `chemistry_r4` at
$r = 4$), not to $n_\\text{qubits}$. This is the asymptotic reason the
manuscript recommends the matchgate-shadow protocol for chemistry-scale
registers: the certificate's runtime decouples from the JW dressing
overhead that breaks the random-Pauli route.

## Catalog-size scaling

Cost scales linearly in the number of catalog words at fixed $r$. The
shipped catalog (`Catalog.chemistry_r4`) has 5 words; the underlying
partition-lattice constants grow as Bell numbers in $r$:

| $r$ | $M_r$ | $B_r$ | $\\widehat B^\\text{charge}_r$ on chemistry catalog |
| ---: | ---: | ---: | --- |
| 3 | 6 | 1 | n/a (catalog is empty) |
| 4 | 26 | 105 | $\\in \\{1, 1, 3, 1, 5\\}$ across the 5 words |
| 5 | 150 | 227,251 | $\\in \\{1, 1, 3, 1, 5\\}$ (length-4 words; r=5 doesn't add catalog entries here) |

In practice the chemistry catalog at $r = 4$ has been the operating point
for everything in the manuscript and these examples.

## Memory

All three entry points are NumPy-only; no GPU, no sparse-matrix backend.
Peak memory at the matchgate path is dominated by the caller's
`majorana_moments` dict; on `chemistry_r4` at $r = 4$ this is ~120 entries
of $(\\mathrm{complex}, \\mathrm{float})$ pairs. Under typical Python
overhead this is <100 kB. The random-Pauli path's dense Pauli-string
union grows with $3^n$ so the memory footprint grows along with the wall
clock above.

## Reproducing these numbers

```bash
git clone https://github.com/kootru-repo/cumulant-residual-cert
cd cumulant-residual-cert
uv sync --extra dev
uv run python tools/benchmark.py
```

For machine-readable output (for plots, regressions, your own report):

```bash
uv run python tools/benchmark.py --json > bench.json
```

The script prints environment metadata (Python version, platform,
processor, NumPy version) so a comparison across machines is interpretable.

## When to retune

These numbers are stable enough that you can build a workflow around them.
The matchgate path's flat-in-$n$ scaling won't degrade until either:

- the catalog grows enough that the Majorana decomposition produces a much
  larger product set than the current 120 (e.g. extending past $r = 4$ on
  the chemistry catalog), or
- the moment-radius propagation in the Mobius assembly becomes the
  bottleneck (unlikely on any catalog small enough to fit in working memory).

If you observe ms-to-seconds jumps that are not explained by either, file
an issue with `tools/benchmark.py --json` output attached.
