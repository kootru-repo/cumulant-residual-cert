"""Microbenchmarks for the public API of cumulant_residual_cert.

Run from the repo root:

    uv run python tools/benchmark.py

Prints wall-clock medians (over a few repeats) for the three entry points
exposed in the README's 30-second example, plus the matchgate moment path.
The numbers feed docs/performance.md and are intended to give a researcher
a sense of cost-vs-input scaling before they commit to a workflow.

Deterministic seed; no randomness aside from the synthetic shadow / moment
inputs we generate ourselves.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time

import numpy as np

from cumulant_residual_cert import (
    Catalog,
    certify,
    delta_ucb,
    delta_ucb_from_majorana_moments,
)
from cumulant_residual_cert.diagnostic import collect_shadows


def _median_seconds(fn, *, repeats: int = 5) -> tuple[float, float]:
    """Return (median_s, min_s) over `repeats` runs."""
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def _state_for_n(n: int, seed: int = 0) -> np.ndarray:
    """A small two-electron superposition on n spin-orbitals; correlated for theta=pi/5."""
    rng = np.random.default_rng(seed)
    psi = np.zeros(2 ** n, dtype=complex)
    bits1 = (1 << (n - 1)) | (1 << (n - 2))  # |1100...>
    bits2 = (1 << 1) | 1  # |...0011>
    theta = math.pi / 5
    psi[bits1] = math.cos(theta)
    psi[bits2] = math.sin(theta)
    # tiny phase noise so the state isn't pathologically aligned
    psi = psi * np.exp(1j * rng.uniform(-1e-6, 1e-6, size=2 ** n))
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def benchmark_certify() -> dict:
    """`certify` on chemistry_r4 with a user-supplied delta. Should be sub-ms."""
    catalog = Catalog.chemistry_r4()

    def run():
        return certify(catalog, delta=0.01, level="block_refined")

    median, fastest = _median_seconds(run, repeats=20)
    return {
        "call": "certify(chemistry_r4, delta=0.01)",
        "catalog_size": len(catalog),
        "median_s": median,
        "min_s": fastest,
    }


def benchmark_delta_ucb(n_qubits: int, n_shots: int) -> dict:
    """`delta_ucb` over the chemistry_r4 catalog with synthetic shadow shots."""
    catalog = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]
    rho = _state_for_n(n_qubits)
    shadow_seed = 42
    shadows = collect_shadows(rho, n=n_qubits, M=n_shots, seed=shadow_seed)

    def run():
        return delta_ucb(
            shadow_samples=shadows,
            catalog=catalog,
            sites_per_word=sites_per_word,
            n_qubits=n_qubits,
            confidence=0.95,
        )

    median, fastest = _median_seconds(run, repeats=3)
    return {
        "call": f"delta_ucb(n_qubits={n_qubits}, n_shots={n_shots}, chemistry_r4)",
        "n_qubits": n_qubits,
        "n_shots": n_shots,
        "median_s": median,
        "min_s": fastest,
    }


def benchmark_delta_ucb_matchgate(n_qubits: int) -> dict:
    """`delta_ucb_from_majorana_moments` with synthetic moment estimates over chemistry_r4."""
    from cumulant_residual_cert._majorana import word_majorana_decomposition
    from itertools import combinations

    catalog = Catalog.chemistry_r4()
    sites_per_word = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4),
    ]

    # Build the COMPLETE set of Majorana products the catalog touches and
    # supply zero-mean tiny-radius placeholders so the UCB pipeline runs.
    needed: set[tuple[int, ...]] = set()
    for w, sites in zip(catalog, sites_per_word):
        m = w.length
        for k in range(1, m + 1):
            for B in combinations(range(1, m + 1), k):
                key = tuple(sorted(B))
                sub_letters = tuple(w.letters[i - 1] for i in key)
                sub_sites = tuple(sites[i - 1] for i in key)
                decomp = word_majorana_decomposition(sub_letters, sub_sites)
                needed.update(decomp.keys())

    moments: dict[tuple[int, ...], tuple[complex, float]] = {(): (1.0 + 0j, 0.0)}
    for idx in needed:
        if idx == ():
            continue
        moments[idx] = (0.0 + 0j, 1e-4)

    def run():
        return delta_ucb_from_majorana_moments(
            majorana_moments=moments,
            catalog=catalog,
            sites_per_word=sites_per_word,
            confidence=0.95,
            n_protocol_terms=len(moments),
        )

    median, fastest = _median_seconds(run, repeats=5)
    return {
        "call": f"delta_ucb_from_majorana_moments(n_qubits={n_qubits}, chemistry_r4)",
        "n_qubits": n_qubits,
        "n_majorana_products": len(moments),
        "median_s": median,
        "min_s": fastest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit JSON to stdout (for tooling)"
    )
    args = parser.parse_args()

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }

    results = []
    print("== certify() ==", flush=True)
    r = benchmark_certify()
    results.append(r)
    print(
        f"  {r['call']:<60s} median={r['median_s']*1e3:7.3f} ms  min={r['min_s']*1e3:7.3f} ms",
        flush=True,
    )

    print("\n== delta_ucb() (random-Pauli shadows) ==", flush=True)
    # Random-Pauli enumeration is dense in n_qubits (3^|P| per subword); the
    # built-in pipeline caps at n_qubits=10 by design. Skip past n=6 here so
    # the benchmark terminates in finite time on a developer laptop.
    for n_qubits, n_shots in [(4, 500), (4, 1000), (4, 2000), (6, 500)]:
        r = benchmark_delta_ucb(n_qubits, n_shots)
        results.append(r)
        print(f"  {r['call']:<60s} median={r['median_s']*1e3:9.3f} ms", flush=True)

    print("\n== delta_ucb_from_majorana_moments() ==", flush=True)
    for n_qubits in [4, 6, 8]:
        r = benchmark_delta_ucb_matchgate(n_qubits)
        results.append(r)
        print(
            f"  {r['call']:<60s} median={r['median_s']*1e3:9.3f} ms  "
            f"({r['n_majorana_products']} Majorana products)",
            flush=True,
        )

    print("\n== environment ==")
    for k, v in env.items():
        print(f"  {k}: {v}")

    if args.json:
        json.dump({"env": env, "results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
