# cumulant-residual-cert

Deterministic bias certificates for charge-neutral fermionic-word observables on
$U(1)$-invariant states.

Given a cumulant-truncated estimate of a chemistry-relevant fermionic-word
observable, this library produces a rigorous upper bound on the truncation
bias, in one function call.

[Open the quickstart in Google Colab](https://colab.research.google.com/github/kootru-repo/cumulant-residual-cert/blob/main/notebooks/01_quickstart.ipynb).

## What this is

A small Python package implementing three integer-valued partition-lattice
sums and a Bonferroni-corrected upper-confidence diagnostic on the
high-cumulant envelope:

- universal $B_r$ (loosest),
- charge-filtered $B^{\mathrm{charge}}_r(W)$,
- block-refined $\widehat B^{\mathrm{charge}}_r(W)$ (tightest).

For the chemistry catalog at $r = 4$, the tightening from universal to
block-refined is

$$
105 \;\longrightarrow\; \{1, 3, 5\}.
$$

Multiply the chosen constant by a state-dependent envelope $\Delta_{r, U(1)}^{\mathrm{cat}}(\rho)$
and you have a certified bias bar for the cumulant-truncated estimate of every
word in the catalog.

## What this is not

- Not a shadow-tomography implementation. Bring your own shadow data from
  OpenFermion, PennyLane, Qiskit, or your own pipeline.
- Not a measurement-advantage tool. Variance and shot count are governed by
  whatever shadow protocol you use.
- Not error mitigation. The certificate operates downstream of hardware, on
  the cumulant-truncation step of the post-processing pipeline.
