# The Jordan-Wigner range caveat

The library's UCB diagnostic, `cumulant_residual_cert.delta_ucb`, defaults to
random-Pauli shadows. Random-Pauli shadows are the textbook starting point but
they carry a $3^{|P|}$ range factor on the per-Pauli Hoeffding radius, where
$|P|$ is the Pauli weight of the operator being estimated.

For fermionic-word observables of length $r$, the Jordan-Wigner string can
extend the Pauli weight beyond $r$, so the per-Pauli range factor scales as
$3^{|P|}$ with $|P|$ potentially much larger than $r$.

**Consequence.** Estimating $\Delta$ via random-Pauli shadows to a target
precision $\epsilon$ needs

$$
M \;\sim\; \frac{3^{2 |P|_{\max}}}{\epsilon^2}
$$

shadow shots. For $r = 3$ with widely separated sites, $3^{2 |P|_{\max}}$ can
easily exceed $10^6$.

## Mitigations

1. **Matchgate / fermionic-Gaussian shadows.** Replace the random-Pauli range
   factor with the matchgate range factor, which avoids the JW penalty entirely.
   Wired in via `cumulant_residual_cert.adapters.openfermion.delta_ucb_from_matchgate_shadows`
   in v0.3.

2. **Compute $\Delta$ from RDMs analytically.** For workflows whose post-HF
   state has 1- and 2-RDMs available, evaluating the catalog cumulants directly
   sidesteps shadow estimation entirely. Available via the PySCF adapter
   `from_rdms()` in v0.3.

3. **Use $\Delta$ as a go/no-go signal at a chosen tolerance.** If you know
   only that $\Delta \le 0.05$ for your state, plug $0.05$ into `certify()`
   and read the certified bound. Workflows whose tolerance is met under the
   loose $\Delta$ assumption do not require shadow estimation at all.

## Scoping

The random-Pauli implementation carries a Jordan-Wigner range factor, so it is
positioned as a *validity primitive* for fermionic-Gaussian or matchgate shadow
measurements rather than as an *efficiency claim*.

The diagnostic is correct for any shadow protocol that supplies per-Pauli mean
estimates and Hoeffding (or tighter) per-Pauli radii; the random-Pauli default
is the universal starting point but not the right protocol for $r \ge 3$
chemistry-scale applications.
