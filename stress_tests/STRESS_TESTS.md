# Extended validation: adversarial stress tests of the rigidity theorem

Post-publication extended validation of the entanglement-spectrum rigidity
theorem (the paper's Theorem 1). Roughly sixty independent exact-diagonalisation
tests designed to *break* the theorem, plus an adversarial audit of the test
battery itself. All scripts here are self-contained (numpy/scipy only) and run
on a workstation in minutes:

```
python stress_A_models.py    # models x sizes x cuts x drives x states
python stress_B_evenness.py  # all-orders evenness, scaling laws
python stress_C_mixed.py     # mixed states, degeneracies, antiunitary probe, QFI
python stress_D_dose.py      # dose-response, N=12 spot check
python stress_E_swap.py      # the swapping mechanism + its falsifiers
```

## Verdict

Zero violations. Every case satisfying the theorem's hypotheses is rigid;
every control violating a hypothesis fails, and fails *proportionally*
(dose-response linear in the violating admixture).

## What was tested

**A — breadth** (26 cases): transverse-field Ising (paramagnet and deep
ferromagnet), XXZ at two anisotropies, the cluster (SPT) chain, long-range
1/r² Ising, disordered symmetric chains; N = 8–12; cuts L_A = 2 … N/2; drives
from a single site to random dense symmetry-odd matrices; ground, excited
(k = 1…5), and odd-parity (⟨G⟩ = −1) eigenstates. Deep-ferromagnet case:
coherence response ~8×10⁷ while the population response stays at ~10⁻¹⁵ — a
22-order separation under an enormous response.

**B — all-orders structure**: for exact ground families of H ± εV with V odd,
|S(ε) − S(−ε)| and the full reduced-spectrum difference are zero to machine
precision at finite ε up to 0.4, while the genuine second-order response is
large (ΔS ≈ −0.12): the antisymmetric-family structure holds to all orders,
not just the first derivative. The (symmetry-breaking)² law for the
population Fisher information generalises beyond the Schwinger model
(transverse-field Ising: log-log slope 1.965). Adding an odd part to an even
drive changes the population response by exactly nothing.

**C — mixed states, degeneracies, antiunitary, QFI**: thermal states at four
temperatures and random non-thermal symmetric mixed families — exact
finite-ε evenness (internal mechanism, as the theorem's mixed-state scope
claims). Engineered exact reduced degeneracy (Bell-pair chain, cut through a
pair) with a straddling odd drive: the reduced state moves hard
(‖∂ρ_A‖ = 1.41) and the degenerate cluster *splits* (±1), while the block
trace stays exactly 0 — only the trace is protected, precisely the theorem's
degenerate form. Independent SLD-QFI cross-check: diagonal share 8×10⁻²⁸ on a
rigid case.

**D — dose-response**: rotating the drive continuously from odd to even,
|∂S|/sin θ is constant to five digits and exactly zero at θ = 0 — the
violation is strictly linear in the even admixture, the signature of a
mechanism rather than a coincidence.

**E — the swapping mechanism and its falsifiers**: nine
(reflection-symmetric model) × (reflection-odd drive) combinations —
including mirror-symmetric *disordered* couplings and excited pure
states — rigid to ≤ 5×10⁻¹⁵ at finite ε. Two falsifiers behave exactly as
the theorem requires: a *thermal* (mixed) family under a swap-odd drive is
NOT rigid (|S(ε)−S(−ε)| = 0.096 at ε = 0.1) — demonstrating that the purity
hypothesis of the swapping mechanism is load-bearing — and an asymmetric cut
(which reflection does not map to itself) is not rigid either.

## Observations beyond the paper's claims (numerical; to be developed elsewhere)

1. **Excited states**: rigidity holds for any parity eigenstate, not only
   ground states (tested k = 1…5).
2. **Odd-parity states**: states with ⟨G⟩ = −1 are equally rigid (the proof
   needs only GρG† = ρ).
3. **An antiunitary mechanism**: a K-odd drive (purely imaginary Hermitian,
   e.g. current-like operators) on a real, time-reversal-symmetric state
   gives ∂S = 0 exactly, since ρ_A(−ε) = conj(ρ_A(ε)) is isospectral —
   time-reversal-odd perturbations are also spectrum-rigid.

These are reported here as verified numerical observations with elementary
proof sketches; a systematic treatment is deferred to follow-up work.

## Audit trail

The battery itself was adversarially audited before publication. Three flaws
were found and fixed: an earlier version of the degeneracy test was vacuous
(its drive did not move the reduced state at all — replaced by the straddling
drive above); the dose-response battery existed only as an unrecorded run
(now `stress_D_dose.py`, with the drive-normalisation convention that fixes
its constant documented); and the swapping mechanism was under-tested (now
battery E). The audit also confirmed that the machine-zero evenness results
compare genuinely different matrices (‖ρ_A(ε) − ρ_A(−ε)‖ = O(1); only the
spectra coincide) and that no rigid case is response-vacuous. One known
numerical artifact: in the near-degenerate deep-ferromagnet case the entropy
derivative sits at ~5×10⁻⁹ rather than 10⁻¹¹ — eigensolver conditioning
(symmetry contamination of the numerical response at 10⁻⁹), not a theorem
effect.
