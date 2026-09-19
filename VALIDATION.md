# Numerical validation: the executed tests

Every number below comes from a calculation actually executed for this work
with the scripts in this repository; the tables regenerate with
`python run_all.py`. Conversely, the proposals in the paper's outlook section
(symmetry-odd quench dynamics, entanglement-Hamiltonian rigidity, metrological
protocols) have **not** been tested and are stated in the paper only as
falsifiable predictions.

Notation: F_cl = population (spectral) Fisher information of the reduced
state, computed with the degeneracy-safe cluster form; F_coh = coherence
(eigenbasis-rotation) Fisher information; ∂S = first derivative of the
entanglement entropy along the perturbation.

## 1. The model-independent battery (`theorems/rigidity_general.py`)

Setup: spin chains of N = 8 sites; transverse-field Ising chain
H = −Σ XᵢXᵢ₊₁ − g Σ Zᵢ at g = 1.5; isotropic Heisenberg chain; region A =
first 4 sites (3 for the off-cut control); thermal state at β = 2; random
bipartite states with d_A = d_B = 4 and fixed seeds. The response direction is
the exact ground-state tangent χ = −(H−E₀)⁺(O−⟨O⟩)ψ₀ built from the full
spectral resolvent. Entropy derivatives are central differences with step
h = 10⁻⁵ (10⁻⁴ thermal), so values of order 10⁻¹¹ are at the
finite-difference floor and consistent with zero.

| case | ∂S | F_cl | F_coh | verdict |
|---|---|---|---|---|
| random, G = G_A⊗G_B, G-odd drive | 0 (exact) | 0 (exact) | 0.94 | rigid |
| Ising, ∏Z, drive ΣX (odd) | +4.6×10⁻¹¹ | 3.3×10⁻²⁹ | 13.36 | rigid |
| Heisenberg, ∏X, staggered Z (odd) | +5.6×10⁻¹² | 3.07 | 8.95 | entropy rigid; F_cl not (degenerate spectrum) |
| Ising, reflection (swaps A↔B), refl.-odd Z | −1.5×10⁻¹¹ | 3.0×10⁻²⁰ | 0.091 | rigid |
| Ising thermal β = 2, ∏Z, ΣX (odd, mixed state) | 0 (exact) | — | — | entropy rigid |
| random, G-**even** drive (control) | +2.1×10⁻¹ | 1.1 | 0.61 | not rigid |
| Ising, X₁X₂ (G-**even**, control) | −1.9×10⁻³ | 1.3×10⁻⁵ | 0.051 | not rigid |
| reflection, off-cut region L_A = 3 (control) | −1.5×10⁻¹ | 6.1×10⁻² | 0.059 | not rigid |

Reading: the five rigid rows all have a genuine, non-zero response
(F_coh = O(1)) — rigidity is pure eigenbasis rotation, not the absence of a
response. The Heisenberg row is the degenerate exception: the SU(2)-degenerate
reduced spectrum keeps the entropy rigid while F_cl picks up the unprotected
intra-cluster splitting. The three controls violate one hypothesis each
(drive not odd; symmetry not bipartition-permuting) and fail as the theorem
requires.

## 2. The Schwinger instance

All at w = 1, α = 1/2, nine flux values ℓ ∈ {−4,…,4}; Hilbert dimensions
630 (N = 8), 2268 (N = 10), 8316 (N = 12) for the dense runs.
Scripts: `lattice/schwinger_mechanism.py`, `lattice/suppression.py`,
`lattice/investigate_gap.py`, `lattice/gap_certificate.py`,
`lattice/carrier_gap_n22.py`, `analysis/verify_tower.py`,
`analysis/central_charge.py`; frozen large-N results in `data/`.

**Λ-point residue** (m = J = 0, cluster form):
F_cl/F_Q = 1.9×10⁻⁸ (N = 8), 3.7×10⁻⁷ (N = 10), against F_Q = 4.15 and 6.47.

**Mass-axis law** (J = 0, m ∈ [0.005, 0.04], degeneracy-free):
log-log slope of F_cl(m) is 1.995 (N = 8) and 1.987 (N = 10) — the
(symmetry-breaking)² law.

**Degenerate-cluster counterexample** (N = 10, Λ point, cluster tolerance
10⁻⁴): a degenerate cluster with maximal diagonal element 8.7×10⁻¹⁸ but
maximal off-diagonal element 2.3×10⁻⁶; the cluster form gives
F_cl = 2.378×10⁻⁶ versus 2.368×10⁻⁶ for the diagonal-only formula, while
every cluster trace is conserved to < 10⁻¹⁶. Only the trace is protected.

**Region/exterior decomposition** (m = J = 1/4); entries are (F_cl, F_coh)
for the drive restricted to the region mass, the exterior mass, and the full
mass:

| N | region | exterior | full |
|---|---|---|---|
| 10 | (0.108, 3.97) | (0.157, 0.97) | (0.503, 8.08) |
| 12 | (0.141, 4.12) | (0.326, 1.84) | (0.821, 10.27) |
| 14 | (0.167, 4.25) | (0.495, 2.84) | (1.077, 12.47) |

The exterior ("antenna") contribution grows with N; the region contribution
is flat.

**Carrier dominance** (complete dense sum rule ‖z‖² = Σₙ(cₙ/Δₙ)², matched to
1.000000): the lowest C-even excitation carries 84.10%, 87.46%, 88.58% of
‖z‖² at N = 8, 10, 12.

**The floor is not a constant** (N = 12, J = 1/4): F_cl/F_Q = 1.92% at
m = 0.156 (dip), 7.404% at m = 1/4, 15.63% at m = 0.32.

**Spectral tower** (N = 12, m → 0.339): C-parities of the three lowest
states +1.0000, −1.0000, +1.0000; soft-to-carrier gap ratio 0.181 → 0.102
across the scan (Ising σ/ε = 0.125); two-parameter Cardy fits of E₀(N) over
N = 8–18 give c = 0.524 → 0.498 as m → 0.339.

**Gap certificate** (`lattice/gap_certificate.py`): 16·H reproduces integers
exactly (integer error 0), so Sylvester-inertia counts are decidable in exact
arithmetic. Sharpness check: N = 8 (dim 630) certifies δ ≥ 0.20749600 and
N = 10 (dim 2268) δ ≥ 0.17090573, both equal to the true gaps to displayed
precision.

**N = 22, conditional certificate** (`data/cert_N22.json`, dim 6,348,888):
ρ₀ = −12.501282298322087; carrier Rayleigh quotient −11.984857800924843 with
residual 1.2×10⁻¹²; soft-state overlap |⟨soft|(M−μ)ψ₀⟩| = 2.9×10⁻¹³;
conditional bound δ_eff ≥ 0.5164244974 (conditional on the eigenvalue-
exclusion count and an interval-arithmetic evaluation, as stated in the
paper).

**N = 28** (`data/gpulob_N28.json`, dim 361,049,400): E₀ = −15.8994280558,
carrier gap 0.4220377195; its dimensionless form 0.9404 fell inside the
pre-registered band [0.93, 0.96].
