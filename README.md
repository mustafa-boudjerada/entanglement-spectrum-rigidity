# Entanglement-spectrum rigidity, and the origin of the small information floor in the lattice Schwinger model at θ = π

Reproducibility bundle (code + data) for the paper

> **Rigidity of entanglement spectra under symmetry-odd perturbations, and the origin of the small information floor in the lattice Schwinger model at θ = π**
> Mustafa Boudjerada (2026)
>
> Code & data: [github.com/mustafa-boudjerada/entanglement-spectrum-rigidity](https://github.com/mustafa-boudjerada/entanglement-spectrum-rigidity)

This repository reproduces every numerical statement in the paper: the
model-independent verification of the rigidity theorem, the Schwinger-model
symmetries and the population/coherence decomposition of the reduced state, the
mass-axis scaling of the population Fisher information, the region/exterior
decomposition, and the conditional gap certificate. The companion study that
first observed the small floor is *Erasability floors for parameter information
and the coherence structure of criticality in the lattice Schwinger model at
θ = π* — SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7359319 ·
code: https://github.com/mustafa-boudjerada/erasability-floor-schwinger

## What is here

| Path | Contents |
|------|----------|
| `theorems/rigidity_general.py` | Model-independent verification of the rigidity theorem (Table 1): random states, transverse-field Ising, Heisenberg, reflection (bipartition swap), and thermal (mixed) states, with the falsifiers. Reports the entropy derivative ∂S (the universally protected quantity) and the population Fisher F_cl. Self-contained (numpy/scipy). |
| `lattice/model.py` | The θ=π staggered Schwinger rotor Hamiltonian (matrix-free `Op`; NumPy/CuPy). J = 1/4, α = 1/2, w = 1 by construction. |
| `lattice/analyze.py` | Per-charge-block Schmidt decomposition of the reduced state into population (F_cl) and coherence (F_coh) Fisher information (raises on exact degeneracy). |
| `lattice/schwinger_mechanism.py` | J-tunable model, the particle-hole ⊗ flux-reflection operator Λ, the [M,C] reframing, the population Fisher at the Λ-symmetric point, and the mass-axis F_cl ∝ m² scaling. |
| `lattice/suppression.py` | Single-excitation dominance of the mass response and the region/exterior decomposition of F_cl, F_coh. |
| `lattice/gap_certificate.py` | Lemma L: certified spectral-gap lower bound (Temple + Sylvester inertia; 16·H is an exact integer matrix). Exact demonstration at small N. |
| `lattice/carrier_gap_n22.py` | The N = 22 conditional certificate (ρ0, the carrier gap, δ_eff ≥ 0.5164). |
| `lattice/investigate_gap.py` | Verification that the mass response lives in the C-even sector (soft-state decoupling). |
| `analysis/central_charge.py` | Finite-size central charge and carrier-gap checks (Ising c ≈ 1/2). |
| `analysis/verify_tower.py` | The {1, σ, ε} C-parity tower and σ/ε gap ratio. |
| `analysis/make_figures.py` | Regenerates the paper's figure from `theorems/rigidity_general.py`. |
| `data/` | Frozen results: `lob_N12.json`, `gpulob_N28.json` (E0 = −15.8994, carrier gap = 0.4220), `cert_N22.json` (ρ0 = −12.50128, δ_eff = 0.5164). |

## Requirements

- Python 3.10+
- `numpy`, `scipy` (and `matplotlib` for the figure script)

The theorem verification (`theorems/rigidity_general.py`) and the small-N
Schwinger checks run on a workstation in seconds to minutes. The N = 22–28
Schwinger runs used matrix-free Krylov diagonalisation; the frozen outputs are
in `data/` so the analysis scripts reproduce without rerunning them.

## Reproduce

```
python theorems/rigidity_general.py      # Table 1 (main theorem)
python run_all.py                         # theorem + small-N Schwinger checks
python analysis/make_figures.py           # the figure
```

## Note on data hygiene

An earlier unconverged Lanczos smoke-test file is deliberately **not** included;
the N = 12 numbers come from `data/lob_N12.json` and the eigensolver gates.

## Citation

See `CITATION.cff`. Licensed CC BY 4.0 (`LICENSE`).
