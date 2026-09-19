Reproducibility bundle for:
  "Rigidity of entanglement spectra under symmetry-odd perturbations, and the
   origin of the small information floor in the lattice Schwinger model at
   theta = pi", Mustafa Boudjerada (2026).

Requirements: Python 3.10+, numpy, scipy (matplotlib for the figure).

Quick start:
  python theorems/rigidity_general.py   # Table 1: the rigidity theorem (self-contained)
  python run_all.py                     # theorem + small-N Schwinger checks
  python analysis/make_figures.py       # figure

Layout:
  theorems/  model-independent verification of Theorem 1
  lattice/   Schwinger model, population/coherence decomposition, gap certificate
  analysis/  central charge, tower, figures
  data/      frozen results (lob_N12, gpulob_N28, cert_N22)

Licensed CC BY 4.0 (see LICENSE). Citation in CITATION.cff.
