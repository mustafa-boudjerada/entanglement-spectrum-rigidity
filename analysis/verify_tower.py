"""MATH CAMPAIGN 1, step O1(b): tower-pattern test of C <-> emergent Ising Z2.
Verification only (N=12, CPU). Prints the low spectrum with C-parities and
gap ratios, to test against Ising {1:(C+,0), sigma:(C-,1/8), eps:(C+,1),...}.
No new physics campaign; existing solver, seconds of compute."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'lattice'))
import numpy as np
from scipy.sparse.linalg import eigsh
import model as gr
from model import Op


def parity_C(op, v, N):
    occ = np.asarray(op.occ, np.int64); mask = (1 << N) - 1
    rot = ((occ << 1) & mask) | (occ >> (N - 1)); yC = mask ^ rot
    iC = np.searchsorted(occ, yC); nl = (occ >> (N - 1)) & 1
    nf, nocc = op.nf, op.nocc; V = v.reshape(nf, nocc); pC = 0.0
    for b in (0, 1):
        sel = np.flatnonzero(nl == b); ids = iC[sel]
        for li in range(nf):
            lip = -li + 6 + b
            if 0 <= lip < nf:
                pC += float(np.dot(V[lip][ids], V[li][sel]))
    return pC


def tower(N, m, k=8):
    op = Op(N, m)
    ev, vecs = eigsh(op.linop(), k=k, which='SA', tol=1e-10, ncv=2*k+10)
    o = np.argsort(ev); ev = ev[o]; vecs = vecs[:, o]
    E0 = ev[0]
    # carrier = lowest C-even excited (C ~ +1); find it for ratio normalisation
    Cs, dE = [], ev - E0
    for r in range(k):
        v = vecs[:, r] / np.linalg.norm(vecs[:, r])
        Cs.append(parity_C(op, v, N))
    # locate carrier (first excited with C>0.5) and soft (first excited C<-0.5)
    carrier_i = next((r for r in range(1, k) if Cs[r] > 0.5), None)
    soft_i = next((r for r in range(1, k) if Cs[r] < -0.5), None)
    dC = dE[carrier_i] if carrier_i else np.nan
    print(f"\n=== N={N}  m={m}  (m_c~0.339) ===")
    print(f"{'n':>2} {'dE':>10} {'C-parity':>9} {'dE/carrier':>11} "
          f"{'x=dE*N/4pi':>11}")
    for r in range(k):
        ratio = dE[r] / dC if carrier_i else np.nan
        x = dE[r] * N / (2 * np.pi * 2.0)      # v=2
        tag = ''
        if r == carrier_i: tag = ' <- carrier (eps? x->1)'
        if r == soft_i:    tag = ' <- soft (sigma? x->1/8)'
        print(f"{r:>2} {dE[r]:>10.5f} {Cs[r]:>+9.4f} {ratio:>11.4f} "
              f"{x:>11.4f}{tag}")
    if carrier_i and soft_i:
        print(f"  soft/carrier gap ratio = {dE[soft_i]/dC:.4f}  "
              f"(Ising sigma/eps = 1/8 = 0.125 at criticality)")
    # C-parity pattern of the ascending tower
    patt = ''.join('+' if c > 0 else '-' for c in Cs)
    print(f"  C-parity sequence (ascending): {patt}   "
          f"(Ising expects +,-,+,... = 1,sigma,eps)")
    return dE, Cs


if __name__ == '__main__':
    for m in [0.25, 0.30, 0.32, 0.339]:
        tower(12, m, k=8)
