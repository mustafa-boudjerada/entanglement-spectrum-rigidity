"""MATH CAMPAIGN 1, O1(a)+O2: central charge vs fixed mass + gap scaling.
Verification only (N<=18, CPU). For each FIXED m, fit
  E0(N) = eps_inf*N + e1 - (pi c v/6)/N   (periodic Cardy, v=2)
and report c(m). The m where c=1/2 locates m_c AND confirms Ising (O1a).
At that m, carrier x_direct(N) -> x_eps with its correction (O2).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'lattice'))
import numpy as np
from scipy.sparse.linalg import eigsh
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


def run(N, m, k=6):
    op = Op(N, m)
    ev, vecs = eigsh(op.linop(), k=k, which='SA', tol=1e-11, ncv=2 * k + 12)
    o = np.argsort(ev); ev = ev[o]; vecs = vecs[:, o]
    E0 = ev[0]; ci = None
    for r in range(1, k):
        vv = vecs[:, r] / np.linalg.norm(vecs[:, r])
        if parity_C(op, vv, N) > 0.5:
            ci = r; break
    return E0, (ev[ci] - E0 if ci else np.nan)


def fit_c(Ns, E0s, v=2.0, const=True):
    Ns = np.asarray(Ns, float)
    cols = [Ns, -1.0 / Ns] + ([np.ones_like(Ns)] if const else [])
    A = np.vstack(cols).T
    p, *_ = np.linalg.lstsq(A, E0s, rcond=None)
    c = p[1] * 6.0 / (np.pi * v)
    rms = np.sqrt(np.mean((E0s - A @ p) ** 2))
    return c, rms


if __name__ == '__main__':
    v = 2.0
    Ns = [8, 10, 12, 14, 16, 18]
    masses = [0.26, 0.28, 0.30, 0.32, 0.339]
    print("central charge c(m) from E0 = eps_inf*N + e1 - (pi c v/6)/N :")
    table = {}
    for m in masses:
        E0s, dgs = [], []
        for N in Ns:
            E0, dg = run(N, m)
            E0s.append(E0); dgs.append(dg)
        table[m] = (np.array(E0s), np.array(dgs))
        c3, r3 = fit_c(Ns, np.array(E0s), v, const=True)
        c2, r2 = fit_c(Ns, np.array(E0s), v, const=False)
        print(f"  m={m:.3f}:  c(3-param)={c3:+.4f} rms={r3:.1e}   "
              f"c(2-param)={c2:+.4f} rms={r2:.1e}")

    # pick m closest to giving c=1/2 (3-param), report its gap scaling
    print("\ncarrier x_direct(N) at each mass (v=2):")
    for m in masses:
        _, dgs = table[m]
        x = dgs * np.array(Ns, float) / (2 * np.pi * v)
        xs = "  ".join(f"{xx:.3f}" for xx in x)
        # 1/N and 1/N^2 + 1/N combined extrapolation
        Na = np.array(Ns, float)
        A1 = np.vstack([np.ones_like(Na), 1 / Na]).T
        xinf1 = np.linalg.lstsq(A1, x, rcond=None)[0][0]
        A2 = np.vstack([np.ones_like(Na), 1 / Na, 1 / Na**2]).T
        xinf2 = np.linalg.lstsq(A2, x, rcond=None)[0][0]
        print(f"  m={m:.3f}: [{xs}]  x_inf(1/N)={xinf1:.3f}  "
              f"x_inf(1/N+1/N^2)={xinf2:.3f}")
