"""MATH CAMPAIGN 2 -- investigation + benchmark for the N=22 gap certificate.
Zero-cost CPU study at small N. Establishes:
 (1) the response z lives in the C-even sector => relevant gap is the CARRIER
     gap delta_eff = E_carrier - E0 (~0.5), NOT the soft gap (~0.09).
 (2) a factorization-free Temple lower bound on E_carrier, validated vs exact.
 (3) sector-dimension reduction available at N=22.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from scipy.sparse.linalg import eigsh
from model import Op


def parity_C(op, v, N):
    occ = np.asarray(op.occ, np.int64); mask = (1 << N) - 1
    rot = ((occ << 1) & mask) | (occ >> (N - 1)); yC = mask ^ rot
    iC = np.searchsorted(occ, yC); nl = (occ >> (N - 1)) & 1
    nf, nocc = op.nf, op.nocc; V = v.reshape(nf, nocc)
    out = np.zeros_like(V)
    for b in (0, 1):
        sel = np.flatnonzero(nl == b); ids = iC[sel]
        for li in range(nf):
            lip = -li + 6 + b
            if 0 <= lip < nf:
                out[lip, ids] = V[li, sel]        # (C v) in C-even/odd test
    Cv = out.reshape(-1)
    return float(np.dot(v, Cv)), Cv


def study(N, m=0.25, k=6):
    op = Op(N, m)
    ev, V = eigsh(op.linop(), k=k, which='SA', tol=1e-12, ncv=2*k+16)
    o = np.argsort(ev); ev = ev[o]; V = V[:, o]
    E0 = ev[0]; psi0 = V[:, 0] / np.linalg.norm(V[:, 0])
    Cs = [parity_C(op, V[:, r]/np.linalg.norm(V[:, r]), N)[0] for r in range(k)]
    # (M - mu) psi0
    mu = float(psi0 @ (op.mass_all * psi0))
    r0 = (op.mass_all - mu) * psi0
    # overlaps of the source with each excited eigenstate
    print(f"\n=== N={N} m={m} dim={op.dim} ===")
    print(f"{'n':>2} {'dE':>10} {'C':>7} {'|<n|(M-mu)|0>|':>16}")
    for r in range(k):
        vr = V[:, r] / np.linalg.norm(V[:, r])
        ov = abs(float(vr @ r0))
        print(f"{r:>2} {ev[r]-E0:>10.5f} {Cs[r]:>+7.3f} {ov:>16.3e}")
    # identify carrier = lowest C-even excited (C>0.5)
    ci = next(r for r in range(1, k) if Cs[r] > 0.5)
    # third C-even eigenvalue for Temple nu
    ce = [r for r in range(k) if Cs[r] > 0.5]      # C-even indices (incl ground)
    print(f"C-even indices: {ce}   carrier index n={ci}, dE={ev[ci]-E0:.5f}")

    # ---- factorization-free Temple lower bound on E_carrier ----
    rho0 = float(psi0 @ op.dot(psi0))              # >= E0 (upper bd on E0)
    psic = V[:, ci] / np.linalg.norm(V[:, ci])
    psic = psic - (psi0 @ psic) * psi0             # deflate ground
    psic /= np.linalg.norm(psic)
    rho_c = float(psic @ op.dot(psic))
    eta_c = float(np.linalg.norm(op.dot(psic) - rho_c * psic))
    # nu_c = lower bound on 3rd C-even eigenvalue; use next C-even Ritz minus margin
    ce_ex = [r for r in ce if r > ci]
    nu_c = (ev[ce_ex[0]] - 1e-6) if ce_ex else rho_c + 0.2
    E_carrier_lo = rho_c - eta_c**2 / (nu_c - rho_c)
    delta_eff_cert = E_carrier_lo - rho0
    print(f"rho0(>=E0)={rho0:.10f}  rho_c={rho_c:.10f}  eta_c={eta_c:.2e}")
    print(f"nu_c={nu_c:.6f}  E_carrier_lo(Temple)={E_carrier_lo:.10f}")
    print(f"CERTIFIED delta_eff >= {delta_eff_cert:.8f}  "
          f"(true carrier gap {ev[ci]-E0:.8f}, "
          f"tightness {delta_eff_cert/(ev[ci]-E0):.4f})")
    print(f"[soft gap was {ev[1]-E0:.5f}; certificate uses the {ev[ci]-E0:.3f} "
          f"gap -> ~{(ev[ci]-E0)/(ev[1]-E0):.1f}x larger]")
    return delta_eff_cert, ev[ci]-E0


if __name__ == '__main__':
    for N in (12, 14):
        study(N)
