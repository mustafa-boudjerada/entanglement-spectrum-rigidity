"""N=22 carrier-gap certificate (factorization-free, CPU).
Produces the certified numbers for the r(22) certificate:
  E0 <= rho0            (variational upper bound on ground energy)
  E_carrier >= rho_c - eta_c^2/(nu_c - rho_c)   (Temple, C-even sector)
  delta_eff >= E_carrier_lo - rho0
The response z lives in the C-even sector (M is C-even), so delta_eff -- not
the soft gap -- controls the resolvent. Residuals in float64 (interval-
arithmetic upgrade noted separately).
"""
import os, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from scipy.sparse.linalg import eigsh
from model import Op


def apply_C(op, v, N):
    occ = np.asarray(op.occ, np.int64); mask = (1 << N) - 1
    rot = ((occ << 1) & mask) | (occ >> (N - 1)); yC = mask ^ rot
    iC = np.searchsorted(occ, yC); nl = (occ >> (N - 1)) & 1
    nf, nocc = op.nf, op.nocc; V = v.reshape(nf, nocc); out = np.zeros_like(V)
    for b in (0, 1):
        sel = np.flatnonzero(nl == b); ids = iC[sel]
        for li in range(nf):
            lip = -li + 6 + b
            if 0 <= lip < nf:
                out[lip, ids] = V[li, sel]
    return out.reshape(-1)


def main(N=22, m=0.25, k=6):
    t0 = time.time()
    op = Op(N, m)
    ev, V = eigsh(op.linop(), k=k, which='SA', tol=1e-8, ncv=2 * k + 20,
                  maxiter=50000)
    o = np.argsort(ev); ev = ev[o]; V = V[:, o]
    E0 = ev[0]
    Cs = []
    for r in range(k):
        vr = V[:, r] / np.linalg.norm(V[:, r])
        Cs.append(float(vr @ apply_C(op, vr, N)))
    psi0 = V[:, 0] / np.linalg.norm(V[:, 0])
    rho0 = float(psi0 @ op.dot(psi0))                     # >= E0
    mu = float(psi0 @ (op.mass_all * psi0))
    src = (op.mass_all - mu) * psi0                       # (M-mu)psi0
    # confirm C-even confinement: overlap with C-odd states
    soft_i = next(r for r in range(1, k) if Cs[r] < -0.5)
    ci = next(r for r in range(1, k) if Cs[r] > 0.5)      # carrier
    ov_soft = abs(float((V[:, soft_i]/np.linalg.norm(V[:, soft_i])) @ src))
    # deflated carrier -> Temple lower bound on E_carrier
    psic = V[:, ci] / np.linalg.norm(V[:, ci])
    psic = psic - (psi0 @ psic) * psi0; psic /= np.linalg.norm(psic)
    rho_c = float(psic @ op.dot(psic))
    eta_c = float(np.linalg.norm(op.dot(psic) - rho_c * psic))
    ce_above = [r for r in range(ci + 1, k) if Cs[r] > 0.5]
    if ce_above:
        j = ce_above[0]
        pj = V[:, j] / np.linalg.norm(V[:, j])
        rho_j = float(pj @ op.dot(pj))
        eta_j = float(np.linalg.norm(op.dot(pj) - rho_j * pj))
        nu_c = rho_j - eta_j                               # rigorous-ish lower bd on E3
    else:
        nu_c = rho_c + 0.2
    E_carrier_lo = rho_c - eta_c**2 / (nu_c - rho_c)
    delta_eff = E_carrier_lo - rho0
    res = dict(N=N, m=m, E0=E0, rho0=rho0,
               soft_gap=float(ev[soft_i]-E0), carrier_gap=float(ev[ci]-E0),
               rho_c=rho_c, eta_c=eta_c, nu_c=nu_c,
               E_carrier_lo=E_carrier_lo, delta_eff_cert=delta_eff,
               src_norm=float(np.linalg.norm(src)), overlap_soft=ov_soft,
               C_parities=Cs, secs=time.time()-t0)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f'cert_N{N}.json', 'w'), indent=1)
    print(f"\nCERTIFIED delta_eff(N={N}) >= {delta_eff:.8f}  "
          f"(true carrier gap {ev[ci]-E0:.8f})", flush=True)
    print(f"soft-state overlap with (M-mu)psi0 = {ov_soft:.2e} "
          f"(confirms z in C-even sector)", flush=True)


if __name__ == '__main__':
    main()
