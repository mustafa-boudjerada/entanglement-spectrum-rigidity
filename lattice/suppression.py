"""MATH CAMPAIGN 4 -- is the small spectral fraction r SUPPRESSION or CANCELLATION?
Exact small-N (CPU). F_sp = 4 sum_a T_aa^2, T_aa = -sum_n (c_n/Delta_n) tau_{a,n}
(signed sum over global excitations n).  Metrics per (N,m):
  F_sp            : coherent (actual) = 4 sum_a (sum_n g_{a,n})^2
  F_sp_incoh      : 4 sum_a (sum_n |g_{a,n}|)^2   (no cancellation)
  F_sp_carrier    : carrier excitation alone
  cancel_ratio    = F_sp / F_sp_incoh   (~1 => suppression; <<1 => cancellation)
  carrier_share   = F_sp_carrier / F_sp
Also z-weight carried by the carrier, and r itself, all exact.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from scipy.sparse.linalg import eigsh
from model import Op
import analyze as A


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


def per_block_diag(op, psi, vec, N):
    """diagonal Schmidt-basis response T_aa of `vec` in ground `psi` SVD basis,
    concatenated over regional-charge blocks. Returns list of arrays."""
    flux = np.asarray(op.flux)
    out = []
    for k in range(7):
        if N // 2 - k < 0 or N // 2 - k > N - 6:
            continue
        B, Zk = A.reshape_charge(psi, vec, op.occ, flux, N, k)
        if np.sum(B * B) == 0:
            continue
        U, s, Vh = np.linalg.svd(B, full_matrices=False)
        T = U.T @ Zk @ Vh.T
        out.append(np.diag(T).copy())
    return out


def solve_ground_tangent(op):
    from model import solve
    psi, z, E, mu, gr, rr, info, secs = solve(op)
    return np.asarray(psi), np.asarray(z), E, mu


def study(N, m=0.25, kexc=16):
    op = Op(N, m)
    psi, z, E, mu = solve_ground_tangent(op)
    # exact regional decomposition (coherent, actual)
    import json, tempfile, os as _os
    # analyze needs an npz; build in-memory dict via A.block_fisher directly:
    flux = np.asarray(op.flux); blocks = []
    for k in range(7):
        if N//2-k < 0 or N//2-k > N-6: continue
        B, Zk = A.reshape_charge(psi, z, op.occ, flux, N, k)
        if np.sum(B*B) == 0: continue
        blocks.append(A.block_fisher(B, Zk, k))
    Fsp = sum(b['spectral'] for b in blocks)
    Frot = sum(b['rotation'] for b in blocks)
    r = Fsp/(Fsp+Frot)
    # excited states for the per-excitation decomposition
    ev, Vv = eigsh(op.linop(), k=kexc, which='SA', tol=1e-9, ncv=2*kexc+20)
    o = np.argsort(ev); ev = ev[o]; Vv = Vv[:, o]
    massall = np.asarray(op.mass_all)
    cs, Dl, gvecs, Cn = [], [], [], []
    for n in range(1, kexc):
        vn = Vv[:, n]/np.linalg.norm(Vv[:, n])
        cn = float(vn @ (massall*psi)); dn = ev[n]-ev[0]
        cs.append(cn); Dl.append(dn)
        Cn.append(float(vn @ apply_C(op, vn, N)))
        # per-block diagonal of -(cn/dn)|n>
        gvecs.append(per_block_diag(op, psi, -(cn/dn)*vn, N))
    cs = np.array(cs); Dl = np.array(Dl)
    # carrier = largest |c_n/Delta_n| among C-even (C>0.5)
    weight = (cs/Dl)**2
    carrier = int(np.argmax([weight[i] if Cn[i] > 0.5 else -1 for i in range(len(cs))]))
    zw_carrier = weight[carrier]/np.sum(weight)         # z-weight share of carrier
    # aggregate coherent / incoherent / carrier-only F_sp over blocks
    nblk = len(gvecs[0])
    Fsp_coh = Fsp_inc = Fsp_car = 0.0
    for b in range(nblk):
        d = len(gvecs[0][b])
        coh = np.zeros(d); inc = np.zeros(d)
        for i in range(len(gvecs)):
            g = gvecs[i][b]
            coh += g; inc += np.abs(g)
        Fsp_coh += 4*float(coh @ coh)
        Fsp_inc += 4*float(inc @ inc)
        Fsp_car += 4*float(gvecs[carrier][b] @ gvecs[carrier][b])
    print(f"=== N={N} m={m} ===  (exact)")
    print(f"  r = Fsp/FQ = {r:.4%}   Fsp={Fsp:.5g} Frot={Frot:.5g}")
    print(f"  carrier: n-index dE={ev[carrier+1]-ev[0]:.4f} C={Cn[carrier]:+.3f} "
          f"c={cs[carrier]:.4f}  z-weight share={zw_carrier:.4%}")
    print(f"  F_sp (coherent, from excitations) = {Fsp_coh:.5g}  "
          f"(vs analyzer {Fsp:.5g}; match={Fsp_coh/Fsp:.3f})")
    print(f"  F_sp (incoherent, |.| sum)        = {Fsp_inc:.5g}")
    print(f"  F_sp (carrier alone)              = {Fsp_car:.5g}")
    print(f"  CANCELLATION ratio Fsp/Fsp_incoh  = {Fsp_coh/Fsp_inc:.4f}  "
          f"(~1 => SUPPRESSION, <<1 => cancellation)")
    print(f"  carrier share Fsp_car/Fsp         = {Fsp_car/Fsp_coh:.4f}\n")
    return r


if __name__ == '__main__':
    for N in (8, 10, 12):
        study(N)
