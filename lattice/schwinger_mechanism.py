"""MATH CAMPAIGN 5 -- WHY is the carrier's spectral projection small?
Exact small-N (CPU, no fit, no paid compute).
 Part 1: verify single-mode dominance via a COMPLETE norm sum rule
         ||z||^2 = sum_n (c_n/Delta_n)^2, and carrier's own F_sp, F_rot.
 Part 2: single-mode from [H,M] moments  M_k = <0|(M-mu)(H-E0)^k(M-mu)|0>.
 Part 3: the Lambda identity -- F_sp = 0 at the cut-factorizing Lambda point
         (m=0,J=0); physical F_sp is the Lambda-breaking remnant.
J is tunable here (gpu_response.Op hardwires J=1/4).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import itertools, math
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from numpy.linalg import eigh, svd
import analyze as A


def build_J(N, m, J, alpha=0.5, w=1.0, ell_min=-4, ell_max=4):
    occ = np.fromiter((sum(1 << j for j in c)
                       for c in itertools.combinations(range(N), N//2)),
                      dtype=np.uint32, count=math.comb(N, N//2)); occ.sort()
    nocc = len(occ); flux = np.arange(ell_min, ell_max+1, dtype=np.int32)
    mass_by_occ = np.zeros(nocc); prefix = np.zeros((N, nocc), dtype=np.int16)
    for j in range(N):
        o = ((occ >> np.uint32(j)) & np.uint32(1)).astype(np.int16)
        mass_by_occ += (1 if j % 2 == 0 else -1)*o; prefix[j] = o-(j % 2)
    np.cumsum(prefix, axis=0, out=prefix)
    diag = np.stack([J*np.sum((float(l)+alpha+prefix)**2, axis=0) for l in flux]) \
        + m*mass_by_occ
    nb = []
    for j in range(N-1):
        active = (((occ >> j) ^ (occ >> (j+1))) & 1).astype(bool)
        sel = np.flatnonzero(active); tgt = occ[sel] ^ np.uint32(3 << j)
        idx = np.searchsorted(occ, tgt); nb.append((sel, idx))
    flip = np.uint32(1 | (1 << (N-1))); first = occ & 1; last = (occ >> (N-1)) & 1
    refs = {}
    for name, cond in (('up', (first == 1) & (last == 0)), ('dn', (first == 0) & (last == 1))):
        sel = np.flatnonzero(cond); tgt = occ[sel] ^ flip
        refs[name] = (sel, np.searchsorted(occ, tgt))
    return occ, flux, mass_by_occ, diag, nb, refs, nocc


class OpJ:
    def __init__(s, N, m, J):
        s.occ, s.flux, s.mass_by_occ, s.diag, s.nb, s.refs, s.nocc = build_J(N, m, J)
        s.N = N; s.w = 1.0; s.nf = len(s.flux); s.dim = s.nf*s.nocc
        s.mass_all = np.tile(s.mass_by_occ, s.nf)

    def dot(s, x):
        X = x.reshape(s.nf, s.nocc); out = s.diag*X
        for sel, idx in s.nb:
            out[:, sel] -= s.w*X[:, idx]
        usel, uidx = s.refs['up']; dsel, didx = s.refs['dn']
        out[:-1, usel] -= s.w*X[1:, uidx]; out[1:, dsel] -= s.w*X[:-1, didx]
        return out.ravel()

    def dense(s):
        D = s.dim; H = np.zeros((D, D)); e = np.eye(D)
        for j in range(D):
            H[:, j] = s.dot(e[:, j])
        return 0.5*(H+H.T)


def lam_apply(op, v, N):
    """Lambda = (per-site particle-hole n->1-n) x (flux reflection l->-l)."""
    occ = np.asarray(op.occ, np.int64); mask = (1 << N)-1
    comp = (~occ) & mask; idx = np.searchsorted(occ, comp)
    nf, nocc = op.nf, op.nocc; V = v.reshape(nf, nocc); out = np.zeros_like(V)
    for li in range(nf):
        out[nf-1-li, :] = V[li, idx]        # flux l->-l is li->nf-1-li; occ->complement
    return out.reshape(-1)


def fsp_frot(op, psi, z, N):
    flux = np.asarray(op.flux); blocks = []
    for k in range(7):
        if N//2-k < 0 or N//2-k > N-6: continue
        B, Zk = A.reshape_charge(psi, z, op.occ, flux, N, k)
        if np.sum(B*B) == 0: continue
        blocks.append(A.block_fisher(B, Zk, k))
    Fsp = sum(b['spectral'] for b in blocks); Frot = sum(b['rotation'] for b in blocks)
    return Fsp, Frot


def ground_tangent(op):
    H = op.dense(); w, V = eigh(H); E0 = w[0]; psi = V[:, 0]
    if psi.sum() < 0: psi = -psi
    mu = float(psi @ (op.mass_all*psi)); src = (op.mass_all-mu)*psi
    # tangent z = -(H-E0)^-1 src on {psi}^perp (pseudo-inverse via spectral)
    z = np.zeros_like(psi)
    for n in range(1, len(w)):
        cn = float(V[:, n] @ src); z += -(cn/(w[n]-E0))*V[:, n]
    return H, w, V, E0, psi, mu, src, z


def part1_2(N, m=0.25, J=0.25):
    op = OpJ(N, m, J); H, w, V, E0, psi, mu, src, z = ground_tangent(op)
    # complete norm sum rule
    cs = np.array([float(V[:, n] @ src) for n in range(len(w))]); Dl = w-E0
    weights = np.zeros(len(w)); weights[1:] = (cs[1:]/Dl[1:])**2
    znorm2 = float(z @ z)
    Ceven = []
    for n in range(1, len(w)):
        vn = V[:, n]; Ceven.append(float(vn @ lam_apply(op, vn, N)))  # Lambda-parity here
    carrier = 1+int(np.argmax(weights[1:]))
    car_share = weights[carrier]/np.sum(weights)
    print(f"[N={N} m={m} J={J}] PART 1 dominance (complete sum rule):")
    print(f"   ||z||^2={znorm2:.6g}  sum_n (c_n/D_n)^2={np.sum(weights):.6g}  "
          f"match={np.sum(weights)/znorm2:.6f}")
    print(f"   carrier n={carrier} dE={Dl[carrier]:.4f} c={cs[carrier]:.4f} "
          f"Lambda-parity={Ceven[carrier-1]:+.3f}  carrier z-share={car_share:.6%}")
    Fsp, Frot = fsp_frot(op, psi, z, N)
    zc = -(cs[carrier]/Dl[carrier])*V[:, carrier]
    Fspc, Frotc = fsp_frot(op, psi, zc, N)
    print(f"   FULL   Fsp={Fsp:.5g} Frot={Frot:.5g} r={Fsp/(Fsp+Frot):.4%}")
    print(f"   CARRIER-only Fsp={Fspc:.5g} Frot={Frotc:.5g} r_c={Fspc/(Fspc+Frotc):.4%}")
    # PART 2 moments
    r0 = src.copy(); M0 = float(r0 @ r0)
    r1 = op.dot(r0)-E0*r0; M1 = float(r0 @ r1); M2 = float(r1 @ r1)
    Dbar = M1/M0; spread = math.sqrt(max(M2/M0-Dbar**2, 0))
    print(f"   PART 2 moments: M0={M0:.4f} Dbar=M1/M0={Dbar:.4f} "
          f"spread={spread:.4f} rel={spread/Dbar:.3f}  (carrier dE={Dl[carrier]:.4f})")


def part3_lambda(N, m=0.25, J=0.25):
    """Test F_sp -> 0 as (m,J) -> Lambda-symmetric point (0,0)."""
    print(f"\n[N={N}] PART 3  Lambda-protection test: F_sp along paths to (m,J)=(0,0)")
    # verify Lambda commutes with T (J=0,m=0) and Lambda psi0 = psi0 there
    op0 = OpJ(N, 0.0, 0.0); H0 = op0.dense()
    v = np.random.default_rng(0).standard_normal(op0.dim)
    comm = np.linalg.norm(op0.dot(lam_apply(op0, v, N)) - lam_apply(op0, op0.dot(v), N))
    w0, V0 = eigh(H0); psi0 = V0[:, 0]
    lov = float(psi0 @ lam_apply(op0, psi0, N))
    print(f"   ||[T,Lambda]v||={comm:.2e} (expect 0)   <psi0|Lambda|psi0>(J=0,m=0)={lov:+.4f}")
    print(f"   {'(m,J)':>14} {'Fsp':>12} {'Frot':>10} {'r':>9}")
    for (mm, JJ) in [(0.25,0.25),(0.1,0.1),(0.05,0.05),(0.02,0.02),(0.01,0.01),
                     (0.0,0.05),(0.05,0.0),(0.02,0.0),(0.0,0.02)]:
        op = OpJ(N, mm, JJ)
        H, w, V, E0, psi, mu, src, z = ground_tangent(op)
        Fsp, Frot = fsp_frot(op, psi, z, N)
        print(f"   {str((mm,JJ)):>14} {Fsp:>12.4e} {Frot:>10.4f} {Fsp/(Fsp+Frot):>8.3%}")


if __name__ == '__main__':
    for N in (8, 10):
        part1_2(N)
    part3_lambda(8)
    part3_lambda(10)
