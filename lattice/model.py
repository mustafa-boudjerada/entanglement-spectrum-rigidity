#!/usr/bin/env python3
"""Exact-tangent solver for the finite theta=pi Schwinger rotor model in the
HANDOFF sign gauge (all off-diagonal elements -w; flux-major layout), on GPU
(CuPy) with NumPy fallback. Solves the ground state psi and the mass tangent
z, saves a handoff-schema npz (occ, psi, z, mass, flux, metadata), runs the
bundled analyzer, and enforces bridge gates.

Floating-point diagnostics only; no interval certificates. The target
fraction is never an input.

Usage:
  python model.py --N 12 --mass 0.25 --gate     # bridge gate
  python model.py --N 22 --mass 0.25 --gate     # certified-E0 gate
  python model.py --N 26 --mass 0.2375
"""
import argparse
import itertools
import json
import math
import sys
import time

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import (LinearOperator as XLinOp,
                                           eigsh as x_eigsh, cg as x_cg)
    xp = cp
    BACKEND = 'cupy'
except Exception:
    from scipy.sparse.linalg import (LinearOperator as XLinOp,
                                     eigsh as x_eigsh, cg as x_cg)
    xp = np
    BACKEND = 'numpy'

# E0 bridge gates (this session's validated values, handoff gauge).
E0_GATES = {(12, 0.25): -6.849875138213032,
            (22, 0.25): -12.501282298322119}
# certified interval at N=22: (-12.501282300, -12.501282298)
FRAC_GATES = {(12, 0.25): 0.07403785308, (22, 0.25): 0.06913242726}
# Independent cross-check (PAPER77 GPU tower, different sign gauge; energies
# are gauge-invariant; their k=1 two-pass Lanczos accuracy ~1e-8): warn-only.
E0_CROSS = {(26, 0.25): -14.7664374352}


def build(N, m, alpha=0.5, w=1.0, J=0.25, ell_min=-4, ell_max=4):
    """Handoff-convention model build (mirrors FiniteSchwinger exactly)."""
    t0 = time.time()
    occ = np.fromiter((sum(1 << j for j in c)
                       for c in itertools.combinations(range(N), N // 2)),
                      dtype=np.uint32, count=math.comb(N, N // 2))
    occ.sort()
    nocc = len(occ)
    flux = np.arange(ell_min, ell_max + 1, dtype=np.int32)
    mass_by_occ = np.zeros(nocc)
    prefix = np.zeros((N, nocc), dtype=np.int16)
    for j in range(N):
        occupied = ((occ >> np.uint32(j)) & np.uint32(1)).astype(np.int16)
        mass_by_occ += (1 if j % 2 == 0 else -1) * occupied
        prefix[j, :] = occupied - (j % 2)
    np.cumsum(prefix, axis=0, out=prefix)
    assert np.all(prefix[-1] == 0), 'neutrality failure'
    diag = np.stack([J * np.sum((float(l) + alpha + prefix) ** 2, axis=0)
                     for l in flux]) + m * mass_by_occ
    del prefix
    nb = []
    for j in range(N - 1):
        active = (((occ >> j) ^ (occ >> (j + 1))) & 1).astype(bool)
        sel = np.flatnonzero(active).astype(np.int64)
        tgt = occ[sel] ^ np.uint32(3 << j)
        idx = np.searchsorted(occ, tgt)
        assert np.array_equal(occ[idx], tgt)
        nb.append((sel, idx.astype(np.int64)))
    last = (occ >> (N - 1)) & 1
    first = occ & 1
    flip = np.uint32(1 | (1 << (N - 1)))
    refs = {}
    for name, cond in (('up', (first == 1) & (last == 0)),
                       ('dn', (first == 0) & (last == 1))):
        sel = np.flatnonzero(cond).astype(np.int64)
        tgt = occ[sel] ^ flip
        idx = np.searchsorted(occ, tgt)
        assert np.array_equal(occ[idx], tgt)
        refs[name] = (sel, idx.astype(np.int64))
    print(f'[build] N={N} nocc={nocc} dim={nocc*len(flux)} '
          f'({time.time()-t0:.1f}s)', flush=True)
    return occ, flux, mass_by_occ, diag, nb, refs


class Op:
    def __init__(self, N, m, w=1.0):
        occ, flux, mvec, diag, nb, refs = build(N, m, w=w)
        self.N, self.w = N, w
        self.occ, self.flux, self.mass_by_occ = occ, flux, mvec
        self.nf, self.nocc = len(flux), len(occ)
        self.dim = self.nf * self.nocc
        self.diag = xp.asarray(diag)
        self.nb = [(xp.asarray(s), xp.asarray(i)) for s, i in nb]
        self.up = tuple(xp.asarray(a) for a in refs['up'])
        self.dn = tuple(xp.asarray(a) for a in refs['dn'])
        self.mass_all = xp.asarray(np.tile(mvec, self.nf))

    def dot(self, x):
        X = x.reshape(self.nf, self.nocc)
        out = self.diag * X
        for sel, idx in self.nb:
            out[:, sel] -= self.w * X[:, idx]
        usel, uidx = self.up
        dsel, didx = self.dn
        # (n0=1, nN-1=0): ell -> ell+1 ; reverse for down
        out[:-1, usel] -= self.w * X[1:, uidx]
        out[1:, dsel] -= self.w * X[:-1, didx]
        return out.ravel()

    def linop(self):
        return XLinOp((self.dim, self.dim), matvec=self.dot,
                      dtype=np.float64)


def solve(op, ncv=16, tol=1e-11, maxiter=200000, cg_rtol=1e-11):
    t0 = time.time()
    try:
        vals, vecs = x_eigsh(op.linop(), k=1, which='SA', ncv=ncv,
                             maxiter=maxiter, tol=tol)
    except Exception:   # cupy eigsh without which='SA' support
        neg = XLinOp((op.dim, op.dim), matvec=lambda v: -op.dot(v),
                     dtype=np.float64)
        vals, vecs = x_eigsh(neg, k=1, which='LA', ncv=ncv,
                             maxiter=maxiter, tol=tol)
    psi = vecs[:, 0].copy()
    if float(psi.sum()) < 0:
        psi = -psi
    psi /= xp.linalg.norm(psi)
    E = float(psi @ op.dot(psi))
    # inverse-iteration refinement to solver-grade residual
    for _ in range(6):
        res = float(xp.linalg.norm(op.dot(psi) - E * psi))
        print(f'[ground] E={E:.12f} residual={res:.3e} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if res < 5e-10:
            break
        Ec = E

        def shifted(y):
            return op.dot(y) - Ec * y + psi * xp.dot(psi, y)

        sop = XLinOp((op.dim, op.dim), matvec=shifted, dtype=np.float64)
        y, info = x_cg(sop, psi, rtol=1e-8, atol=0, maxiter=50000)
        psi = y / xp.linalg.norm(y)
        if float(psi.sum()) < 0:
            psi = -psi
        E = float(psi @ op.dot(psi))
    mu = float(psi @ (op.mass_all * psi))
    v = (op.mass_all - mu) * psi
    Ef = E

    def shifted2(y):
        return op.dot(y) - Ef * y + psi * xp.dot(psi, y)

    sop = XLinOp((op.dim, op.dim), matvec=shifted2, dtype=np.float64)
    z, info = x_cg(sop, -v, rtol=cg_rtol, atol=0, maxiter=200000)
    z = z - psi * xp.dot(psi, z)
    gr = float(xp.linalg.norm(op.dot(psi) - E * psi))
    rr = float(xp.linalg.norm(op.dot(z) - E * z + v))
    print(f'[tangent] ||z||={float(xp.linalg.norm(z)):.10f} '
          f'ground_res={gr:.3e} response_res={rr:.3e} cg_info={int(info)} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return psi, z, E, mu, gr, rr, int(info), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--mass', type=float, required=True)
    ap.add_argument('--hopping', type=float, default=1.0)
    ap.add_argument('--ncv', type=int, default=16)
    ap.add_argument('--gate', action='store_true',
                    help='enforce E0/fraction bridge gates; exit 1 on fail')
    args = ap.parse_args()
    print(f'backend={BACKEND}', flush=True)
    op = Op(args.N, args.mass, w=args.hopping)
    psi, z, E, mu, gr, rr, cginfo, secs = solve(op, ncv=args.ncv)
    meta = dict(N=args.N, mass=args.mass, alpha=0.5, hopping=args.hopping,
                electric_coefficient=0.25, dimension=op.dim,
                ground_rayleigh=E, mass_expectation=mu,
                derivative_norm=float(xp.linalg.norm(z)),
                ground_residual_l2=gr, response_residual_l2=rr,
                cg_info=cginfo, solve_seconds=secs, backend=BACKEND,
                claim='Floating-point Ritz/CG diagnostics only')
    path = f'n{args.N}_m{args.mass:.4f}.npz'
    tonp = (cp.asnumpy if BACKEND == 'cupy' else np.asarray)
    np.savez_compressed(path, occ=op.occ, psi=tonp(psi), z=tonp(z),
                        mass=op.mass_by_occ, flux=np.asarray(op.flux),
                        metadata=np.array(json.dumps(meta)))
    print('SAVED', path, flush=True)
    import analyze as A
    r = A.analyze(path)
    out = f'n{args.N}_m{args.mass:.4f}_analysis.json'
    with open(out, 'w') as f:
        json.dump(r, f, indent=1)
    frac = r['spectral_fraction']
    print(f'N={args.N} m={args.mass} FRACTION={frac:.9%} '
          f"FQ={r['regional_Fisher']:.10g} Fsp={r['spectral_Fisher']:.10g} "
          f"Fch={r['charge_Fisher']:.10g} "
          f"first={r['first_mode_rotation_fraction']:.6%}", flush=True)
    key = (args.N, args.mass)
    if key in E0_CROSS:
        dev = abs(E - E0_CROSS[key])
        print(f'[cross] E0 vs PAPER77 tower: dev={dev:.2e} '
              f'({"OK" if dev < 1e-6 else "WARN: investigate"})', flush=True)
    if args.gate:
        ok = True
        if key in E0_GATES:
            d = abs(E - E0_GATES[key])
            ok &= d < 1e-8
            print(f'[gate] E0 dev={d:.2e} {"PASS" if d < 1e-8 else "FAIL"}',
                  flush=True)
        if key in FRAC_GATES:
            d = abs(frac - FRAC_GATES[key])
            ok &= d < 1e-6
            print(f'[gate] fraction dev={d:.2e} '
                  f'{"PASS" if d < 1e-6 else "FAIL"}', flush=True)
        if not ok:
            sys.exit(1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
