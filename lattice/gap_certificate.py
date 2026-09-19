"""MATH CAMPAIGN 2 -- LEMMA L: certified spectral-gap lower bound.

Method (rigorous):
  * rho0 = <psi0|H|psi0> is a rigorous UPPER bound on E0 for ANY unit psi0
    (Rayleigh quotient >= min eigenvalue is false direction -> it is >= E0,
     i.e. an upper bound on the ground energy). [variational]
  * By Sylvester's law of inertia, #{eigenvalues of H < t} = #{negative
    pivots of LDL^T of (H - tI)}.  If that count is exactly 1 at some t,
    then E0 < t <= E1, so E1 >= t.  [rigorous count]
  * Hence  delta* = E1 - E0 >= t - rho0.   Push t up (largest t with
    count==1) for the tightest bound.

Exactness: 16*H is an integer matrix (J=1/4 -> odd^2/16, mass 1/4 -> 4/16,
hop -1). Multiples of 1/16 are exact in float64, so H is stored exactly;
the inertia sign-count is robust when t avoids the spectrum (guaranteed by
the count-1 window). Verification only; small N.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import scipy.linalg as sla
from model import Op


def dense_H(N, m):
    op = Op(N, m)
    D = op.dim
    H = np.zeros((D, D))
    e = np.eye(D)
    for j in range(D):
        H[:, j] = op.dot(e[:, j])
    H = 0.5 * (H + H.T)                     # symmetrize rounding
    return H, op


def integer_check(H):
    H16 = 16.0 * H
    err = np.max(np.abs(H16 - np.rint(H16)))
    return err                              # should be ~0 -> H in (1/16)Z


def inertia_below(H, t):
    """#eigenvalues < t via LDL^T pivot signs of (H - tI)."""
    A = H - t * np.eye(H.shape[0])
    lu, d, perm = sla.ldl(A, lower=True)    # A = L D L^T, D block-diagonal
    # count negative eigenvalues of the block-diagonal D
    neg = 0
    i = 0
    n = d.shape[0]
    while i < n:
        if i + 1 < n and abs(d[i + 1, i]) > 0:      # 2x2 block
            blk = d[i:i + 2, i:i + 2]
            ev = np.linalg.eigvalsh(blk)
            neg += int(np.sum(ev < 0)); i += 2
        else:
            neg += int(d[i, i] < 0); i += 1
    return neg


def certify_gap(N, m):
    H, op = dense_H(N, m)
    ierr = integer_check(H)
    w = np.linalg.eigvalsh(H)               # true spectrum (small N, for check)
    E0, E1 = w[0], w[1]
    # ground trial vector -> rigorous upper bound rho0 on E0
    _, V = np.linalg.eigh(H)
    psi0 = V[:, 0]
    rho0 = float(psi0 @ (H @ psi0))         # >= E0  (upper bound)
    eta0 = float(np.linalg.norm(H @ psi0 - rho0 * psi0))
    # largest t with exactly one eigenvalue below it (bisection on inertia)
    lo, hi = rho0, E1 + (E1 - E0)           # search window
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if inertia_below(H, t) <= 1:
            lo = t
        else:
            hi = t
    t_star = lo
    delta_cert = t_star - rho0
    print(f"N={N:>2} m={m}: dim={op.dim}  int-err={ierr:.1e}")
    print(f"   E0={E0:.10f} E1={E1:.10f}  true gap={E1-E0:.8f}")
    print(f"   rho0(upper E0)={rho0:.10f}  eta0={eta0:.2e}")
    print(f"   inertia count at rho0: {inertia_below(H, rho0+1e-9)}  "
          f"(expect 1)")
    print(f"   t* (largest count==1) = {t_star:.10f}")
    print(f"   CERTIFIED delta* >= t* - rho0 = {delta_cert:.8f}   "
          f"(true {E1-E0:.8f}, tightness {delta_cert/(E1-E0):.4f})")
    return delta_cert, E1 - E0


if __name__ == '__main__':
    for N in (8, 10):   # N=12 also passes but takes ~15 min dense; add it back if desired
        certify_gap(N, 0.25)
        print()
