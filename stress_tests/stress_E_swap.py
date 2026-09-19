"""STRESS TEST E: the SWAPPING mechanism, properly exercised (audit follow-up).
The internal mechanism is an easy algebraic identity; the swap mechanism
(reflection exchanging A and B, pure families) is the nontrivial half and was
under-tested. Here: several reflection-symmetric models x several
reflection-odd drives x finite-eps exact evenness + excited states + the
purity falsifier (mixed swap family SHOULD fail).
"""
import numpy as np
from numpy.linalg import eigh

I2 = np.eye(2); X = np.array([[0., 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1])


def kl(ops):
    o = np.array([[1.]])
    for m in ops: o = np.kron(o, m)
    return o


def on(op, i, N): return kl([op if j == i else I2 for j in range(N)])


def rho_A(p, LA, N):
    M = p.reshape(2**LA, 2**(N-LA)); return M @ M.conj().T


def ptr(rho, LA, N):
    R = rho.reshape(2**LA, 2**(N-LA), 2**LA, 2**(N-LA))
    return np.einsum('aibi->ab', R)


def vN(r):
    w = np.linalg.eigvalsh(r); w = w[w > 1e-13]; return float(-np.sum(w*np.log(w)))


def reflection(N):
    perm = np.zeros(2**N, dtype=int)
    for b in range(2**N):
        rb = 0
        for k in range(N):
            rb |= ((b >> k) & 1) << (N-1-k)
        perm[b] = rb
    P = np.zeros((2**N, 2**N)); P[np.arange(2**N), perm] = 1.0
    return P


def eig_state(Hm, k=0):
    w, V = eigh(Hm); return V[:, k].astype(complex)


if __name__ == '__main__':
    N = 8; LA = 4
    P = reflection(N)
    c = np.array([np.sign(i-(N-1)/2) for i in range(N)])   # antisymmetric profile

    # reflection-SYMMETRIC models
    models = {}
    models["TFIM g=1.5"] = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
        - 1.5*sum(on(Z, i, N) for i in range(N))
    models["XXZ D=0.7"] = sum(on(X, i, N)@on(X, i+1, N)
                              + np.real(on(Y, i, N)@on(Y, i+1, N))
                              + 0.7*on(Z, i, N)@on(Z, i+1, N) for i in range(N-1))
    Jsym = np.array([1.3, 0.8, 1.1, 0.6, 1.1, 0.8, 1.3])   # mirror-symmetric couplings
    models["mirror-disordered Ising"] = \
        -sum(Jsym[i]*on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
        - 1.2*sum(on(Z, i, N) for i in range(N))

    # reflection-ODD drives (antisymmetric under j -> N-1-j)
    drives = {
        "antisym Z profile": sum(c[i]*on(Z, i, N) for i in range(N)),
        "antisym X profile": sum(c[i]*on(X, i, N) for i in range(N)),
        "antisym bond XX": sum(c[i]*on(X, i, N)@on(X, i+1, N) for i in range(N-1)),
        "single asym pair Z0-Z7": on(Z, 0, N) - on(Z, N-1, N),
    }

    print("== E1: exact finite-eps evenness, swap mechanism, ground states ==")
    print(f"{'model':>24} | {'drive':>22} | {'|S(e)-S(-e)| e=0.2':>18} {'max|dlam|':>10}")
    for mname, H in models.items():
        Psym = float(np.real(eig_state(H).conj() @ P @ eig_state(H)))
        for dname, V in drives.items():
            odd = np.linalg.norm(P@V@P.T + V)/np.linalg.norm(V)
            e = 0.2
            pp = eig_state(H+e*V); pm = eig_state(H-e*V)
            rp = rho_A(pp, LA, N); rm = rho_A(pm, LA, N)
            lp = np.sort(np.linalg.eigvalsh(rp)); lm = np.sort(np.linalg.eigvalsh(rm))
            print(f"{mname:>24} | {dname:>22} | {abs(vN(rp)-vN(rm)):>18.2e} "
                  f"{np.max(np.abs(lp-lm)):>10.2e}"
                  + ("" if odd < 1e-9 else "  [drive not odd!]"))

    print("\n== E2: swap mechanism on an EXCITED pure state (TFIM, k=2) ==")
    H = models["TFIM g=1.5"]; V = drives["antisym Z profile"]
    for k in (1, 2, 3):
        e = 0.15
        pp = eig_state(H+e*V, k); pm = eig_state(H-e*V, k)
        rp = rho_A(pp, LA, N); rm = rho_A(pm, LA, N)
        print(f"   k={k}: |S(e)-S(-e)|={abs(vN(rp)-vN(rm)):.2e}")

    print("\n== E3: purity FALSIFIER -- mixed (thermal) family under swap-odd drive ==")
    # theorem's swap mechanism requires pure families; thermal swap-odd family
    # is predicted to be UNPROTECTED (rho_A, rho_B not isospectral off pure).
    beta = 2.0
    def rA_th(e):
        w, Vv = eigh(H+e*V); p = np.exp(-beta*(w-w.min())); p /= p.sum()
        return ptr((Vv*p)@Vv.conj().T, LA, N)
    for e in (0.1, 0.2):
        print(f"   thermal beta=2, eps={e}: |S(e)-S(-e)|={abs(vN(rA_th(e))-vN(rA_th(-e))):.3e}"
              "   (nonzero PREDICTED: purity hypothesis is load-bearing)")

    print("\n== E4: asymmetric-cut FALSIFIER (LA=3, swap does not permute {A,B}) ==")
    LAx = 3
    e = 0.2
    pp = eig_state(H+e*V); pm = eig_state(H-e*V)
    rp = rho_A(pp, LAx, N); rm = rho_A(pm, LAx, N)
    print(f"   LA=3: |S(e)-S(-e)|={abs(vN(rp)-vN(rm)):.3e}   (nonzero predicted)")
