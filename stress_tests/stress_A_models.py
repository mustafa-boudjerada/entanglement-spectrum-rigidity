"""STRESS TEST A: rigidity theorem across models, sizes, cuts, drives, states.
Adversarial: try to find a (symmetric state, odd drive, permuted cut) that
VIOLATES first-order spectrum rigidity. Metrics: |dS/de| (universal), F_cl
(cluster form), F_coh. FD floor: h=1e-5 central => |dS| ~< 1e-9 means zero.
"""
import numpy as np
from numpy.linalg import eigh, eigvalsh

I2 = np.eye(2); X = np.array([[0., 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1])


def kl(ops):
    o = np.array([[1.]])
    for m in ops: o = np.kron(o, m)
    return o


def on(op, i, N): return kl([op if j == i else I2 for j in range(N)])


def rho_A(psi, LA, N):
    M = psi.reshape(2**LA, 2**(N-LA)); return M @ M.conj().T


def vN(r):
    w = np.linalg.eigvalsh(r); w = w[w > 1e-13]; return float(-np.sum(w*np.log(w)))


def fisher_split(r0, dr, tol=1e-6):
    lam, U = eigh(r0); T = U.conj().T @ dr @ U
    o = np.argsort(lam); lam = lam[o]; T = T[np.ix_(o, o)]
    n = len(lam); Fcl = 0.0; i = 0
    while i < n:
        j = i+1
        while j < n and abs(lam[j]-lam[i]) <= tol*max(1, abs(lam[i])): j += 1
        idx = list(range(i, j))
        if lam[i] > 1e-13:
            Fcl += float(np.sum(np.abs(eigvalsh(T[np.ix_(idx, idx)]))**2))/lam[i]
        i = j
    Fcoh = sum(abs(T[a, b])**2/(lam[a]+lam[b]) for a in range(n) for b in range(n)
               if a != b and lam[a]+lam[b] > 1e-10 and abs(lam[a]-lam[b]) > tol)
    return Fcl, Fcoh


def state_response(H, O, N, which=0):
    """which-th eigenstate family response chi = -(H-E_k)^+(O-<O>)psi_k."""
    w, V = eigh(H); psi = V[:, which].astype(complex); Ek = w[which]
    mu = float(np.real(psi.conj() @ O @ psi))
    src = (O - mu*np.eye(2**N)) @ psi
    chi = np.zeros_like(psi)
    for n in range(2**N):
        if n == which: continue
        d = w[n]-Ek
        if abs(d) < 1e-12: continue          # skip exact degeneracies (documented)
        chi += -V[:, n]*(V[:, n].conj() @ src)/d
    return psi, chi, w


def check(H, G, O, LA, N, label, which=0):
    psi, chi, w = state_response(H, O, N, which)
    gs = complex(psi.conj() @ G @ psi)
    godd = np.linalg.norm(G@O@G.conj().T + O)/max(np.linalg.norm(O), 1e-12)
    h = 1e-5
    def SA(e):
        p = psi+e*chi; p = p/np.linalg.norm(p); return vN(rho_A(p, LA, N))
    dS = (SA(h)-SA(-h))/(2*h)
    dr = (rho_A(psi+h*chi, LA, N)-rho_A(psi-h*chi, LA, N))/(2*h)
    Fcl, Fcoh = fisher_split(rho_A(psi, LA, N), dr)
    ok = abs(dS) < 1e-8
    print(f"{label:58s} <G>={gs.real:+.2f} odd={godd<1e-9!s:>5} "
          f"|dS|={abs(dS):.1e} F_cl={Fcl:.1e} F_coh={Fcoh:.2f} "
          f"{'RIGID' if ok else '*** VIOLATION? ***'}")
    return ok


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    results = []

    print("== A1: models x sizes x cuts (internal G, odd drives) ==")
    for N in (8, 10):
        Gz = kl([Z]*N); Gx = kl([X]*N)
        # TFIM paramagnet and deep ferromagnet (near-degenerate global pair)
        for g, tag in ((1.5, "para"), (0.5, "ferro")):
            H = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
                - g*sum(on(Z, i, N) for i in range(N))
            for LA in (2, N//2):
                results.append(check(H, Gz, sum(on(X, i, N) for i in range(N)),
                                     LA, N, f"TFIM {tag} N={N} LA={LA} sumX"))
        # XXZ at Delta=0.5 and 2.0, G=prodX(spin flip), staggered Z drive
        for D in (0.5, 2.0):
            H = sum(on(X, i, N)@on(X, i+1, N)+np.real(on(Y, i, N)@on(Y, i+1, N))
                    + D*on(Z, i, N)@on(Z, i+1, N) for i in range(N-1))
            results.append(check(H, Gx, sum((-1)**i*on(Z, i, N) for i in range(N)),
                                 N//2, N, f"XXZ D={D} N={N} stagZ"))
        # cluster model (SPT): H=-sum Z X Z ; G=prod X ; drive sum Z (odd)
        H = -sum(on(Z, i, N)@on(X, i+1, N)@on(Z, i+2, N) for i in range(N-2))
        results.append(check(H, Gx, sum(on(Z, i, N) for i in range(N)),
                             N//2, N, f"cluster(SPT) N={N} sumZ"))
        # long-range Ising 1/r^2 + field, G=prodZ, drive sumX
        H = -sum((1.0/abs(i-j)**2)*on(X, i, N)@on(X, j, N)
                 for i in range(N) for j in range(i+1, N)) \
            - 1.2*sum(on(Z, i, N) for i in range(N))
        results.append(check(H, Gz, sum(on(X, i, N) for i in range(N)),
                             N//2, N, f"long-range Ising N={N} sumX"))
        # disordered symmetric chain: random J_i XX + random h_i Z (keeps prodZ)
        J = rng.uniform(0.5, 1.5, N-1); hz = rng.uniform(0.5, 1.5, N)
        H = -sum(J[i]*on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
            - sum(hz[i]*on(Z, i, N) for i in range(N))
        results.append(check(H, Gz, sum(on(X, i, N) for i in range(N)),
                             N//2, N, f"disordered Ising N={N} sumX"))

    print("\n== A2: drive locality extremes (TFIM g=1.5, G=prodZ) ==")
    N = 10; Gz = kl([Z]*N)
    H = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
        - 1.5*sum(on(Z, i, N) for i in range(N))
    results.append(check(H, Gz, on(X, 3, N), 5, N, "single-site X_3 (odd)"))
    results.append(check(H, Gz, on(X, 0, N)@on(Z, 4, N)@on(X, 7, N), 5, N,
                         "nonlocal X0 Z4 X7 (odd: two X's? -> check parity)"))
    # random G-odd Hermitian drive (project random Herm onto odd sector)
    d = 2**N
    R = rng.standard_normal((d, d))+1j*rng.standard_normal((d, d)); R = (R+R.conj().T)/2
    Rodd = (R - Gz@R@Gz)/2
    results.append(check(H, Gz, Rodd, 5, N, "random dense G-odd Hermitian drive"))

    print("\n== A3: excited and G-odd eigenstates (rigidity beyond the ground state) ==")
    w, V = eigh(H)
    # find a G-even and a G-odd low excited state
    for k in range(1, 8):
        p = float(np.real(V[:, k] @ Gz @ V[:, k]))
        if abs(abs(p)-1) < 1e-8:
            results.append(check(H, Gz, sum(on(X, i, N) for i in range(N)), 5, N,
                                 f"excited state k={k} (parity {p:+.0f}) sumX", which=k))
        if k >= 5: break

    n_ok = sum(results); n_tot = len(results)
    print(f"\nA SUMMARY: {n_ok}/{n_tot} rigid as predicted"
          + ("" if n_ok == n_tot else "  *** INVESTIGATE FAILURES ***"))
