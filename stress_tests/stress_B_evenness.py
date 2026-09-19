"""STRESS TEST B: the all-orders predictions.
B1: EXACT evenness S(eps)=S(-eps) at finite eps for exact eigenstate families
    (much stronger than dS=0: tests the antisymmetric-family structure itself).
B2: full reduced SPECTRUM even at finite eps (max_a |lam_a(e)-lam_a(-e)|).
B3: breaking-squared law in OTHER models: pre-break the symmetry by eta*B,
    measure F_cl(eta) of the odd drive -> expect F_cl ~ eta^2.
B4: mixed-parity drive decomposition: F_cl(V_odd+V_even) vs F_cl(V_even).
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


def ground(Hm):
    w, V = eigh(Hm); return V[:, 0].astype(complex)


def fisher_cl(r0, dr, tol=1e-6):
    lam, U = eigh(r0); T = U.conj().T @ dr @ U
    o = np.argsort(lam); lam = lam[o]; T = T[np.ix_(o, o)]
    n = len(lam); F = 0.0; i = 0
    while i < n:
        j = i+1
        while j < n and abs(lam[j]-lam[i]) <= tol*max(1, abs(lam[i])): j += 1
        idx = list(range(i, j))
        if lam[i] > 1e-13:
            F += float(np.sum(np.abs(eigvalsh(T[np.ix_(idx, idx)]))**2))/lam[i]
        i = j
    return F


if __name__ == '__main__':
    N = 10; LA = 5
    Gz = kl([Z]*N)
    H0 = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
         - 1.5*sum(on(Z, i, N) for i in range(N))
    Vodd = sum(on(X, i, N) for i in range(N))

    print("== B1/B2: EXACT finite-eps evenness (TFIM g=1.5, V=sumX odd) ==")
    print(f"{'eps':>6} {'|S(e)-S(-e)|':>13} {'max|lam(e)-lam(-e)|':>20} {'S(e)-S(0)':>12}")
    for e in (0.05, 0.1, 0.2, 0.4):
        pp = ground(H0+e*Vodd); pm = ground(H0-e*Vodd); p0 = ground(H0)
        rp = rho_A(pp, LA, N); rm = rho_A(pm, LA, N)
        lp = np.sort(np.linalg.eigvalsh(rp)); lm = np.sort(np.linalg.eigvalsh(rm))
        print(f"{e:>6.2f} {abs(vN(rp)-vN(rm)):>13.2e} {np.max(np.abs(lp-lm)):>20.2e} "
              f"{vN(rp)-vN(rho_A(p0,LA,N)):>+12.3e}")
    print("  [prediction: cols 2,3 machine-zero at ALL eps; col 4 = genuine")
    print("   second-order response, even in eps]")

    print("\n== B1b: same, swapping mechanism (reflection-odd drive) ==")
    perm = np.zeros(2**N, dtype=int)
    for b in range(2**N):
        rb = 0
        for k in range(N): rb |= ((b >> k) & 1) << (N-1-k)
        perm[b] = rb
    P = np.zeros((2**N, 2**N)); P[np.arange(2**N), perm] = 1.0
    c = np.array([np.sign(i-(N-1)/2) for i in range(N)])
    Vrefl = sum(c[i]*on(Z, i, N) for i in range(N))
    for e in (0.1, 0.3):
        pp = ground(H0+e*Vrefl); pm = ground(H0-e*Vrefl)
        rp = rho_A(pp, LA, N); rm = rho_A(pm, LA, N)
        lp = np.sort(np.linalg.eigvalsh(rp)); lm = np.sort(np.linalg.eigvalsh(rm))
        print(f"  eps={e}: |S(e)-S(-e)|={abs(vN(rp)-vN(rm)):.2e}  "
              f"max|dlam|={np.max(np.abs(lp-lm)):.2e}")

    print("\n== B3: breaking-squared law, TFIM (pre-break with eta*X_2) ==")
    print("   F_cl of the sumX drive at symmetry-breaking eta; expect slope 2")
    etas = np.array([0.02, 0.04, 0.08, 0.16])
    Fs = []
    for eta in etas:
        Hb = H0 + eta*on(X, 2, N)          # breaks prodZ weakly
        w, V = eigh(Hb); psi = V[:, 0].astype(complex); E0 = w[0]
        mu = float(np.real(psi.conj() @ Vodd @ psi))
        src = (Vodd - mu*np.eye(2**N)) @ psi
        chi = np.zeros_like(psi)
        for n in range(1, 2**N):
            chi += -V[:, n]*(V[:, n].conj() @ src)/(w[n]-E0)
        h = 1e-5
        dr = (rho_A(psi+h*chi, LA, N)-rho_A(psi-h*chi, LA, N))/(2*h)
        F = fisher_cl(rho_A(psi, LA, N), dr)
        Fs.append(F)
        print(f"   eta={eta:.2f}  F_cl={F:.3e}")
    p = np.polyfit(np.log(etas), np.log(Fs), 1)[0]
    print(f"   log-log slope = {p:.3f}  (breaking^2 predicts 2)")

    print("\n== B4: mixed-parity drive: F_cl(V_odd + V_even) vs F_cl(V_even) ==")
    Veven = on(Z, 1, N) + on(X, 3, N)@on(X, 6, N)
    for tag, V in (("V_even only", Veven), ("V_odd+V_even", Vodd+Veven)):
        w, Vv = eigh(H0); psi = Vv[:, 0].astype(complex); E0 = w[0]
        mu = float(np.real(psi.conj() @ V @ psi))
        src = (V - mu*np.eye(2**N)) @ psi
        chi = np.zeros_like(psi)
        for n in range(1, 2**N):
            chi += -Vv[:, n]*(Vv[:, n].conj() @ src)/(w[n]-E0)
        h = 1e-5
        dr = (rho_A(psi+h*chi, LA, N)-rho_A(psi-h*chi, LA, N))/(2*h)
        F = fisher_cl(rho_A(psi, LA, N), dr)
        print(f"   {tag:14s} F_cl={F:.6e}")
    print("   [prediction: identical to first order -- odd part contributes 0 to F_cl]")
