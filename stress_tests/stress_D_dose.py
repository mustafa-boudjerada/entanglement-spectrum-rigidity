"""STRESS TEST D: dose-response and size check.
D1: rotate the drive continuously from G-odd to G-even,
    V(theta) = cos(theta) V_odd + sin(theta) V_even  (both Frobenius-normalised
    -- the constant |dS|/sin(theta) depends on this normalisation convention).
    Prediction: |dS/de| = C * sin(theta) exactly (linear in the even
    admixture), with C the even drive's own response; exactly 0 at theta=0.
D2: N=12 spot check, odd drive vs even control.
"""
import numpy as np
from numpy.linalg import eigh

I2 = np.eye(2); X = np.array([[0., 1], [1, 0]]); Z = np.diag([1., -1])


def kl(ops):
    o = np.array([[1.]])
    for m in ops: o = np.kron(o, m)
    return o


def on(op, i, N): return kl([op if j == i else I2 for j in range(N)])


def rho_A(p, LA, N):
    M = p.reshape(2**LA, 2**(N-LA)); return M @ M.conj().T


def vN(r):
    w = np.linalg.eigvalsh(r); w = w[w > 1e-13]; return float(-np.sum(w*np.log(w)))


def dS_of(H, V, N, LA, h=1e-5):
    w, Vv = eigh(H); psi = Vv[:, 0].astype(complex); E0 = w[0]
    mu = float(np.real(psi.conj() @ V @ psi))
    src = (V - mu*np.eye(2**N)) @ psi
    chi = np.zeros_like(psi)
    for n in range(1, 2**N):
        chi += -Vv[:, n]*(Vv[:, n].conj() @ src)/(w[n]-E0)

    def SA(e):
        p = psi+e*chi; p = p/np.linalg.norm(p); return vN(rho_A(p, LA, N))
    return (SA(h)-SA(-h))/(2*h)


if __name__ == '__main__':
    print("== D1: dose-response (TFIM g=1.5, N=10, LA=5; drives normalised) ==")
    N = 10; LA = 5
    H = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
        - 1.5*sum(on(Z, i, N) for i in range(N))
    Vodd = sum(on(X, i, N) for i in range(N))
    Veven = on(Z, 1, N) + on(X, 3, N)@on(X, 6, N)
    Vodd = Vodd/np.linalg.norm(Vodd); Veven = Veven/np.linalg.norm(Veven)
    th = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    dSs = []
    for t in th:
        d = dS_of(H, np.cos(t)*Vodd+np.sin(t)*Veven, N, LA)
        dSs.append(abs(d))
        print(f"   theta={t:>4.2f}  |dS/de|={abs(d):.4e}")
    r = np.array(dSs[1:])/np.sin(th[1:])
    print(f"   |dS|/sin(theta) = {np.round(r, 5)}  (flat => strictly linear in the")
    print("   even admixture; the value is the normalised even drive's response)")

    print("\n== D2: N=12 spot check ==")
    N = 12; LA = 6
    H = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
        - 1.5*sum(on(Z, i, N) for i in range(N))
    print(f"   sumX (odd):    |dS/de|={abs(dS_of(H, sum(on(X,i,N) for i in range(N)), N, LA)):.2e}")
    print(f"   Z_1 (even ctl):|dS/de|={abs(dS_of(H, on(Z,1,N), N, LA)):.2e}")
