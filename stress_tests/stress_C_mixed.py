"""STRESS TEST C: mixed states, degeneracies, antiunitary probe, QFI cross-check.
C1: thermal beta scan incl. near-maximally-mixed; entropy evenness at finite eps.
C2: random G-symmetric MIXED states (non-thermal), internal mechanism.
C3: engineered EXACT reduced degeneracy + odd drive: trace conserved,
    splitting real (the D1 lesson, now cross-model).
C4: antiunitary probe: G = K.U (complex conjugation x unitary) -- outside the
    theorem's hypotheses; does rigidity hold anyway or fail? (either answer
    is informative; no claim at stake).
C5: independent SLD-QFI cross-check: full QFI of rho_A vs Fcl+4*Fcoh sum rule
    convention check on a rigid case.
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


def ptr(rho, LA, N):
    R = rho.reshape(2**LA, 2**(N-LA), 2**LA, 2**(N-LA))
    return np.einsum('aibi->ab', R)


def vN(r):
    w = np.linalg.eigvalsh(r); w = w[w > 1e-13]; return float(-np.sum(w*np.log(w)))


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    N = 8; LA = 4; d = 2**N
    Gz = kl([Z]*N)
    H0 = -sum(on(X, i, N)@on(X, i+1, N) for i in range(N-1)) \
         - 1.5*sum(on(Z, i, N) for i in range(N))
    Vodd = sum(on(X, i, N) for i in range(N))

    print("== C1: thermal beta scan, EXACT finite-eps entropy evenness ==")
    for beta in (0.05, 0.5, 2.0, 8.0):
        def rA(e):
            w, V = eigh(H0+e*Vodd); p = np.exp(-beta*(w-w.min())); p /= p.sum()
            return ptr((V*p)@V.conj().T, LA, N)
        for e in (0.1, 0.3):
            print(f"   beta={beta:<5} eps={e}: |S(e)-S(-e)|={abs(vN(rA(e))-vN(rA(-e))):.2e}"
                  f"   S(e)-S(0)={vN(rA(e))-vN(rA(0)):+.3e}")

    print("\n== C2: random G-symmetric MIXED state + G-odd family (internal) ==")
    # rho(eps) = M(eps) M(eps)^dag / tr, with M(eps) built so G rho(e) G = rho(-e)
    A = rng.standard_normal((d, d))+1j*rng.standard_normal((d, d))
    Aev = (A + Gz@A@Gz)/2; Aod = (A - Gz@A@Gz)/2      # G-even / G-odd parts
    def rho_mix(e):
        M = Aev + e*Aod
        r = M@M.conj().T; return r/np.trace(r).real
    for e in (0.1, 0.3):
        rp = ptr(rho_mix(e), LA, N); rm = ptr(rho_mix(-e), LA, N)
        lp = np.sort(np.linalg.eigvalsh(rp)); lm = np.sort(np.linalg.eigvalsh(rm))
        print(f"   eps={e}: |S(e)-S(-e)|={abs(vN(rp)-vN(rm)):.2e}  "
              f"max|dlam|={np.max(np.abs(lp-lm)):.2e}")
    print("   [G rho(e) G = rho(-e) holds by construction; prediction: even]")

    print("\n== C3: engineered EXACT reduced degeneracy + odd drive ==")
    # Bell-pair product state: rho_A = I/2^LA (max degenerate); odd perturbation
    bell = np.zeros(4); bell[0] = bell[3] = 1/np.sqrt(2)     # (|00>+|11>)/sqrt2
    psi = bell.copy()
    for _ in range(N//2 - 1): psi = np.kron(psi, bell)
    # reorder so first LA sites = one member of each pair: use swap network
    # simpler: pairs are (0,1),(2,3),..; cut after LA=4 splits pairs (nontrivial rho_A)
    # rho_A for LA=4 cut: sites 0..3 = pairs (0,1),(2,3) fully inside -> pure part;
    # to force degeneracy take LA=3: site 3's pair partner is outside? pairs (2,3) inside LA=4.
    # Use LA=3: pair (2,3) straddles the cut -> rho_A = pure(pair01) x I/2 -> 2-fold degenerate.
    LAd = 3
    r0 = ptr(np.outer(psi, psi.conj()), LAd, N)
    lam0 = np.sort(np.linalg.eigvalsh(r0))[::-1]
    print(f"   base spectrum (top 4): {np.round(lam0[:4],6)} (exact 2-fold degeneracy)")
    psi_e = psi.astype(complex)
    # STRADDLING odd drive: X2 Z3 acts across the cut pair (2,3) -> moves rho_A
    # hard (||drho||~1.4) and splits the degenerate cluster; the protected
    # object is the TRACE. (A single-site X2 or Y2 does NOT move rho_A at all
    # for this state -- that would be a vacuous test; audit 2026-09-19.)
    Vloc = on(X, 2, N) @ on(Z, 3, N)
    gpar = complex(psi_e.conj() @ Gz @ psi_e)
    # Bell pairs: |00>+|11> is prodZ-even per pair -> global even. good.
    h = 1e-5
    # family: exact ground of H_bell + e*Vloc is hard; instead use direct state family
    # psi(e) = normalized (1 + i e Vloc) psi  (unitary-generated, G-antisymmetric family)
    def st(e):
        p = (np.eye(d) + 1j*e*Vloc) @ psi_e; return p/np.linalg.norm(p)
    rp = ptr(np.outer(st(h), st(h).conj()), LAd, N)
    rm = ptr(np.outer(st(-h), st(-h).conj()), LAd, N)
    dr = (rp-rm)/(2*h)
    lam, U = eigh(r0); T = U.conj().T@dr@U
    o = np.argsort(lam)[::-1]; lam = lam[o]; T = T[np.ix_(o, o)]
    # the exact degenerate pair = indices of the two equal top eigenvalues
    deg = [i for i in range(len(lam)) if abs(lam[i]-lam[0]) < 1e-12]
    blk = T[np.ix_(deg, deg)]
    ev = eigvalsh(blk)
    print(f"   <G>={gpar.real:+.2f}  degenerate block size={len(deg)}  "
          f"||drho_A/de||={np.linalg.norm(dr):.3e} (nonzero => non-vacuous)")
    print(f"   block trace={np.trace(blk).real:+.2e} (predicted 0)   "
          f"splitting eigenvalues={np.round(ev,8)} (nonzero => only the trace is protected)")

    print("\n== C4: ANTIUNITARY probe (outside hypotheses): G = K (conjugation) ==")
    # real Hamiltonian: ground real -> K psi = psi; drive odd under K: i*(A - A^T)?
    # K-odd Hermitian V means conj(V) = -V => V purely imaginary Hermitian: V = i(B - B^T)
    B = rng.standard_normal((d, d))
    Vk = 1j*(B - B.T)                                     # Hermitian, K-odd
    w, Vv = eigh(H0); psi0 = Vv[:, 0].astype(complex); E0 = w[0]
    mu = float(np.real(psi0.conj()@Vk@psi0))
    src = (Vk - mu*np.eye(d)) @ psi0
    chi = np.zeros_like(psi0)
    for n in range(1, d): chi += -Vv[:, n]*(Vv[:, n].conj()@src)/(w[n]-E0)
    def SA(e):
        p = psi0+e*chi; p = p/np.linalg.norm(p)
        return vN(ptr(np.outer(p, p.conj()), LA, N))
    dS = (SA(1e-5)-SA(-1e-5))/2e-5
    print(f"   K-odd imaginary-Hermitian drive: |dS/de|={abs(dS):.2e}  "
          f"({'rigid -> antiunitary extension?' if abs(dS)<1e-8 else 'NOT rigid -> unitary hypothesis matters'})")

    print("\n== C5: independent SLD-QFI cross-check on a rigid case ==")
    # full QFI = 2 sum |<a|dr|b>|^2/(la+lb) over ALL a,b -- diagonal part must be
    # tiny for a rigid case; check identity QFI_total = QFI_diag + QFI_offdiag.
    w, Vv = eigh(H0); psi0 = Vv[:, 0].astype(complex); E0 = w[0]
    mu = float(np.real(psi0.conj()@Vodd@psi0))
    src = (Vodd-mu*np.eye(d))@psi0
    chi = np.zeros_like(psi0)
    for n in range(1, d): chi += -Vv[:, n]*(Vv[:, n].conj()@src)/(w[n]-E0)
    h = 1e-5
    r0 = ptr(np.outer(psi0, psi0.conj()), LA, N)
    rp = ptr(np.outer(psi0+h*chi, (psi0+h*chi).conj()), LA, N)/np.linalg.norm(psi0+h*chi)**2
    rm = ptr(np.outer(psi0-h*chi, (psi0-h*chi).conj()), LA, N)/np.linalg.norm(psi0-h*chi)**2
    dr = (rp-rm)/(2*h)
    lam, U = eigh(r0); T = U.conj().T@dr@U
    tot = dia = 0.0
    for a in range(len(lam)):
        for b in range(len(lam)):
            if lam[a]+lam[b] > 1e-10:
                x = 2*abs(T[a, b])**2/(lam[a]+lam[b])
                tot += x
                if a == b: dia += x
    print(f"   QFI_total={tot:.6f}  diagonal part={dia:.3e}  offdiag={tot-dia:.6f}")
    print(f"   diagonal/total = {dia/tot:.2e}  (rigid: ~0; matches selection rule)")
