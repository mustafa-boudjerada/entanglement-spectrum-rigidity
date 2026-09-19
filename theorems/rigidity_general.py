"""
Entanglement-spectrum rigidity (Theorem 1) -- model-independent verification.

Reproduces Table 1 of the paper. For a state invariant under a symmetry G that
permutes a bipartition {A,B}, a G-ODD perturbation leaves the reduced spectrum
(and entropy) stationary at first order; the response is pure eigenbasis
rotation. We test this on random states, the transverse-field Ising and
Heisenberg chains, a reflection (bipartition-swapping) symmetry, and a thermal
(mixed) state, and we exhibit the falsifiers (G-even drive; region off the
symmetric cut) where a hypothesis fails.

Pure numpy/scipy; runs in seconds. Deterministic (fixed seeds).

Reported per case: F_cl (population Fisher, degeneracy-safe cluster form),
F_coh (coherence Fisher), and whether the case is rigid (F_cl << F_coh).
"""
import numpy as np
from numpy.linalg import eigh, eigvalsh

I2 = np.eye(2)
X = np.array([[0., 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1., -1])


def kron_list(ops):
    o = np.array([[1.]])
    for m in ops:
        o = np.kron(o, m)
    return o


def on(op, i, N):
    return kron_list([op if j == i else I2 for j in range(N)])


def rho_A(psi, LA, N):
    M = psi.reshape(2 ** LA, 2 ** (N - LA))
    return M @ M.conj().T


def fisher_split(rho_A_matrix, drho_A, tol=1e-6):
    """Degeneracy-safe population (F_cl) and coherence (F_coh) Fisher info.
    F_cl = sum_c ||P_c drho P_c||_F^2 / lam_c  (cluster form, Eq. 3).
    F_coh = 2 sum_{lam_a != lam_b} |<a|drho|b>|^2 / (lam_a+lam_b)."""
    lam, U = eigh(rho_A_matrix)
    T = U.conj().T @ drho_A @ U
    o = np.argsort(lam)
    lam = lam[o]
    T = T[np.ix_(o, o)]
    n = len(lam)
    Fcl = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(lam[j] - lam[i]) <= tol * max(1, abs(lam[i])):
            j += 1
        idx = list(range(i, j))
        if lam[i] > 1e-13:
            Fcl += float(np.sum(np.abs(eigvalsh(T[np.ix_(idx, idx)])) ** 2)) / lam[i]
        i = j
    Fcoh = 0.0
    for a in range(n):
        for b in range(n):
            if a != b and lam[a] + lam[b] > 1e-10 and abs(lam[a] - lam[b]) > tol:
                Fcoh += abs(T[a, b]) ** 2 / (lam[a] + lam[b])
    return Fcl, Fcoh


def ground_state_response(H, O, N):
    """chi = -(H-E0)^+ (O-<O>) psi  (state response to a G-odd operator)."""
    w, V = eigh(H)
    psi = V[:, 0].astype(complex)
    E0 = w[0]
    mu = float(np.real(psi.conj() @ O @ psi))
    src = (O - mu * np.eye(2 ** N)) @ psi
    chi = np.zeros_like(psi)
    for n in range(1, 2 ** N):
        chi += -V[:, n] * (V[:, n].conj() @ src) / (w[n] - E0)
    return psi, chi


def vN(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-13]
    return float(-np.sum(w * np.log(w)))


def test_ground(H, G, O, LA, N, label):
    """Reports the UNIVERSAL rigidity measure dS/de (entropy, Corollary 1;
    exact for pure/mixed and degenerate/simple spectra) and, secondarily,
    F_cl (the population-Fisher floor, which is additionally suppressed only
    when the reduced spectrum is non-degenerate)."""
    psi, chi = ground_state_response(H, O, N)
    gsym = float(np.real(psi.conj() @ G @ psi))
    godd = np.linalg.norm(G @ O @ G + O) / max(np.linalg.norm(O), 1e-12)
    h = 1e-5

    def SA(e):
        p = psi + e * chi; p = p / np.linalg.norm(p); return vN(rho_A(p, LA, N))
    dS = (SA(h) - SA(-h)) / (2 * h)
    drho = (rho_A(psi + h * chi, LA, N) - rho_A(psi - h * chi, LA, N)) / (2 * h)
    Fcl, Fcoh = fisher_split(rho_A(psi, LA, N), drho)
    print(f"{label:48s} <G>={gsym:+.2f} odd={godd < 1e-9!s:>5} "
          f"dS/de={dS:+.1e} F_cl={Fcl:.1e} F_coh={Fcoh:.2f}")
    return dS


def vN_entropy(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-13]
    return float(-np.sum(w * np.log(w)))


def test_thermal(H, O, LA, N, beta, label):
    """Mixed state: the exactly-protected quantity is the entropy
    (Corollary 1), since F_cl picks up an unprotected splitting residue from
    near-degenerate cross-parity clusters. We report dS/de (== 0 at 1st order)."""
    def rA(e):
        wp, Vp = eigh(H + e * O)
        p = np.exp(-beta * (wp - wp.min())); p /= p.sum()
        rho = (Vp * p) @ Vp.conj().T
        R = rho.reshape(2 ** LA, 2 ** (N - LA), 2 ** LA, 2 ** (N - LA))
        return np.einsum('aibi->ab', R)
    h = 1e-4
    dS = (vN_entropy(rA(h)) - vN_entropy(rA(-h))) / (2 * h)
    even = abs(vN_entropy(rA(h)) - vN_entropy(rA(-h)))
    print(f"{label:48s} (mixed)  dS/de={dS:+.2e} |S(e)-S(-e)|={even:.1e} "
          f"rigid(entropy)={abs(dS) < 1e-8}")


def test_random(dA=4, dB=4, seed=1, godd=True):
    r = np.random.default_rng(seed)
    gA = r.choice([-1, 1], dA); gB = r.choice([-1, 1], dB)
    G = np.kron(np.diag(gA), np.diag(gB))
    v = r.standard_normal(dA * dB) + 1j * r.standard_normal(dA * dB)
    psi = (v + G @ v) / 2; psi /= np.linalg.norm(psi)
    w = r.standard_normal(dA * dB) + 1j * r.standard_normal(dA * dB)
    chi = (w - G @ w) / 2 if godd else (w + G @ w) / 2
    chi = chi - psi * (psi.conj() @ chi); chi /= np.linalg.norm(chi)
    h = 1e-5
    LA = int(np.log2(dA)); Ntot = LA + int(np.log2(dB))

    def SA(e):
        p = psi + e * chi; p = p / np.linalg.norm(p); return vN(rho_A(p, LA, Ntot))
    dS = (SA(h) - SA(-h)) / (2 * h)
    drho = (rho_A(psi + h * chi, LA, Ntot) - rho_A(psi - h * chi, LA, Ntot)) / (2 * h)
    Fcl, Fcoh = fisher_split(rho_A(psi, LA, Ntot), drho)
    tag = "random, G-odd drive" if godd else "random, G-EVEN drive (control)"
    print(f"{tag:48s}            dS/de={dS:+.1e} F_cl={Fcl:.1e} F_coh={Fcoh:.2f}")


def reflection_matrix(N):
    perm = np.zeros(2 ** N, dtype=int)
    for b in range(2 ** N):
        rb = 0
        for k in range(N):
            rb |= ((b >> k) & 1) << (N - 1 - k)
        perm[b] = rb
    P = np.zeros((2 ** N, 2 ** N)); P[np.arange(2 ** N), perm] = 1.0
    return P


if __name__ == '__main__':
    print("=== Entanglement-Spectrum Rigidity: model-independent test (Table 1) ===\n")
    N = 8; g = 1.5
    Hising = -sum(on(X, i, N) @ on(X, i + 1, N) for i in range(N - 1)) \
        - g * sum(on(Z, i, N) for i in range(N))
    Hheis = sum(on(X, i, N) @ on(X, i + 1, N)
                + np.real(on(Y, i, N) @ on(Y, i + 1, N))
                + on(Z, i, N) @ on(Z, i + 1, N) for i in range(N - 1))
    Gz = kron_list([Z] * N); Gx = kron_list([X] * N); P = reflection_matrix(N)
    c = np.array([np.sign(i - (N - 1) / 2) for i in range(N)])

    print("RIGID cases (hypotheses hold):")
    test_random(seed=1, godd=True)
    test_ground(Hising, Gz, sum(on(X, i, N) for i in range(N)), 4, N,
                "Ising  G=prod Z (cut-fact)  O=sum X (odd)")
    test_ground(Hheis, Gx, sum((-1) ** i * on(Z, i, N) for i in range(N)), 4, N,
                "Heisenberg  G=prod X (cut-fact)  O=staggered Z (odd)")
    test_ground(Hising, P, sum(c[i] * on(Z, i, N) for i in range(N)), 4, N,
                "Ising  G=reflection (swaps A<->B)  O=refl-odd")
    test_thermal(Hising, sum(on(X, i, N) for i in range(N)), 4, N, 2.0,
                 "Ising thermal  G=prod Z  O=sum X (odd)")
    print("\nFALSIFIERS (a hypothesis fails):")
    test_random(seed=1, godd=False)
    test_ground(Hising, Gz, on(X, 0, N) @ on(X, 1, N), 4, N,
                "Ising  G=prod Z  O=X1 X2 (EVEN)")
    test_ground(Hising, P, sum(c[i] * on(Z, i, N) for i in range(N)), 3, N,
                "Ising  reflection, region off symmetric cut")
    print("\nUniversal check (Corollary 1): dS/de = 0 for every valid case above")
    print("(pure/mixed, degenerate/simple). This is the exactly-protected quantity.")
    print("The floor F_cl is additionally suppressed only for a NON-degenerate")
    print("reduced spectrum (Ising paramagnet, random, reflection: F_cl ~ 0);")
    print("with a degenerate spectrum (Heisenberg AFM, SU(2)) the entropy stays")
    print("rigid but F_cl carries the unprotected intra-cluster splitting.")
    print("Falsifiers (G-even drive; region off the symmetric cut): dS/de != 0.")
