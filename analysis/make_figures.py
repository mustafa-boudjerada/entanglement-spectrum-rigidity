"""Regenerate the paper figure: entanglement-spectrum rigidity across models.

Shows the entropy derivative |dS/de| (the universally protected quantity) for
the cases where the theorem's hypotheses hold versus the falsifiers. Rigid
cases sit at machine precision; falsifiers are order 0.1. Reproduces the visual
content of Table 1. Output: figure_rigidity.pdf/.png and figure_data.csv.
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "theorems"))
import rigidity_general as R  # noqa: E402


def collect():
    N = 8
    g = 1.5
    on = R.on; X = R.X; Y = R.Y; Z = R.Z; kl = R.kron_list
    Hising = -sum(on(X, i, N) @ on(X, i + 1, N) for i in range(N - 1)) \
        - g * sum(on(Z, i, N) for i in range(N))
    Hheis = sum(on(X, i, N) @ on(X, i + 1, N)
                + np.real(on(Y, i, N) @ on(Y, i + 1, N))
                + on(Z, i, N) @ on(Z, i + 1, N) for i in range(N - 1))
    Gz = kl([Z] * N); Gx = kl([X] * N); P = R.reflection_matrix(N)
    c = np.array([np.sign(i - (N - 1) / 2) for i in range(N)])

    def dS(H, O, LA):
        psi, chi = R.ground_state_response(H, O, N)
        h = 1e-5

        def SA(e):
            p = psi + e * chi; p = p / np.linalg.norm(p)
            return R.vN(R.rho_A(p, LA, N))
        return abs((SA(h) - SA(-h)) / (2 * h))

    rows = [
        ("Ising, $\\prod Z$, $\\sum X$", dS(Hising, sum(on(X, i, N) for i in range(N)), 4), "rigid"),
        ("Heisenberg, staggered $Z$", dS(Hheis, sum((-1) ** i * on(Z, i, N) for i in range(N)), 4), "rigid"),
        ("Ising, reflection", dS(Hising, sum(c[i] * on(Z, i, N) for i in range(N)), 4), "rigid"),
        ("Ising, $X_1X_2$ (even)", dS(Hising, on(X, 0, N) @ on(X, 1, N), 4), "falsifier"),
        ("reflection, off-cut region", dS(Hising, sum(c[i] * on(Z, i, N) for i in range(N)), 3), "falsifier"),
    ]
    return rows


def main():
    rows = collect()
    with open(os.path.join(HERE, "figure_data.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "abs_dS_de", "class"])
        for name, v, cls in rows:
            w.writerow([name, f"{v:.3e}", cls])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; wrote figure_data.csv only.")
        return
    labels = [r[0] for r in rows]
    vals = [max(r[1], 1e-16) for r in rows]
    colors = ["#2c7fb8" if r[2] == "rigid" else "#d95f0e" for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_yscale("log")
    ax.axhline(1e-8, ls="--", lw=0.8, color="grey")
    ax.set_ylabel(r"$|\partial_\lambda S_A|$")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Entanglement-spectrum rigidity: entropy derivative")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"figure_rigidity.{ext}"), dpi=200)
    print("wrote figure_rigidity.pdf/.png and figure_data.csv")


if __name__ == "__main__":
    main()
