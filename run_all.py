#!/usr/bin/env python3
"""Reproduce the verification tables end to end (workstation, minutes)."""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
def run(rel):
    path = os.path.join(here, rel)
    print("\n" + "=" * 70 + f"\n RUN  {rel}\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable, path], cwd=os.path.dirname(path), check=False)
if __name__ == "__main__":
    # 1. The main theorem, model-independent (self-contained).
    run("theorems/rigidity_general.py")
    # 2. Schwinger instance, small N (needs lattice/model.py, analyze.py).
    for s in ("lattice/investigate_gap.py",
              "lattice/gap_certificate.py",
              "analysis/verify_tower.py"):
        run(s)
    print("\nDone. See the paper for the N>=22 numbers (frozen in data/).")
