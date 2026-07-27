#!/usr/bin/env python3
"""
01_generate_lattices.py

Generate the five {10,3} hyperbolic-lattice system sizes used throughout
the paper (Table I): N = 490, 2880, 16810, 98000, 571210, corresponding to
polygon-BFS generations 2-6 (Fig. 4 convention). Lattices are saved as
.npz files (complex128 vertex positions on the Poincare disk + integer
edge list) under lattices/.

Runtime: a few seconds for the small sizes, ~2.5 min for the largest
(N=571210), dominated by the high-precision (mpmath) Mobius-group
arithmetic needed to keep vertex/polygon deduplication reliable at depth,
since points crowd exponentially close to the disk boundary.

As a built-in correctness check, the printed cumulative vertex counts
after each shell should read exactly:
    shell 2: 490      shell 3: 2880     shell 4: 16810
    shell 5: 98000    shell 6: 571210
matching Table I of the paper.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hyperbolic_lattice import generate_lattice

P, Q = 10, 3
N_SHELLS = 6
TARGET_SHELLS = {2, 3, 4, 5, 6}
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "lattices")

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    generate_lattice(P, Q, n_shells=N_SHELLS, snapshot_shells=TARGET_SHELLS,
                      snapshot_dir=OUTDIR, verbose=True)
