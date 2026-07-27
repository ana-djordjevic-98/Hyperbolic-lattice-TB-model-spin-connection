#!/usr/bin/env python3
"""
02_bfs_shell_distances.py

Breadth-first search graph distance d from the 10 vertices of the central
polygon (generation 0), on the bond graph of the N=16810 lattice, needed
for the Appendix J shell-resolved analysis (Fig. A1 / Table II).

By construction in hyperbolic_lattice.generate_lattice(), the first p=10
vertices added to the lattice (indices 0..9) are exactly the vertices of
the central polygon, so they are used directly as multi-source BFS seeds.

As a correctness check, the resulting shell populations should exactly
match Table II of the paper:
  d:      0   1   2   3   4    5    6    7    8    9    10   11   12   13   14   15  16
  sites: 10  10  20  40  80  140  270  510  960 1640 2440 3000 3010 2400 1480  640 160
"""
import os
import numpy as np
from collections import deque, Counter

LATTICE_PATH = os.path.join(os.path.dirname(__file__), "..", "lattices",
                             "lattice_gen4_N16810.npz")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                         "bfs_dist_N16810.npy")


def main():
    d = np.load(LATTICE_PATH)
    pos, edges = d["pos"], d["edges"]
    N = len(pos)

    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist = -np.ones(N, dtype=int)
    q = deque()
    for v in range(10):  # central-polygon vertices, generation 0
        dist[v] = 0
        q.append(v)
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)

    assert (dist >= 0).all(), "lattice graph is disconnected!"
    max_d = dist.max()
    counts = Counter(dist.tolist())
    print(f"max BFS distance: {max_d}")
    print(f"{'d':>3} {'sites':>7}")
    for dd in range(max_d + 1):
        print(f"{dd:>3} {counts[dd]:>7}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, dist)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
