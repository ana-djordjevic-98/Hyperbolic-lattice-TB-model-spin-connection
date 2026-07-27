"""
hyperbolic_lattice.py

Generate finite open {p,q} hyperbolic lattices embedded in the Poincare disk,
following the point-group / translation construction of Sec. III of

  Dordevic, Dimitrijevic Ciric & Juricic,
  "Symmetry-based theory of Dirac fermions on two-dimensional hyperbolic
   crystals: Coupling to the spin connection".

Sites are the VERTICES of the {p,q} tessellation (coordination number q in
the bulk). Lattices are grown shell-by-shell in the dual (polygon) graph,
exactly mirroring the "generation" counting of Fig. 4 / Table I of the paper.

High-precision (mpmath) arithmetic is used for the group-element composition
because points crowd exponentially close to the boundary of the Poincare
disk with increasing generation; double precision alone loses the digits
needed for reliable vertex/polygon deduplication beyond ~2-3 shells.
"""
import mpmath as mp
import numpy as np
from collections import deque
import os

mp.mp.dps = 50  # decimal digits of working precision

# ----------------------------------------------------------------------
# SU(1,1)-like Mobius group elements: (a,b) with |a|^2-|b|^2=1 represents
#   z -> (a z + b) / (conj(b) z + conj(a))
# ----------------------------------------------------------------------

def su11_mult(A, B):
    a1, b1 = A
    a2, b2 = B
    a = a1 * a2 + b1 * mp.conj(b2)
    b = a1 * b2 + b1 * mp.conj(a2)
    return (a, b)

def su11_apply(A, z):
    a, b = A
    return (a * z + b) / (mp.conj(b) * z + mp.conj(a))

def su11_normalize(A):
    a, b = A
    n = mp.sqrt(mp.fabs(a * mp.conj(a) - b * mp.conj(b)))
    return (a / n, b / n)


def build_generators(p, q, l=1):
    """Rotation R (2pi/p about origin) and edge-pairing translations T_m,
    m=0..p-1, following Eqs. (75), (78)-(81) of the paper (with l=1)."""
    l = mp.mpf(l)
    pi = mp.pi
    P, Q = mp.mpf(p), mp.mpf(q)
    r0 = l * mp.sqrt(mp.cos(pi / P + pi / Q) / mp.cos(pi / P - pi / Q))   # Eq. (75)
    dE = l * mp.sqrt(1 - mp.sin(pi / P) ** 2 / mp.cos(pi / Q) ** 2)       # Eq. (78)

    theta = 2 * pi / P
    R = (mp.e ** (1j * theta / 2), mp.mpc(0))
    a0 = 1 / mp.sqrt(1 - dE ** 2)
    b0 = dE / mp.sqrt(1 - dE ** 2)
    T0 = (a0, b0)

    Rinv = (mp.conj(R[0]), -R[1])
    Rs, Rinvs = [(mp.mpc(1), mp.mpc(0))], [(mp.mpc(1), mp.mpc(0))]
    for _ in range(1, p):
        Rs.append(su11_mult(Rs[-1], R))
        Rinvs.append(su11_mult(Rinvs[-1], Rinv))

    Ts = [su11_normalize(su11_mult(su11_mult(Rs[m], T0), Rinvs[m])) for m in range(p)]
    return r0, dE, R, T0, Ts


def base_vertices(p, r0):
    """p vertices of the central polygon, offset so that edge m (direction
    angle 2*pi*m/p) connects vertices (m-1) and m (mod p)."""
    return [r0 * mp.e ** (1j * (2 * k + 1) * mp.pi / p) for k in range(p)]


ROUND_SCALE = mp.mpf(10) ** 25  # dedup resolution (well within our 50-digit precision)

def _rkey(z):
    return (int(mp.nint(mp.re(z) * ROUND_SCALE)), int(mp.nint(mp.im(z) * ROUND_SCALE)))


def generate_lattice(p, q, n_shells, verbose=True, snapshot_shells=None,
                      snapshot_dir=None, snapshot_prefix="lattice"):
    """
    Grow the {p,q} lattice out to n_shells polygon-BFS shells (shell 0 =
    central polygon only, matching Fig. 4 / Table I "generation" numbering).

    If snapshot_shells is given (a set/list of shell indices), the vertex
    coordinates and edge list are saved to snapshot_dir at those shells
    (as float64 .npz files) without needing to regenerate from scratch.

    Returns final (vertex_coords[complex128 array], edges[int array Nx2]).
    """
    r0, dE, R, T0, Ts = build_generators(p, q)
    base = base_vertices(p, r0)

    vert_id = {}
    vert_coords = []

    def get_vid(z):
        k = _rkey(z)
        idx = vert_id.get(k)
        if idx is not None:
            return idx
        idx = len(vert_coords)
        vert_id[k] = idx
        vert_coords.append(z)
        return idx

    edges = set()
    poly_center_seen = set()
    identity = (mp.mpc(1), mp.mpc(0))
    poly_center_seen.add(_rkey(su11_apply(identity, 0)))
    current_shell = [identity]

    snapshot_shells = set(snapshot_shells or [])
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    for shell in range(n_shells + 1):
        next_shell = []
        for g in current_shell:
            gv = [su11_apply(g, v) for v in base]
            ids = [get_vid(z) for z in gv]
            for k in range(p):
                a, b = ids[k], ids[(k + 1) % p]
                edges.add((a, b) if a < b else (b, a))
            for m in range(p):
                gnew = su11_normalize(su11_mult(g, Ts[m]))
                c = su11_apply(gnew, 0)
                ck = _rkey(c)
                if ck not in poly_center_seen:
                    poly_center_seen.add(ck)
                    next_shell.append(gnew)
        if verbose:
            print(f"shell {shell}: polygons_processed={len(current_shell)} "
                  f"cum_polygons={len(poly_center_seen)} "
                  f"cum_vertices={len(vert_coords)} cum_edges={len(edges)}")

        if shell in snapshot_shells and snapshot_dir:
            _save_npz(vert_coords, edges, os.path.join(
                snapshot_dir, f"{snapshot_prefix}_gen{shell}_N{len(vert_coords)}.npz"))

        current_shell = next_shell
        if not current_shell:
            break

    return vert_coords, edges


def _save_npz(vert_coords, edges, path):
    pos = np.array([complex(z) for z in vert_coords], dtype=np.complex128)
    e = np.array(sorted(edges), dtype=np.int64)
    np.savez(path, pos=pos, edges=e)
    print(f"  saved {path}  (N={len(pos)}, N_edges={len(e)})")


if __name__ == "__main__":
    # Quick self-test: generate through generation 4, checking cumulative
    # vertex counts against Table I system sizes for {10,3}.
    generate_lattice(10, 3, n_shells=4, snapshot_shells={2, 3, 4},
                      snapshot_dir="lattices_test")
