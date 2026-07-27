"""
hamiltonian.py

Build the reference tight-binding Hamiltonian H0 (Eq. 89) and the
spin-connection-dressed Hamiltonian H_sc (Eq. 91) on a finite {p,q}
hyperbolic lattice, using the exact geodesic Wilson-line phase (Eq. 95-96).

Key simplification (see notes below): because H_sc is block diagonal in the
spin index sigma=+/- with blocks H_+ and H_- = conj(H_+) = H_+^T, the two
blocks have identical spectra. The physical (per-state-normalized) DOS
rho_sc(E) used in the paper is therefore exactly equal to the DOS of the
single N x N complex-Hermitian block H_+ built with hopping
   -t * exp(i f_ij/2)
on each bond, where f_ij = 2*angle(1 - z_i * conj(z_j))  (l=1).
This halves the memory/compute cost relative to building the full 2N x 2N
spinor Hamiltonian, with no loss of accuracy.
"""
import numpy as np
import scipy.sparse as sp


def wilson_phase(z_i, z_j):
    """Exact geodesic Wilson-line phase f_ij, Eq. (95)-(96), l=1.
    f_ij = -i log[(1 - z_i conj(z_j))/(1 - conj(z_i) z_j)] = 2*angle(1 - z_i conj(z_j))
    (the ratio is automatically a unit-modulus number, w/conj(w))."""
    w = 1.0 - z_i * np.conj(z_j)
    return 2.0 * np.angle(w)


def build_H0(pos, edges, t=1.0):
    """Reference spinless nearest-neighbor Hamiltonian, Eq. (89)."""
    N = len(pos)
    i = edges[:, 0]
    j = edges[:, 1]
    data = np.concatenate([-t * np.ones(len(i)), -t * np.ones(len(i))])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    H0 = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)
    return H0


def build_Hsc_block(pos, edges, t=1.0):
    """Single spin block H_+ of the spin-connection Hamiltonian (Eq. 91),
    with the geodesic Wilson-line phase f_ij/2 on each bond. This has the
    same density of states as the full 2N-dim H_sc (see module docstring)."""
    N = len(pos)
    i = edges[:, 0]
    j = edges[:, 1]
    zi = pos[i]
    zj = pos[j]
    f = wilson_phase(zi, zj)  # phase for the i->j direction
    amp_ij = -t * np.exp(1j * f / 2.0)
    amp_ji = np.conj(amp_ij)
    data = np.concatenate([amp_ij, amp_ji])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    Hsc = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)
    return Hsc


def check_hermitian(H, tol=1e-10):
    diff = (H - H.getH())
    return abs(diff).max() < tol if diff.nnz > 0 else True


def check_chiral_bipartite(edges, N):
    """The {p,q} vertex graph is bipartite (all faces even for q odd? -- we
    verify directly by 2-coloring), which underlies sigma_z H sigma_z = -H
    (zero on-site energy, chiral symmetry) used in the paper to guarantee
    rho(E)=rho(-E)."""
    color = -np.ones(N, dtype=np.int8)
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    from collections import deque
    for start in range(N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False, None
    return True, color


if __name__ == "__main__":
    d = np.load("lattices_test/lattice_gen2_N490.npz")
    pos, edges = d["pos"], d["edges"]
    H0 = build_H0(pos, edges)
    Hsc = build_Hsc_block(pos, edges)
    print("N =", len(pos), " N_edges =", len(edges))
    print("H0 hermitian:", check_hermitian(H0))
    print("Hsc hermitian:", check_hermitian(Hsc))
    bip, color = check_chiral_bipartite(edges, len(pos))
    print("bipartite:", bip)
    if bip:
        # verify sigma_z H sigma_z = -H, i.e. no same-color hopping and
        # zero on-site energy (true here by construction: only edges
        # contribute, no diagonal terms).
        s = np.where(color == 0, 1, -1)
        S = sp.diags(s)
        resid0 = abs((S @ H0 @ S) - (-H0)).max()
        residsc = abs((S @ Hsc @ S) - (-Hsc)).max()
        print("chiral check H0 residual:", resid0)
        print("chiral check Hsc residual:", residsc)
