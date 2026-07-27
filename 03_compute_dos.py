#!/usr/bin/env python3
"""
03_compute_dos.py

Compute rho_0(E) (Eq. 89, no spin connection) and rho_sc(E) (Eq. 91,
geodesic Wilson-line spin connection) via the Kernel Polynomial Method,
for one lattice generation, with M=4096 Chebyshev moments and R = 150, 72,
36, 12, 12 stochastic vectors for N = 490, 2880, 16810, 98000, 571210
respectively (Table I).

Unlike a naive implementation that only accumulates the running sum of
Chebyshev moments over the R random-phase vectors, this script keeps each
vector's individual contribution (storage cost is trivial: at most
R x M complex128 ~ 150*4096*16 bytes ~ 10 MB). That per-vector ensemble is
what makes the following honest rather than assumed or omitted:

  - Standard error. rho(0) (and the full rho(E) curve) gets a proper
    standard error of the mean, sigma/sqrt(R), computed from the actual
    spread of the R independent KPM estimates -- not stated without a
    number, and not implying six-decimal precision from R=12 samples.
  - Particle-hole symmetrization. H0 and H_sc are exactly chiral-symmetric
    (sigma_z H sigma_z = -H, verified in hamiltonian.py), so the true
    trace obeys rho(E)=rho(-E) exactly. A finite-R stochastic estimate
    does not, and that residual asymmetry is pure sampling noise. Both
    rho(E) and the symmetrized rho_sym(E)=(rho(E)+rho(-E))/2 are saved;
    04_make_figures_6_7_8.py plots the symmetrized curve and reports the
    unsymmetrized asymmetry as a noise diagnostic rather than silently
    discarding it.

A second, easy-to-miss noise source is removed as a prerequisite for the
above to mean what it says: scipy's ARPACK-based spectral-radius estimate
(used to set the KPM rescaling a_scale) draws an unseeded starting vector
internally, so naively it returns a slightly different a_scale on every
call. Because the Chebyshev recursion is carried to M~4000 (where T_n(x)
has derivative ~n^2 near |x|=1), even a ~1e-7 relative change in a_scale
can shift the reconstructed rho(0) by a percent-level amount -- large
enough to contaminate an R-vector error budget with an unrelated noise
source. src/kpm.py's estimate_spectral_radius() now seeds its ARPACK call
(seed=0 by default), making a_scale exactly reproducible so that R-vector
sampling is the only remaining source of run-to-run variation captured by
the standard error computed here.

USAGE
-----
Small/medium lattices (finishes in well under a minute to ~1 min):
    python3 03_compute_dos.py --N 490
    python3 03_compute_dos.py --N 2880
    python3 03_compute_dos.py --N 16810
    python3 03_compute_dos.py --N 98000

Large lattice (N=571210): ~13-15 minutes in one call. If your environment
has a shorter execution-time limit per call, use --chunk to process a
handful of random vectors per invocation; progress is checkpointed to
results/checkpoints/ and the script auto-resumes:

    python3 03_compute_dos.py --N 571210 --chunk 4   # repeat until done
    python3 03_compute_dos.py --N 571210 --chunk 4
    ...
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hamiltonian import build_H0, build_Hsc_block
from kpm import estimate_spectral_radius, reconstruct_dos

ROOT = os.path.join(os.path.dirname(__file__), "..")
LATTICE_TMPL = {
    490:    "lattice_gen2_N490.npz",
    2880:   "lattice_gen3_N2880.npz",
    16810:  "lattice_gen4_N16810.npz",
    98000:  "lattice_gen5_N98000.npz",
    571210: "lattice_gen6_N571210.npz",
}
R_TABLE = {490: 150, 2880: 72, 16810: 36, 98000: 12, 571210: 12}  # Table I
M_DEFAULT = 4096
E_GRID = np.linspace(-0.15, 0.15, 6001)
I0 = int(np.argmin(np.abs(E_GRID)))  # index of E=0


def lattice_path(N):
    return os.path.join(ROOT, "lattices", LATTICE_TMPL[N])


def checkpoint_path(N, which):
    d = os.path.join(ROOT, "results", "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"moments_all_N{N}_{which}.npz")


def out_path(N):
    d = os.path.join(ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"dos_N{N}.npz")


def build_H(which, pos, edges):
    return build_H0(pos, edges) if which == "H0" else build_Hsc_block(pos, edges)


def one_vector_moments(Hhat, M, seed, N):
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=N))
    r = phases / np.sqrt(N)
    a0 = r.copy()
    a1 = Hhat @ a0
    mu = np.empty(M, dtype=np.complex128)
    mu[0] = np.vdot(r, a0)
    mu[1] = np.vdot(r, a1)
    a_prev, a_cur = a0, a1
    for n in range(2, M):
        a_next = 2 * (Hhat @ a_cur) - a_prev
        mu[n] = np.vdot(r, a_next)
        a_prev, a_cur = a_cur, a_next
    return mu


def run_hamiltonian(which, N, pos, edges, M, R_target, n_vectors_this_call, seed_base):
    path = checkpoint_path(N, which)
    H = build_H(which, pos, edges)
    try:
        ck = np.load(path)
        moments_all = list(ck["moments_all"])
        a_scale = float(ck["a_scale"])
    except FileNotFoundError:
        a_scale = estimate_spectral_radius(H, verify_with_power_iteration=False) * 1.01
        moments_all = []

    Hhat = H / a_scale
    R_done = len(moments_all)
    n_todo = min(n_vectors_this_call, R_target - R_done) if n_vectors_this_call else (R_target - R_done)

    if n_todo > 0:
        t0 = time.time()
        for k in range(n_todo):
            seed = seed_base + R_done + k
            mu = one_vector_moments(Hhat, M, seed, N)
            moments_all.append(mu)
        dt = time.time() - t0
        R_done = len(moments_all)
        print(f"  [{which}] computed {n_todo} vector(s) in {dt:.1f}s "
              f"-> {R_done}/{R_target} done", flush=True)
        np.savez(path, moments_all=np.array(moments_all), a_scale=a_scale, M=M)
    else:
        print(f"  [{which}] already at {R_done}/{R_target}", flush=True)

    return np.array(moments_all), R_done, a_scale


def per_vector_curves(moments_all, a_scale, energies):
    """rho(E) reconstructed separately from each vector's own moments,
    stacked as an (R, n_energies) array."""
    return np.array([reconstruct_dos(mu, a_scale, 0.0, energies) for mu in moments_all]).real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True, choices=list(R_TABLE.keys()))
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--chunk", type=int, default=0,
                     help="Number of random vectors to compute THIS call, "
                          "per Hamiltonian (0 = do all R at once).")
    args = ap.parse_args()

    N, M = args.N, args.M
    R_target = R_TABLE[N]

    d = np.load(lattice_path(N))
    pos, edges = d["pos"], d["edges"]
    assert len(pos) == N

    print(f"N={N}  R_target={R_target}  M={M}  chunk={args.chunk or 'ALL'}")
    mus0, R0, a0 = run_hamiltonian("H0", N, pos, edges, M, R_target, args.chunk, seed_base=1000)
    mussc, Rsc, asc = run_hamiltonian("Hsc", N, pos, edges, M, R_target, args.chunk, seed_base=2000)

    if R0 < R_target or Rsc < R_target:
        print(f"Not yet complete: H0 {R0}/{R_target}, Hsc {Rsc}/{R_target}. "
              f"Re-run this command again to continue.")
        return

    curves0 = per_vector_curves(mus0, a0, E_GRID)     # (R, n_E)
    curvessc = per_vector_curves(mussc, asc, E_GRID)

    rho0_mean = curves0.mean(axis=0)
    rho0_sem = curves0.std(axis=0, ddof=1) / np.sqrt(R0)
    rhosc_mean = curvessc.mean(axis=0)
    rhosc_sem = curvessc.std(axis=0, ddof=1) / np.sqrt(Rsc)

    # Particle-hole symmetrization + residual-asymmetry diagnostic (see
    # module docstring). E_GRID is symmetric about 0, so E_GRID[::-1] ==
    # -E_GRID and curve[::-1] directly gives rho(-E) on the same grid.
    rho0_sym = 0.5 * (rho0_mean + rho0_mean[::-1])
    rhosc_sym = 0.5 * (rhosc_mean + rhosc_mean[::-1])
    asym0 = 0.5 * (rho0_mean - rho0_mean[::-1])
    asymsc = 0.5 * (rhosc_mean - rhosc_mean[::-1])

    print(f"rho0(0)  = {rho0_mean[I0]:.4e} +/- {rho0_sem[I0]:.1e}")
    print(f"rhosc(0) = {rhosc_mean[I0]:.4f} +/- {rhosc_sem[I0]:.4f}")
    print(f"max |asymmetry|: rho0 {np.max(np.abs(asym0)):.2e}, "
          f"rhosc {np.max(np.abs(asymsc)):.2e} "
          f"({100*np.max(np.abs(asymsc))/np.max(np.abs(rhosc_mean)):.2f}% of peak)")

    np.savez(out_path(N), N=N, R=R_target, E=E_GRID,
             rho0=rho0_mean, rho0_sem=rho0_sem,
             rhosc=rhosc_mean, rhosc_sem=rhosc_sem,
             rho0_sym=rho0_sym, rhosc_sym=rhosc_sym,
             asym0=asym0, asymsc=asymsc, a0=a0, asc=asc)
    print("saved", out_path(N))


if __name__ == "__main__":
    main()
