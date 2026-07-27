"""
Standalone, separate-checkpoint computation of extra H0 stochastic
vectors for N=571210, specifically to get a cleaner rho_0(E) curve for
the near-E=0 scaling analysis (Fig. 6 inset). Deliberately does NOT
touch the official Table I checkpoint (results/checkpoints/), to keep
Table I's stated R=12 methodology untouched and self-consistent.

Usage:
    python3 07_scaling_analysis_H0.py <n_vectors_to_add>   # accumulate
    python3 07_scaling_analysis_H0.py --analyze            # fit + save
"""
import sys, os, time
sys.path.insert(0, "src")
import numpy as np
from hamiltonian import build_H0
from kpm import estimate_spectral_radius, reconstruct_dos

CKPT = "results/scaling_analysis/moments_H0_N571210_extra.npz"
LATTICE = "lattices/lattice_gen6_N571210.npz"
M = 4096


def one_vector_moments(Hhat, M, seed, N):
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2*np.pi, size=N))
    r = phases / np.sqrt(N)
    a0 = r.copy()
    a1 = Hhat @ a0
    mu = np.empty(M, dtype=np.complex128)
    mu[0] = np.vdot(r, a0); mu[1] = np.vdot(r, a1)
    a_prev, a_cur = a0, a1
    for n in range(2, M):
        a_next = 2*(Hhat @ a_cur) - a_prev
        mu[n] = np.vdot(r, a_next)
        a_prev, a_cur = a_cur, a_next
    return mu


def main(n_this_call):
    d = np.load(LATTICE)
    pos, edges = d["pos"], d["edges"]
    N = len(pos)
    H0 = build_H0(pos, edges, t=1.0)

    try:
        ck = np.load(CKPT)
        moments = list(ck["moments_all"])
        a_scale = float(ck["a_scale"])
    except FileNotFoundError:
        a_scale = estimate_spectral_radius(H0, verify_with_power_iteration=False) * 1.01
        moments = []

    Hhat = H0 / a_scale
    R_done = len(moments)
    seed_base = 5000
    t0 = time.time()
    for k in range(n_this_call):
        seed = seed_base + R_done + k
        moments.append(one_vector_moments(Hhat, M, seed, N))
        R_done = len(moments)
        np.savez(CKPT, moments_all=np.array(moments), a_scale=a_scale, M=M)
        print(f"  vector {k+1}/{n_this_call} done ({time.time()-t0:.1f}s elapsed) -> R_total={R_done}", flush=True)
    dt = time.time() - t0
    print(f"computed {n_this_call} vectors in {dt:.1f}s -> R_total={R_done}", flush=True)


def analyze():
    """Reconstruct rho_0(E) from the accumulated checkpoint and fit the
    near-E=0 power law rho_0(E) ~ |E|^alpha over the symmetric window
    -0.01<E<0.01, using BOTH signs of E together (this doubles the
    statistics relative to a one-sided fit and serves as a built-in
    left/right consistency check -- see the two one-sided numbers
    printed below). Uncertainty is a bootstrap over the R individual
    stochastic vectors' own fitted slopes (mean +/- SEM across vectors),
    not a naive per-energy-point least-squares fit, since the ~thousands
    of energy-grid points making up one reconstructed curve are not
    independent measurements of anything -- they are one smooth curve
    with only R=len(moments) independent random degrees of freedom.
    """
    d = np.load(CKPT)
    moments, a_scale = d["moments_all"], float(d["a_scale"])
    R = len(moments)
    resolution_floor = a_scale / M  # ~7e-4 for N=571210

    E = np.linspace(-0.01, 0.01, 2001)
    curves = np.array([reconstruct_dos(mu, a_scale, 0.0, E).real for mu in moments])

    def fit(curve_1d, mask):
        Ep, rp = np.abs(E[mask]), curve_1d[mask]
        valid = rp > 0
        logE, logR_ = np.log(Ep[valid]), np.log(rp[valid])
        A = np.vstack([logE, np.ones_like(logE)]).T
        return np.linalg.lstsq(A, logR_, rcond=None)[0]

    both = np.abs(E) > resolution_floor
    pos = E > resolution_floor
    neg = E < -resolution_floor

    fits_both = np.array([fit(curves[r], both) for r in range(R)])
    fits_pos = np.array([fit(curves[r], pos)[0] for r in range(R)])
    fits_neg = np.array([fit(curves[r], neg)[0] for r in range(R)])

    alpha, alpha_sem = fits_both[:, 0].mean(), fits_both[:, 0].std(ddof=1) / np.sqrt(R)
    intercept = fits_both[:, 1].mean()
    print(f"E>0 only:  alpha = {fits_pos.mean():.3f} +/- {fits_pos.std(ddof=1)/np.sqrt(R):.3f}")
    print(f"E<0 only:  alpha = {fits_neg.mean():.3f} +/- {fits_neg.std(ddof=1)/np.sqrt(R):.3f}")
    print(f"combined:  alpha = {alpha:.3f} +/- {alpha_sem:.3f}")

    rho_mean = curves.mean(axis=0)
    rho_sem = curves.std(axis=0, ddof=1) / np.sqrt(R)
    np.savez("results/scaling_analysis/dos_H0_N571210_symmetric.npz",
             E=E, rho0=rho_mean, rho0_sem=rho_sem,
             alpha=alpha, alpha_sem=alpha_sem, intercept=intercept)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze()
    else:
        main(int(sys.argv[1]))
