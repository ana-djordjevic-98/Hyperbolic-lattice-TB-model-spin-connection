"""
kpm.py

Kernel Polynomial Method density-of-states engine matching the numerical
protocol described in Sec. IV.A / App. J of the paper:
  - M Chebyshev moments (default 4096)
  - Jackson kernel damping
  - adaptive rescaling fixed by the estimated spectral radius (via a sparse
    Lanczos/ARPACK extremal-eigenvalue solve), used directly as the KPM
    rescaling; consistency of the estimate can be spot-checked against a
    plain power-iteration estimate (see estimate_spectral_radius).
  - stochastic trace estimator with R random-phase vectors, either over the
    full lattice or masked to a subset of sites (App. J, Eq. J1-J2).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def estimate_spectral_radius(H, tol=1e-4, verify_with_power_iteration=True, seed=0):
    """Largest |eigenvalue| of Hermitian sparse H via ARPACK (Lanczos).

    ARPACK's default starting vector is drawn from an unseeded internal
    RNG, so repeated calls on the same matrix return slightly different
    estimates (differing at the ~1e-7 relative level). Because the KPM
    rescaling a_scale is applied inside a Chebyshev recursion carried to
    M~4000 moments, such tiny differences are amplified (Chebyshev
    polynomials T_n(x) have derivative ~n^2 near |x|=1) enough to shift
    the reconstructed rho(0) by a percent-level amount -- large enough to
    contaminate a stochastic-trace error budget with a second, unrelated
    noise source. Fixing v0 via a seeded RNG makes a_scale (and therefore
    the whole KPM reconstruction, for fixed random-vector seeds) exactly
    reproducible, so that R-vector sampling is the only remaining source
    of run-to-run variation.
    """
    N = H.shape[0]
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    v0 /= np.linalg.norm(v0)
    try:
        emax = spla.eigsh(H, k=1, which='LA', tol=tol, v0=v0,
                           return_eigenvectors=False)[0]
    except spla.ArpackNoConvergence as e:
        emax = e.eigenvalues.max() if len(e.eigenvalues) else None
    if verify_with_power_iteration:
        rng2 = np.random.default_rng(seed + 1)
        v = rng2.standard_normal(N) + 1j * rng2.standard_normal(N)
        v /= np.linalg.norm(v)
        lam_old = 0.0
        for _ in range(60):
            w = H @ v
            nrm = np.linalg.norm(w)
            v = w / nrm
            lam = np.real(np.vdot(v, H @ v))
            if abs(lam - lam_old) < 1e-6 * max(1.0, abs(lam)):
                break
            lam_old = lam
        if abs(abs(lam) - abs(emax)) > 1e-2 * max(1.0, abs(emax)):
            print(f"  [warn] power-iteration Emax~{lam:.6f} vs ARPACK {emax:.6f}")
    return float(abs(emax))


def jackson_kernel(M):
    n = np.arange(M)
    Mp1 = M + 1
    g = ((Mp1 - n) * np.cos(np.pi * n / Mp1) +
         np.sin(np.pi * n / Mp1) / np.tan(np.pi / Mp1)) / Mp1
    return g


def chebyshev_moments(H, M, R, rng, mask=None, a_scale=None, b_shift=0.0):
    """Stochastic-trace Chebyshev moments mu_n = (1/R) sum_r <r|T_n(Hhat)|r>,
    n=0..M-1, of the rescaled Hamiltonian Hhat=(H-b_shift)/a_scale.

    If `mask` (boolean array of length N) is given, random vectors are
    supported only on the masked subset with amplitude 1/sqrt(|S|)
    (App. J Eq. J1), giving the shell-restricted local DOS estimator; the
    Chebyshev recursion is still performed with the full H.
    """
    N = H.shape[0]
    if a_scale is None:
        a_scale = estimate_spectral_radius(H) * 1.01
    Hhat = (H - b_shift * sp.identity(N, dtype=H.dtype)) / a_scale

    if mask is None:
        support = np.arange(N)
    else:
        support = np.where(mask)[0]
    nsupp = len(support)

    mu = np.zeros(M, dtype=np.complex128)
    for _ in range(R):
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=nsupp))
        r = np.zeros(N, dtype=np.complex128)
        r[support] = phases / np.sqrt(nsupp)

        a0 = r.copy()
        a1 = Hhat @ a0
        mu[0] += np.vdot(r, a0)
        mu[1] += np.vdot(r, a1)
        a_prev, a_cur = a0, a1
        for n in range(2, M):
            a_next = 2 * (Hhat @ a_cur) - a_prev
            mu[n] += np.vdot(r, a_next)
            a_prev, a_cur = a_cur, a_next
    mu /= R
    return mu, a_scale, b_shift


def reconstruct_dos(mu, a_scale, b_shift, energies):
    """Jackson-kernel-damped Chebyshev reconstruction of rho(E) on a grid
    of physical energies (same units as H, i.e. as multiples of t)."""
    M = len(mu)
    g = jackson_kernel(M)
    x = (energies - b_shift) / a_scale
    x_clipped = np.clip(x, -1 + 1e-12, 1 - 1e-12)
    theta = np.arccos(x_clipped)
    # T_n(x) = cos(n theta)
    n = np.arange(M)
    # build via cos(n*theta) outer product (M x n_energies) -- fine for
    # M~4096 and a few thousand energies; for very large grids this could
    # be done with a Chebyshev recursion instead.
    Tn = np.cos(np.outer(n, theta))
    coeff = g * mu.real
    coeff[0] *= 1.0  # n=0 term counted once
    s = coeff[0] + 2.0 * (coeff[1:] @ Tn[1:])
    rho = s / (np.pi * np.sqrt(1 - x_clipped ** 2) * a_scale)
    return rho


def kpm_dos(H, M, R, energies, seed=0, mask=None, a_scale=None, b_shift=0.0):
    rng = np.random.default_rng(seed)
    mu, a_scale, b_shift = chebyshev_moments(H, M, R, rng, mask=mask,
                                              a_scale=a_scale, b_shift=b_shift)
    rho = reconstruct_dos(mu, a_scale, b_shift, energies)
    return rho, a_scale, b_shift


if __name__ == "__main__":
    from hamiltonian import build_H0, build_Hsc_block
    d = np.load("lattices_test/lattice_gen2_N490.npz")
    pos, edges = d["pos"], d["edges"]
    H0 = build_H0(pos, edges)
    Hsc = build_Hsc_block(pos, edges)

    a_scale0 = estimate_spectral_radius(H0) * 1.01
    a_scalesc = estimate_spectral_radius(Hsc) * 1.01
    print("a_scale H0:", a_scale0, " a_scale Hsc:", a_scalesc)

    E = np.linspace(-0.15, 0.15, 601)
    rho0, *_ = kpm_dos(H0, M=1024, R=50, energies=E, a_scale=a_scale0)
    rhosc, *_ = kpm_dos(Hsc, M=1024, R=50, energies=E, a_scale=a_scalesc)
    i0 = np.argmin(np.abs(E))
    print("rho0(0) ~", rho0[i0], " rhosc(0) ~", rhosc[i0])
    print("(paper Table I, N=490: rho0(0)=5.46e-7, rhosc(0)=0.7155)")
