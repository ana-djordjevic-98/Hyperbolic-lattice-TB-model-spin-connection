# Spin connection on the {10,3} hyperbolic lattice

Code for the numerical results in Sec. IV / App. J:
*"Symmetry-based theory of Dirac fermions on two-dimensional hyperbolic
crystals: Coupling to the spin connection."*

A discrete spin connection is constructed on the {10,3} hyperbolic
tight-binding lattice via the geodesic Wilson line and implemented as a
spin-dependent nearest-neighbor link factor. The Kernel Polynomial Method
is used to show that this spin connection produces a robust, finite
zero-energy density of states, in contrast to the reference model without
the spin-connection link factor, whose zero-energy DOS is strongly
suppressed.

## Contents

| Script | Produces |
|---|---|
| `scripts/01_generate_lattices.py` | The five {10,3} lattice generations (Table I) |
| `scripts/02_bfs_shell_distances.py` | The 17 BFS shells used in App. J (Table II) |
| `scripts/03_compute_dos.py` | `rho_0(E)`, `rho_sc(E)` per generation, with per-vector standard errors and particle-hole-symmetrized curves (Table I) |
| `scripts/04_make_figures_6_7_8.py` | Figs. 6, 7, 8 (symmetrized, with error bars) and Table I (uncertainty-aware digits) |
| `scripts/05_compute_shell_resolved.py` | App. J masked-trace shell analysis |
| `scripts/06_make_figureA1.py` | Fig. A1 and Table II |

## Method summary

**Lattice.** Sites are the vertices of the {p,q}={10,3} tessellation of the
Poincaré disk. The lattice is grown shell-by-shell in the dual (polygon)
graph using the point-group rotation and edge-pairing translation
generators of Sec. III (`src/hyperbolic_lattice.py`), represented as
SU(1,1)-type Möbius transformations. High-precision arithmetic (`mpmath`,
50 decimal digits) is used for the group-element composition, because
points crowd exponentially close to the boundary of the Poincaré disk with
increasing generation — ordinary double precision loses the digits needed
for reliable vertex/polygon deduplication after only a couple of shells.

**Hamiltonians** (`src/hamiltonian.py`). `H0` is the plain nearest-neighbor
hopping model (Eq. 89). `H_sc` uses the exact closed-form geodesic
Wilson-line phase (Eq. 95–96),

```
f_ij = 2 * angle(1 - z_i * conj(z_j))    (l = 1)
```

as a spin-dependent hopping phase ±f_ij/2 (Eq. 91). Because `H_sc` is block
diagonal in the spin index with blocks `H_+` and `H_- = conj(H_+)`, which
have identical spectra, the physical (per-state) DOS is exactly equal to
the DOS of the single N×N complex-Hermitian block `H_+`. This code builds
only that block, at half the memory/compute cost of the full 2N×2N spinor
Hamiltonian. Both `H0` and `H_sc` are checked to be Hermitian and to
satisfy the exact chiral symmetry σ_z H σ_z = −H (zero on-site energy,
bipartite lattice).

**DOS.** Kernel Polynomial Method (`src/kpm.py`): Jackson-kernel-damped
Chebyshev expansion, adaptive rescaling by a seeded (fully reproducible)
ARPACK spectral-radius estimate, and a stochastic trace estimator with
R = 150, 72, 36, 12, 12 random-phase vectors for the five system sizes and
M = 4096 moments throughout. The App. J masked/shell-restricted trace
estimator (`scripts/05_compute_shell_resolved.py`) restricts only the
*support of the random vector*, not the Hamiltonian used in the Chebyshev
recursion, per Eq. J1–J2.

**Uncertainty quantification.** `scripts/03_compute_dos.py` keeps each of
the R stochastic-trace vectors' individual contribution (not just their
sum), which is what makes the following honest rather than assumed:

- *Standard errors.* Table I and Fig. 8 report the standard error of the
  mean, σ/√R, computed from the actual spread of the R independent KPM
  estimates. Table I's digits are set by `round_to_uncertainty()` in
  `scripts/04_make_figures_6_7_8.py` rather than a fixed decimal count, so
  e.g. R=12 at the two largest sizes is reported to 3–4 significant
  figures with an explicit ± term instead of implying six-decimal
  precision.
- *Particle-hole symmetrization.* `H0` and `H_sc` are exactly chiral
  symmetric (verified in `hamiltonian.py`), so the true trace obeys
  ρ(E)=ρ(−E) exactly; a finite-R stochastic estimate does not. Figs. 6–7
  plot ρ_sym(E)=(ρ(E)+ρ(−E))/2. The unsymmetrized residual asymmetry is
  not discarded — it's written to `figures/asymmetry_diagnostic.csv` as a
  direct, quantitative noise diagnostic (it shrinks from ~6% of peak
  height at N=490 to <1% at N=571210, exactly tracking 1/√R as expected
  for sampling noise rather than a physical effect).
- *A second, easy-to-miss noise source.* scipy's ARPACK-based
  spectral-radius estimate draws an unseeded starting vector internally,
  so naively it returns a slightly different KPM rescaling `a_scale` on
  every call. Because the Chebyshev recursion runs to M~4000 (where
  T_n(x) has derivative ~n² near |x|=1), even a ~1e-7 relative change in
  `a_scale` can shift the reconstructed ρ(0) by a percent-level amount —
  enough to contaminate an R-vector error budget with an unrelated noise
  source. `estimate_spectral_radius()` now seeds its ARPACK call so that
  `a_scale` is exactly reproducible, leaving R-vector sampling as the only
  source of run-to-run variation captured by the reported standard error.

## Usage


```bash
pip install -r requirements.txt

# 1. Generate all five lattice generations (~3 min total; the N=571210
#    case is the slow one, ~2.5 min, due to the high-precision arithmetic).
python3 scripts/01_generate_lattices.py

# 2. BFS shell distances needed for Appendix J.
python3 scripts/02_bfs_shell_distances.py

# 3. DOS for each generation (M=4096, Table I's R values).
python3 scripts/03_compute_dos.py --N 490
python3 scripts/03_compute_dos.py --N 2880
python3 scripts/03_compute_dos.py --N 16810
python3 scripts/03_compute_dos.py --N 98000
python3 scripts/03_compute_dos.py --N 571210   # ~13-15 min in one shot

# 4. Figures 6, 7, 8 and Table I.
python3 scripts/04_make_figures_6_7_8.py

# 5. Appendix J shell-resolved analysis (~30-40 min in one shot).
python3 scripts/05_compute_shell_resolved.py

# 6. Figure A1 and Table II.
python3 scripts/06_make_figureA1.py
```

### Long-running steps on time-limited environments

The N=571210 DOS (step 3) and the shell-resolved analysis (step 5) are the
two expensive parts of the pipeline (dominated by ~0.6-1 s per stochastic
vector per Chebyshev moment set, sparse-matrix-vector-product bound). Both
scripts accept a `--chunk` argument that processes only a handful of
random vectors per invocation and checkpoints progress to
`results/checkpoints/`, so they can be safely re-run repeatedly (e.g. from
a notebook cell, a cron job, or a shell script loop) until complete:

```bash
python3 scripts/03_compute_dos.py --N 571210 --chunk 4     # run repeatedly
python3 scripts/05_compute_shell_resolved.py --chunk 200   # run repeatedly
```

Each call reports how many of the required stochastic vectors have been
completed so far; the final DOS/figures are written automatically once the
target is reached.

## Repository layout

```
src/
  hyperbolic_lattice.py    # {p,q} Poincare-disk lattice generation (Sec. III)
  hamiltonian.py            # H0, H_sc builders + Hermiticity/chirality checks
  kpm.py                     # KPM DOS engine (Jackson kernel, stochastic trace)
scripts/
  01_generate_lattices.py
  02_bfs_shell_distances.py
  03_compute_dos.py
  04_make_figures_6_7_8.py
  05_compute_shell_resolved.py
  06_make_figureA1.py
figures/                    # output figures + Table I / Table II CSVs (committed)
lattices/                   # generated lattice .npz files (regenerate; not committed)
results/                    # generated DOS/checkpoint data (regenerate; not committed)
```

`lattices/` and `results/` are regenerated by the scripts above and are
excluded via `.gitignore` (the N=571210 lattice alone is ~19 MB; keeping
generated data out of version control keeps the repository small and the
provenance unambiguous — everything follows from `src/` and `scripts/`
alone). `figures/` (final PNGs and the two CSV tables) is small and is
committed as a reference for what a correct run produces.

## Notes

- The DOS uses a stochastic trace estimator, so exact bit-for-bit
  agreement across separate runs is not expected — but this is now a
  quantified statement rather than a caveat: Table I reports the actual
  standard error for each `rho(0)`, and Fig. 8 plots it as error bars.
  With `a_scale` seeded (see Method summary), the reported standard error
  reflects R-vector sampling only, so re-running with the same R and
  seeds reproduces the same numbers, and re-running with a larger R
  should shrink the error bars roughly as 1/√R.
- The lattice site counts (Table I) and BFS shell populations (Table II)
  are not stochastic and are exact functions of the {10,3} geometry and
  generation number.
- `scripts/01_generate_lattices.py` uses 50-digit `mpmath` precision by
  default (`mpmath.mp.dps` in `src/hyperbolic_lattice.py`); this is
  comfortably more than needed through generation 6 but should be
  increased if the lattice generation is extended further.
- Table II and Fig. A1 (`scripts/05_compute_shell_resolved.py` /
  `06_make_figureA1.py`) now carry the same per-vector standard errors as
  Table I / Fig. 8, computed the same way (each stochastic vector's own
  reconstructed ρ(0) is kept, not just the running sum). No particle-hole
  symmetrization is needed there: ρ(0) is evaluated at E=0, its own
  mirror point, so ρ(0)=ρ(−0) trivially for any single estimate — the
  symmetrization in Figs. 6–7 only matters away from E=0.
