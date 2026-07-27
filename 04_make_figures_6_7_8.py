#!/usr/bin/env python3
"""
04_make_figures_6_7_8.py

Build Figs. 6-8 and Table I from the DOS data saved by 03_compute_dos.py
(results/dos_N*.npz, which include per-vector standard errors and
particle-hole-symmetrized curves -- see 03_compute_dos.py's docstring).
Run 03_compute_dos.py for all five sizes first.

Compared to a naive version of this script:
  - Figs. 6 and 7 plot the particle-hole symmetrized rho_sym(E) rather
    than the raw stochastic estimate, since the underlying Hamiltonians
    are exactly chiral symmetric and the true trace obeys rho(E)=rho(-E).
    The unsymmetrized data is not discarded: the maximum residual
    asymmetry (a direct measure of stochastic-sampling noise) is printed
    and written to figures/asymmetry_diagnostic.csv.
  - Fig. 8 gets error bars: the standard error of rho(0) computed from
    the actual spread of the R independent stochastic-trace estimates
    (see 03_compute_dos.py), not omitted.
  - Table I reports rho(0) to a number of digits set by its own
    uncertainty (via round_to_uncertainty below) instead of a fixed six
    decimal places that would overstate the precision available from,
    e.g., R=12 stochastic vectors at the two largest sizes.
"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from uncertainty import round_to_uncertainty, format_linear, format_scientific

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

SIZES = [490, 2880, 16810, 98000, 571210]
GENS = {490: 2, 2880: 3, 16810: 4, 98000: 5, 571210: 6}

# High-contrast, colorblind-safe qualitative palette (Okabe-Ito based),
# chosen for clear separation on screen, in print, and in grayscale.
PALETTE = ["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7"]


def load_all():
    data = {}
    missing = []
    for N in SIZES:
        path = os.path.join(RESULTS, f"dos_N{N}.npz")
        if not os.path.exists(path):
            missing.append(N)
            continue
        d = np.load(path)
        data[N] = {k: d[k] for k in d.files}
    if missing:
        raise SystemExit(f"Missing results for N={missing}. "
                          f"Run 03_compute_dos.py --N <size> for each first "
                          f"(this version of the script requires the "
                          f"per-vector-error dos_N*.npz format).")
    return data


def fig6(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c, N in zip(PALETTE, SIZES):
        E = data[N]["E"]
        ax.plot(E, data[N]["rho0_sym"], "--", color=c, lw=1.0, alpha=0.9,
                 label=rf"$\rho_0$, $N$={N}")
        ax.plot(E, data[N]["rhosc_sym"], "-", color=c, lw=1.2,
                 label=rf"$\rho_{{sc}}$, $N$={N}")
    ax.set_xlabel(r"$E/t$"); ax.set_ylabel(r"$\rho(E)$")
    ax.set_xlim(-0.15, 0.15); ax.set_ylim(0, None)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig6_dos.png")
    fig.savefig(out, dpi=200); print("saved", out)


def fig7(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c, N in zip(PALETTE, SIZES):
        E = data[N]["E"]
        delta = data[N]["rhosc_sym"] - data[N]["rho0_sym"]
        ax.plot(E, delta, "-", color=c, lw=1.1, label=f"N={N}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$E/t$"); ax.set_ylabel(r"$\delta\rho(E)=\rho_{sc}(E)-\rho_0(E)$")
    ax.set_xlim(-0.15, 0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig7_delta_dos.png")
    fig.savefig(out, dpi=200); print("saved", out)


def fig8(data):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    Ns = np.array(SIZES)
    rho0_0, rho0_err, rhosc_0, rhosc_err = [], [], [], []
    for N in SIZES:
        E = data[N]["E"]; i0 = int(np.argmin(np.abs(E)))
        rho0_0.append(data[N]["rho0"][i0])
        rho0_err.append(data[N]["rho0_sem"][i0])
        rhosc_0.append(data[N]["rhosc"][i0])
        rhosc_err.append(data[N]["rhosc_sem"][i0])
    ax.errorbar(Ns, rhosc_0, yerr=rhosc_err, fmt="s-", color="tab:blue",
                capsize=3, label=r"$\rho_{sc}(0)$")
    ax.errorbar(Ns, rho0_0, yerr=rho0_err, fmt="^-", color="black",
                capsize=3, label=r"$\rho_0(0)$")
    ax.set_xscale("log")
    ax.set_xlabel("N (number of lattice sites)"); ax.set_ylabel(r"$\rho(0)$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig8_zero_energy_dos.png")
    fig.savefig(out, dpi=200); print("saved", out)


def table1(data):
    out = os.path.join(FIGDIR, "table1.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gen", "N", "R", "rho0(0)", "rho0(0)_stderr",
                    "rhosc(0)", "rhosc(0)_stderr", "delta_rho(0)_formatted"])
        for N in SIZES:
            E = data[N]["E"]; i0 = int(np.argmin(np.abs(E)))
            r0, r0e = data[N]["rho0"][i0], data[N]["rho0_sem"][i0]
            rsc, rsce = data[N]["rhosc"][i0], data[N]["rhosc_sem"][i0]
            R = int(data[N]["R"])
            m_v, m_u, exp = format_scientific(r0, r0e)
            rsc_v, rsc_u = format_linear(rsc, rsce)
            w.writerow([GENS[N], N, R,
                        f"({m_v}\u00b1{m_u})e{exp:+03d}", f"{r0e:.2e}",
                        f"{rsc_v}\u00b1{rsc_u}", f"{rsce:.4f}",
                        f"{rsc_v}\u00b1{rsc_u}"])
    print("saved", out)


def asymmetry_diagnostic(data):
    """Residual particle-hole asymmetry of the unsymmetrized rho(E) --
    quantitative evidence that it is stochastic sampling noise (it shrinks
    steadily with R and N) rather than a physical effect, retained here
    instead of only asserted in prose."""
    out = os.path.join(FIGDIR, "asymmetry_diagnostic.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Gen", "N", "R", "max|asym rho0|", "max|asym rhosc|",
                    "max|asym rhosc| as % of peak rhosc"])
        for N in SIZES:
            R = int(data[N]["R"])
            a0 = np.max(np.abs(data[N]["asym0"]))
            asc = np.max(np.abs(data[N]["asymsc"]))
            pct = 100 * asc / np.max(np.abs(data[N]["rhosc"]))
            w.writerow([GENS[N], N, R, f"{a0:.2e}", f"{asc:.2e}", f"{pct:.2f}"])
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    data = load_all()
    fig6(data); fig7(data); fig8(data); table1(data); asymmetry_diagnostic(data)
