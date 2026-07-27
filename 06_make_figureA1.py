#!/usr/bin/env python3
"""
06_make_figureA1.py

Build Fig. A1 and Table II from results/shell_resolved_N16810.npz
(produced by 05_compute_shell_resolved.py, which keeps each stochastic
vector's individual rho(0) estimate per shell -- not just the running
sum -- so the standard error of the mean, sigma/sqrt(R), is computed
from the actual spread of the R independent estimates at each shell.
Table II's digit count is set by that uncertainty (via
round_to_uncertainty in src/uncertainty.py), and both panels of Fig. A1
carry the corresponding error bars.

No particle-hole symmetrization is needed here (contrast
04_make_figures_6_7_8.py's treatment of Figs. 6-7): rho(0) is evaluated
at E=0, its own mirror point, so rho(0)=rho(-0) trivially for any single
stochastic estimate.
"""
import os, csv, sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from uncertainty import format_linear, format_scientific

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d_data = np.load(os.path.join(RESULTS, "shell_resolved_N16810.npz"))
    d_arr = d_data["d"]
    rho0_shell, rhosc_shell = d_data["rho0"], d_data["rhosc"]
    rho0_sem, rhosc_sem = d_data["rho0_sem"], d_data["rhosc_sem"]
    max_d = int(d_arr.max())

    dist = np.load(os.path.join(RESULTS, "bfs_dist_N16810.npy"))
    counts = Counter(dist.tolist())

    # ---- Table II ----
    out_csv = os.path.join(FIGDIR, "tableII.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "sites", "rho0(0)", "rho0(0)_stderr",
                    "rhosc(0)", "rhosc(0)_stderr"])
        for d in range(max_d + 1):
            m_v, m_u, exp = format_scientific(rho0_shell[d], rho0_sem[d])
            rsc_v, rsc_u = format_linear(rhosc_shell[d], rhosc_sem[d])
            w.writerow([d, counts[d], f"({m_v}\u00b1{m_u})e{exp:+03d}",
                        f"{rho0_sem[d]:.2e}", f"{rsc_v}\u00b1{rsc_u}",
                        f"{rhosc_sem[d]:.4f}"])
    print("saved", out_csv)

    # ---- Figure A1 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.errorbar(d_arr[1:], rhosc_shell[1:], yerr=rhosc_sem[1:], fmt="o-",
                color="tab:blue", ms=4, capsize=2)
    ax.errorbar(d_arr[0:1], rhosc_shell[0:1], yerr=rhosc_sem[0:1], fmt="o",
                mfc="none", mec="tab:blue", ecolor="tab:blue", ms=6, capsize=2)
    ax.set_xlabel("BFS shell $d$ (graph distance from central polygon)")
    ax.set_ylabel(r"$\rho_{sc}(0)$")

    ax2 = axes[1]
    ax2.set_yscale("log")
    ax2.errorbar(d_arr[1:], np.abs(rho0_shell[1:]), yerr=rho0_sem[1:], fmt="s-",
                 color="tab:red", ms=4, capsize=2)
    ax2.errorbar(d_arr[0:1], np.abs(rho0_shell[0:1]), yerr=rho0_sem[0:1], fmt="s",
                 mfc="none", mec="tab:red", ecolor="tab:red", ms=6, capsize=2)
    ax2.set_xlabel("BFS shell $d$ (graph distance from central polygon)")
    ax2.set_ylabel(r"$\rho_0(0)$")

    fig.tight_layout()
    out_png = os.path.join(FIGDIR, "figA1_shell_resolved.png")
    fig.savefig(out_png, dpi=200)
    print("saved", out_png)

    print("\nd=0 excluded from the bulk-uniformity statement (10-site cluster; "
          "intrinsic level spacing not resolved at this KPM resolution, see App. J).")
    print(f"rho_sc(0) range d=1..{max_d}: "
          f"[{rhosc_shell[1:].min():.4f}, {rhosc_shell[1:].max():.4f}]")
    print(f"rho_0(0) range d=1..{max_d}:  "
          f"[{np.abs(rho0_shell[1:]).min():.2e}, {np.abs(rho0_shell[1:]).max():.2e}]")
    print(f"median standard error: rho_sc(0) {np.median(rhosc_sem[1:]):.4f}  "
          f"rho_0(0) {np.median(rho0_sem[1:]):.2e}")


if __name__ == "__main__":
    main()
