#!/usr/bin/env python3
"""
05_compute_shell_resolved.py

Appendix J: shell-resolved zero-energy DOS via a masked stochastic trace
estimator (Eq. J1-J2). Random-phase vectors are supported only on the
sites of a given BFS shell (graph distance d from the 10 central-polygon
vertices, from 02_bfs_shell_distances.py), while the Chebyshev recursion
still uses the full N=16810 lattice Hamiltonian -- only the final trace
evaluation is restricted to the shell.

Matches the paper's protocol: M=4096 moments, R=200 random vectors for
the two smallest shells (d=0,1), R=60 for all other shells (d=2..16).

As in 03_compute_dos.py, each stochastic vector's own reconstructed
rho(0) is kept (not just the running sum of moments), so the standard
error of the mean at each shell, sigma/sqrt(R), is computed from the
actual spread of the R independent estimates rather than omitted. Note
that no particle-hole symmetrization is needed here (unlike Figs. 6-7):
rho(0) is evaluated at E=0 exactly, which is its own mirror point, so
rho(0)=rho(-0) trivially for any single estimate.

USAGE
-----
    python3 05_compute_shell_resolved.py                # run everything
    python3 05_compute_shell_resolved.py --chunk 200     # resumable chunks

Progress is checkpointed to results/checkpoints/shell_ckpt_{H0,Hsc}.npz;
re-running with --chunk simply continues from where it left off. A full
run needs ~1300 stochastic vectors per Hamiltonian (2600 total); at
roughly 0.6-1s per vector on N=16810 this is ~30-40 minutes total, hence
the --chunk option for environments with short per-call time limits.
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hamiltonian import build_H0, build_Hsc_block
from kpm import estimate_spectral_radius, reconstruct_dos

ROOT = os.path.join(os.path.dirname(__file__), "..")
LATTICE_PATH = os.path.join(ROOT, "lattices", "lattice_gen4_N16810.npz")
DIST_PATH = os.path.join(ROOT, "results", "bfs_dist_N16810.npy")
CKPT_DIR = os.path.join(ROOT, "results", "checkpoints")
M = 4096
R_SMALL_SHELLS = 200   # d = 0, 1
R_OTHER_SHELLS = 60    # d = 2..16
E0 = np.array([0.0])


def shell_R(d):
    return R_SMALL_SHELLS if d in (0, 1) else R_OTHER_SHELLS


def build_task_list(max_d):
    tasks = []
    for d in range(max_d + 1):
        for k in range(shell_R(d)):
            tasks.append((d, k))
    return tasks


def ckpt_path(which):
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, f"shell_ckpt_{which}.npz")


def run_which(which, dist, pos, edges, n_tasks_this_call):
    N = len(dist)
    max_d = int(dist.max())
    tasks = build_task_list(max_d)
    n_total = len(tasks)

    H = build_H0(pos, edges) if which == "H0" else build_Hsc_block(pos, edges)
    path = ckpt_path(which)
    try:
        ck = np.load(path)
        mu_sum = ck["mu_sum"]
        rho0_all = ck["rho0_all"]            # flat, one entry per task
        next_task = int(ck["next_task"])
        a_scale = float(ck["a_scale"])
    except FileNotFoundError:
        a_scale = estimate_spectral_radius(H, verify_with_power_iteration=False) * 1.01
        mu_sum = np.zeros((max_d + 1, M), dtype=np.complex128)
        rho0_all = np.full(n_total, np.nan)
        next_task = 0

    Hhat = H / a_scale
    shell_support = {d: np.where(dist == d)[0] for d in range(max_d + 1)}

    n_todo = n_tasks_this_call if n_tasks_this_call else (n_total - next_task)
    end_task = min(next_task + n_todo, n_total)

    t0 = time.time()
    for ti in range(next_task, end_task):
        d, k = tasks[ti]
        support = shell_support[d]
        nsupp = len(support)
        seed = 10_000_000 * (0 if which == "H0" else 1) + d * 100_000 + k
        rng = np.random.default_rng(seed)
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=nsupp))
        r = np.zeros(N, dtype=np.complex128)
        r[support] = phases / np.sqrt(nsupp)

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

        mu_sum[d] += mu
        rho0_all[ti] = reconstruct_dos(mu, a_scale, 0.0, E0)[0].real
    dt = time.time() - t0
    done = end_task
    print(f"  [{which}] tasks {next_task}..{end_task-1} ({end_task-next_task}) "
          f"in {dt:.1f}s -> {done}/{n_total}", flush=True)
    np.savez(path, mu_sum=mu_sum, rho0_all=rho0_all, next_task=done,
              a_scale=a_scale, n_total=n_total, max_d=max_d)
    return done, n_total, max_d, a_scale, mu_sum, rho0_all, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0,
                     help="Number of shell-tasks to process THIS call, "
                          "per Hamiltonian (0 = run to completion).")
    args = ap.parse_args()

    dist = np.load(DIST_PATH)
    d_lat = np.load(LATTICE_PATH)
    pos, edges = d_lat["pos"], d_lat["edges"]

    done0, total0, max_d, a0, mu0, rho0_all_0, tasks = run_which(
        "H0", dist, pos, edges, args.chunk)
    donesc, totalsc, _, asc, musc, rho0_all_sc, _ = run_which(
        "Hsc", dist, pos, edges, args.chunk)

    if done0 >= total0 and donesc >= totalsc:
        # Per-shell arrays of the individual per-vector rho(0) estimates,
        # sliced out of the flat task-ordered rho0_all using the same
        # deterministic task list (d, k) built by build_task_list().
        task_d = np.array([d for d, k in tasks])

        rho0_shell = np.zeros(max_d + 1)
        rho0_shell_sem = np.zeros(max_d + 1)
        rhosc_shell = np.zeros(max_d + 1)
        rhosc_shell_sem = np.zeros(max_d + 1)
        for d in range(max_d + 1):
            mask = task_d == d
            R = shell_R(d)
            vals0 = rho0_all_0[mask]
            valssc = rho0_all_sc[mask]
            rho0_shell[d] = vals0.mean()
            rho0_shell_sem[d] = vals0.std(ddof=1) / np.sqrt(R)
            rhosc_shell[d] = valssc.mean()
            rhosc_shell_sem[d] = valssc.std(ddof=1) / np.sqrt(R)

        outdir = os.path.join(ROOT, "results")
        np.savez(os.path.join(outdir, "shell_resolved_N16810.npz"),
                  d=np.arange(max_d + 1),
                  rho0=rho0_shell, rho0_sem=rho0_shell_sem,
                  rhosc=rhosc_shell, rhosc_sem=rhosc_shell_sem)
        print("DONE. Saved results/shell_resolved_N16810.npz")
        print(f"{'d':>3} {'rho0(0)':>12} {'+/-':>10} {'rhosc(0)':>10} {'+/-':>8}")
        for d in range(max_d + 1):
            print(f"{d:>3} {rho0_shell[d]:>12.3e} {rho0_shell_sem[d]:>10.2e} "
                  f"{rhosc_shell[d]:>10.4f} {rhosc_shell_sem[d]:>8.4f}")
    else:
        print("Not yet complete; re-run this command again to continue.")


if __name__ == "__main__":
    main()
