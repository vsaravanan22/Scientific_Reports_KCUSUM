"""Controlled KCUSUM benchmark (Results, 'Controlled benchmark experiments').

Outputs (in --outdir):
    mmd2_estimates.csv          squared MMD per task
    benchmark_results.csv       time to false alarm and delay for every (task, h)
    log_fit.csv                 linear fit of delay on log(time to false alarm)
                                over the six largest thresholds
    benchmark_delay_vs_false_alarm.png   Figure 1

Usage:
    python run_benchmark.py                 # full settings of the paper
    python run_benchmark.py --quick         # small smoke test (minutes)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from kcusum_core import (TASK_NAMES, arl_bound, delay_bound, estimate_mmd2,
                         kcusum_many_runs)

SIGMA2 = 1.0
CAP = 10 ** 9
SETTINGS = {
    # task: (delta, thresholds)
    1: (2.0 ** -7, [2, 4, 8, 12, 16, 20, 24, 28, 32]),
    2: (2.0 ** -7, [2, 4, 8, 12, 16, 20, 24, 28, 32]),
    3: (2.0 ** -7, [2, 4, 8, 12, 16, 20, 24, 28, 32]),
    4: (2.0 ** -9, [4, 8, 16, 24, 32, 40, 48, 56, 64]),
}


def seed_for(task, h, change, n_runs):
    """Fixed, distinct seed block for every (task, threshold, change) cell.
    No-change runs for tasks 1-3 share one block because they are identical."""
    t = 0 if (not change and task in (1, 2, 3)) else task
    # Stays below 2**32, the range accepted by np.random.seed.
    return 100_000_000 * t + 1_000_000 * int(h) + (500_000 if change else 0)


def summarize(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    sd = x.std(ddof=1) if n > 1 else 0.0
    return x.mean(), sd, 1.96 * sd / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--mmd-pairs", type=int, default=4_000_000)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true",
                    help="200 runs, thresholds capped at 16, 2e5 MMD pairs")
    ap.add_argument("--outdir", default="results/benchmark")
    args = ap.parse_args()
    if args.quick:
        args.runs = 200
        args.mmd_pairs = 200_000
    os.makedirs(args.outdir, exist_ok=True)

    # Squared MMD per task
    mmd_rows = []
    mmd2 = {}
    for task in args.tasks:
        m, se = estimate_mmd2(task, SIGMA2, args.mmd_pairs, 7_000 + task)
        mmd2[task] = m
        delta = SETTINGS[task][0]
        mmd_rows.append(dict(task=task, name=TASK_NAMES[task], mmd2=m, se=se,
                             delta=delta, detectable=m > delta))
        print(f"task {task}: d_k^2 = {m:.4f} (s.e. {se:.4f}), delta = {delta:.4f}")
    pd.DataFrame(mmd_rows).to_csv(os.path.join(args.outdir, "mmd2_estimates.csv"),
                                  index=False)

    rows = []
    arl_cache = {}
    for task in args.tasks:
        delta, hs = SETTINGS[task]
        if args.quick:
            hs = [h for h in hs if h <= 16]
        for h in hs:
            t0 = time.time()
            key = ("shared" if task in (1, 2, 3) else task, h)
            if key not in arl_cache:
                arl_cache[key] = kcusum_many_runs(
                    task, False, float(h), delta, SIGMA2, CAP, args.runs,
                    seed_for(task, h, False, args.runs))
            fa = arl_cache[key]
            dl = kcusum_many_runs(task, True, float(h), delta, SIGMA2, CAP,
                                  args.runs, seed_for(task, h, True, args.runs))
            n_cens = int((fa < 0).sum() + (dl < 0).sum())
            fa_m, fa_sd, fa_ci = summarize(fa[fa > 0])
            dl_m, dl_sd, dl_ci = summarize(dl[dl > 0])
            rows.append(dict(
                task=task, name=TASK_NAMES[task], delta=delta, h=h,
                ttfa_mean=fa_m, ttfa_sd=fa_sd, ttfa_ci95=fa_ci,
                delay_mean=dl_m, delay_sd=dl_sd, delay_ci95=dl_ci,
                delay_bound=delay_bound(h, delta, mmd2[task]),
                arl_bound=arl_bound(h, delta), censored=n_cens,
                n_runs=args.runs))
            print(f"task {task} h={h:>3}: TTFA {fa_m:>11.1f} +- {fa_sd:>10.1f} | "
                  f"delay {dl_m:>9.1f} +- {dl_sd:>8.1f} | "
                  f"bound {rows[-1]['delay_bound']:.3g} | {time.time()-t0:.1f}s")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.outdir, "benchmark_results.csv"), index=False)

    # Delay against log(time to false alarm) over the six largest thresholds
    fit_rows = []
    for task, grp in res.groupby("task"):
        g = grp.sort_values("h").tail(6)
        if len(g) < 3:
            continue
        xlog = np.log(g["ttfa_mean"].to_numpy())
        y = g["delay_mean"].to_numpy()
        slope, intercept = np.polyfit(xlog, y, 1)
        pred = slope * xlog + intercept
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fit_rows.append(dict(task=task, slope=slope, intercept=intercept, r2=r2,
                             thresholds=",".join(map(str, g["h"]))))
        print(f"task {task}: slope {slope:.1f} per unit log TTFA, R^2 = {r2:.3f}")
    pd.DataFrame(fit_rows).to_csv(os.path.join(args.outdir, "log_fit.csv"),
                                  index=False)

    plot_benchmark(res, os.path.join(args.outdir,
                                     "benchmark_delay_vs_false_alarm.png"))


def plot_benchmark(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for task, grp in res.groupby("task"):
        g = grp.sort_values("h")
        ax.errorbar(g["ttfa_mean"], g["delay_mean"],
                    xerr=g["ttfa_ci95"], yerr=g["delay_ci95"],
                    marker=markers.get(task, "o"), ms=5, capsize=2, lw=1.2,
                    label=f"({task}) {TASK_NAMES[task]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean time to false alarm (observations)")
    ax.set_ylabel("Mean detection delay (observations)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
