"""One-at-a-time sensitivity of KCUSUM on the MD trajectories
(Results, 'Sensitivity to detector settings'; Supplementary Table S1).

Baseline: h = 0.015, delta = 0.1, sigma^2 = 1. Each parameter is varied alone
while the other two stay at baseline:

    h       in {0.01, 0.015, 0.02}
    delta   in {0.05, 0.1, 0.2}
    sigma^2 in {0.5, 1, 2}

giving 7 configurations. Everything else is identical to the main analysis:
the detector loop of the public kcusum.py (pairs at 0-based indices i = 2, 4, ...),
the `fitter` reference fitted once per signal to the calibration frames, and 200
runs with seeds 0-199, so the baseline row reproduces Table 3. The rigid-body
diagnostic and R_g do not depend on detector settings and are not re-run.

Input: a per-frame signal CSV with a `frame` column (1-based) and one column per
signal. Defaults: mean_z, rmsd_from_first, f2f_rmsd, rg (override with --columns).
For 1H9T, --xyz computes the signals (including R_g) directly from the trajectory.

Usage:
    python run_sensitivity.py --dataset 1h9t --xyz data/1H9T.xyz
    python run_sensitivity.py --dataset 1h9t --signals results/1h9t/signals.csv
    python run_sensitivity.py --dataset 1fme --signals 1fme_signals.csv

Outputs (in --outdir, default results/sensitivity_<dataset>):
    sensitivity_first_alarms.csv   one row per signal x configuration
    sensitivity_continuous.csv     1FME only: alarm fraction per R_g regime
    sensitivity_summary.md         compact table for the manuscript
"""

import argparse
import os

import numpy as np
import pandas as pd

from kcusum_md import fit_reference, sample_reference, summarize_alarms

BASE = dict(h=0.015, delta=0.1, sigma2=1.0)
VARIATIONS = [("baseline", BASE)]
for key, values in (("h", (0.01, 0.02)), ("delta", (0.05, 0.2)),
                    ("sigma2", (0.5, 2.0))):
    for v in values:
        VARIATIONS.append((key, {**BASE, key: v}))

DATASETS = {
    # calibration frames (inclusive), first frame given to the detectors,
    # reference event used for interpretation
    "1h9t": dict(calib=(1, 80), start=1, event=100,
                 event_label="translation between frames 99 and 100"),
    "1fme": dict(calib=(1001, 2000), start=1001, event=4439,
                 event_label="first structural transition (expansion)",
                 regimes=[("calibration end to expansion", 2001, 4438),
                          ("expanded episode", 4439, 13615),
                          ("after re-compaction", 13616, None)]),
}

DEFAULT_COLUMNS = {
    "mean_z": "Mean Z",
    "rmsd_from_first": "RMSD from frame 1",
    "f2f_rmsd": "Frame-to-frame RMSD after superposition",
    "rg": "Radius of gyration",
}


# ---------------------------------------------------------------- detector
def increments(x, y, delta, sigma2):
    """g - delta at 0-based indices i = 2, 4, ... (public kcusum.py pairing),
    with k(a, b) = exp(-(a - b)^2 / (2 sigma^2)). x has shape (m,), y (runs, m)."""
    i = np.arange(2, x.shape[-1], 2)

    def k(a, b):
        return np.exp(-((a - b) ** 2) / (2.0 * sigma2))

    g = (k(x[i], x[i - 1]) + k(y[:, i], y[:, i - 1])
         - k(x[i], y[:, i - 1]) - k(x[i - 1], y[:, i]))
    return g - delta, i


def first_alarms(v, h):
    """First pair at which Z > h, with Z_n = max(0, Z_{n-1} + v_n), via the closed
    form Z_n = S_n - min(0, min_{k<=n} S_k). Returns pair index or -1."""
    s = np.cumsum(v, axis=1)
    z = s - np.minimum(0.0, np.minimum.accumulate(s, axis=1))
    hit = z > h
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def continuous_alarms(v, h):
    """Run without stopping, resetting Z to 0 after each alarm. Returns a boolean
    array (runs, pairs) marking alarm pairs."""
    try:
        from numba import njit
    except ImportError:
        njit = None

    def loop(v, h):
        out = np.zeros(v.shape, dtype=np.bool_)
        for r in range(v.shape[0]):
            z = 0.0
            for j in range(v.shape[1]):
                z += v[r, j]
                if z < 0.0:
                    z = 0.0
                if z > h:
                    out[r, j] = True
                    z = 0.0
        return out

    fn = njit(cache=False)(loop) if njit else loop
    return fn(np.ascontiguousarray(v), float(h))


# ---------------------------------------------------------------- signals
def signals_from_xyz(path):
    from md_io import frame_means, read_trajectory
    from rigid_body import diagnostics
    traj = read_trajectory(path)
    diag = diagnostics(traj)
    centred = traj - traj.mean(axis=1, keepdims=True)
    rg = np.sqrt((centred ** 2).sum(axis=2).mean(axis=1))
    n = traj.shape[0]
    return pd.DataFrame(dict(
        frame=np.arange(1, n + 1),
        mean_z=frame_means(traj)[:, 2],
        rmsd_from_first=diag["rmsd_from_first"],
        f2f_rmsd=np.concatenate(([np.nan], diag["f2f_rmsd"])),
        rg=rg))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--signals", help="per-frame signal CSV")
    src.add_argument("--xyz", help="trajectory (signals computed here)")
    ap.add_argument("--columns", nargs="+", default=None,
                    help="signal columns (default: mean_z rmsd_from_first "
                         "f2f_rmsd rg)")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    outdir = args.outdir or f"results/sensitivity_{args.dataset}"
    os.makedirs(outdir, exist_ok=True)

    df = signals_from_xyz(args.xyz) if args.xyz else pd.read_csv(args.signals)
    cols = args.columns or [c for c in DEFAULT_COLUMNS if c in df.columns]
    missing = [c for c in (args.columns or DEFAULT_COLUMNS) if c not in df.columns]
    if missing:
        print(f"[warn] columns not found and skipped: {missing}")

    rows, cont_rows = [], []
    for col in cols:
        d = df[["frame", col]].dropna()
        d = d[d["frame"] >= cfg["start"]]
        frames = d["frame"].to_numpy(int)
        x = d[col].to_numpy(float)
        calib = (frames >= cfg["calib"][0]) & (frames <= cfg["calib"][1])
        name, params = fit_reference(x[calib])
        y = np.stack([sample_reference(name, params, len(x), s)
                      for s in range(args.runs)])
        label = DEFAULT_COLUMNS.get(col, col)
        print(f"\n{label}: reference = {name}, {len(x)} frames")

        for varied, p in VARIATIONS:
            v, idx = increments(x, y, p["delta"], p["sigma2"])
            first = first_alarms(v, p["h"])
            alarms = [None if j < 0 else int(frames[idx[j]]) for j in first]
            fired = np.array([a for a in alarms if a is not None])
            rows.append(dict(
                signal=label, varied=varied, h=p["h"], delta=p["delta"],
                sigma2=p["sigma2"], reference=name,
                summary=summarize_alarms(alarms),
                n_no_alarm=sum(a is None for a in alarms),
                min_alarm=fired.min() if fired.size else np.nan,
                median_alarm=np.median(fired) if fired.size else np.nan,
                max_alarm=fired.max() if fired.size else np.nan,
                frac_before_event=(fired < cfg["event"]).sum() / args.runs,
                frac_at_first_post_event_pair=(
                    (fired == cfg["event"] + 1).sum() / args.runs)))
            print(f"  {varied:8s} h={p['h']:<6} delta={p['delta']:<5} "
                  f"sigma2={p['sigma2']:<4} -> {rows[-1]['summary'][:70]}")

            if args.dataset == "1fme" and col == "rg":
                hits = continuous_alarms(v, p["h"])
                pair_frames = frames[idx]
                for reg, lo, hi in cfg["regimes"]:
                    m = (pair_frames >= lo) & (
                        pair_frames <= (hi if hi else pair_frames.max()))
                    cont_rows.append(dict(
                        varied=varied, h=p["h"], delta=p["delta"],
                        sigma2=p["sigma2"], regime=reg,
                        alarm_fraction=hits[:, m].mean() if m.any() else np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "sensitivity_first_alarms.csv"), index=False)
    if cont_rows:
        cont = pd.DataFrame(cont_rows)
        cont.to_csv(os.path.join(outdir, "sensitivity_continuous.csv"), index=False)
        print("\nContinuous KCUSUM on R_g, fraction of pairs alarmed (mean of runs):")
        print(cont.pivot_table(index=["varied", "h", "delta", "sigma2"],
                               columns="regime", values="alarm_fraction",
                               sort=False).round(3).to_string())

    summ = res[["signal", "varied", "h", "delta", "sigma2", "summary",
                "frac_before_event"]]
    with open(os.path.join(outdir, "sensitivity_summary.md"), "w") as fh:
        fh.write(f"# KCUSUM sensitivity, {args.dataset.upper()} "
                 f"(event: {cfg['event_label']}, frame {cfg['event']})\n\n")
        fh.write(summ.to_string(index=False))
    print(f"\nWritten to {outdir}/")


if __name__ == "__main__":
    main()
