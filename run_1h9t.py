"""1H9T case study (Results, 'The 1H9T transition is a rigid-body translation' and
'Detector behaviour on coordinate-derived and structural signals').

Usage:
    python run_1h9t.py path/to/1H9T.xyz [--runs 200] [--outdir results/1h9t]

Outputs (in --outdir):
    signals.csv                       mean X/Y/Z, RMSD from frame 1, f2f RMSD
    step_diagnostics.csv              per-step unaligned and superposed change
    alarms_table.csv / alarms_table.md   Table 3
    kcusum_alarms_raw.csv             alarm frame of every KCUSUM run
    1H9T_rigid_body_diagnostics.png   Figure 2
    1H9T_method_comparison.png        Figure 3
"""

import argparse
import os

import numpy as np
import pandas as pd

from bocpd import bocpd_alarms
from kcusum_md import run_kcusum_config, summarize_alarms
from kliep import kliep_detector
from md_io import frame_means, read_trajectory
from rigid_body import diagnostics, plot_diagnostics, report_jump

CALIB_LAST = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--h", type=float, default=0.015)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--jump", type=int, default=None,
                    help="1-based frame after the jump; detected if omitted")
    ap.add_argument("--outdir", default="results/1h9t")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    traj = read_trajectory(args.xyz)
    n, n_atoms, _ = traj.shape
    print(f"{n} complete frames, {n_atoms} atoms")
    frames = np.arange(1, n + 1)

    # ---------------- rigid-body diagnostics ----------------
    means = frame_means(traj)
    diag = diagnostics(traj)
    mz = means[:, 2]
    jump = args.jump or int(np.argmax(np.abs(np.diff(mz)))) + 2
    sd_cal = mz[:CALIB_LAST].std(ddof=1)
    dz = mz[jump - 1] - mz[jump - 2]
    print(f"Largest mean-Z step: frames {jump-1}->{jump}, change {dz:.3f} A, "
          f"{abs(dz)/sd_cal:.0f} x s.d. over frames 1-{CALIB_LAST} ({sd_cal:.4f} A)")
    report_jump(diag, jump - 1, jump)
    if n >= 98:
        print(f"Mean-Z drift over frames 90-98: {mz[97] - mz[89]:+.4f} A")

    f2f_frames = frames[1:]
    pd.DataFrame(dict(frame=frames, mean_x=means[:, 0], mean_y=means[:, 1],
                      mean_z=mz, rmsd_from_first=diag["rmsd_from_first"],
                      f2f_rmsd=np.concatenate(([np.nan], diag["f2f_rmsd"])))
                 ).to_csv(os.path.join(args.outdir, "signals.csv"), index=False)
    pd.DataFrame(dict(step_to_frame=f2f_frames,
                      median_disp=diag["median_step_disp"],
                      max_disp=diag["max_step_disp"],
                      superposed_rmsd=diag["f2f_rmsd"])
                 ).to_csv(os.path.join(args.outdir, "step_diagnostics.csv"),
                          index=False)
    plot_diagnostics(means, diag, jump,
                     os.path.join(args.outdir, "1H9T_rigid_body_diagnostics.png"))

    # ---------------- detectors ----------------
    signals = {
        "Mean Z": (mz, frames),
        "RMSD from frame 1": (diag["rmsd_from_first"], frames),
        "Frame-to-frame RMSD after superposition": (diag["f2f_rmsd"], f2f_frames),
    }
    rows, raw = [], {}

    # KCUSUM with the public configuration (reference fitted to all frames)
    a, dist = run_kcusum_config(mz, frames, np.ones(n, bool), args.runs, args.h,
                                args.delta, "Mean Z, reference from all frames")
    raw["meanZ_allframes"] = a
    rows.append({"Signal": "Mean Z, reference from all frames (public code)",
                 "KCUSUM": summarize_alarms(a), "Reference": dist,
                 "BOCPD": "--", "KLIEP": "--"})

    outputs = {}
    for name, (s, fr) in signals.items():
        a, dist = run_kcusum_config(s, fr, fr <= CALIB_LAST, args.runs, args.h,
                                    args.delta, f"{name}, reference frames 1-80")
        raw[name] = a
        b = bocpd_alarms(s, fr)
        k = kliep_detector(s, fr, calib_last=CALIB_LAST)
        outputs[name] = (b, k, a)
        bo = "none" if b["first_alarm"] is None else str(b["first_alarm"])
        if b["first_alarm_after_restrict"] is not None and \
                b["first_alarm_after_restrict"] != b["first_alarm"]:
            bo += f" ({b['first_alarm_after_restrict']} if restricted to frames >80)"
        label = "Mean Z, reference from frames 1-80" if name == "Mean Z" else name
        rows.append({"Signal": label, "KCUSUM": summarize_alarms(a),
                     "Reference": dist, "BOCPD": bo,
                     "KLIEP": "none" if k["first_alarm"] is None
                     else str(k["first_alarm"])})
        print(f"[BOCPD] {name}: first alarm {b['first_alarm']}, after 80: "
              f"{b['first_alarm_after_restrict']}, frames flagged: {b['n_flagged']}")
        print(f"[KLIEP] {name}: threshold {k['threshold']:.3f}, "
              f"first alarm {k['first_alarm']}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.outdir, "alarms_table.csv"), index=False)
    with open(os.path.join(args.outdir, "alarms_table.md"), "w") as fh:
        fh.write(table.to_markdown(index=False) if _has_tabulate()
                 else table.to_string(index=False))
    print("\n" + table.to_string(index=False))

    pd.DataFrame({k: [np.nan if v is None else v for v in vals]
                  for k, vals in raw.items()}
                 ).to_csv(os.path.join(args.outdir, "kcusum_alarms_raw.csv"),
                          index_label="seed")

    b, k, a = outputs["Mean Z"]
    plot_comparison(mz, frames, b, k, a,
                    os.path.join(args.outdir, "1H9T_method_comparison.png"))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def plot_comparison(mz, frames, b, k, kcusum_alarms, path):
    """Figure 3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [v for v in kcusum_alarms if v is not None]
    kc = int(pd.Series(vals).mode()[0]) if vals else None
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.0), sharex=True)
    axes[0].plot(frames, mz, color="black", lw=1.1)
    axes[0].set_ylabel("Mean Z (\u00c5)")
    axes[1].plot(frames, b["map_rl"], color="tab:blue", lw=1.1)
    axes[1].axhline(10, color="grey", ls="--", lw=1)
    axes[1].set_ylabel("BOCPD MAP\nrun length")
    axes[2].plot(frames, k["scores"], color="tab:green", lw=1.1)
    axes[2].axhline(k["threshold"], color="grey", ls="--", lw=1)
    axes[2].axvspan(1, CALIB_LAST, color="grey", alpha=0.12)
    axes[2].set_ylabel("KLIEP log score")
    axes[2].set_xlabel("Frame")

    marks = [(kc, "tab:red", "KCUSUM"),
             (b["first_alarm"], "tab:blue", "BOCPD"),
             (b["first_alarm_after_restrict"], "tab:blue", None),
             (k["first_alarm"], "tab:green", "KLIEP")]
    for ax in axes:
        for f, c, _ in marks:
            if f is not None:
                ax.axvline(f, color=c, ls="--", lw=0.9, alpha=0.8)
    handles = [plt.Line2D([], [], color=c, ls="--", label=l)
               for f, c, l in marks if l and f is not None]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, ncol=3)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.13, 1.0, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=12, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
