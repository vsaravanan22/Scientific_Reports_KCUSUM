"""1FME check (Results, '1FME trajectory').

Streams the trajectory, computes the centre of geometry of every frame and reports
the largest frame-to-frame step overall and within +-window frames of each
candidate change point from the earlier analysis.

NOTE: the candidates were found on 26,283 frames. Until it is confirmed which part
of the 100,000-frame file those frames correspond to, pass --offset to map them
(default 0, i.e. candidate frame numbers are taken as frame numbers of this file).

Usage:
    python run_1fme.py path/to/1FME.xyz [--offset 0] [--window 50]
"""

import argparse
import os

import numpy as np
import pandas as pd

from md_io import iter_frames

CANDIDATES = [12122, 12566, 13496, 22472, 24110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--outdir", default="results/1fme")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cog = []
    n_atoms = None
    for f, xyz in iter_frames(args.xyz, args.max_frames):
        n_atoms = xyz.shape[0]
        cog.append(xyz.mean(axis=0))
        if f % 10000 == 0:
            print(f"  read {f} frames")
    cog = np.array(cog)
    n = len(cog)
    print(f"{n} frames, {n_atoms} atoms")

    step = np.linalg.norm(np.diff(cog, axis=0), axis=1)  # step i: frame i+1 -> i+2
    to_frame = np.arange(2, n + 1)
    pd.DataFrame(dict(frame=np.arange(1, n + 1), cog_x=cog[:, 0], cog_y=cog[:, 1],
                      cog_z=cog[:, 2])).to_csv(
        os.path.join(args.outdir, "centre_of_geometry.csv"), index=False)

    print(f"Largest centre-of-geometry step: {step.max():.3f} A "
          f"(into frame {to_frame[step.argmax()]})")
    rows = []
    for c in CANDIDATES:
        f = c + args.offset
        sel = (to_frame >= f - args.window) & (to_frame <= f + args.window)
        m = float(step[sel].max()) if sel.any() else np.nan
        rows.append(dict(candidate=c, mapped_frame=f, max_step_within_window=m))
        print(f"  candidate {c} (frame {f}): max step within +-{args.window} "
              f"frames = {m:.3f} A")
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "candidate_steps.csv"),
                              index=False)


if __name__ == "__main__":
    main()
