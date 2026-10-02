"""Write a small synthetic VMD XYZ trajectory for testing the pipeline.

It mimics the features discussed for 1H9T: small thermal motion, slow structural
drift, a rigid-body translation between frames 99 and 100, and an incomplete final
frame. It is NOT a substitute for the real data.

Usage:
    python make_test_trajectory.py test.xyz [--atoms 500] [--frames 151]
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--atoms", type=int, default=500)
    ap.add_argument("--frames", type=int, default=151)
    ap.add_argument("--jump-frame", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = rng.normal(scale=10.0, size=(args.atoms, 3))
    internal = np.zeros_like(base)
    shift = np.array([-0.96, 3.37, -0.48])
    with open(args.out, "w") as fh:
        for f in range(1, args.frames + 2):
            internal += rng.normal(scale=0.08, size=base.shape)
            internal *= 0.97
            xyz = base + internal + 0.01 * f
            if f >= args.jump_frame:
                xyz = xyz + shift
            n_write = args.atoms if f <= args.frames else args.atoms // 3
            fh.write(f"{args.atoms}\n frame {f}\n")
            for a in range(n_write):
                fh.write(f"C {xyz[a,0]:.4f} {xyz[a,1]:.4f} {xyz[a,2]:.4f}\n")


if __name__ == "__main__":
    main()
