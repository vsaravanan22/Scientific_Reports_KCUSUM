"""Per-frame signals for the 1FME trajectory, computed in one streaming pass.

Signals (all atoms, equal weights; frames numbered from 1):
    mean_x, mean_y, mean_z   frame-averaged coordinates (no alignment)
    rmsd_from_first          Kabsch RMSD from frame 1
    f2f_rmsd                 Kabsch RMSD between frames t-1 and t (NaN at t = 1)
    rg                       radius of gyration about the centre of geometry

Usage:
    python compute_1fme_signals.py data/1FME.xyz[.gz] [--out 1fme_signals.csv]
"""

import argparse
import gzip
import subprocess

import numpy as np
import pandas as pd


def batched_kabsch_rmsd(P, Q):
    """RMSD after optimal superposition between P (f, n, 3) and Q (f, n, 3)."""
    p = P - P.mean(axis=1, keepdims=True)
    q = Q - Q.mean(axis=1, keepdims=True)
    H = np.einsum("fai,faj->fij", p, q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S[:, -1] *= d
    msd = ((p ** 2).sum(axis=(1, 2)) + (q ** 2).sum(axis=(1, 2))
           - 2.0 * S.sum(axis=1)) / P.shape[1]
    return np.sqrt(np.clip(msd, 0.0, None))


def coordinate_stream(path):
    """Coordinate lines only (4 fields), via awk for speed."""
    cat = f"zcat '{path}'" if path.endswith(".gz") else f"cat '{path}'"
    proc = subprocess.Popen(f"{cat} | awk 'NF==4{{print $2, $3, $4}}'",
                            shell=True, stdout=subprocess.PIPE)
    return proc.stdout


def n_atoms_of(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return int(fh.readline().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out", default="1fme_signals.csv")
    ap.add_argument("--chunk", type=int, default=5000, help="frames per chunk")
    args = ap.parse_args()

    n_atoms = n_atoms_of(args.xyz)
    reader = pd.read_csv(coordinate_stream(args.xyz), sep=" ", header=None,
                         dtype=np.float64, engine="c",
                         chunksize=args.chunk * n_atoms)
    first, last = None, None
    out = {k: [] for k in ("mean_x", "mean_y", "mean_z", "rmsd_from_first",
                           "f2f_rmsd", "rg")}
    for chunk in reader:
        a = chunk.to_numpy()
        if a.shape[0] % n_atoms:
            a = a[: a.shape[0] - a.shape[0] % n_atoms]
            print("[warn] incomplete final frame dropped")
        X = a.reshape(-1, n_atoms, 3)
        if first is None:
            first = X[0].copy()
        means = X.mean(axis=1)
        out["mean_x"].append(means[:, 0])
        out["mean_y"].append(means[:, 1])
        out["mean_z"].append(means[:, 2])
        c = X - means[:, None, :]
        out["rg"].append(np.sqrt((c ** 2).sum(axis=2).mean(axis=1)))
        out["rmsd_from_first"].append(
            batched_kabsch_rmsd(np.broadcast_to(first, X.shape), X))
        prev = np.concatenate(([last] if last is not None else [], X[:-1])) \
            if last is not None else X[:-1]
        f2f = batched_kabsch_rmsd(prev, X if last is not None else X[1:])
        out["f2f_rmsd"].append(f2f if last is not None
                               else np.concatenate(([np.nan], f2f)))
        last = X[-1].copy()
        print(f"  {sum(len(v) for v in out['rg'])} frames", flush=True)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})
    df.insert(0, "frame", np.arange(1, len(df) + 1))
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(df)} frames, {n_atoms} atoms -> {args.out}")


if __name__ == "__main__":
    main()
