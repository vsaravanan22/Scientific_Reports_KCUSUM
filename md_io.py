"""Reading VMD-format XYZ trajectories.

Each frame is
    <number of atoms>
    <comment line>
    <element> x y z      (one line per atom)

The reader streams frames one at a time and stops cleanly at an incomplete final
frame (the 1H9T file supplied to us ends part-way through frame 152).
"""

import io
import itertools

import numpy as np


def iter_frames(path, max_frames=None, dtype=np.float64):
    """Yield (frame_number, coords[n_atoms, 3]) with 1-based frame numbers."""
    with open(path, "r") as fh:
        frame = 0
        while max_frames is None or frame < max_frames:
            header = fh.readline()
            if not header:
                return
            header = header.strip()
            if not header:
                continue
            n_atoms = int(header)
            fh.readline()  # comment line
            lines = list(itertools.islice(fh, n_atoms))
            if len(lines) < n_atoms:
                print(f"[md_io] incomplete frame after frame {frame} "
                      f"({len(lines)} of {n_atoms} atom lines); stopping.")
                return
            try:
                xyz = np.loadtxt(io.StringIO("".join(lines)), usecols=(1, 2, 3),
                                 dtype=dtype, ndmin=2)
            except ValueError:
                print(f"[md_io] unparseable frame after frame {frame}; stopping.")
                return
            if xyz.shape != (n_atoms, 3):
                print(f"[md_io] malformed frame after frame {frame}; stopping.")
                return
            frame += 1
            yield frame, xyz


def read_trajectory(path, max_frames=None, cache=True):
    """Load a whole trajectory as an array (n_frames, n_atoms, 3).

    With cache=True the parsed array is stored next to the input as
    <path>.npy and reused on later calls."""
    cache_path = path + ".npy"
    if cache and max_frames is None:
        try:
            return np.load(cache_path)
        except (FileNotFoundError, OSError):
            pass
    frames = [xyz for _, xyz in iter_frames(path, max_frames)]
    traj = np.stack(frames)
    if cache and max_frames is None:
        try:
            np.save(cache_path, traj)
        except OSError:
            pass
    return traj


def frame_means(traj):
    """Frame-averaged X, Y and Z coordinates over all atoms, without unwrapping
    or alignment (as in the public kcusum.py)."""
    return traj.mean(axis=1)
