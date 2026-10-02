"""Rigid-body diagnostics (Methods, 'Rigid-body diagnostics and structural signals').

For consecutive frames we compute the median atomic displacement without alignment
and the RMSD after optimal superposition (Kabsch algorithm, all atoms, equal
weights). We also compute the superposed RMSD of each frame from frame 1 and the
largest single-step displacement of any atom (periodic-wrapping check).
"""

import numpy as np


def kabsch_rmsd(P, Q):
    """RMSD between coordinate sets P and Q (n_atoms x 3) after optimal
    translation and rotation (proper rotations only)."""
    p = P - P.mean(axis=0)
    q = Q - Q.mean(axis=0)
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = S.copy()
    S[-1] *= d
    # ||p R - q||^2 = |p|^2 + |q|^2 - 2 tr(S')
    msd = ((p ** 2).sum() + (q ** 2).sum() - 2.0 * S.sum()) / P.shape[0]
    return float(np.sqrt(max(msd, 0.0)))


def diagnostics(traj):
    """Return a dict of per-frame and per-step diagnostics.

    Arrays indexed by step have length n_frames - 1; element i describes the step
    from frame i+1 to frame i+2 (1-based)."""
    n = traj.shape[0]
    step_disp = np.linalg.norm(np.diff(traj, axis=0), axis=2)  # (n-1, atoms)
    median_disp = np.median(step_disp, axis=1)
    max_disp = step_disp.max(axis=1)
    mean_vec = np.diff(traj, axis=0).mean(axis=1)               # (n-1, 3)
    f2f_rmsd = np.array([kabsch_rmsd(traj[i], traj[i + 1]) for i in range(n - 1)])
    rmsd_first = np.array([kabsch_rmsd(traj[0], traj[i]) for i in range(n)])
    return dict(
        median_step_disp=median_disp,
        max_step_disp=max_disp,
        min_step_disp=step_disp.min(axis=1),
        mean_step_vector=mean_vec,
        f2f_rmsd=f2f_rmsd,
        rmsd_from_first=rmsd_first,
    )


def report_jump(diag, frame_before, frame_after):
    """Print the comparison reported in the Results for one step (1-based)."""
    i = frame_before - 1
    others = np.delete(diag["f2f_rmsd"], i)
    print(f"Step {frame_before}->{frame_after}:")
    print(f"  atomic displacement range  {diag['min_step_disp'][i]:.2f}-"
          f"{diag['max_step_disp'][i]:.2f} A")
    print(f"  mean displacement vector   {np.round(diag['mean_step_vector'][i], 2)} A")
    print(f"  median atomic displacement {diag['median_step_disp'][i]:.3f} A "
          f"(typical step {np.median(np.delete(diag['median_step_disp'], i)):.3f} A)")
    print(f"  superposed RMSD            {diag['f2f_rmsd'][i]:.3f} A "
          f"(median of other steps {np.median(others):.3f} A)")
    print(f"  RMSD from frame 1          {diag['rmsd_from_first'][frame_before-1]:.3f}"
          f" -> {diag['rmsd_from_first'][frame_after-1]:.3f} A")
    print(f"  largest single-atom step in trajectory "
          f"{diag['max_step_disp'].max():.2f} A")


def plot_diagnostics(means, diag, jump_frame, path):
    """Figure 2: (a) frame-averaged coordinates, (b) unaligned vs superposed
    frame-to-frame change, (c) RMSD from frame 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = means.shape[0]
    frames = np.arange(1, n + 1)
    steps = np.arange(2, n + 1)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)

    rel = means - means[0]
    for k, lab in enumerate("XYZ"):
        axes[0].plot(frames, rel[:, k], lw=1.2, label=lab)
    axes[0].set_ylabel("Mean coordinate\nrelative to frame 1 (\u00c5)")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)

    axes[1].semilogy(steps, diag["median_step_disp"], color="black", lw=1.1,
                     label="Median atomic displacement (unaligned)")
    axes[1].semilogy(steps, diag["f2f_rmsd"], color="tab:orange", lw=1.1,
                     label="RMSD after superposition")
    axes[1].set_ylabel("Frame-to-frame\nchange (\u00c5)")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(frames, diag["rmsd_from_first"], color="tab:blue", lw=1.2)
    axes[2].set_ylabel("RMSD from\nframe 1 (\u00c5)")
    axes[2].set_xlabel("Frame")

    for ax, letter in zip(axes, "abc"):
        ax.axvline(jump_frame, color="grey", ls=":", lw=1)
        ax.text(-0.13, 1.0, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=12, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
