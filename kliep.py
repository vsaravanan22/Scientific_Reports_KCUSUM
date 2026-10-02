"""Adjacent-window KLIEP detector (Methods, 'Adjacent-window KLIEP comparator').

An online adaptation of the KLIEP change-point score of Liu et al. (2013).
At frame t a reference window (t-19 .. t-10) is compared with a test window
(t-9 .. t). The density ratio w(x) = sum_l alpha_l K(x, c_l) uses Gaussian basis
functions centred on the ten test samples and is fitted by projected gradient
ascent under the constraint that w averages to one over the reference window.
The score is the mean log ratio over the test window.
"""

import numpy as np


def gauss_basis(x, centres, bw):
    return np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2.0 * bw ** 2))


def kliep_score(ref, test, bw, n_iter=2000, step=1e-3):
    K_te = gauss_basis(test, test, bw)          # (n_te, n_basis)
    b = gauss_basis(ref, test, bw).mean(axis=0)  # constraint vector
    alpha = np.ones(len(test))
    alpha /= b @ alpha
    for _ in range(n_iter):
        alpha = alpha + step * K_te.T @ (1.0 / (K_te @ alpha))
        alpha = alpha + (1.0 - b @ alpha) * b / (b @ b)
        alpha = np.maximum(alpha, 0.0)
        alpha = alpha / (b @ alpha)
    return float(np.mean(np.log(K_te @ alpha)))


def median_abs_pairwise_diff(v):
    d = np.abs(v[:, None] - v[None, :])
    return float(np.median(d[np.triu_indices(len(v), k=1)]))


def kliep_detector(series, frames, calib_last=80, win=10, calib_first_end=20,
                   quantile=95, n_iter=2000, step=1e-3):
    """Scores for every frame with complete windows, the calibrated threshold and
    the first alarm after the calibration interval."""
    series = np.asarray(series, dtype=float)
    frames = np.asarray(frames)
    calib = frames <= calib_last
    mu, sd = series[calib].mean(), series[calib].std(ddof=1)
    z = (series - mu) / sd
    bw = median_abs_pairwise_diff(z[calib])

    scores = np.full(len(z), np.nan)
    for i in range(2 * win - 1, len(z)):
        ref = z[i - 2 * win + 1: i - win + 1]
        test = z[i - win + 1: i + 1]
        scores[i] = kliep_score(ref, test, bw, n_iter, step)

    cal = (frames >= calib_first_end) & (frames <= calib_last) & ~np.isnan(scores)
    thr = float(np.percentile(scores[cal], quantile))
    after = (frames > calib_last) & (scores > thr)
    first = int(frames[after][0]) if after.any() else None
    return dict(scores=scores, threshold=thr, bandwidth=bw, first_alarm=first)
