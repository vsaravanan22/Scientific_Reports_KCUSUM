"""Core KCUSUM routines for the controlled benchmark experiments.

Implements Algorithm 2 of the manuscript (Flynn & Yoo, 2019) for observations in
R^d with a Gaussian kernel k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).

Tasks (pre-change distribution N(0, I_4 / 2) in every case):
    1  mean shift to (1, 1, 1, 1), unchanged covariance
    2  global variance increase to covariance 2 I_4
    3  one randomly chosen component of a pre-change sample scaled by 2
       (component drawn independently for every observation)
    4  independent uniform components on [-sqrt(3/2), sqrt(3/2)]
       (same mean and variance as the pre-change components)
"""

import numpy as np
from numba import njit, prange

DIM = 4
PRE_SD = np.sqrt(0.5)
UNIF_HALF_WIDTH = np.sqrt(1.5)
TASK_NAMES = {
    1: "Mean shift",
    2: "Global variance",
    3: "Single component",
    4: "Shape (uniform)",
}


@njit(cache=True)
def _kernel(a, b, inv_two_sigma2):
    s = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return np.exp(-s * inv_two_sigma2)


@njit(cache=True)
def g_pair(x0, x1, y0, y1, inv_two_sigma2):
    """Linear-time MMD^2 term g((x0, x1), (y0, y1)), equation (6)."""
    return (_kernel(x0, x1, inv_two_sigma2) + _kernel(y0, y1, inv_two_sigma2)
            - _kernel(x0, y1, inv_two_sigma2) - _kernel(x1, y0, inv_two_sigma2))


@njit(cache=True)
def sample_pre(out):
    for i in range(out.shape[0]):
        out[i] = PRE_SD * np.random.standard_normal()


@njit(cache=True)
def sample_post(task, out):
    d = out.shape[0]
    if task == 1:
        for i in range(d):
            out[i] = 1.0 + PRE_SD * np.random.standard_normal()
    elif task == 2:
        for i in range(d):
            out[i] = np.sqrt(2.0) * np.random.standard_normal()
    elif task == 3:
        for i in range(d):
            out[i] = PRE_SD * np.random.standard_normal()
        j = np.random.randint(0, d)
        out[j] *= 2.0
    elif task == 4:
        for i in range(d):
            out[i] = np.random.uniform(-UNIF_HALF_WIDTH, UNIF_HALF_WIDTH)
    else:
        raise ValueError("unknown task")


@njit(cache=True)
def kcusum_single_run(task, change, h, delta, sigma2, cap, seed):
    """One KCUSUM run.

    change = False: all observations from the pre-change distribution
                    (returns the time to false alarm).
    change = True : change at t = 1, all observations post-change
                    (returns the number of observations until the alarm).
    Returns -1 if the cap is reached (censored run).
    """
    np.random.seed(seed)
    inv = 1.0 / (2.0 * sigma2)
    x_prev = np.empty(DIM)
    x_cur = np.empty(DIM)
    y_prev = np.empty(DIM)
    y_cur = np.empty(DIM)

    # n = 1
    if change:
        sample_post(task, x_prev)
    else:
        sample_pre(x_prev)
    sample_pre(y_prev)
    z = 0.0
    n = 1
    while n < cap:
        n += 1
        if change:
            sample_post(task, x_cur)
        else:
            sample_pre(x_cur)
        sample_pre(y_cur)
        if n % 2 == 0:
            v = g_pair(x_prev, x_cur, y_prev, y_cur, inv) - delta
            z = max(0.0, z + v)
            if z >= h:
                return n
        x_prev, x_cur = x_cur, x_prev
        y_prev, y_cur = y_cur, y_prev
    return -1


@njit(parallel=True, cache=True)
def kcusum_many_runs(task, change, h, delta, sigma2, cap, n_runs, base_seed):
    """n_runs independent runs. Each run seeds its own generator, so results do
    not depend on the number of threads."""
    out = np.empty(n_runs, dtype=np.int64)
    for r in prange(n_runs):
        out[r] = kcusum_single_run(task, change, h, delta, sigma2, cap,
                                   base_seed + r)
    return out


@njit(cache=True)
def estimate_mmd2(task, sigma2, n_pairs, seed):
    """Monte Carlo estimate of d_k^2(p0, p1) by averaging g over independent
    pairs (x0, x1) ~ p1 and (y0, y1) ~ p0."""
    np.random.seed(seed)
    inv = 1.0 / (2.0 * sigma2)
    x0 = np.empty(DIM)
    x1 = np.empty(DIM)
    y0 = np.empty(DIM)
    y1 = np.empty(DIM)
    s = 0.0
    s2 = 0.0
    for _ in range(n_pairs):
        sample_post(task, x0)
        sample_post(task, x1)
        sample_pre(y0)
        sample_pre(y1)
        v = g_pair(x0, x1, y0, y1, inv)
        s += v
        s2 += v * v
    mean = s / n_pairs
    se = np.sqrt((s2 / n_pairs - mean * mean) / n_pairs)
    return mean, se


def arl_bound(h, delta, k_inf=1.0):
    """False-alarm lower bound, equation (8)."""
    return 2.0 * np.exp(h / (4.0 * k_inf) * np.log1p(delta / (4.0 * k_inf)))


def delay_bound(h, delta, mmd2, k_inf=1.0):
    """Detection-delay upper bound, equation (9). Returns inf if mmd2 <= delta."""
    gap = mmd2 - delta
    if gap <= 0:
        return np.inf
    return 2.0 * h / gap + 8.0 * k_inf ** 2 / gap ** 2
