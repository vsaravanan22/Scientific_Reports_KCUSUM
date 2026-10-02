"""KCUSUM on 1H9T signals (Methods, 'KCUSUM on 1H9T').

The detector loop reproduces KCUSUMDetector.run from the public kcusum.py
(https://github.com/HPCAI-lab/kcusum-cpd) without its plotting. With 0-based
indices it evaluates, for i = 2, 4, 6, ...,

    g = k(x[i], x[i-1]) + k(y[i], y[i-1]) - k(x[i], y[i-1]) - k(x[i-1], y[i])

with k(a, b) = exp(-(a - b)^2 / 2), i.e. sigma^2 = 1 on unstandardized values,
adds g - delta to Z, resets Z to 0 when negative and stops when Z > h. The
alarm index i is 0-based, so the reported 1-based frame is i + 1 (always odd).

The reference distribution is fitted with the `fitter` package (common
distributions, ranked by sum of squared errors), either to the full series (as in
the public script) or to the calibration frames only.
"""

import warnings

import numpy as np
from scipy.stats import distributions


def fit_reference(values):
    """Best common distribution by sum of squared errors, as in kcusum.py."""
    from fitter import Fitter, get_common_distributions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Fitter(np.asarray(values, dtype=float),
                   distributions=get_common_distributions(), verbose=False)
        f.fit()
        best = f.get_best(method="sumsquare_error")
    name, params = list(best.items())[0]
    return name, dict(params)


def sample_reference(name, params, size, seed):
    """Draw reference samples from the fitted distribution (as
    generate_random_samples in kcusum.py, but with an explicit seed)."""
    dist = getattr(distributions, name)
    loc = params.get("loc", 0)
    scale = params.get("scale", 1)
    args = [params[k] for k in params if k not in ("loc", "scale")]
    rng = np.random.default_rng(seed)
    return dist.rvs(*args, loc=loc, scale=scale, size=size, random_state=rng)


def gk(x, y):
    return np.exp(-((x - y) ** 2) / 2.0)


def kcusum_public(x, y, h=0.015, delta=0.1):
    """Detector loop of the public implementation. Returns (alarm_index_0based or
    None, statistic trace, indices of the trace)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = 0.0
    trace, idx = [], []
    for i in range(2, len(x), 2):
        g = gk(x[i], x[i - 1]) + gk(y[i], y[i - 1]) - gk(x[i], y[i - 1]) \
            - gk(x[i - 1], y[i])
        z += g - delta
        trace.append(z)
        idx.append(i)
        if z < 0:
            z = 0.0
        if z > h:
            return i, np.array(trace), np.array(idx)
    return None, np.array(trace), np.array(idx)


def run_kcusum_config(series, frames, ref_mask, n_runs=200, h=0.015, delta=0.1,
                      label=""):
    """Fit the reference on series[ref_mask] once, then run KCUSUM n_runs times
    with seeds 0..n_runs-1. Returns (alarm frames list with None for no alarm,
    distribution name)."""
    name, params = fit_reference(series[ref_mask])
    alarms = []
    for seed in range(n_runs):
        y = sample_reference(name, params, len(series), seed)
        i, _, _ = kcusum_public(series, y, h=h, delta=delta)
        alarms.append(None if i is None else int(frames[i]))
    print(f"[KCUSUM] {label}: reference = {name}; {summarize_alarms(alarms)}")
    return alarms, name


def summarize_alarms(alarms):
    """Text such as '101 (196 runs), 103 (3), 51 (1)'."""
    vals, counts = np.unique(np.array([-1 if a is None else a for a in alarms]),
                             return_counts=True)
    order = np.argsort(-counts)
    parts = []
    for j, k in enumerate(order):
        v = "none" if vals[k] == -1 else str(vals[k])
        parts.append(f"{v} ({counts[k]}{' runs' if j == 0 else ''})")
    return ", ".join(parts)
