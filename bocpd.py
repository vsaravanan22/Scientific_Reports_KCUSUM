"""Bayesian online changepoint detection (Adams & MacKay, 2007).

Univariate Gaussian observations with unknown mean and variance, conjugate
normal-inverse-gamma prior, constant hazard, untruncated run-length posterior
(Methods, 'BOCPD comparator').
"""

import numpy as np
from scipy.special import gammaln, logsumexp


def student_t_logpdf(x, df, loc, scale2):
    z = (x - loc) ** 2 / (df * scale2)
    return (gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df * scale2) - (df + 1) / 2 * np.log1p(z))


def bocpd(x, mu0, kappa0, alpha0, beta0, hazard=1 / 50):
    """Return the MAP run length after each observation (array of len(x)) and
    the full log run-length posterior as a list of arrays."""
    x = np.asarray(x, dtype=float)
    log_h = np.log(hazard)
    log_1mh = np.log1p(-hazard)

    mu = np.array([mu0])
    kappa = np.array([kappa0], dtype=float)
    alpha = np.array([alpha0], dtype=float)
    beta = np.array([beta0], dtype=float)
    log_r = np.array([0.0])  # P(r_0 = 0) = 1

    map_rl = np.empty(len(x), dtype=int)
    posts = []
    for t, xt in enumerate(x):
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        log_pred = student_t_logpdf(xt, 2 * alpha, mu, scale2)
        growth = log_r + log_pred + log_1mh
        cp = logsumexp(log_r + log_pred + log_h)
        log_r = np.concatenate(([cp], growth))
        log_r -= logsumexp(log_r)
        posts.append(log_r.copy())
        map_rl[t] = int(np.argmax(log_r))

        # Update sufficient statistics (new run length 0 restarts from the prior)
        mu_new = (kappa * mu + xt) / (kappa + 1)
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate(([mu0], mu_new))
        kappa = np.concatenate(([kappa0], kappa + 1))
        alpha = np.concatenate(([alpha0], alpha + 0.5))
        beta = np.concatenate(([beta0], beta_new))
    return map_rl, posts


def bocpd_alarms(series, frames, prior_frames=20, hazard=1 / 50, max_rl=10,
                 restrict_after=80):
    """Apply the alarm rule of the paper.

    Prior from frames 1..prior_frames: mu0 = sample mean, kappa0 = 1,
    alpha0 = 1, beta0 = sample variance. Alarm at the first frame after
    `prior_frames` whose MAP run length is <= max_rl. Also returns the first such
    frame after `restrict_after` and the number of frames after `prior_frames`
    that satisfy the rule."""
    series = np.asarray(series, dtype=float)
    frames = np.asarray(frames)
    prior = series[frames <= prior_frames]
    mu0 = prior.mean()
    beta0 = prior.var(ddof=1)
    map_rl, _ = bocpd(series, mu0, 1.0, 1.0, beta0, hazard)
    flag = (map_rl <= max_rl) & (frames > prior_frames)
    first = int(frames[flag][0]) if flag.any() else None
    flag_r = flag & (frames > restrict_after)
    first_r = int(frames[flag_r][0]) if flag_r.any() else None
    return dict(map_rl=map_rl, first_alarm=first,
                first_alarm_after_restrict=first_r, n_flagged=int(flag.sum()))
