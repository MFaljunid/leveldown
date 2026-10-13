"""How much of a group gap is explained by a covariate (regression share and coarsened matching)."""
import numpy as np


def zscore(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def ols_gap(y, groups, covars, lo, hi):
    """OLS of y on group dummies (reference `lo`) + covariates; returns the `hi` coefficient."""
    labels = sorted(set(groups.tolist()))
    names = [g for g in labels if g != lo]
    X = np.column_stack([np.ones(len(y))] + [(groups == g).astype(float) for g in names] + list(covars))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1 + names.index(hi)])


def matched_gap(y, groups, strata, lo, hi):
    """Mean over strata (weights = #hi users) of mean_hi - mean_lo; strata with both groups.
    Returns (gap, coverage of hi users)."""
    num, den = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        a, b = y[m & (groups == lo)], y[m & (groups == hi)]
        if len(a) and len(b):
            num += len(b) * (b.mean() - a.mean())
            den += len(b)
    return (num / den if den else np.nan), den / max(int((groups == hi).sum()), 1)


def deciles(v):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))


def decompose(y, groups, covar, lo, hi, B=1000, seed=0):
    """Raw gap, covariate-adjusted gap, share explained with bootstrap 95% CI, matched gap (deciles)."""
    z = zscore(covar)
    raw = ols_gap(y, groups, [], lo, hi)
    adj = ols_gap(y, groups, [z], lo, hi)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        r = ols_gap(y[i], groups[i], [], lo, hi)
        a = ols_gap(y[i], groups[i], [z[i]], lo, hi)
        bs.append(1 - a / r if r != 0 else np.nan)
    lo_ci, hi_ci = np.nanpercentile(bs, [2.5, 97.5])
    mg, cov = matched_gap(y, groups, deciles(covar), lo, hi)
    return {"raw": raw, "adjusted": adj, "share": 1 - adj / raw if raw != 0 else np.nan,
            "share_ci": [float(lo_ci), float(hi_ci)], "matched": mg, "coverage": cov}
