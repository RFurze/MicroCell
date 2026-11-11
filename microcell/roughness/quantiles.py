"""NumPy-only quantile functions for the height-distribution (HPD) families.

The old generator (EHLBase ``microscale/src/functions/rough_surface.py``) took
its quantiles from ``scipy.stats``.  This package must import with NumPy only
(Case API plan §6.1, gate G3), so the five families it supported are evaluated
here instead:

========== ================================================= ===================
family     quantile used                                     method
========== ================================================= ===================
gaussian   ``norm.ppf``                                      Cephes ``ndtri``
gennorm    ``gennorm.ppf(p, beta)``                          incomplete-gamma
                                                             inversion (Newton,
                                                             bracketed)
weibull    ``weibull_min.ppf(p, c)``                         closed form
johnsonsu  ``johnsonsu.ppf(p, a, b)``                        closed form + ndtri
skewnorm   ``skewnorm.ppf(p, a)``                            Owen's T + bracketed
                                                             Newton
========== ================================================= ===================

``ndtri`` is the Cephes rational approximation that SciPy itself evaluates, so
the Gaussian family reproduces the old generator's sorted target heights to
the last bit or within one ulp (see ``tests/roughness``).  The others agree
with SciPy to round-off (``1e-13`` relative or better over the tested shape
ranges); they are not bit-for-bit.

Only ``math`` and ``numpy`` are imported.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "norm_ppf", "norm_cdf", "gammainc_pq", "gennorm_ppf", "gennorm_cdf",
    "weibull_min_ppf", "johnsonsu_ppf", "owens_t", "skewnorm_cdf",
    "skewnorm_ppf",
]

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Normal distribution: Cephes ndtri (the routine behind scipy.special.ndtri)
# ---------------------------------------------------------------------------
_S2PI = 2.50662827463100050242e0          # sqrt(2 pi)
_EXPM2 = 0.13533528323661269189           # exp(-2)

# approximation for 0 <= |y - 0.5| <= 3/8
_P0 = (-5.99633501014107895267e1, 9.80010754185999661536e1,
       -5.66762857469070293439e1, 1.39312609387279679503e1,
       -1.23916583867381258016e0)
_Q0 = (1.95448858338141759834e0, 4.67627912898881538453e0,
       8.63602421390890590575e1, -2.25462687854119370527e2,
       2.00260212380060660359e2, -8.20372256168333339912e1,
       1.59056225126211695515e1, -1.18331621121330003142e0)
# z = sqrt(-2 log y) between 2 and 8
_P1 = (4.05544892305962419923e0, 3.15251094599893866154e1,
       5.71628192246421288162e1, 4.40805073893200834700e1,
       1.46849561928858024014e1, 2.18663306850790267539e0,
       -1.40256079171354495875e-1, -3.50424626827848203418e-2,
       -8.57456785154685413611e-4)
_Q1 = (1.57799883256466749731e1, 4.53907635128879210584e1,
       4.13172038254672030440e1, 1.50425385692907503408e1,
       2.50464946208309415979e0, -1.42182922854787788574e-1,
       -3.80806407691578277194e-2, -9.33259480895457427372e-4)
# z = sqrt(-2 log y) between 8 and 64
_P2 = (3.23774891776946035970e0, 6.91522889068984211695e0,
       3.93881025292474443415e0, 1.33303460815807542389e0,
       2.01485389549179081538e-1, 1.23716634817820021358e-2,
       3.01581553508235416007e-4, 2.65806974686737550832e-6,
       6.23974539184983293730e-9)
_Q2 = (6.02427039364742014255e0, 3.67983563856160859403e0,
       1.37702099489081330271e0, 2.16236993594496635890e-1,
       1.34204006088543189037e-2, 3.28014464682127739104e-4,
       2.89247864745380683936e-6, 6.79019408009981274425e-9)


def _polevl(x: np.ndarray, coef) -> np.ndarray:
    ans = np.full_like(x, coef[0])
    for c in coef[1:]:
        ans = ans * x + c
    return ans


def _p1evl(x: np.ndarray, coef) -> np.ndarray:
    ans = x + coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans


def norm_ppf(p) -> np.ndarray:
    """Standard-normal quantile (Cephes ``ndtri``), vectorised.

    ``p`` must lie in [0, 1]; 0 and 1 give -inf and +inf.
    """
    y0 = np.asarray(p, dtype=float)
    scalar = y0.ndim == 0
    y0 = np.atleast_1d(y0)
    if np.any((y0 < 0.0) | (y0 > 1.0)) or np.any(~np.isfinite(y0)):
        raise ValueError("norm_ppf: p must lie in [0, 1]")
    out = np.empty_like(y0)
    upper = y0 > (1.0 - _EXPM2)
    y = np.where(upper, 1.0 - y0, y0)
    central = y > _EXPM2
    if np.any(central):
        yc = y[central] - 0.5
        y2 = yc * yc
        x = yc + yc * (y2 * _polevl(y2, _P0) / _p1evl(y2, _Q0))
        out[central] = x * _S2PI
    tail = ~central
    if np.any(tail):
        yt = y[tail]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.sqrt(-2.0 * np.log(yt))
            x0 = x - np.log(x) / x
            z = 1.0 / x
            x1 = np.where(x < 8.0,
                          z * _polevl(z, _P1) / _p1evl(z, _Q1),
                          z * _polevl(z, _P2) / _p1evl(z, _Q2))
            x = x0 - x1
        # Cephes: code = 1 (lower tail) flips the sign
        x = np.where(upper[tail], x, -x)
        x = np.where(yt == 0.0, np.where(upper[tail], np.inf, -np.inf), x)
        out[tail] = x
    return out[0] if scalar else out


_ERFC = np.frompyfunc(math.erfc, 1, 1)


def norm_cdf(x) -> np.ndarray:
    """Standard-normal CDF via ``math.erfc`` (exact to ~1 ulp, scalar loop)."""
    x = np.asarray(x, dtype=float)
    return (0.5 * _ERFC(-x / math.sqrt(2.0)).astype(float)).reshape(x.shape)


# ---------------------------------------------------------------------------
# Regularised incomplete gamma P(a, x), Q(a, x) for scalar a, array x
# ---------------------------------------------------------------------------
_FPMIN = 1e-300
_MAXIT = 2000


def gammainc_pq(a: float, x, logx=None) -> Tuple[np.ndarray, np.ndarray]:
    """Regularised lower/upper incomplete gamma ``(P(a,x), Q(a,x))``.

    Series for ``x < a + 1``, modified-Lentz continued fraction otherwise
    (Numerical Recipes 3rd ed. §6.2).  ``logx`` may be passed when ``x``
    itself would underflow (``x = exp(logx)``).
    """
    a = float(a)
    if not a > 0.0:
        raise ValueError("gammainc_pq: a must be > 0")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if logx is None:
        with np.errstate(divide="ignore"):
            logx = np.log(x)
    else:
        logx = np.atleast_1d(np.asarray(logx, dtype=float))
    P = np.zeros_like(x)
    Q = np.ones_like(x)
    lg = math.lgamma(a)
    pos = np.isfinite(logx)
    ser = pos & (x < a + 1.0)
    cf = pos & ~ser
    if np.any(ser):
        xs = x[ser]
        ap = a
        d = np.full_like(xs, 1.0 / a)
        s = d.copy()
        for _ in range(_MAXIT):
            ap += 1.0
            d = d * xs / ap
            s = s + d
            if np.all(np.abs(d) <= np.abs(s) * _EPS):
                break
        pv = s * np.exp(-xs + a * logx[ser] - lg)
        P[ser] = pv
        Q[ser] = 1.0 - pv
    if np.any(cf):
        xc = x[cf]
        b = xc + 1.0 - a
        c = np.full_like(xc, 1.0 / _FPMIN)
        d = 1.0 / b
        h = d.copy()
        for i in range(1, _MAXIT):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
            c = b + an / c
            c = np.where(np.abs(c) < _FPMIN, _FPMIN, c)
            d = 1.0 / d
            de = d * c
            h = h * de
            if np.all(np.abs(de - 1.0) <= _EPS):
                break
        qv = np.exp(-xc + a * logx[cf] - lg) * h
        Q[cf] = qv
        P[cf] = 1.0 - qv
    return P, Q


# ---------------------------------------------------------------------------
# Generalised normal (scipy.stats.gennorm, scale 1)
# ---------------------------------------------------------------------------
def _gennorm_abs_cdf(w: np.ndarray, beta: float):
    """P(|Z| <= w), P(|Z| > w) for gennorm(beta): P(1/beta, w**beta)."""
    a = 1.0 / beta
    with np.errstate(divide="ignore"):
        logx = beta * np.log(w)
    x = np.exp(logx)
    return gammainc_pq(a, x, logx)


def gennorm_cdf(x, beta: float) -> np.ndarray:
    beta = float(beta)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    P, _ = _gennorm_abs_cdf(np.abs(x), beta)
    return 0.5 + 0.5 * np.sign(x) * P


def _gamma_inv_guess(a: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Starting values for P(a, x) = p (Numerical Recipes 3rd ed. §6.2.1)."""
    if a > 1.0:
        pp = np.minimum(p, q)
        with np.errstate(divide="ignore"):
            t = np.sqrt(-2.0 * np.log(np.maximum(pp, 1e-300)))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        x = np.where(p < 0.5, -x, x)
        return np.maximum(1e-3, a * (1.0 - 1.0 / (9.0 * a) - x / (3.0 * math.sqrt(a))) ** 3)
    t = 1.0 - a * (0.253 + a * 0.12)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_lo = np.power(p / t, 1.0 / a)
        x_hi = 1.0 - np.log(np.maximum(q, 1e-300) / (1.0 - t))
    return np.where(p < t, x_lo, x_hi)


def _gennorm_abs_ppf(p_abs: np.ndarray, q_abs: np.ndarray,
                     beta: float) -> np.ndarray:
    """Solve P(1/beta, w**beta) = p_abs for w >= 0 (q_abs = 1 - p_abs given
    separately so the upper tail keeps its relative precision)."""
    a = 1.0 / beta
    lg = math.lgamma(a)
    n = p_abs.size
    w = np.zeros(n)
    todo = p_abs > 0.0
    if not np.any(todo):
        return w
    use_q = q_abs < p_abs                      # residual on the smaller tail
    qmin = float(np.min(q_abs[todo])) if np.any(todo) else 0.5
    if qmin <= 0.0:
        raise ValueError("gennorm_ppf: p must lie strictly inside (0, 1)")
    x_hi = 2.0 * (-math.log(qmin)) + 4.0 * a + 20.0
    while gammainc_pq(a, np.array([x_hi]))[1][0] >= qmin:
        x_hi *= 2.0
    lo = np.zeros(n)
    hi = np.full(n, x_hi ** (1.0 / beta))
    # initial guess: Numerical Recipes 3rd ed. gammpinv starting values
    w = _gamma_inv_guess(a, p_abs, q_abs) ** (1.0 / beta)
    w = np.minimum(np.maximum(w, 0.0), hi)
    active = todo.copy()
    for _ in range(300):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        wi = w[idx]
        P, Q = _gennorm_abs_cdf(wi, beta)
        r = np.where(use_q[idx], q_abs[idx] - Q, P - p_abs[idx])   # increasing in w
        lo[idx] = np.where(r < 0.0, wi, lo[idx])
        hi[idx] = np.where(r > 0.0, wi, hi[idx])
        g = beta * np.exp(-(wi ** beta) - lg)                      # dP/dw
        with np.errstate(divide="ignore", invalid="ignore"):
            wn = wi - r / g
        bad = ~np.isfinite(wn) | (wn <= lo[idx]) | (wn >= hi[idx])
        wn = np.where(bad, 0.5 * (lo[idx] + hi[idx]), wn)
        step = np.abs(wn - wi)
        w[idx] = wn
        done = (step <= 4.0 * _EPS * np.abs(wn)) | (r == 0.0) | \
               (hi[idx] - lo[idx] <= 4.0 * _EPS * np.abs(wn))
        active[idx[done]] = False
    return w


def gennorm_ppf(p, beta: float) -> np.ndarray:
    """Quantile of the generalised normal ``exp(-|z|**beta)`` (scale 1)."""
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("gennorm_ppf: beta must be > 0")
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("gennorm_ppf: p must lie strictly inside (0, 1)")
    c = np.sign(p - 0.5)
    lower = np.minimum(p, 1.0 - p)             # exact for p from (i+0.5)/n
    q_abs = 2.0 * lower                         # P(|Z| > w); P(|Z| <= w) = 1 - q_abs
    # |Z| depends on p only through q_abs: solve each distinct value once
    # (a symmetric probability grid halves the work)
    uq, inv = np.unique(q_abs, return_inverse=True)
    up = np.abs(1.0 - uq)
    w = _gennorm_abs_ppf(up, uq, beta)[inv.ravel()]
    return c * w


# ---------------------------------------------------------------------------
# Weibull (scipy.stats.weibull_min) and Johnson SU: closed forms
# ---------------------------------------------------------------------------
def weibull_min_ppf(p, c: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.power(-np.log1p(-p), 1.0 / float(c))


def johnsonsu_ppf(p, a: float, b: float) -> np.ndarray:
    return np.sinh((norm_ppf(p) - float(a)) / float(b))


# ---------------------------------------------------------------------------
# Skew normal (scipy.stats.skewnorm): Owen's T + bracketed Newton
# ---------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _owens_t_small(h: np.ndarray, a: float) -> np.ndarray:
    """T(h, a) for |a| <= 1 by 64-point Gauss-Legendre on [0, a]."""
    t = 0.5 * a * (_GL_X + 1.0)                       # nodes on [0, a]
    wt = 0.5 * a * _GL_W
    f = np.exp(-0.5 * np.multiply.outer(h * h, 1.0 + t * t)) / (1.0 + t * t)
    return f @ wt / (2.0 * math.pi)


def owens_t(h, a: float) -> np.ndarray:
    """Owen's T function T(h, a) (scalar a, array h)."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    a = float(a)
    if a == 0.0:
        return np.zeros_like(h)
    if a < 0.0:
        return -owens_t(h, -a)
    if a <= 1.0:
        return _owens_t_small(h, a)
    # T is even in h; for h >= 0, a > 1:
    # T(h,a) = (Phi(h) + Phi(ah))/2 - Phi(h) Phi(ah) - T(ah, 1/a)
    h = np.abs(h)
    ph = norm_cdf(h)
    pah = norm_cdf(a * h)
    return 0.5 * ph + 0.5 * pah - ph * pah - _owens_t_small(a * h, 1.0 / a)


def skewnorm_cdf(x, a: float) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return norm_cdf(x) - 2.0 * owens_t(x, a)


def skewnorm_ppf(p, a: float) -> np.ndarray:
    """Quantile of the skew normal with shape ``a`` (location 0, scale 1)."""
    a = float(a)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("skewnorm_ppf: p must lie strictly inside (0, 1)")
    if a < 0.0:
        return -skewnorm_ppf(1.0 - p, -a)
    if a == 0.0:
        return norm_ppf(p)
    # a > 0: Phi^-1(p) <= x <= Phi^-1((1+p)/2)
    lo = norm_ppf(p)
    hi = norm_ppf(0.5 * (1.0 + p))
    x = 0.5 * (lo + hi)
    active = np.ones(p.size, dtype=bool)
    for _ in range(200):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        xi = x[idx]
        r = skewnorm_cdf(xi, a) - p[idx]
        lo[idx] = np.where(r < 0.0, xi, lo[idx])
        hi[idx] = np.where(r > 0.0, xi, hi[idx])
        pdf = 2.0 * np.exp(-0.5 * xi * xi) / math.sqrt(2.0 * math.pi) * norm_cdf(a * xi)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            xn = xi - r / pdf
        bad = ~np.isfinite(xn) | (xn <= lo[idx]) | (xn >= hi[idx])
        xn = np.where(bad, 0.5 * (lo[idx] + hi[idx]), xn)
        step = np.abs(xn - xi)
        x[idx] = xn
        done = (step <= 8.0 * _EPS * np.maximum(1.0, np.abs(xn))) | (r == 0.0)
        active[idx[done]] = False
    return x
