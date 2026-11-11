"""NumPy-only standard-normal quantile for the Gaussian height distribution (HPD).

The old generator (EHLBase ``microscale/src/functions/rough_surface.py``) took
its quantiles from ``scipy.stats``.  This package imports NumPy only, so
``norm.ppf`` is evaluated here by the Cephes rational approximation ``ndtri``
that SciPy itself evaluates: the Gaussian family reproduces the old
generator's sorted target heights to the last bit or within one ulp.

[MicroCell WP3] The parent's quantiles of the other HPD families (``gennorm``,
``weibull``, ``johnsonsu``, ``skewnorm``) and the functions they need
(``norm_cdf``, the incomplete gamma, Owen's T) were removed with those
families; they are in the parent (``ehl/roughness/quantiles.py``).

Only ``numpy`` is imported.
"""
from __future__ import annotations

import numpy as np

__all__ = ["norm_ppf"]


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
