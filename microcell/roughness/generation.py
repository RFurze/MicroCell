"""Random rough-surface generation: IAAFT port and the filtered Gaussian generator.

This is a port of the numerical core of the old tree's generator
(EHLBase-DispatchHeavy ``microscale/src/functions/rough_surface.py``,
sha256 ``3d9b481d…0590``, 15 July 2026), which implements

    Pérez-Ràfols & Almqvist, *Generating randomly rough surfaces with given
    height probability distribution and power spectrum*, Tribology
    International 131 (2019) 591-604,

plus the scipy-free Gaussian generator of the refactor tree's
``microscale/src/functions/generated_ht_cell.py`` (``gaussian_surface``) and the
radial-PSD / empirical-HPD file inputs of the old ``generated_surface.py``.

What is the same as the old code (verified by ``tests/roughness``):

* PSD models, Parseval normalisation, the IAAFT loop (seed draw, stable
  arg-sort rank remap, iteration order), ``take='Zh'`` / ``'Zs'``;
* the five HPD families, now evaluated by :mod:`ehl.roughness.quantiles`
  (NumPy only) instead of ``scipy.stats``;
* the Sm fit (bisection on measured summit spacing, legacy *maxima* rule);
* ``generate_rough_surface`` (old top-level entry) for parity.

What is deliberately different (Case API plan §2.3, §4):

* Radial-PSD tables are evaluated on the true rectangular frequency grid with
  zero power outside the measured band by default.  The old behaviour (one
  axis length for both directions, queries clipped to the band edges) is the
  explicit ``out_of_band="legacy_clip"`` mode, square cells only.
* The Gaussian generator returns the canonical ``z[j_y, i_x]`` layout; the old
  function's ``z[i_x, j_y]`` array is ``layout="legacy_xy"``.
* Error metrics: the old ``err_h`` / ``err_s`` are kept under ``*_legacy``
  names.  ``err_s_legacy`` is an unweighted relative error over every mode
  above 1e-4 of the peak amplitude and is dominated by the weak high-q tail of
  Gaussian-ACF spectra (1.21 for the WTC26 gaussian surface); the energy-
  weighted :func:`spectral_energy_error` is the one to put tolerances on.

All arrays are ``z[j_y, i_x]`` on the half-open periodic grid, SI units.
Only NumPy is imported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from . import quantiles as _q

__all__ = [
    "GENERATOR_REVISION", "LEGACY_SOURCE", "HDS_CORR_LEN",
    "wavenumber_grid", "wavevector_grids", "normalise_amplitude",
    "psd_self_affine", "psd_exponential_acf", "psd_gaussian_acf",
    "PSD_MODELS", "PSD_FIT_PARAM", "amplitude_from_radial_table",
    "hpd_sorted_sample", "heights_from_samples", "HPD_FAMILIES",
    "HPD_SHAPE_BRACKET", "sa_over_sq_of_hpd", "fit_hpd_shape",
    "IAAFTResult", "iaaft", "rank_remap", "err_h_legacy", "err_s_legacy",
    "spectral_energy_error", "fit_psd_for_sm", "SurfaceResult",
    "generate_rough_surface", "gaussian_filtered_surface",
]

#: Version tag of this module's numerical behaviour; part of recipe identity.
GENERATOR_REVISION = "ehl.roughness.generation/1"

#: Provenance of the ported code.
LEGACY_SOURCE = {
    "iaaft": {
        "path": "EHLBase-DispatchHeavy/microscale/src/functions/rough_surface.py",
        "sha256": "3d9b481db07b7b97a0e04d06af7e448e8438c6ee684ae1ae23c7a9d07e980590",
    },
    "file_inputs": {
        "path": "EHLBase-DispatchHeavy/microscale/src/functions/generated_surface.py",
        "sha256": "080b3a85cea515cc173934cd6f86e58b7b78a663c8c9ee7e74f5f124a3f5e4bf",
    },
    "gaussian_filter": {
        "path": "MultiscaleMixedBallinSocket/microscale/src/functions/generated_ht_cell.py",
        "sha256": "015f89d227a573d4dff9f3ae351bc4667924bd2420a6c43f5038b719d2a1f6bc",
    },
}

#: Correlation length fitted for the WTC26 HDS gaussian family (Sm target
#: 0.78 um on the 5 um, N = 64 cell; ``wtc26_hds/meta.json``).
HDS_CORR_LEN = 3.2633555700856686e-07


# ---------------------------------------------------------------------------
# Wavenumber grids
# ---------------------------------------------------------------------------
def wavenumber_grid(ny: int, nx: int, Ly: float, Lx: float) -> np.ndarray:
    """Radial wavenumber |q| = 2*pi*sqrt(fx^2 + fy^2) [rad/m], shape (ny, nx),
    FFT-ordered (q[0, 0] = 0).  Same signature and values as the old code."""
    fx = np.fft.fftfreq(nx, d=Lx / nx)      # cycles per metre
    fy = np.fft.fftfreq(ny, d=Ly / ny)
    QX, QY = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy, indexing="xy")
    return np.sqrt(QX ** 2 + QY ** 2)


def wavevector_grids(ny: int, nx: int, Ly: float, Lx: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(QX, QY) [rad/m], each shape (ny, nx), FFT-ordered."""
    fx = np.fft.fftfreq(nx, d=Lx / nx)
    fy = np.fft.fftfreq(ny, d=Ly / ny)
    return np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy, indexing="xy")


# ---------------------------------------------------------------------------
# PSD models -> amplitude arrays |Z(q)|, normalised to a unit-rms surface
# ---------------------------------------------------------------------------
def normalise_amplitude(A: np.ndarray) -> np.ndarray:
    """Scale an amplitude array so the realised surface has unit rms
    (Parseval for NumPy's unnormalised FFT); zero mean."""
    A = np.array(A, dtype=float)
    A[0, 0] = 0.0
    n = A.size
    s = np.sqrt(np.sum(A ** 2)) / n
    if s == 0.0:
        raise ValueError("PSD is identically zero over the grid band.")
    return A / s


def psd_self_affine(q: np.ndarray, H: float = 0.8,
                    q0: Optional[float] = None,
                    q1: Optional[float] = None) -> np.ndarray:
    """Isotropic self-affine PSD C(q) = (q/q0)^(-2(1+H)) on [q0, q1], else 0."""
    qpos = q[q > 0]
    if q0 is None:
        q0 = float(qpos.min())
    if q1 is None:
        q1 = float(qpos.max())
    C = np.zeros_like(q)
    band = (q >= q0) & (q <= q1)
    C[band] = (q[band] / q0) ** (-2.0 * (1.0 + H))
    return normalise_amplitude(np.sqrt(C))


def psd_exponential_acf(q: np.ndarray, corr_len: float = 1e-4,
                        q1: Optional[float] = None) -> np.ndarray:
    """PSD of the exponential ACF R(r) = exp(-|r|/corr_len):
    C(q) ~ (beta^2 + q^2)^(-3/2), beta = 1/corr_len."""
    beta = 1.0 / float(corr_len)
    C = (beta ** 2 + q ** 2) ** (-1.5)
    C[0, 0] = 0.0
    if q1 is not None:
        C[q > q1] = 0.0
    return normalise_amplitude(np.sqrt(C))


def psd_gaussian_acf(q: np.ndarray, corr_len: float = 1e-4,
                     q1: Optional[float] = None) -> np.ndarray:
    """PSD of the Gaussian ACF R(r) = exp(-(r/corr_len)^2):
    C(q) ~ exp(-q^2 corr_len^2 / 4)."""
    C = np.exp(-(q ** 2) * (float(corr_len) ** 2) / 4.0)
    C[0, 0] = 0.0
    if q1 is not None:
        C[q > q1] = 0.0
    return normalise_amplitude(np.sqrt(C))


PSD_MODELS: Dict[str, Callable[..., np.ndarray]] = {
    "self_affine": psd_self_affine,
    "exponential_acf": psd_exponential_acf,
    "gaussian_acf": psd_gaussian_acf,
}

#: PSD keyword used as the length-scale knob when fitting Sm
PSD_FIT_PARAM: Dict[str, str] = {
    "self_affine": "H",
    "exponential_acf": "corr_len",
    "gaussian_acf": "corr_len",
}


def amplitude_from_radial_table(f: np.ndarray, P: np.ndarray, ny: int, nx: int,
                                Ly: float, Lx: float, gamma_tilt: float = 0.0,
                                out_of_band: str = "zero") -> np.ndarray:
    """Target |Z| on the FFT grid from a radial PSD table (shape only).

    ``f`` [1/m] must be strictly increasing and positive, ``P`` positive and
    finite, same length >= 2.  ``P`` is used as a *shape*: its absolute scale
    and its density convention cancel in the unit-rms normalisation, so an
    old ``pooled_psd.npz`` (f converted to 1/m, P left in the exporter's
    units) is acceptable here and nowhere that needs absolute power.

    Log-log interpolation onto the grid, spectral tilt
    ``C *= (f/f0)^(-2*gamma_tilt)`` with ``f0 = sqrt(f_min*f_max)``, then
    unit-rms normalisation.

    ``out_of_band``:
      * ``"zero"`` (default): no power outside [f_min, f_max];
      * ``"legacy_clip"``: the old behaviour -- queries clipped to the band
        edges, and the x-axis frequencies used for both directions (square
        cells only).
    """
    f = np.asarray(f, dtype=float)
    P = np.asarray(P, dtype=float)
    if f.ndim != 1 or P.ndim != 1 or f.size != P.size or f.size < 2:
        raise ValueError("radial PSD table: f and P must be 1-D, equal length >= 2")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(P))):
        raise ValueError("radial PSD table: non-finite values")
    if np.any(f <= 0.0) or np.any(np.diff(f) <= 0.0):
        raise ValueError("radial PSD table: f must be positive and strictly increasing")
    if np.any(P <= 0.0):
        raise ValueError("radial PSD table: P must be positive")
    if out_of_band == "legacy_clip":
        if nx != ny or abs(Lx - Ly) > 1e-12 * Lx:
            raise ValueError("out_of_band='legacy_clip' reproduces the old square-cell "
                             "code only; use 'zero' for rectangular cells")
        fx = np.fft.fftfreq(nx, d=Lx / nx)
        FR = np.hypot(fx[:, None], fx[None, :])
        Cbase = np.exp(np.interp(np.log(np.clip(FR, f.min(), f.max())),
                                 np.log(f), np.log(P)))
        inband = np.ones_like(FR, dtype=bool)
    elif out_of_band == "zero":
        fx = np.fft.fftfreq(nx, d=Lx / nx)
        fy = np.fft.fftfreq(ny, d=Ly / ny)
        FR = np.hypot(fy[:, None], fx[None, :])           # [j_y, i_x]
        inband = (FR >= f.min()) & (FR <= f.max())
        with np.errstate(divide="ignore"):
            Cbase = np.where(inband,
                             np.exp(np.interp(np.log(np.where(FR > 0, FR, f.min())),
                                              np.log(f), np.log(P))), 0.0)
    else:
        raise ValueError(f"unknown out_of_band policy {out_of_band!r}")
    f0 = np.sqrt(f.min() * f.max())
    FR_safe = np.where(FR > 0.0, FR, f0)
    amp = np.sqrt(Cbase * (FR_safe / f0) ** (-2.0 * float(gamma_tilt)))
    amp = np.where(inband, amp, 0.0)
    amp[0, 0] = 0.0
    return normalise_amplitude(amp)


# ---------------------------------------------------------------------------
# HPD models -> sorted, standardised height vectors
# ---------------------------------------------------------------------------
HPD_FAMILIES = ("gaussian", "gennorm", "skewnorm", "weibull", "johnsonsu")

#: (lo, hi) search bracket of each family's shape parameter (old code)
HPD_SHAPE_BRACKET: Dict[str, Tuple[float, float]] = {
    "gennorm": (0.4, 30.0),
    "skewnorm": (-30.0, 30.0),
    "weibull": (0.6, 25.0),
}


def hpd_sorted_sample(n: int, family: str = "gaussian",
                      shape: Any = None) -> np.ndarray:
    """Sorted, standardised (mean 0, rms 1) quantile vector of an HPD family
    at probabilities (i + 0.5)/n -- the old ``hpd_sorted_sample``.

    ``gaussian`` (no shape), ``gennorm`` (beta), ``skewnorm`` (a),
    ``weibull`` (k; k > 0 is negated so the long tail points down, as the old
    code did -- pass -k for the un-negated form), ``johnsonsu`` ((a, b)).
    """
    p = (np.arange(n) + 0.5) / n
    fam = family.lower()
    if fam == "gaussian":
        z = _q.norm_ppf(p)
    elif fam == "gennorm":
        z = _q.gennorm_ppf(p, float(shape))
    elif fam == "skewnorm":
        z = _q.skewnorm_ppf(p, float(shape))
    elif fam == "weibull":
        k = float(shape)
        sgn = -1.0 if k > 0 else 1.0
        z = sgn * _q.weibull_min_ppf(p, abs(k))
        z = np.sort(z)
    elif fam == "johnsonsu":
        a, b = shape
        z = _q.johnsonsu_ppf(p, float(a), float(b))
    else:
        raise ValueError(f"unknown HPD family {family!r}")
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"HPD family {family!r} shape={shape!r} produced "
                         "non-finite quantiles; shape parameter out of range.")
    z = z - z.mean()
    z = z / np.sqrt(np.mean(z ** 2))
    return np.sort(z)


def heights_from_samples(samples: np.ndarray, n: int) -> np.ndarray:
    """Sorted, standardised n-vector from measured height samples
    (deterministic linear quantile resampling at (i + 0.5)/n, as the old
    ``_heights_from_file`` / make_cells.py).  Non-finite samples are an error
    here (the old loader dropped them silently)."""
    ph = np.sort(np.asarray(samples, dtype=float).ravel())
    if not np.all(np.isfinite(ph)):
        raise ValueError("HPD samples contain non-finite values")
    if ph.size < 2 or ph[0] == ph[-1]:
        raise ValueError("HPD samples: need >= 2 distinct finite values")
    q = (np.arange(n) + 0.5) / n
    th = np.quantile(ph, q)
    th = th - th.mean()
    th /= np.sqrt(np.mean(th ** 2))
    return np.sort(th)


def sa_over_sq_of_hpd(family: str, shape: Any, n: int = 200_000) -> float:
    z = hpd_sorted_sample(n, family, shape)
    return float(np.mean(np.abs(z)))


def fit_hpd_shape(target_sa_over_sq: float, family: str = "gennorm",
                  n: int = 200_000, tol: float = 1e-5,
                  max_iter: int = 100) -> float:
    """Legacy Sa/Sq shape fit (bisection on a fixed n-point sample).

    Kept for parity with the old generator.  New preparation uses
    :func:`ehl.roughness.targets.fit_gennorm_ratio`, which fits on the exact
    final-grid quantile vector (Case API plan §5.2).
    """
    if family.lower() == "gaussian":
        got = math.sqrt(2.0 / math.pi)
        if abs(got - target_sa_over_sq) > 5e-3:
            raise ValueError(
                f"gaussian HPD fixes Sa/Sq = {got:.4f}; target "
                f"{target_sa_over_sq:.4f} is unreachable. Use family='gennorm'.")
        return float("nan")
    lo, hi = HPD_SHAPE_BRACKET[family.lower()]
    f_lo = sa_over_sq_of_hpd(family, lo, n) - target_sa_over_sq
    f_hi = sa_over_sq_of_hpd(family, hi, n) - target_sa_over_sq
    if f_lo * f_hi > 0:
        r_lo = f_lo + target_sa_over_sq
        r_hi = f_hi + target_sa_over_sq
        raise ValueError(
            f"target Sa/Sq = {target_sa_over_sq:.4f} is outside the range "
            f"attainable with family={family!r}: "
            f"[{min(r_lo, r_hi):.4f}, {max(r_lo, r_hi):.4f}]. "
            "Pick a different HPD family.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = sa_over_sq_of_hpd(family, mid, n) - target_sa_over_sq
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# IAAFT (paper Section 2)
# ---------------------------------------------------------------------------
@dataclass
class IAAFTResult:
    Zh: np.ndarray          # exact HPD, approximate PSD
    Zs: np.ndarray          # exact PSD, approximate HPD
    n_iter: int
    err_h: float            # legacy histogram L1 error of Zs (paper eq. 2)
    err_s: float            # legacy relative rms spectral error of Zh (eq. 3)
    history: list = field(default_factory=list)


def _apply_order(order: np.ndarray, target_sorted: np.ndarray,
                 shape: Tuple[int, int]) -> np.ndarray:
    out = np.empty(order.size, dtype=float)
    out[order] = target_sorted
    return out.reshape(shape)


def rank_remap(Z: np.ndarray, target_sorted: np.ndarray) -> np.ndarray:
    """Replace the values of Z by ``target_sorted`` in Z's rank order
    (stable arg-sort, as the old code)."""
    order = np.argsort(Z, axis=None, kind="stable")
    return _apply_order(order, target_sorted, Z.shape)


def iaaft(amp: np.ndarray, target_sorted: np.ndarray,
          rng: np.random.Generator, n_iter: int = 200,
          tol: float = 0.0) -> IAAFTResult:
    """Iterative amplitude-adjusted Fourier transform on a periodic 2-D grid
    (identical to the old ``rough_surface.iaaft``)."""
    ny, nx = amp.shape
    npts = ny * nx
    if target_sorted.size != npts:
        raise ValueError("target_sorted length must equal amp.size")
    F = np.fft.fft2(rng.standard_normal((ny, nx)))
    Zs = np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(F))))
    Zh = rank_remap(Zs, target_sorted)
    history = []
    prev_order = None
    used = n_iter
    for it in range(n_iter):
        F = np.fft.fft2(Zh)
        Zs = np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(F))))
        order = np.argsort(Zs, axis=None, kind="stable")
        Zh = _apply_order(order, target_sorted, (ny, nx))
        if tol > 0.0 and prev_order is not None:
            changed = np.mean(order != prev_order)
            history.append(changed)
            if changed < tol:
                used = it + 1
                break
        prev_order = order
    return IAAFTResult(Zh=Zh, Zs=Zs, n_iter=used,
                       err_h=err_h_legacy(Zs, target_sorted),
                       err_s=err_s_legacy(Zh, amp), history=history)


def err_h_legacy(Zs: np.ndarray, target_sorted: np.ndarray) -> float:
    """Old ``_err_h``: L1 distance between the histograms of Zs and of the
    target heights on Scott's-rule bins (probability-mass difference)."""
    zs = Zs.ravel()
    zh = target_sorted
    n = zh.size
    w = 3.49 * np.std(zh) * n ** (-1.0 / 3.0)
    lo, hi = zh.min(), zh.max()
    nb = max(int(np.ceil((hi - lo) / w)), 2)
    edges = np.linspace(min(lo, zs.min()), max(hi, zs.max()), nb + 1)
    nh, _ = np.histogram(zh, bins=edges, density=False)
    ns, _ = np.histogram(zs, bins=edges, density=False)
    return float(np.sum(np.abs(nh / n - ns / zs.size)))


def err_s_legacy(Zh: np.ndarray, amp: np.ndarray, band_rel: float = 1e-4) -> float:
    """Old ``_err_s``: unweighted relative rms error of |FFT(Zh)| against the
    target over modes with ``amp > band_rel * amp.max()``.  Not an
    acceptance metric (see module docstring)."""
    A = np.abs(np.fft.fft2(Zh))
    band = amp > band_rel * amp.max()
    if not np.any(band):
        return float("nan")
    rel = (A[band] - amp[band]) / amp[band]
    return float(np.sqrt(np.mean(rel ** 2)))


def spectral_energy_error(z: np.ndarray, amp: np.ndarray) -> float:
    """Energy-weighted spectral amplitude error of a unit-rms-normalised
    field against a unit-rms target amplitude:
    ``sqrt(sum((|Z| - amp)^2) / sum(amp^2))`` (DC excluded)."""
    zc = np.asarray(z, dtype=float) - np.mean(z)
    rms = float(np.sqrt(np.mean(zc ** 2)))
    if rms == 0.0:
        return float("nan")
    A = np.abs(np.fft.fft2(zc / rms))
    A[0, 0] = 0.0
    a = np.array(amp, dtype=float)
    a[0, 0] = 0.0
    return float(np.sqrt(np.sum((A - a) ** 2) / np.sum(a ** 2)))


# ---------------------------------------------------------------------------
# Sm fit (old fit_psd_for_sm, legacy maxima summit rule)
# ---------------------------------------------------------------------------
def _build_surface(n: int, amp: np.ndarray, target: np.ndarray, seed: int,
                   n_iter: int, take: str):
    res = iaaft(amp, target, np.random.default_rng(seed), n_iter=n_iter)
    z = res.Zh if take == "Zh" else res.Zs
    return z, res


def fit_psd_for_sm(target_Sm: float, n: int, Lx: float, Ly: float,
                   psd_model: str, psd_kwargs: Dict[str, Any],
                   hpd_family: str, hpd_shape: Any,
                   n_probe: int = 2, probe_seed: int = 12345,
                   n_iter: int = 40, take: str = "Zh",
                   peak_rule: str = "8-neighbour",
                   peak_threshold: Optional[float] = None,
                   rtol: float = 0.02, max_iter: int = 30,
                   verbose: bool = False,
                   target_sorted: Optional[np.ndarray] = None
                   ) -> Tuple[float, float]:
    """Bisect the PSD length-scale parameter so the measured Sm hits the
    target (old ``fit_psd_for_sm``).  Returns ``(param, Sm_at_fit)``.

    Sm here is the *legacy* spacing: 1/sqrt(density of local maxima of the
    array).  ``target_sorted`` (optional) fixes the actual final height vector
    instead of ``hpd_family``/``hpd_shape``; the old runtime resolver fitted
    with a Gaussian HPD even when the final HPD was measured (plan §2.3.8).
    """
    from .stats import summit_spacing

    q = wavenumber_grid(n, n, Ly, Lx)
    pname = PSD_FIT_PARAM[psd_model]
    psd_fn = PSD_MODELS[psd_model]
    target = (hpd_sorted_sample(n * n, hpd_family, hpd_shape)
              if target_sorted is None else np.asarray(target_sorted, dtype=float))

    def measure(pval: float) -> float:
        kw = dict(psd_kwargs)
        kw[pname] = pval
        amp = psd_fn(q, **kw)
        vals = []
        for k in range(n_probe):
            z, _ = _build_surface(n, amp, target, probe_seed + k, n_iter, take)
            Sm, _ = summit_spacing(z, Lx, Ly, rule=peak_rule,
                                   threshold=peak_threshold, kind="maxima")
            vals.append(Sm)
        return float(np.mean(vals))

    dx = Lx / n
    if pname == "corr_len":
        lo, hi = 1.0 * dx, 0.25 * min(Lx, Ly)
        guess = float(psd_kwargs.get("corr_len", 0.05 * min(Lx, Ly)))
        lo = max(lo, 1e-3 * guess)
        hi = min(hi, 1e3 * guess) if guess > 0 else hi
        lo, hi = min(lo, hi * 0.999), hi
    else:
        lo, hi = 0.05, 0.98
    f_lo = measure(lo) - target_Sm
    f_hi = measure(hi) - target_Sm
    if verbose:
        print(f"  bracket {pname}: [{lo:.4g}, {hi:.4g}] -> "
              f"Sm [{f_lo + target_Sm:.4g}, {f_hi + target_Sm:.4g}]")
    if f_lo * f_hi > 0:
        raise ValueError(
            f"target Sm = {target_Sm:.4g} m is unreachable with "
            f"psd_model={psd_model!r} on this grid: the attainable range over "
            f"{pname} in [{lo:.4g}, {hi:.4g}] is "
            f"[{min(f_lo, f_hi) + target_Sm:.4g}, "
            f"{max(f_lo, f_hi) + target_Sm:.4g}] m. "
            "Sm is bandwidth-limited -- change n, the domain size, or the PSD family.")
    best = 0.5 * (lo + hi)
    for it in range(max_iter):
        mid = math.sqrt(lo * hi) if pname == "corr_len" else 0.5 * (lo + hi)
        f_mid = measure(mid) - target_Sm
        best = mid
        if verbose:
            print(f"  it {it:2d}: {pname}={mid:.5g}  Sm={f_mid + target_Sm:.5g}")
        if abs(f_mid) <= rtol * target_Sm:
            return float(mid), float(f_mid + target_Sm)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(best), float(measure(best))


# ---------------------------------------------------------------------------
# Old top-level entry (kept for parity; new code prepares from a spec)
# ---------------------------------------------------------------------------
@dataclass
class SurfaceResult:
    z: np.ndarray
    Lx: float
    Ly: float
    stats: Dict[str, Any]
    amp: np.ndarray
    psd_model: str
    psd_kwargs: Dict[str, Any]
    hpd_family: str
    hpd_shape: Any
    iaaft: IAAFTResult
    seed: Optional[int]


def generate_rough_surface(
    n: int = 512, Lx: float = 1e-3, Ly: Optional[float] = None, *,
    Sq: float = 0.5e-6, Sa: Optional[float] = None, Sm: Optional[float] = None,
    seed: Optional[int] = None, hpd_family: str = "gennorm", hpd_shape: Any = None,
    psd_model: str = "gaussian_acf", psd_kwargs: Optional[Dict[str, Any]] = None,
    n_iter: int = 200, take: str = "Zh", peak_rule: str = "8-neighbour",
    peak_threshold: Optional[float] = None, fit_sm: bool = True,
    sm_rtol: float = 0.02, sm_probes: int = 2, verbose: bool = False,
) -> SurfaceResult:
    """The old ``generate_rough_surface`` (same arguments, same arithmetic).

    ``seed`` is required here: the old code drew an unrecorded random seed
    when it was None, which this package does not allow.
    """
    from .stats import legacy_surface_stats

    if seed is None:
        raise ValueError("seed is required (no unrecorded random seeds)")
    Ly = Lx if Ly is None else Ly
    psd_kwargs = dict(psd_kwargs or {})
    if psd_model not in PSD_MODELS:
        raise ValueError(f"unknown psd_model {psd_model!r}")
    if take not in ("Zh", "Zs"):
        raise ValueError("take must be 'Zh' or 'Zs'")
    if hpd_shape is None and Sa is not None and hpd_family.lower() != "gaussian":
        hpd_shape = fit_hpd_shape(Sa / Sq, family=hpd_family)
    elif hpd_shape is None and hpd_family.lower() == "gennorm":
        hpd_shape = 2.0
    if psd_model in ("gaussian_acf", "exponential_acf") and "corr_len" not in psd_kwargs:
        psd_kwargs["corr_len"] = (Sm if Sm else 0.05 * Lx)
    if Sm is not None and fit_sm:
        pname = PSD_FIT_PARAM[psd_model]
        pval, _ = fit_psd_for_sm(
            Sm, n, Lx, Ly, psd_model, psd_kwargs, hpd_family, hpd_shape,
            n_probe=sm_probes, n_iter=min(n_iter, 40), take=take,
            peak_rule=peak_rule, peak_threshold=peak_threshold,
            rtol=sm_rtol, verbose=verbose)
        psd_kwargs[pname] = pval
    q = wavenumber_grid(n, n, Ly, Lx)
    amp = PSD_MODELS[psd_model](q, **psd_kwargs)
    target = hpd_sorted_sample(n * n, hpd_family, hpd_shape)
    z, res = _build_surface(n, amp, target, int(seed), n_iter, take)
    z = z - z.mean()
    rms = float(np.sqrt(np.mean(z ** 2)))
    z = z * (Sq / rms)
    stats = legacy_surface_stats(z, Lx, Ly, peak_rule=peak_rule,
                                 peak_threshold=peak_threshold)
    return SurfaceResult(z=z, Lx=Lx, Ly=Ly, stats=stats, amp=amp,
                         psd_model=psd_model, psd_kwargs=psd_kwargs,
                         hpd_family=hpd_family, hpd_shape=hpd_shape,
                         iaaft=res, seed=int(seed))


# ---------------------------------------------------------------------------
# Filtered Gaussian generator (no IAAFT)
# ---------------------------------------------------------------------------
def gaussian_filtered_surface(N: int = 64, Lx: float = 5e-6,
                              corr_len: float = HDS_CORR_LEN, seed: int = 1000,
                              layout: str = "canonical") -> np.ndarray:
    """Periodic Gaussian random field with Gaussian ACF exp(-r^2/corr_len^2):
    white noise filtered by sqrt(PSD), zero mean, unit-free amplitude.

    Port of ``generated_ht_cell.gaussian_surface`` (the surface of the FFT
    contact prototype, ``study3.py``).  That function builds its array as
    ``z[i_x, j_y]`` (``indexing="ij"``) and its sampler reads index 0 as x;
    ``layout="canonical"`` returns the same physical surface as
    ``z[j_y, i_x]`` (the transpose), ``layout="legacy_xy"`` the old array.
    Square cells only, as the old function.
    """
    if corr_len <= 0.0:
        raise ValueError("corr_len must be > 0")
    rng = np.random.default_rng(int(seed))
    t = (np.arange(N) - N // 2) * (float(Lx) / N)
    X, Y = np.meshgrid(t, t, indexing="ij")
    acf = np.exp(-(X ** 2 + Y ** 2) / corr_len ** 2)
    psd = np.fft.fft2(np.fft.ifftshift(acf)).real
    amp = np.sqrt(np.maximum(psd, 0.0))
    z = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * amp).real
    z = z - z.mean()
    if layout == "legacy_xy":
        return z
    if layout == "canonical":
        return np.ascontiguousarray(z.T)
    raise ValueError(f"unknown layout {layout!r}")
