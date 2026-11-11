"""Random rough-surface generation: the IAAFT port, Gaussian case only.

This is a port of the numerical core of the old tree's generator
(EHLBase-DispatchHeavy ``microscale/src/functions/rough_surface.py``,
sha256 ``3d9b481d…0590``, 15 July 2026), which implements

    Pérez-Ràfols & Almqvist, *Generating randomly rough surfaces with given
    height probability distribution and power spectrum*, Tribology
    International 131 (2019) 591-604.

What is the same as the old code: the Gaussian-ACF PSD model, the Parseval
normalisation, the IAAFT loop (seed draw, stable arg-sort rank remap,
iteration order) and the Gaussian height distribution, evaluated by
:mod:`microcell.roughness.quantiles` (NumPy only) instead of ``scipy.stats``.

[MicroCell WP3] Kept: what ``Roughness.generated`` calls (the Gaussian-ACF
spectrum, the Gaussian HPD, IAAFT and its two legacy error metrics).  Removed:
the other PSD models and radial-PSD tables, the other HPD families and their
shape fit, the Sm fit, ``generate_rough_surface``, the filtered Gaussian
generator, ``spectral_energy_error`` and the revision / provenance constants;
they are in the parent (``ehl/roughness/generation.py``).

All arrays are ``z[j_y, i_x]`` on the half-open periodic grid, SI units.
Only NumPy is imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from . import quantiles as _q

__all__ = [
    "HDS_CORR_LEN", "wavenumber_grid", "normalise_amplitude", "psd_gaussian_acf",
    "hpd_sorted_sample", "IAAFTResult", "iaaft", "rank_remap", "err_h_legacy",
    "err_s_legacy",
]

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


def psd_gaussian_acf(q: np.ndarray, corr_len: float = 1e-4,
                     q1: Optional[float] = None) -> np.ndarray:
    """PSD of the Gaussian ACF R(r) = exp(-(r/corr_len)^2):
    C(q) ~ exp(-q^2 corr_len^2 / 4)."""
    C = np.exp(-(q ** 2) * (float(corr_len) ** 2) / 4.0)
    C[0, 0] = 0.0
    if q1 is not None:
        C[q > q1] = 0.0
    return normalise_amplitude(np.sqrt(C))


# ---------------------------------------------------------------------------
# HPD models -> sorted, standardised height vectors
# ---------------------------------------------------------------------------
def hpd_sorted_sample(n: int, family: str = "gaussian",
                      shape: Any = None) -> np.ndarray:
    """Sorted, standardised (mean 0, rms 1) quantile vector of an HPD family
    at probabilities (i + 0.5)/n -- the old ``hpd_sorted_sample``.

    Only ``gaussian`` (no shape) is in this repo; the parent's other families
    (``gennorm``, ``skewnorm``, ``weibull``, ``johnsonsu``) were removed in
    MicroCell WP3.
    """
    p = (np.arange(n) + 0.5) / n
    fam = family.lower()
    if fam == "gaussian":
        z = _q.norm_ppf(p)
    else:
        raise ValueError(f"unknown HPD family {family!r}")
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"HPD family {family!r} shape={shape!r} produced "
                         "non-finite quantiles; shape parameter out of range.")
    z = z - z.mean()
    z = z / np.sqrt(np.mean(z ** 2))
    return np.sort(z)


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
