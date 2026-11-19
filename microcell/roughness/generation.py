"""Random rough-surface generation by IAAFT, for a Gaussian autocorrelation function (ACF) and a
Gaussian height probability distribution (HPD).

The method is that of Pérez-Ràfols & Almqvist, *Generating randomly rough surfaces with given
height probability distribution and power spectrum*, Tribology International 131 (2019) 591-604.
``Roughness.generated`` calls ``psd_gaussian_acf``, ``hpd_sorted_sample`` and ``iaaft`` in turn.

All arrays are ``z[j_y, i_x]`` on the half-open periodic grid, in SI units.  NumPy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from . import quantiles as _q

__all__ = [
    "HDS_CORR_LEN", "wavenumber_grid", "normalise_amplitude", "psd_gaussian_acf",
    "hpd_sorted_sample", "IAAFTResult", "iaaft", "rank_remap",
]

#: default correlation length of the Gaussian ACF [m]: fitted for a mean spacing Sm of 0.78 um on
#: the 5 um cell with 64 points per side
HDS_CORR_LEN = 3.2633555700856686e-07


# Wavenumber grid
def wavenumber_grid(ny: int, nx: int, Ly: float, Lx: float) -> np.ndarray:
    """Radial wavenumber |q| = 2*pi*sqrt(fx^2 + fy^2) [rad/m] of an ``ny x nx`` grid on an
    ``Ly x Lx`` cell [m].  Shape (ny, nx), FFT order (q[0, 0] = 0)."""
    fx = np.fft.fftfreq(nx, d=Lx / nx)      # cycles per metre
    fy = np.fft.fftfreq(ny, d=Ly / ny)
    QX, QY = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy, indexing="xy")
    return np.sqrt(QX ** 2 + QY ** 2)


# Power spectrum: amplitude array |Z(q)| of a surface with unit rms
def normalise_amplitude(A: np.ndarray) -> np.ndarray:
    """Amplitude array scaled so that the surface has zero mean and unit rms (Parseval, for the
    unnormalised FFT of NumPy)."""
    A = np.array(A, dtype=float)
    A[0, 0] = 0.0
    n = A.size
    s = np.sqrt(np.sum(A ** 2)) / n
    if s == 0.0:
        raise ValueError("PSD is identically zero over the grid band.")
    return A / s


def psd_gaussian_acf(q: np.ndarray, corr_len: float = 1e-4,
                     q1: Optional[float] = None) -> np.ndarray:
    """Normalised amplitude array of the Gaussian ACF R(r) = exp(-(r/corr_len)^2), whose power
    spectrum is C(q) ~ exp(-q^2 corr_len^2 / 4).

    q: radial wavenumbers [rad/m] from ``wavenumber_grid``; corr_len [m]; q1: optional cut-off
    [rad/m], above which the spectrum is zero.
    """
    C = np.exp(-(q ** 2) * (float(corr_len) ** 2) / 4.0)
    C[0, 0] = 0.0
    if q1 is not None:
        C[q > q1] = 0.0
    return normalise_amplitude(np.sqrt(C))


# Height distribution: sorted, standardised height vector
def hpd_sorted_sample(n: int, family: str = "gaussian",
                      shape: Any = None) -> np.ndarray:
    """Sorted quantiles of the height distribution at probabilities (i + 0.5)/n, i = 0 to n - 1,
    standardised to mean 0 and rms 1.  Only ``family="gaussian"`` is available; ``shape`` is
    not used by it."""
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


# IAAFT
@dataclass
class IAAFTResult:
    """Result of ``iaaft``: two surfaces of unit rms, ``[j_y, i_x]``."""
    Zh: np.ndarray          # exact height distribution, approximate spectrum
    Zs: np.ndarray          # exact spectrum, approximate height distribution
    n_iter: int             # iterations used
    history: list = field(default_factory=list)     # fraction of ranks changed per iteration (tol > 0)


def _apply_order(order: np.ndarray, target_sorted: np.ndarray,
                 shape: Tuple[int, int]) -> np.ndarray:
    """Array of ``shape`` with ``target_sorted`` placed in the rank order ``order``."""
    out = np.empty(order.size, dtype=float)
    out[order] = target_sorted
    return out.reshape(shape)


def rank_remap(Z: np.ndarray, target_sorted: np.ndarray) -> np.ndarray:
    """Z with its values replaced by ``target_sorted`` in the rank order of Z (stable sort)."""
    order = np.argsort(Z, axis=None, kind="stable")
    return _apply_order(order, target_sorted, Z.shape)


def iaaft(amp: np.ndarray, target_sorted: np.ndarray,
          rng: np.random.Generator, n_iter: int = 200,
          tol: float = 0.0) -> IAAFTResult:
    """Iterative amplitude-adjusted Fourier transform on a periodic 2-D grid.

    amp: amplitude array, shape (ny, nx), from ``psd_gaussian_acf``.
    target_sorted: sorted heights, length ny * nx, from ``hpd_sorted_sample``.
    rng: generator of the random starting phases.
    n_iter: iterations; with tol > 0 the loop stops when the fraction of ranks changed in an
        iteration falls below tol.
    """
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
                       history=history)
