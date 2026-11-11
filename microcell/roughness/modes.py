"""Cosine mode-list surfaces: idealised profiles and the seeded band-limited
generator of the roughness-location/phase study.

Port of ``tests/micro_roughness_study/surfaces.py`` (MultiscaleMixedBallinSocket,
sha256 ``73ce9b8e…acd032``), so the study harness can import its surfaces from
this package instead of keeping its own generator (Case API plan §15.3).  The
arithmetic is unchanged: same wavevector enumeration order, same
``SeedSequence(master, spawn_key=(pair_index, member))`` draws, same
``2|c| cos(2 pi (m X + n Y) + arg c)`` modes, same analytic RMS scaling.

A mode list ``[(m, n, a, phi), ...]`` is the surface

    r(x, y) = sum a * cos(2*pi*(m*x/Lx + n*y/Ly) + phi)

with integer wavevectors on the half-plane (n > 0, or n == 0 and m > 0), so
the mean is exactly zero and ``rms = sqrt(sum a^2 / 2)``.  It is evaluated
through the canonical evaluator with
:meth:`ehl.roughness.evaluator.FourierSurface.from_modes` (exact; no grid).

Only NumPy is imported.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["Mode", "IDEAL_PROFILES", "MEMBER_CODE", "ideal_modes", "rms_analytic",
           "scale_to_rms", "translate_modes", "evaluate_modes", "band_wavevectors",
           "psd_shape", "surface_seed", "band_limited_modes", "LEGACY_SOURCE"]

Mode = Tuple[int, int, float, float]          # (m, n, amplitude [m], phase [rad])

LEGACY_SOURCE = {
    "path": "MultiscaleMixedBallinSocket/tests/micro_roughness_study/surfaces.py",
    "sha256": "73ce9b8ed77ec444f98a102e11181559c80097b180c1ebafb120579b52acd032",
}

IDEAL_PROFILES = ("I0", "I1", "I2", "I3", "I4")
MEMBER_CODE = {"A": 0, "B": 1, "": 0}


def ideal_modes(profile: str) -> List[Mode]:
    """Raw (unit-amplitude) modes of the study's idealised profiles I0-I4."""
    if profile == "I0":
        return []
    if profile == "I1":
        return [(1, 0, 1.0, 0.0), (0, 1, 1.0, 0.0)]
    if profile == "I2":
        return [(0, 1, 1.0, 0.0)]
    if profile == "I3":
        return [(1, 0, 1.0, 0.0)]
    if profile == "I4":
        return [(1, 0, 1.0, 0.0), (0, 1, 1.0, 0.0),
                (2, 0, 0.25, math.pi / 3.0), (0, 2, 0.25, -math.pi / 5.0)]
    raise ValueError(profile)


def _check_half_plane(modes: Sequence[Mode]) -> None:
    seen = set()
    for m, n, _a, _p in modes:
        if not (n > 0 or (n == 0 and m > 0)):
            raise ValueError(f"mode ({m},{n}) is not on the half-plane")
        if (m, n) in seen:
            raise ValueError(f"duplicate wavevector ({m},{n})")
        seen.add((m, n))


def rms_analytic(modes: Sequence[Mode]) -> float:
    _check_half_plane(modes)
    return math.sqrt(sum(a * a for _m, _n, a, _p in modes) / 2.0) if modes else 0.0


def scale_to_rms(modes: Sequence[Mode], sigma: float) -> List[Mode]:
    if not modes:
        return []
    r = rms_analytic(modes)
    if r <= 0.0:
        raise ValueError("cannot scale a zero surface to a nonzero RMS")
    f = sigma / r
    return [(m, n, a * f, p) for m, n, a, p in modes]


def translate_modes(modes: Sequence[Mode], xi: float, eta: float) -> List[Mode]:
    """Modes of r(X - xi, Y - eta) (normalised displacement)."""
    return [(m, n, a, p - 2.0 * np.pi * (m * xi + n * eta)) for m, n, a, p in modes]


def evaluate_modes(modes: Sequence[Mode], X: np.ndarray, Y: np.ndarray,
                   xi: float = 0.0, eta: float = 0.0) -> np.ndarray:
    """r(X - xi, Y - eta) on normalised coordinates X = x/Lx, Y = y/Ly."""
    out = np.zeros(np.broadcast(X, Y).shape, dtype=float)
    for m, n, a, p in modes:
        out += a * np.cos(2.0 * np.pi * (m * (X - xi) + n * (Y - eta)) + p)
    return out


def band_wavevectors(Lx: float, Ly: float, lambda_min: float,
                     lambda_max: float) -> List[Tuple[int, int]]:
    """Half-plane integer wavevectors with 2pi/lambda_max <= |k| <= 2pi/lambda_min."""
    kmin, kmax = 2 * np.pi / lambda_max, 2 * np.pi / lambda_min
    m_max = int(math.ceil(kmax * Lx / (2 * np.pi))) + 1
    n_max = int(math.ceil(kmax * Ly / (2 * np.pi))) + 1
    out = []
    for n in range(0, n_max + 1):
        for m in range(-m_max, m_max + 1):
            if not (n > 0 or (n == 0 and m > 0)):
                continue
            k = 2 * np.pi * math.hypot(m / Lx, n / Ly)
            if kmin - 1e-12 <= k <= kmax + 1e-12:
                out.append((m, n))
    return out


def psd_shape(m: int, n: int, Lx: float, Ly: float, ell_x: float, ell_y: float) -> float:
    kx, ky = 2 * np.pi * m / Lx, 2 * np.pi * n / Ly
    return (1.0 + (ell_x * kx) ** 2 + (ell_y * ky) ** 2) ** -2


def surface_seed(master_seed: int, pair_index: int, member: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(master_seed),
                                  spawn_key=(int(pair_index), MEMBER_CODE[member]))


def band_limited_modes(Lx: float, Ly: float, lambda_min: float, lambda_max: float,
                       ell_x: float, ell_y: float, master_seed: int,
                       pair_index: int, member: str) -> List[Mode]:
    """Seeded raw modes (before RMS scaling): the study's ``fourier_modes``.

    PSD shape ``[1 + (ell_x kx)^2 + (ell_y ky)^2]^-2`` inside the band; each
    mode gets ``c = (g1 + i g2) sqrt(S)`` with independent standard normals
    drawn in wavevector order, and becomes ``2|c| cos(... + arg c)``.
    """
    wv = band_wavevectors(Lx, Ly, lambda_min, lambda_max)
    if not wv:
        raise ValueError("no wavevector inside the band")
    rng = np.random.default_rng(surface_seed(master_seed, pair_index, member))
    g = rng.standard_normal((len(wv), 2))
    modes: List[Mode] = []
    for (m, n), (g1, g2) in zip(wv, g):
        s = psd_shape(m, n, Lx, Ly, ell_x, ell_y)
        c = complex(g1, g2) * math.sqrt(s)
        modes.append((m, n, 2.0 * abs(c), float(np.angle(c))))
    return modes
