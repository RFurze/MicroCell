"""Areal statistics of a gap-perturbation height array.

Canonical convention (R0 note §3): the array is a *gap perturbation*
``z[j_y, i_x]`` on the half-open periodic grid, ``h = H0 + HT*t + z``, so
``z < 0`` is an asperity (thinner film) and the film-reducing extreme is
``-min(z)``.

:func:`surface_stats` reports, for that convention:

=================== =========================================================
key                 meaning
=================== =========================================================
Sa, Sq              mean |z|, rms z (after centring once)
Ssk, Sku            skewness, kurtosis **of the gap perturbation**;
                    ``Ssk_protrusion = -Ssk`` is the skewness of the same
                    surface measured as protrusion into the film
Sp, Sv              ``max(z)`` and ``-min(z)``, both >= 0 (ISO-style magnitudes
                    of the array as given: Sp is the deepest *valley* of the
                    gap, Sv the film-reducing extreme)
Sz                  ``Sp + Sv``
z_min, z_max        signed extremes [m]
film_reducing_extreme  ``max(0, -z_min)``  (the cosine's ``Ah`` counterpart)
n_summits           film-reducing summits: strict local minima of ``z``
                    (asperities), periodic, 8-neighbour by default
Sm                  ``sqrt(Lx*Ly / n_summits)`` = 1/sqrt(summit density)
n_maxima, Sm_maxima local maxima of ``z`` -- what the old
                    ``rough_surface.surface_stats`` counted (valleys of the gap)
=================== =========================================================

:func:`legacy_surface_stats` reproduces the old ``rough_surface.surface_stats``
dictionary (signed ``Sv = min(z)``, maxima-based Sm) for parity checks only.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["find_summits", "summit_spacing", "profile_rsm", "spectral_moments",
           "surface_stats", "legacy_surface_stats", "height_moments"]


def find_summits(z: np.ndarray, rule: str = "8-neighbour",
                 threshold: Optional[float] = None,
                 kind: str = "minima") -> np.ndarray:
    """Boolean mask of strict periodic local extrema.

    ``kind="minima"``: film-reducing summits of a gap perturbation (default).
    ``kind="maxima"``: the old generator's summits (local maxima of the array).
    ``threshold`` (optional): minima must lie below ``threshold``, maxima
    above it.
    """
    if kind not in ("minima", "maxima"):
        raise ValueError(f"kind must be 'minima' or 'maxima', got {kind!r}")
    zz = z if kind == "maxima" else -np.asarray(z)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if rule == "8-neighbour":
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif rule != "4-neighbour":
        raise ValueError(f"unknown peak rule {rule!r}")
    mask = np.ones_like(zz, dtype=bool)
    for dy, dx in offs:
        mask &= zz > np.roll(np.roll(zz, dy, axis=0), dx, axis=1)
    if threshold is not None:
        mask &= (zz > threshold) if kind == "maxima" else (zz > -threshold)
    return mask


def summit_spacing(z: np.ndarray, Lx: float, Ly: float,
                   rule: str = "8-neighbour",
                   threshold: Optional[float] = None,
                   kind: str = "minima") -> Tuple[float, float]:
    """``(Sm, density)``: Sm = 1/sqrt(summits per unit area)."""
    n = int(find_summits(z, rule=rule, threshold=threshold, kind=kind).sum())
    if n == 0:
        return float("inf"), 0.0
    density = n / (Lx * Ly)
    return float(1.0 / math.sqrt(density)), float(density)


def profile_rsm(z: np.ndarray, Lx: float, Ly: float) -> float:
    """Mean profile element width (old ``profile_rsm``): mean distance
    between successive upward mean-line crossings over all rows and columns.
    Note it takes rows with ``Lx`` and columns with ``Ly`` as the old code."""
    widths = []
    for arr, L in ((z, Lx), (z.T, Ly)):
        a = arr - arr.mean()
        up = (a[:, :-1] <= 0) & (a[:, 1:] > 0)
        counts = up.sum(axis=1)
        counts = counts[counts > 0]
        if counts.size:
            widths.append(np.mean(L / counts))
    return float(np.mean(widths)) if widths else float("nan")


def spectral_moments(z: np.ndarray, Lx: float, Ly: float) -> Dict[str, float]:
    """Directional spectral moments of the array (variance, mean-square
    slopes and curvatures) from its DFT -- diagnostic, not a radius."""
    zc = np.asarray(z, dtype=float) - np.mean(z)
    ny, nx = zc.shape
    P = np.abs(np.fft.fft2(zc) / zc.size) ** 2
    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    QX, QY = np.meshgrid(qx, qy, indexing="xy")
    return {
        "m0": float(P.sum()),
        "m2_x": float((P * QX ** 2).sum()),
        "m2_y": float((P * QY ** 2).sum()),
        "m4_xx": float((P * QX ** 4).sum()),
        "m4_yy": float((P * QY ** 4).sum()),
    }


def height_moments(z: np.ndarray) -> Dict[str, float]:
    zc = np.asarray(z, dtype=float) - np.mean(z)
    Sq = float(np.sqrt(np.mean(zc ** 2)))
    Sa = float(np.mean(np.abs(zc)))
    Ssk = float(np.mean(zc ** 3) / Sq ** 3) if Sq > 0 else float("nan")
    Sku = float(np.mean(zc ** 4) / Sq ** 4) if Sq > 0 else float("nan")
    return {"Sa": Sa, "Sq": Sq, "Ssk": Ssk, "Sku": Sku}


def surface_stats(z: np.ndarray, Lx: float, Ly: float,
                  rule: str = "8-neighbour",
                  threshold: Optional[float] = None) -> Dict[str, float]:
    """Areal parameters of a gap-perturbation array (see module docstring)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("surface_stats expects a 2-D array z[j_y, i_x]")
    mean_removed = float(np.mean(z))
    zc = z - mean_removed
    out: Dict[str, float] = {"convention": "gap_perturbation"}
    out.update(height_moments(zc))
    zmin, zmax = float(zc.min()), float(zc.max())
    n_min = int(find_summits(zc, rule=rule, threshold=threshold, kind="minima").sum())
    n_max = int(find_summits(zc, rule=rule, threshold=threshold, kind="maxima").sum())
    A = float(Lx) * float(Ly)
    out.update({
        "Sa_over_Sq": out["Sa"] / out["Sq"] if out["Sq"] > 0 else float("nan"),
        "Ssk_protrusion": -out["Ssk"],
        "Sp": zmax, "Sv": -zmin, "Sz": zmax - zmin,
        "z_min": zmin, "z_max": zmax,
        "film_reducing_extreme": max(0.0, -zmin),
        "n_summits": n_min,
        "Sm": math.sqrt(A / n_min) if n_min else float("inf"),
        "summit_density": n_min / A,
        "n_maxima": n_max,
        "Sm_maxima": math.sqrt(A / n_max) if n_max else float("inf"),
        "summit_rule": rule,
        "nx": int(z.shape[1]), "ny": int(z.shape[0]),
        "Lx": float(Lx), "Ly": float(Ly),
        "mean_removed": mean_removed,
    })
    return out


def legacy_surface_stats(z: np.ndarray, Lx: float, Ly: float,
                         peak_rule: str = "8-neighbour",
                         peak_threshold: Optional[float] = None) -> Dict[str, float]:
    """The old ``rough_surface.surface_stats`` dictionary, unchanged
    (signed ``Sv = min(z)``, summits = local maxima).  Parity use only."""
    zc = z - z.mean()
    Sq = float(np.sqrt(np.mean(zc ** 2)))
    Sa = float(np.mean(np.abs(zc)))
    Ssk = float(np.mean(zc ** 3) / Sq ** 3) if Sq > 0 else float("nan")
    Sku = float(np.mean(zc ** 4) / Sq ** 4) if Sq > 0 else float("nan")
    Sm, dens = summit_spacing(zc, Lx, Ly, rule=peak_rule,
                              threshold=peak_threshold, kind="maxima")
    return {
        "Sq": Sq, "Sa": Sa,
        "Sa_over_Sq": Sa / Sq if Sq > 0 else float("nan"),
        "Ssk": Ssk, "Sku": Sku,
        "Sp": float(zc.max()), "Sv": float(zc.min()),
        "Sz": float(zc.max() - zc.min()),
        "Sm_summit": Sm, "summit_density": dens,
        "n_summits": int(dens * Lx * Ly + 0.5),
        "RSm_profile": profile_rsm(zc, Lx, Ly),
    }
