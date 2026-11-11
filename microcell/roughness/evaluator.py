"""The canonical surface evaluator: band-limited Fourier reconstruction.

Decision D4 (16 October 2026, FFT contact plan): one evaluator for every
consumer -- FE nodal refill, the contact solve, the predictor and reports.
Bilinear refill survives only as the labelled legacy path
(:func:`bilinear_nodal`).

Reconstruction contract (R0 note §4)
------------------------------------
A source array ``z[j, i]`` on the half-open periodic grid
``x_i = i*Lx/Nx``, ``y_j = j*Ly/Ny`` defines the real trigonometric polynomial

    z(x, y) = sum_{(kx, ky) in K} c(kx, ky) exp(2*pi*i*(kx*x/Lx + ky*y/Ly))

with ``c`` from ``fft2(z)/(Nx*Ny)`` and integer wavenumbers
``|kx| <= Nx/2``, ``|ky| <= Ny/2``.  On an even axis the Nyquist coefficient
is split equally between ``+N/2`` and ``-N/2`` (``nyquist="split"``): the
symmetric, minimum-energy choice that keeps the polynomial real for every
shift and derivative and reproduces the source exactly at the source nodes
(adding ``sin(pi*N*x/L)`` would change no nodal value, so the split is a
convention and part of the identity).
``nyquist="drop"`` removes those modes instead (then the source is not
reproduced unless it has no Nyquist content).  The retained set ``K`` is
the explicit :class:`Bandwidth`: all source modes by default, or a
rectangular (``kx_max``, ``ky_max``) and/or radial (``q_max`` [rad/m]) cut.

Properties (tested in ``tests/roughness``):

* nodal values on the source grid at zero shift equal the source to round-off;
* a shift by a whole period returns the source;
* a shift by whole grid steps equals ``np.roll`` of the source;
* nodal values on *any* ``mx x my`` grid, for any signed displacement
  ``(sx, sy)`` [m], are exact samples of the polynomial (spectral fold, no
  aliasing error), at FFT cost;
* rectangular cells (``Lx != Ly``, ``Nx != Ny``) are supported.

Translation convention: ``nodal(..., shift=(sx, sy))`` returns
``z(x - sx, y - sy)``: the surface displaced by ``+s``.

The continuous polynomial's mean square is the nodal (Parseval) mean square
minus half the energy of single-axis Nyquist modes and three quarters of the
(N/2, N/2) corner mode; :meth:`FourierSurface.variances` computes it exactly
and reports both.  A Fourier interpolant can overshoot the source extrema between
nodes; :meth:`FourierSurface.extrema` evaluates it on an oversampled grid.

Only NumPy is imported.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["EVALUATOR_REVISION", "Bandwidth", "FourierSurface", "bilinear_nodal",
           "nodal_grid"]

#: Version tag of the reconstruction; part of cache and artefact identity.
EVALUATOR_REVISION = "ehl.roughness.fourier/1"


@dataclass(frozen=True)
class Bandwidth:
    """Which Fourier modes the reconstruction retains.

    ``kx_max`` / ``ky_max``: keep integer wavenumbers ``|k| <= k_max`` along
    each axis (None: all source modes).  ``q_max``: additionally keep only
    ``2*pi*sqrt((kx/Lx)^2 + (ky/Ly)^2) <= q_max`` [rad/m] (None: no radial
    cut).  ``nyquist``: ``"split"`` (default) or ``"drop"`` for the
    ``|k| = N/2`` modes of an even source axis.
    """
    kx_max: Optional[int] = None
    ky_max: Optional[int] = None
    q_max: Optional[float] = None
    nyquist: str = "split"

    def __post_init__(self):
        if self.nyquist not in ("split", "drop"):
            raise ValueError(f"nyquist must be 'split' or 'drop', got {self.nyquist!r}")
        for name in ("kx_max", "ky_max"):
            v = getattr(self, name)
            if v is not None and (int(v) != v or v < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.q_max is not None and not self.q_max > 0.0:
            raise ValueError("q_max must be > 0 or None")

    def to_dict(self) -> dict:
        return asdict(self)


def _extended_axis(n: int) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Integer wavenumbers of the split representation of an n-point axis.

    Returns ``(k_ext, src_index, nyq_pos)``: ``k_ext[e]`` is the wavenumber of
    extended entry ``e``, ``src_index[e]`` the FFT index it takes its
    coefficient from, and ``nyq_pos`` the extended positions of +-N/2 (or
    None for odd n)."""
    k = np.rint(np.fft.fftfreq(n) * n).astype(int)          # FFT order
    src = np.arange(n)
    if n % 2 == 0:
        k_ext = np.concatenate([k, [n // 2]])                # -N/2 at n//2, +N/2 appended
        src_ext = np.concatenate([src, [n // 2]])
        return k_ext, src_ext, (n // 2, n)
    return k, src, None


def nodal_grid(nx: int, ny: int, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    """(X, Y) [m] of the half-open periodic grid, shape (ny, nx): x_i = i*Lx/nx."""
    x = np.arange(nx) * (Lx / nx)
    y = np.arange(ny) * (Ly / ny)
    return np.meshgrid(x, y, indexing="xy")


class FourierSurface:
    """Band-limited trigonometric reconstruction of a periodic height field.

    Construct from a source array ``z[j_y, i_x]`` (metres, any mean -- it is
    centred, and the removed mean is kept in ``mean_removed``) on an
    ``Lx x Ly`` cell, or from a cosine mode list with :meth:`from_modes`.
    Instances are immutable.
    """

    def __init__(self, z: np.ndarray, Lx: float, Ly: float,
                 bandwidth: Bandwidth = Bandwidth()):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or min(z.shape) < 1:
            raise ValueError("FourierSurface: z must be a 2-D array z[j_y, i_x]")
        if not np.all(np.isfinite(z)):
            raise ValueError("FourierSurface: z contains non-finite values")
        if not (Lx > 0.0 and Ly > 0.0):
            raise ValueError("FourierSurface: Lx and Ly must be > 0")
        ny, nx = z.shape
        mean = float(z.mean())
        C = np.fft.fft2(z - mean) / (nx * ny)
        C[0, 0] = 0.0
        self._init_from_fft(C, float(Lx), float(Ly), bandwidth)
        self.mean_removed = mean
        zc = np.ascontiguousarray(z - mean)
        zc.setflags(write=False)
        self._source = zc

    # -- construction helpers ------------------------------------------------
    def _init_from_fft(self, C: np.ndarray, Lx: float, Ly: float,
                       bandwidth: Bandwidth) -> None:
        ny, nx = C.shape
        kx, sx, nyqx = _extended_axis(nx)
        ky, sy, nyqy = _extended_axis(ny)
        Ce = C[np.ix_(sy, sx)].astype(complex)
        if nyqx is not None:
            Ce[:, list(nyqx)] *= 0.5
        if nyqy is not None:
            Ce[list(nyqy), :] *= 0.5
        keep = np.ones(Ce.shape, dtype=bool)
        if bandwidth.kx_max is not None:
            keep &= (np.abs(kx) <= bandwidth.kx_max)[None, :]
        if bandwidth.ky_max is not None:
            keep &= (np.abs(ky) <= bandwidth.ky_max)[:, None]
        if bandwidth.q_max is not None:
            q = 2.0 * np.pi * np.hypot(ky[:, None] / Ly, kx[None, :] / Lx)
            keep &= q <= bandwidth.q_max * (1.0 + 1e-12)
        if bandwidth.nyquist == "drop":
            if nyqx is not None:
                keep[:, list(nyqx)] = False
            if nyqy is not None:
                keep[list(nyqy), :] = False
        self._energy_all = float(np.sum(np.abs(Ce) ** 2))
        Ce = np.where(keep, Ce, 0.0)
        # drop all-zero rows/columns to keep the fold small
        cols = np.nonzero(np.any(Ce != 0.0, axis=0))[0]
        rows = np.nonzero(np.any(Ce != 0.0, axis=1))[0]
        if cols.size == 0:
            cols = np.array([0])
            rows = np.array([0])
        self._C = np.ascontiguousarray(Ce[np.ix_(rows, cols)])
        self._kx = kx[cols].copy()
        self._ky = ky[rows].copy()
        self._C.setflags(write=False)
        self._kx.setflags(write=False)
        self._ky.setflags(write=False)
        self.Lx, self.Ly = Lx, Ly
        self.source_shape = (ny, nx)
        self.bandwidth = bandwidth
        nyq_energy = 0.0
        Cabs2 = np.abs(C) ** 2
        if nx % 2 == 0:
            nyq_energy += float(Cabs2[:, nx // 2].sum())
        if ny % 2 == 0:
            nyq_energy += float(Cabs2[ny // 2, :].sum())
            if nx % 2 == 0:
                nyq_energy -= float(Cabs2[ny // 2, nx // 2])
        self._nodal_energy = float(Cabs2.sum())
        self._nyquist_energy = nyq_energy

    @classmethod
    def from_modes(cls, modes: Iterable[Tuple[int, int, float, float]],
                   Lx: float, Ly: float) -> "FourierSurface":
        """Exact surface ``sum a*cos(2*pi*(m*x/Lx + n*y/Ly) + phi)`` from
        ``(m, n, a, phi)`` modes on the half-plane (n > 0, or n == 0 and m > 0;
        no repeats), as ``tests/micro_roughness_study/surfaces.py`` uses.
        Built on an odd grid, so no Nyquist mode is involved."""
        modes = [(int(m), int(n), float(a), float(p)) for m, n, a, p in modes]
        seen = set()
        for m, n, _a, _p in modes:
            if not (n > 0 or (n == 0 and m > 0)):
                raise ValueError(f"mode ({m},{n}) is not on the half-plane")
            if (m, n) in seen:
                raise ValueError(f"duplicate wavevector ({m},{n})")
            seen.add((m, n))
        mmax = max([abs(m) for m, *_ in modes], default=0)
        nmax = max([abs(n) for _m, n, *_ in modes], default=0)
        nx, ny = 2 * mmax + 1, 2 * nmax + 1
        C = np.zeros((ny, nx), dtype=complex)
        for m, n, a, p in modes:
            c = 0.5 * a * complex(math.cos(p), math.sin(p))
            C[n % ny, m % nx] += c
            C[(-n) % ny, (-m) % nx] += c.conjugate()
        self = cls.__new__(cls)
        self._init_from_fft(C, float(Lx), float(Ly), Bandwidth())
        self.mean_removed = 0.0
        src = np.real(np.fft.ifft2(C)) * (nx * ny)
        src.setflags(write=False)
        self._source = src
        return self

    # -- information ------------------------------------------------------------
    @property
    def source(self) -> np.ndarray:
        """Centred source array (read-only)."""
        return self._source

    @property
    def kx(self) -> np.ndarray:
        return self._kx

    @property
    def ky(self) -> np.ndarray:
        return self._ky

    def variances(self) -> dict:
        """``nodal``: Parseval mean square of the source array;
        ``continuous``: mean square of the retained trigonometric polynomial
        over the cell; ``nyquist_fraction``: share of the nodal variance in
        the Nyquist modes; ``retained_fraction``: continuous / full split
        polynomial."""
        cont = float(np.sum(np.abs(self._C) ** 2))
        nod = self._nodal_energy
        return {
            "nodal": nod,
            "continuous": cont,
            "nyquist_fraction": (self._nyquist_energy / nod) if nod > 0 else 0.0,
            "retained_fraction": (cont / self._energy_all) if self._energy_all > 0 else 1.0,
        }

    def identity(self) -> dict:
        """Reconstruction descriptor for cache/artefact identity."""
        src = np.ascontiguousarray(self._source, dtype="<f8")
        h = hashlib.sha256(f"float64-le:{src.shape[0]}x{src.shape[1]}:".encode()
                           + src.tobytes()).hexdigest()
        return {
            "evaluator": EVALUATOR_REVISION,
            "kind": "fourier_trig",
            "bandwidth": self.bandwidth.to_dict(),
            "source_shape": list(self.source_shape),
            "Lx": self.Lx, "Ly": self.Ly,
            "source_array_sha256": h,
        }

    # -- evaluation -------------------------------------------------------------
    def _shifted(self, shift: Sequence[float], deriv: Tuple[int, int]) -> np.ndarray:
        sx, sy = float(shift[0]), float(shift[1])
        px = np.exp(-2j * np.pi * self._kx * (sx / self.Lx))
        py = np.exp(-2j * np.pi * self._ky * (sy / self.Ly))
        C = self._C * py[:, None] * px[None, :]
        ax, ay = int(deriv[0]), int(deriv[1])
        if ax:
            C = C * ((2j * np.pi * self._kx / self.Lx) ** ax)[None, :]
        if ay:
            C = C * ((2j * np.pi * self._ky / self.Ly) ** ay)[:, None]
        return C

    def _fold(self, C: np.ndarray, mx: int, my: int) -> np.ndarray:
        Fx = np.zeros((self._kx.size, mx))
        Fx[np.arange(self._kx.size), self._kx % mx] = 1.0
        Fy = np.zeros((self._ky.size, my))
        Fy[np.arange(self._ky.size), self._ky % my] = 1.0
        return Fy.T @ (C @ Fx)

    def nodal(self, mx: int, my: Optional[int] = None,
              shift: Sequence[float] = (0.0, 0.0),
              deriv: Tuple[int, int] = (0, 0),
              return_imag: bool = False):
        """Values ``z(x_i - sx, y_j - sy)`` [m] on the ``my x mx`` half-open
        grid ``x_i = i*Lx/mx``, ``y_j = j*Ly/my``; array ``[j_y, i_x]``.

        ``deriv=(ax, ay)`` returns the ``d^ax/dx^ax d^ay/dy^ay`` derivative
        instead.  ``return_imag`` also returns the max |imaginary| residual of
        the inverse FFT (round-off; a diagnostic of conjugate symmetry).
        """
        mx = int(mx)
        my = mx if my is None else int(my)
        if mx < 1 or my < 1:
            raise ValueError("grid sizes must be >= 1")
        T = self._fold(self._shifted(shift, deriv), mx, my)
        v = np.fft.ifft2(T) * (mx * my)
        out = np.ascontiguousarray(v.real)
        if return_imag:
            return out, float(np.max(np.abs(v.imag)))
        return out

    def at_points(self, x, y, shift: Sequence[float] = (0.0, 0.0),
                  deriv: Tuple[int, int] = (0, 0), chunk: int = 4096) -> np.ndarray:
        """Direct evaluation at arbitrary points (x, y) [m] (any shape)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        xb = np.broadcast_to(x, shape).ravel()
        yb = np.broadcast_to(y, shape).ravel()
        C = self._shifted(shift, deriv)
        out = np.empty(xb.size)
        for s in range(0, xb.size, chunk):
            ex = np.exp(2j * np.pi * np.outer(xb[s:s + chunk] / self.Lx, self._kx))
            ey = np.exp(2j * np.pi * np.outer(yb[s:s + chunk] / self.Ly, self._ky))
            out[s:s + chunk] = np.real(np.einsum("pk,kj,pj->p", ey, C, ex))
        return out.reshape(shape)

    def extrema(self, oversample: int = 4,
                shift: Sequence[float] = (0.0, 0.0)) -> Tuple[float, float]:
        """(min, max) of the reconstruction on an ``oversample``-times finer
        grid than the source -- an estimate of the continuous extrema (the
        exact extrema lie between samples)."""
        ny, nx = self.source_shape
        v = self.nodal(oversample * nx, oversample * ny, shift=shift)
        return float(v.min()), float(v.max())


def bilinear_nodal(z: np.ndarray, Lx: float, Ly: float, mx: int,
                   my: Optional[int] = None,
                   shift: Sequence[float] = (0.0, 0.0),
                   layout: str = "canonical") -> np.ndarray:
    """LEGACY periodic-bilinear refill (old ``generated_surface_ht`` refill
    and ``generated_ht_cell.sample_periodic``), on the half-open
    ``my x mx`` grid; returns ``[j_y, i_x]``.

    Kept for reproducing historical runs only: it turns every source extremum
    into a pyramid tip and makes contact loads depend on node alignment
    (FFT contact plan §4.3).  ``layout="legacy_xy"`` reads the source as
    ``z[i_x, j_y]`` like ``generated_ht_cell``.
    """
    z = np.asarray(z, dtype=float)
    if layout == "legacy_xy":
        z = z.T
    elif layout != "canonical":
        raise ValueError(f"unknown layout {layout!r}")
    my = mx if my is None else int(my)
    ny, nx = z.shape
    u = np.arange(mx) / float(mx)
    v = np.arange(my) / float(my)
    U, V = np.meshgrid(u, v, indexing="xy")
    dfx = float(shift[0]) / Lx
    dfy = float(shift[1]) / Ly
    fx = ((U - dfx) % 1.0) * nx
    fy = ((V - dfy) % 1.0) * ny
    i0 = np.floor(fx).astype(int) % nx
    j0 = np.floor(fy).astype(int) % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    return ((1 - tx) * (1 - ty) * z[j0, i0] + tx * (1 - ty) * z[j0, i1]
            + (1 - tx) * ty * z[j1, i0] + tx * ty * z[j1, i1])
