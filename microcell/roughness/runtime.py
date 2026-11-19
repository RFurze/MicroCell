"""The surfaces of a cell, as the solver reads them (NumPy only).

``SurfaceField`` evaluates the summed roughness of the stationary and translating surfaces, one
``FourierSurface`` each, at grid nodes or at points.  Its attributes are set by
``microcell.roughness.frontend.ArrayField``.

``SurfaceMotion`` holds the displacement of the translating surface,
``s(t) = s0 + ratio*(2*U*t + UT*t**2)`` per axis [m], evaluated at the requested time.  The film
callback of ``microcell/film.py`` sets it from the solver.

``SurfaceContactHeight`` is the undeformed film that the contact solve reads, evaluated from
the same field.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pair import pattern_displacement

__all__ = ["SurfaceField", "SurfaceMotion", "SurfaceContactHeight"]

Shift = Tuple[float, float]
_ZERO: Shift = (0.0, 0.0)

_CELL_RTOL = 1e-9       # relative tolerance of the cell-size check


def _close(a: float, b: float) -> bool:
    """True when a equals b to the relative tolerance ``_CELL_RTOL``."""
    return abs(float(a) - float(b)) <= _CELL_RTOL * abs(float(b))


class SurfaceField:
    """The surfaces of one cell and their summed roughness.

    The attributes are set by the subclass ``microcell.roughness.frontend.ArrayField``.
    ``nodal(nx, ny, s)`` returns the summed gap perturbation ``z[j_y, i_x]`` [m] on a grid of
    the cell and ``at_unit(x, y, s)`` the same at unit-cell coordinates.  The stationary
    surface is at its offset and the translating surface at its offset plus the displacement
    ``s = (sx, sy)`` [m].
    """

    def _shift(self, role: str, s: Shift) -> Shift:
        """Displacement of the surface ``role`` [m]: its offset, plus ``s`` when it translates."""
        off = self.members[role][1]
        if role == "translating":
            return (off[0] + float(s[0]), off[1] + float(s[1]))
        return off

    # Evaluation
    def nodal(self, nx: int, ny: Optional[int] = None, s: Shift = _ZERO) -> np.ndarray:
        """Summed roughness ``z[j_y, i_x]`` [m] at ``(i*Lx/nx, j*Ly/ny)`` (``ny = nx`` by default).

        The returned array is read-only.  The stationary part is cached per grid, and the sum
        for the last ``s`` per grid."""
        nx = int(nx)
        ny = nx if ny is None else int(ny)
        key = (nx, ny)
        z0 = self._static_cache.get(key)
        if z0 is None:
            z0 = np.zeros((ny, nx))
            if "stationary" in self.members:
                fs, off = self.members["stationary"]
                z0 += fs.nodal(nx, ny, shift=off)
            z0.setflags(write=False)
            self._static_cache[key] = z0
        if not self.translating:
            return z0
        s = (float(s[0]), float(s[1]))
        hit = self._moving_cache.get(key)
        if hit is not None and hit[0] == s:
            return hit[1]
        fs, _off = self.members["translating"]
        z = z0 + fs.nodal(nx, ny, shift=self._shift("translating", s))
        z.setflags(write=False)
        self._moving_cache[key] = (s, z)
        return z

    def at_unit(self, x, y, s: Shift = _ZERO) -> np.ndarray:
        """Summed roughness [m] at the unit-cell coordinates ``(x, y)``, with the translating
        surface displaced by ``s`` [m].

        Points on a lattice ``(i/nx, j/ny)`` (to 1e-9) are read from ``nodal``; other points are
        evaluated directly."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        xb = np.broadcast_to(x, shape).ravel()
        yb = np.broadcast_to(y, shape).ravel()
        nx, ny = _lattice(xb), _lattice(yb)
        if nx and ny:
            i = np.rint(xb * nx).astype(np.int64) % nx
            j = np.rint(yb * ny).astype(np.int64) % ny
            return self.nodal(nx, ny, s)[j, i].reshape(shape)
        out = np.zeros(xb.size)
        for role, (fs, _off) in self.members.items():
            out += fs.at_points(xb * self.Lx, yb * self.Ly, shift=self._shift(role, s))
        return out.reshape(shape)

    def check_cell(self, xmax, ymax) -> None:
        """Raises ``ValueError`` unless the cell ``xmax x ymax`` [m] is that of the surfaces."""
        if not (_close(float(xmax), self.Lx) and _close(float(ymax), self.Ly)):
            raise ValueError(f"the micro cell {float(xmax):.6g} x {float(ymax):.6g} m is not the prepared "
                             f"surfaces' {self.Lx:.6g} x {self.Ly:.6g} m")


def _lattice(u: np.ndarray, n_max: int = 2048) -> int:
    """Number of lattice points n when the values of ``u`` (modulo 1) are the n values ``k/n``
    to 1e-9, with 0 among them; 0 when they are not, or when n exceeds ``n_max``."""
    if u.size == 0:
        return 0
    v = np.unique(np.round(np.mod(u, 1.0), 12))
    v = v[v < 1.0 - 1e-12]
    if v.size == 0:
        return 0
    n = int(v.size)
    if n > n_max:
        return 0
    if np.allclose(v * n, np.rint(v * n), atol=1e-9, rtol=0) and np.rint(v * n).min() == 0:
        return n
    return 0


class SurfaceMotion:
    """Displacement of the translating surface: ``s(t) = s0 + ratio * (2*U*t + UT*t**2)`` per
    axis [m].

    kin: ``(ratio, Ux, Uy, UxT, UyT)``, the velocity ratio, the mean velocities [m/s] and their
        rates [m/s^2]; None until set.
    s0: starting displacement [m].  ``set`` and ``set_from_solver`` do not change it.
    """

    def __init__(self):
        self.kin: Optional[Tuple[float, float, float, float, float]] = None
        self.s0: Shift = _ZERO

    def set(self, ratio, Ux, Uy, UxT=0.0, UyT=0.0) -> None:
        """Sets the kinematics ``kin``."""
        self.kin = (float(ratio), float(Ux), float(Uy), float(UxT), float(UyT))

    def set_from_solver(self, solver) -> None:
        """Sets the kinematics from ``solver.velocity_ratio`` and ``solver.zeta``."""
        z = solver.zeta
        self.set(solver.velocity_ratio, z.Ux, z.Uy, z.Uxdot, z.Uydot)

    def set_offset(self, s0=None) -> None:
        """Sets the starting displacement ``s0`` [m]; None is (0, 0)."""
        self.s0 = _ZERO if s0 is None else (float(s0[0]), float(s0[1]))

    def displacement(self, t) -> Shift:
        """``(sx, sy)`` [m] at time t [s].  Raises ``RuntimeError`` when the kinematics are not set."""
        if self.kin is None:
            raise RuntimeError("the translating surface's kinematics are not set: the film callback's "
                               "refresh_geometry(solver) must run before the surface is evaluated")
        ratio, Ux, Uy, UxT, UyT = self.kin
        t = float(t)
        sx, sy = pattern_displacement(t, Ux, UxT, ratio), pattern_displacement(t, Uy, UyT, ratio)
        if self.s0 == _ZERO:
            return (sx, sy)
        return (self.s0[0] + sx, self.s0[1] + sy)


class SurfaceContactHeight:
    """Undeformed film of a surface cell, as the contact solve reads it.

    field: the ``SurfaceField``; motion: the ``SurfaceMotion`` of the translating surface.
    The interface is that of ``microcell.film.CosineContactDescriptor``: ``height`` and the
    phase attributes ``phi_x, phi_y, dphi_x, dphi_y``, which are 0.0 for a surface.
    """

    phi_x = 0.0
    phi_y = 0.0
    dphi_x = 0.0
    dphi_y = 0.0

    def __init__(self, field: SurfaceField, motion: Optional[SurfaceMotion] = None):
        self.field = field
        self.motion = motion if motion is not None else SurfaceMotion()

    def height(self, x, y, xmax, ymax, H0, HT=0.0, T=0.0, Ux=0.0, Uy=0.0):
        """Undeformed film ``H0 + HT*T + z`` [m] at the unit-cell coordinates ``(x, y)``.

        xmax, ymax: cell size [m], checked against the surfaces; H0 [m], HT [m/s]: film and its
        rate; T: time [s].  The translating surface is displaced by ``motion.displacement(T)``;
        ``Ux`` and ``Uy`` are not used.
        """
        self.field.check_cell(xmax, ymax)
        base = float(H0) + float(HT) * float(T)
        s = self.motion.displacement(T) if self.field.translating else _ZERO
        return base + self.field.at_unit(x, y, s)

    def __repr__(self) -> str:
        return f"SurfaceContactHeight({self.field.arrangement}, envelope {self.field.envelope_m:.4g} m)"
