"""The surfaces of a cell, as the solver uses them (R4, R5a).

:class:`SurfaceField` holds the evaluation of the cell's member surfaces: one
:class:`~microcell.roughness.evaluator.FourierSurface` per member with its
bandwidth and offset.  It is built in memory by
:class:`microcell.roughness.frontend.ArrayField` (the parent loaded it from a
case descriptor and the artefact store; MicroCell WP3 removed that loader and
the store).  Everything the solver sees of the surface comes from
:meth:`SurfaceField.nodal` / :meth:`SurfaceField.at_unit`, i.e. from the one
Fourier evaluator (decision D4).

:class:`SurfaceContactHeight` is the rigid film ``fft_elastic_v1`` reads
(``ht.contact_cosine.height`` in the contact solver): the interface of
``microcell.film.CosineContactDescriptor`` (``height`` plus four phase attributes,
constant zero here), evaluated from the same field.

Moving surfaces (R5a): the translating member is displaced by
``s(t) = ratio*(2*U*t + UT*t**2)`` (``pair.pattern_displacement``, the
solver's convention) evaluated at the time being evaluated -- history times
included, where the solver's ``T * _Ux_dim`` is wrong when UT != 0
(implementation log finding 11).  :class:`SurfaceMotion` holds the task's
kinematics; the FE adapter sets it from the solver's parameters at every
geometry refresh.  Local-cell policy: every task starts at s(0) = 0 (plus the
member offsets); nothing is carried between tasks.

Only NumPy and the standard library are imported (no FEniCS).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pair import pattern_displacement

__all__ = ["SurfaceField", "SurfaceMotion", "SurfaceContactHeight"]

Shift = Tuple[float, float]
_ZERO: Shift = (0.0, 0.0)

_CELL_RTOL = 1e-9


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= _CELL_RTOL * abs(float(b))


class SurfaceField:
    """The member surfaces of one cell (the attributes are set by the subclass
    :class:`microcell.roughness.frontend.ArrayField`).

    ``nodal(nx, ny, s)`` -> the summed gap perturbation ``z[j_y, i_x]`` [m] on
    the half-open ``ny x nx`` grid of the cell, the stationary member at its
    offset and the translating member at its offset plus the displacement
    ``s = (sx, sy)`` [m]; ``at_unit(x, y, s)`` -> the same at unit-cell
    coordinates (exact lattice look-up when the points lie on an n x m
    lattice, direct Fourier evaluation otherwise).
    """

    def _shift(self, role: str, s: Shift) -> Shift:
        off = self.members[role][1]
        if role == "translating":
            return (off[0] + float(s[0]), off[1] + float(s[1]))
        return off

    # -- evaluation ------------------------------------------------------------
    def nodal(self, nx: int, ny: Optional[int] = None, s: Shift = _ZERO) -> np.ndarray:
        """``z[j_y, i_x]`` [m] at ``(i*Lx/nx, j*Ly/ny)``: the stationary member
        at its offset plus the translating member at its offset + ``s``.
        Read-only; the stationary part is cached per grid, the whole field for
        the last ``s`` per grid (one inverse FFT per new displacement)."""
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
        """``z`` [m] at unit-cell coordinates ``(x, y)`` (arrays of one shape)
        with the translating member displaced by ``s`` [m].

        Points on a lattice ``(i/nx, j/ny)`` (to 1e-9) read the nodal grid
        (the FE refill and the contact grid); anything else is evaluated
        directly."""
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
        if not (_close(float(xmax), self.Lx) and _close(float(ymax), self.Ly)):
            raise ValueError(f"the micro cell {float(xmax):.6g} x {float(ymax):.6g} m is not the prepared "
                             f"surfaces' {self.Lx:.6g} x {self.Ly:.6g} m")


def _lattice(u: np.ndarray, n_max: int = 2048) -> int:
    """n when every value of ``u`` is ``k/n`` for integers k (to 1e-9), with n
    the smallest such that also reproduces the number of distinct values;
    0 when ``u`` is not on a lattice we recognise."""
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
    """The kinematics of the translating member for the current task:
    ``s(t) = ratio * (2*U*t + UT*t**2)`` per axis (``pair.pattern_displacement``).
    Set from the solver's parameters at every geometry refresh."""

    def __init__(self):
        self.kin: Optional[Tuple[float, float, float, float, float]] = None
        # [phase-table WP4, 4 Nov 2026] the task's starting displacement of the translating member [m]
        # (plan D3: the per-node offset of a run with Asperity.phase_average = "contact_table_v1";
        # (0, 0) otherwise).  A separate attribute: ``set_from_solver`` (every geometry refresh)
        # must not touch it, and the worker sets it on EVERY task because one motion object lives per
        # worker (review F7).  displacement(t) = s0 + ratio (2 U t + UT t^2).
        self.s0: Shift = _ZERO

    def set(self, ratio, Ux, Uy, UxT=0.0, UyT=0.0) -> None:
        self.kin = (float(ratio), float(Ux), float(Uy), float(UxT), float(UyT))

    def set_from_solver(self, solver) -> None:
        """The kinematics of a solver's task: its velocity ratio and its ``zeta`` (``s0`` untouched)."""
        z = solver.zeta
        self.set(solver.velocity_ratio, z.Ux, z.Uy, z.Uxdot, z.Uydot)

    def set_offset(self, s0=None) -> None:
        """The starting displacement ``s0`` [m] of this task; None or (0, 0) = none."""
        self.s0 = _ZERO if s0 is None else (float(s0[0]), float(s0[1]))

    def displacement(self, t) -> Shift:
        if self.kin is None:
            raise RuntimeError("the translating surface's kinematics are not set: the film callback's "
                               "refresh_geometry(solver) must run before the surface is evaluated")
        ratio, Ux, Uy, UxT, UyT = self.kin
        t = float(t)
        sx, sy = pattern_displacement(t, Ux, UxT, ratio), pattern_displacement(t, Uy, UyT, ratio)
        if self.s0 == _ZERO:
            return (sx, sy)                   # no offset: the expression as it always was
        return (self.s0[0] + sx, self.s0[1] + sy)


class SurfaceContactHeight:
    """The rigid film for ``fft_elastic_v1`` from the prepared surfaces.

    The contact solver reads ``ht.contact_cosine``: ``height(x, y, xmax, ymax,
    H0, HT, T, Ux, Uy)`` [m] at unit-cell coordinates, and the phase
    attributes ``phi_x, phi_y, dphi_x, dphi_y`` for its state key.  A generated
    surface has no phases (all 0.0).  A translating member is displaced by
    ``motion.displacement(T)`` -- the passed ``Ux, Uy`` (the solver's
    ``_Ux_dim`` effective velocity) are not used."""

    phi_x = 0.0
    phi_y = 0.0
    dphi_x = 0.0
    dphi_y = 0.0

    def __init__(self, field: SurfaceField, motion: Optional[SurfaceMotion] = None):
        self.field = field
        self.motion = motion if motion is not None else SurfaceMotion()

    def height(self, x, y, xmax, ymax, H0, HT=0.0, T=0.0, Ux=0.0, Uy=0.0):
        self.field.check_cell(xmax, ymax)
        base = float(H0) + float(HT) * float(T)
        s = self.motion.displacement(T) if self.field.translating else _ZERO
        return base + self.field.at_unit(x, y, s)

    def __repr__(self) -> str:
        return f"SurfaceContactHeight({self.field.arrangement}, envelope {self.field.envelope_m:.4g} m)"
