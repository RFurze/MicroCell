"""The film callbacks ``ht`` of the micro solver: the analytic cosine cell and the generated surfaces.

MicroCell WP5 (11 Nov 2026): ``ideal_ht.py`` and ``surface_ht.py`` in one file, in that order.  The
code is moved verbatim.  Their module docstrings follow.  Where one names one of the two files,
read this file.

``ideal_ht.py`` (the parent's ``microscale/src/functions/ideal_ht.py``)
------------------------------------------------------------------------------------------
Canonical film-thickness function for all microscale code paths.

Every production and diagnostic microscale code path in this repository
must use :func:`make_ideal_ht` to build the ``ht`` callback passed to
``MicroMixedSolver``.  Keeping the definition in a single module prevents
the historic drift between ``run_microscale.py``, the tests under
``tests/``, and the analysis scripts under ``analysis/``.

Single-surface expression (``two_surface=False``, default)
----------------------------------------------------------
    H(x, y, t) = H0 + HT*T
                + (Ah/2) * cos(kx*2*pi*(x_d - T*Ux)/xmax + phi_x)
                + (Ah/2) * cos(ky*2*pi*(y_d - T*Uy)/ymax + phi_y)

A single roughness pattern translating at velocity (Ux, Uy).  Spatial
mean is H0 + HT*T.  Extremes are H0 + HT*T +/- Ah.

Two-surface expression (``two_surface=True``)
---------------------------------------------
For two surfaces with identical roughness (e.g. ball rotating in a
stationary cup), each carrying the same sinusoidal profile but
translating in opposite directions at +/- (Ux, Uy):

    r_1(x,t) = (Ah/2) * cos(kx*2*pi*(x_d - T*Ux)/xmax + phi_x)  + y-term
    r_2(x,t) = (Ah/2) * cos(kx*2*pi*(x_d + T*Ux)/xmax + phi_x + dphi_x) + y-term

When ``dphi_x = dphi_y = 0`` (both surfaces share the same phase at
t=0), the trig identity cos(A-B) + cos(A+B) = 2*cos(A)*cos(B) collapses
this to:

    H = H0 + HT*T
      + Ah * cos(kx*2*pi*x_d/xmax + phi_x) * cos(kx*2*pi*T*Ux/xmax)
      + Ah * cos(ky*2*pi*y_d/ymax + phi_y) * cos(ky*2*pi*T*Uy/ymax)

When ``dphi != 0``, the two surfaces have different initial phases and
the expression is kept in its general two-term form.  Sweeping ``dphi``
while holding ``phi_x/phi_y`` fixed tests the effect of relative phase
between the surfaces, independent of the absolute spatial position
(which is already handled by periodicity).

``Ah`` is the per-surface amplitude.  Combined extremes range from
H0 + HT*T - 2*Ah  to  H0 + HT*T + 2*Ah  (when dphi=0 and surfaces
are in phase).

Sweep-friendly design
---------------------
``phi_x`` and ``phi_y`` are accepted as either dolfin ``Constant``
objects or Python floats.  **Prefer Constants**: when the caller intends
to sweep ``phi_x`` over multiple values (see
``FilmThicknessParams.phi_x_values``), baking a ``Constant`` into the
UFL expression keeps the form signature stable across the sweep.  FFC
compiles the residual/Jacobian code once per worker and each phase is
just a ``phi_x_const.assign(...)`` away — there is no recompilation
cost, and the Newton solver keeps its previous solution as a warm start
for the next phase.

``surface_ht.py`` (the parent's ``microscale/src/functions/surface_ht.py``)
------------------------------------------------------------------------------------------
Film callback for generated surfaces (R4, R5a): the FE adapter of
:mod:`ehl.roughness.runtime`.

``make_surface_ht(field)`` returns a callback with the signature of
``make_ideal_ht``'s (``ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy)``, the last four
optional), so every solver class takes it unchanged:

    h = H0 + HT*T + z_h

where ``z_h`` is a CG1 ``Function`` on the mesh of ``x`` whose nodal values are
the Fourier reconstruction of the prepared surfaces at the mesh vertices (exact
nodal values, one inverse FFT per refill; decision D4).  One refill per mesh, so
the coarse and fine tiers each have their own (plan 7.4).

Moving surfaces (R5a): ``ht.refresh_geometry(solver)`` -- called by the base
solver's ``_refresh_geometry`` seam at every time the solver evaluates the
film, BDF history times included -- sets the task's kinematics from
``solver.zeta`` and refills ``solver.mesh_m``'s ``z_h`` at the displacement
``s(T)`` of ``solver.T_dim_const`` (``ehl.roughness.pair.pattern_displacement``;
not ``T * _Ux_dim``, which is wrong at history times when UT != 0).  The BDF
history keeps projected copies, so refilling ``z_h`` never alters a stored
history film.  A stationary arrangement never refills.

``ht.contact_cosine`` is the :class:`~ehl.roughness.runtime.SurfaceContactHeight`
of the same field and motion -- the attribute ``fft_elastic_v1`` reads its rigid
film from (``micro_phase_neutral_class_contact``; the name is the contact
solver's).  ``ht.surface_field`` and ``ht.surface_motion`` are exposed.
"""
from __future__ import annotations

import sys
from typing import Any, Callable


class CosineContactDescriptor(tuple):
    """``ht.contact_cosine``: the geometry identity that travels with the callback.

    As a tuple it is exactly ``(Ah, kx, ky, two_surface)`` (:meth:`height`
    and the ripple check of the solver class read the entries).
    [fft_elastic_v1, 16 Oct 2026] It also carries the
    rest of the ``make_ideal_ht`` description -- the four phase objects (dolfin
    ``Constant`` or float, kept by reference and read at call time, so the
    worker's per-task ``.assign`` is seen), and ``static_moving`` -- and
    :meth:`height`, a numpy evaluation of the SAME formula as the UFL ``ht``
    at nodal coordinates.  ``fft_elastic_v1`` builds its nodal rigid film from
    it (plan WP2); gate G2a checks it against the UFL ``ht`` at the vertices.
    """

    def __new__(cls, Ah, kx, ky, two_surface, *, phi_x=0.0, phi_y=0.0, dphi_x=0.0,
                dphi_y=0.0, static_moving=False):
        self = super().__new__(cls, (float(Ah), float(kx), float(ky), bool(two_surface)))
        self.phi_x, self.phi_y = phi_x, phi_y
        self.dphi_x, self.dphi_y = dphi_x, dphi_y
        self.static_moving = bool(static_moving)
        return self

    def __reduce__(self):            # pickle/copy keep the full descriptor
        return (_rebuild_descriptor, (tuple(self), self.phi_x, self.phi_y, self.dphi_x,
                                      self.dphi_y, self.static_moving))

    def height(self, x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        """Dimensional film [m] at unit-cell coordinates ``(x, y)`` (arrays).

        Same argument list as the UFL callback (``MicroPhaseNeutralSolver.
        _call_ht``); every scalar may be a dolfin ``Constant`` and is read
        with ``float()`` now.  Mirrors the UFL expression branch for branch.
        """
        import numpy as np

        Ah, kx, ky, two_surface = tuple(self)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xmax, ymax = float(xmax), float(ymax)
        H0, HT, T, Ux, Uy = (float(v) for v in (H0, HT, T, Ux, Uy))
        phi_x, phi_y = float(self.phi_x), float(self.phi_y)
        base = H0 + HT * T
        if Ah == 0.0:
            return np.full(np.broadcast(x, y).shape, base)
        x_d = x * xmax
        y_d = y * ymax
        pi = np.pi
        if two_surface:
            dphi_x, dphi_y = float(self.dphi_x), float(self.dphi_y)
            alpha_x = kx * 2 * pi / xmax
            alpha_y = ky * 2 * pi / ymax
            if self.static_moving:
                r1 = (0.5 * Ah * np.cos(alpha_x * x_d + phi_x)
                      + 0.5 * Ah * np.cos(alpha_y * y_d + phi_y))
                r2 = (0.5 * Ah * np.cos(alpha_x * (x_d - T * Ux) + phi_x + dphi_x)
                      + 0.5 * Ah * np.cos(alpha_y * (y_d - T * Uy) + phi_y + dphi_y))
            else:
                r1 = (0.5 * Ah * np.cos(alpha_x * (x_d - T * Ux) + phi_x)
                      + 0.5 * Ah * np.cos(alpha_y * (y_d - T * Uy) + phi_y))
                r2 = (0.5 * Ah * np.cos(alpha_x * (x_d + T * Ux) + phi_x + dphi_x)
                      + 0.5 * Ah * np.cos(alpha_y * (y_d + T * Uy) + phi_y + dphi_y))
            return base + r1 + r2
        return base + 0.5 * Ah * (
            np.cos(kx * 2 * pi * (x_d - T * Ux) / xmax + phi_x)
            + np.cos(ky * 2 * pi * (y_d - T * Uy) / ymax + phi_y))


def _rebuild_descriptor(values, phi_x, phi_y, dphi_x, dphi_y, static_moving):
    return CosineContactDescriptor(*values, phi_x=phi_x, phi_y=phi_y, dphi_x=dphi_x,
                                   dphi_y=dphi_y, static_moving=static_moving)


def make_ideal_ht(
    Ah: float,
    kx: float,
    ky: float,
    phi_x_const: Any = 0.0,
    phi_y_const: Any = 0.0,
    dphi_x_const: Any = 0.0,
    dphi_y_const: Any = 0.0,
    two_surface: bool = False,
    static_moving: bool = False,
) -> Callable:
    """Build an ``ht`` callback for the microscale solver.

    Parameters
    ----------
    Ah : float
        Roughness amplitude [m].  For ``two_surface=False`` this is the
        peak-to-mean amplitude of the single pattern.  For
        ``two_surface=True`` this is the *per-surface* amplitude; the
        combined roughness can reach +/- 2*Ah when the two surfaces are
        in phase.
    kx, ky : float
        Number of roughness periods per cell in x and y.
    phi_x_const, phi_y_const : dolfin.Constant | float, optional
        Phase offsets for surface 1 (and shared by surface 2 when
        ``dphi = 0``).  Pass ``dolfin.Constant`` instances if you want
        to mutate the phase between solver runs without triggering a UFL
        form rebuild (the recommended pattern for the production phi_x
        sweep).  Python floats also work but bake the value into the UFL
        tree and therefore force FFC to recompile for every new phase.
    dphi_x_const, dphi_y_const : dolfin.Constant | float, optional
        *Relative* phase offset applied to surface 2 only
        (``two_surface=True``).  Surface 2's phase becomes
        ``phi_x + dphi_x``.  When 0 (default), both surfaces share the
        same phase at t=0 and the expression collapses to the standing-
        wave product form.  Sweep ``dphi_x`` to test how dQ depends on
        the relative alignment of the two surfaces.  Ignored when
        ``two_surface=False``.
    two_surface : bool, optional
        If True, model two identical roughness profiles counter-
        propagating at +/- (Ux, Uy).  Default False (single translating
        roughness).

    Returns
    -------
    ht : callable
        A function with signature
        ``ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy)`` returning a UFL
        expression (dimensional).  Matches the callback signature
        expected by ``MicroMixedSolver._call_ht``.
    """

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        # Lazy dolfin import so importing this module never drags FEniCS
        # into processes that only need configuration metadata.
        from dolfin import pi, sin, cos

        x_d = x * xmax
        y_d = y * ymax

        # Python-side safety check using float projections of the Constants
        # passed in by the solver.
        H0_val = float(H0)
        HT_val = float(HT)
        T_val = float(T)

        if two_surface:
            # Worst case: both surfaces in phase, both cosines at -1.
            hmin_amp = 2.0 * Ah
        else:
            hmin_amp = Ah

        hmin0 = H0_val - hmin_amp
        hmin1 = H0_val - hmin_amp + HT_val * T_val

        if (hmin0 < 0.0) or (hmin1 < 0.0):
            print(
                f"Warning: Negative film thickness at T={T_val:.6e}s, "
                f"hmin0={hmin0:.6e}, hmin1={hmin1:.6e}."
            )
            sys.stdout.flush()

        # Fast path: Ah == 0 reduces to the smooth film.
        if Ah == 0.0:
            return H0 + HT * T

        if two_surface:
            alpha_x = kx * 2 * pi / xmax
            alpha_y = ky * 2 * pi / ymax

            if static_moving:
                # Split model: surface 1 STATIC, surface 2 MOVING at (Ux, Uy).
                # r2 carries the relative phase dphi, which phase_sync advances
                # across macro steps so the moving roughness stays consistent.
                r1 = (
                    0.5 * Ah * cos(alpha_x * x_d + phi_x_const)
                    + 0.5 * Ah * cos(alpha_y * y_d + phi_y_const)
                )
                r2 = (
                    0.5 * Ah * cos(alpha_x * (x_d - T * Ux)
                                   + phi_x_const + dphi_x_const)
                    + 0.5 * Ah * cos(alpha_y * (y_d - T * Uy)
                                     + phi_y_const + dphi_y_const)
                )
            else:
                # Legacy: two identical roughnesses counter-propagating at
                # +/- (Ux, Uy).  Surface 2 carries a relative phase dphi.
                # When dphi=0 this reduces to the standing-wave product form:
                #   cos(k(x-Vt)+phi) + cos(k(x+Vt)+phi) = 2*cos(kx+phi)*cos(kVt)
                # Surface 1: translating at -U (e.g. stationary cup in ball frame)
                r1 = (
                    0.5 * Ah * cos(alpha_x * (x_d - T * Ux) + phi_x_const)
                    + 0.5 * Ah * cos(alpha_y * (y_d - T * Uy) + phi_y_const)
                )
                # Surface 2: translating at +U, with relative phase offset dphi
                r2 = (
                    0.5 * Ah * cos(alpha_x * (x_d + T * Ux)
                                   + phi_x_const + dphi_x_const)
                    + 0.5 * Ah * cos(alpha_y * (y_d + T * Uy)
                                     + phi_y_const + dphi_y_const)
                )
            return H0 + HT * T + r1 + r2
        else:
            # Single roughness translating at (Ux, Uy).
            return (
                H0
                + HT * T
                + 0.5 * Ah * (
                    cos(kx * 2 * pi * (x_d - T * Ux) / xmax + phi_x_const)
                    + cos(ky * 2 * pi * (y_d - T * Uy) / ymax + phi_y_const)
                )
            )

    # Geometry identity travels with the canonical callback, including worker
    # reuse and phase changes. Curvature is independent of a common translation.
    # [fft_elastic_v1] A tuple equal to the historical (Ah, kx, ky, two_surface),
    # carrying the full description and a numpy evaluator (CosineContactDescriptor).
    ht.contact_cosine = CosineContactDescriptor(
        Ah, kx, ky, two_surface, phi_x=phi_x_const, phi_y=phi_y_const,
        dphi_x=dphi_x_const, dphi_y=dphi_y_const, static_moving=static_moving)
    return ht



from typing import Callable

from microcell.roughness.runtime import SurfaceContactHeight, SurfaceField, SurfaceMotion


def _mesh_of(x):
    dom = x.ufl_domain()
    mesh = dom.ufl_cargo() if dom is not None else None
    if mesh is None:
        raise ValueError("make_surface_ht: the callback needs SpatialCoordinate components of a mesh")
    return mesh


def make_surface_ht(field: SurfaceField) -> Callable:
    from dolfin import Function, FunctionSpace

    motion = SurfaceMotion()
    refills: dict = {}            # mesh.id() -> [Function, mesh, dof coordinates, displacement]

    def z_function(mesh, s=(0.0, 0.0)):
        """The CG1 refill of the surfaces on ``mesh`` with the translating member
        displaced by ``s`` [m] (refilled only when ``s`` changes)."""
        s = (float(s[0]), float(s[1])) if field.translating else (0.0, 0.0)
        entry = refills.get(mesh.id())
        if entry is None:
            V = FunctionSpace(mesh, "CG", 1)
            X = V.tabulate_dof_coordinates().reshape(-1, 2)
            entry = [Function(V), mesh, X, None]       # keep the mesh alive with its refill
            refills[mesh.id()] = entry
        if entry[3] != s:
            f, _mesh, X = entry[0], entry[1], entry[2]
            f.vector().set_local(field.at_unit(X[:, 0], X[:, 1], s))
            f.vector().apply("insert")
            entry[3] = s
        return entry[0]

    def ht(x, y, xmax, ymax, H0, HT=0.0, T=0.0, Ux=0.0, Uy=0.0):
        field.check_cell(xmax, ymax)
        mesh = _mesh_of(x)
        entry = refills.get(mesh.id())
        # the Function the form holds; its values follow refresh_geometry
        f = entry[0] if entry is not None else z_function(mesh)
        return H0 + HT * T + f

    def refresh_geometry(solver) -> None:
        """Refill ``solver.mesh_m``'s film at ``s(solver.T_dim_const)`` with the
        solver's task kinematics (a no-op for stationary surfaces)."""
        if not field.translating:
            return
        motion.set_from_solver(solver)
        z_function(solver.mesh_m, motion.displacement(float(solver.T_dim_const)))

    ht.contact_cosine = SurfaceContactHeight(field, motion)
    ht.surface_field = field
    ht.surface_motion = motion
    ht.z_function = z_function
    ht.refresh_geometry = refresh_geometry
    return ht
