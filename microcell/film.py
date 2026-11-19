"""Film callbacks ``ht`` of the micro solver: the analytic cosine cell and the generated surfaces.

A callback has the signature ``ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy)`` and returns the film [m]
as a UFL expression.  ``x``, ``y`` are the unit-cell coordinates, ``xmax``, ``ymax`` the cell size
[m], ``H0`` the mean film [m], ``HT`` its rate [m/s], ``T`` the time [s] and ``Ux``, ``Uy`` the
effective velocity of the moving roughness [m/s] (its displacement divided by ``T``).  The solver
supplies all of them.

Cosine cell (``make_ideal_ht``), with ``x_d = x xmax`` and ``y_d = y ymax``:

    h = H0 + HT T + (Ah/2) cos(2 pi kx (x_d - T Ux) / xmax + phi_x)
                  + (Ah/2) cos(2 pi ky (y_d - T Uy) / ymax + phi_y)

With ``two_surface`` a second pattern of the same amplitude is added, with the phase shifted by
``dphi``.  The first translates at +(Ux, Uy) and the second at -(Ux, Uy), or, with
``static_moving``, the first is at rest and the second translates at +(Ux, Uy).

Generated surfaces (``make_surface_ht``): ``h = H0 + HT T + z_h``, where ``z_h`` is a CG1 function
holding the surface heights at the mesh vertices.  A translating surface is displaced by its own
kinematics, read from the solver, and ``Ux``, ``Uy`` are not used.

Each callback carries ``ht.contact_cosine``, from which the contact solver evaluates the rigid
film at the mesh nodes.
"""
from __future__ import annotations

import sys
from typing import Any, Callable


class CosineContactDescriptor(tuple):
    """``ht.contact_cosine`` of the cosine cell: its description and a numpy evaluator.

    As a tuple it is ``(Ah, kx, ky, two_surface)``.  The attributes ``phi_x``, ``phi_y``,
    ``dphi_x``, ``dphi_y`` (dolfin ``Constant`` or float, held by reference and read at call
    time) and ``static_moving`` complete the arguments of ``make_ideal_ht``.  ``height``
    evaluates the same formula as the UFL callback; the contact solver builds its nodal rigid
    film from it.
    """

    def __new__(cls, Ah, kx, ky, two_surface, *, phi_x=0.0, phi_y=0.0, dphi_x=0.0,
                dphi_y=0.0, static_moving=False):
        self = super().__new__(cls, (float(Ah), float(kx), float(ky), bool(two_surface)))
        self.phi_x, self.phi_y = phi_x, phi_y
        self.dphi_x, self.dphi_y = dphi_x, dphi_y
        self.static_moving = bool(static_moving)
        return self

    def __reduce__(self):            # pickle and copy keep the attributes
        return (_rebuild_descriptor, (tuple(self), self.phi_x, self.phi_y, self.dphi_x,
                                      self.dphi_y, self.static_moving))

    def height(self, x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        """Film [m] at the unit-cell coordinates ``x``, ``y`` (arrays).

        The arguments are those of the UFL callback.  Each scalar may be a dolfin ``Constant``
        and is read with ``float()`` at the call.  Returns an array of the broadcast shape.
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
    """Film callback of the analytic cosine cell.

    Ah: roughness amplitude [m], the largest departure of one surface from the mean film.
        With ``two_surface`` it is the amplitude of each surface, and the film can depart
        from its mean by 2 Ah.
    kx, ky: number of roughness periods per cell in x and y.
    phi_x_const, phi_y_const: phase of surface 1 [rad].  A dolfin ``Constant`` can be
        reassigned between solves without the forms being recompiled; a float is fixed in
        the form.
    dphi_x_const, dphi_y_const: phase of surface 2 relative to surface 1 [rad].  Used with
        ``two_surface`` only.
    two_surface: two surfaces with the same pattern, translating at +(Ux, Uy) and -(Ux, Uy).
    static_moving: with ``two_surface``, surface 1 is at rest and surface 2 translates at
        (Ux, Uy).
    Returns ``ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy)``, the film [m] as a UFL expression,
    with the attribute ``ht.contact_cosine`` (a ``CosineContactDescriptor``).
    """

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        # dolfin is imported at the call, so the module imports without FEniCS
        from dolfin import pi, cos

        x_d = x * xmax
        y_d = y * ymax

        # warning when the minimum film at T = 0 or at T is negative
        H0_val = float(H0)
        HT_val = float(HT)
        T_val = float(T)

        if two_surface:
            # both surfaces in phase, both cosines at -1
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

        # smooth film
        if Ah == 0.0:
            return H0 + HT * T

        if two_surface:
            alpha_x = kx * 2 * pi / xmax
            alpha_y = ky * 2 * pi / ymax

            if static_moving:
                # surface 1 at rest; surface 2 translating at (Ux, Uy) with relative phase dphi
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
                # two surfaces translating in opposite directions.  With dphi = 0 the sum is the
                # standing wave 2 cos(k x + phi) cos(k V t).
                # surface 1: pattern argument x_d - T Ux
                r1 = (
                    0.5 * Ah * cos(alpha_x * (x_d - T * Ux) + phi_x_const)
                    + 0.5 * Ah * cos(alpha_y * (y_d - T * Uy) + phi_y_const)
                )
                # surface 2: pattern argument x_d + T Ux, with relative phase dphi
                r2 = (
                    0.5 * Ah * cos(alpha_x * (x_d + T * Ux)
                                   + phi_x_const + dphi_x_const)
                    + 0.5 * Ah * cos(alpha_y * (y_d + T * Uy)
                                     + phi_y_const + dphi_y_const)
                )
            return H0 + HT * T + r1 + r2
        else:
            # one surface translating at (Ux, Uy)
            return (
                H0
                + HT * T
                + 0.5 * Ah * (
                    cos(kx * 2 * pi * (x_d - T * Ux) / xmax + phi_x_const)
                    + cos(ky * 2 * pi * (y_d - T * Uy) / ymax + phi_y_const)
                )
            )

    # description of the cell and its numpy evaluator, read by the contact solver
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
    """Film callback of generated or supplied surfaces: ``h = H0 + HT T + z_h``.

    field: the ``SurfaceField`` holding the surface heights (``Roughness`` builds it).
    Returns ``ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy)``, the film [m] as a UFL expression,
    with the attributes:
      ``refresh_geometry(solver)``: moves a translating surface to the solver's time;
      ``contact_cosine``: the ``SurfaceContactHeight`` of the same field and motion;
      ``surface_field``, ``surface_motion``, ``z_function``.
    """
    from dolfin import Function, FunctionSpace

    motion = SurfaceMotion()
    refills: dict = {}            # mesh.id() -> [Function, mesh, dof coordinates, displacement]

    def z_function(mesh, s=(0.0, 0.0)):
        """CG1 function of the surface heights [m] on ``mesh``, with the translating surface
        displaced by ``s`` [m].  The values are recomputed only when ``s`` changes."""
        s = (float(s[0]), float(s[1])) if field.translating else (0.0, 0.0)
        entry = refills.get(mesh.id())
        if entry is None:
            V = FunctionSpace(mesh, "CG", 1)
            X = V.tabulate_dof_coordinates().reshape(-1, 2)
            entry = [Function(V), mesh, X, None]       # the entry holds a reference to the mesh
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
        # the Function held by the form; refresh_geometry updates its values
        f = entry[0] if entry is not None else z_function(mesh)
        return H0 + HT * T + f

    def refresh_geometry(solver) -> None:
        """Sets ``z_h`` on ``solver.mesh_m`` to the displacement at ``solver.T_dim_const``,
        with the kinematics of ``solver.zeta``.  Does nothing for stationary surfaces."""
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
