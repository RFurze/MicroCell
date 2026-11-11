"""Film callback for generated surfaces (R4, R5a): the FE adapter of
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
``solver.params`` and refills ``solver.mesh_m``'s ``z_h`` at the displacement
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
        motion.set_from_params(solver.params)
        z_function(solver.mesh_m, motion.displacement(float(solver.T_dim_const)))

    ht.contact_cosine = SurfaceContactHeight(field, motion)
    ht.surface_field = field
    ht.surface_motion = motion
    ht.z_function = z_function
    ht.refresh_geometry = refresh_geometry
    return ht
