"""Samplers of a solved micro cell: the fluid pressure and the fluid traction, and the contact
pressure and the asperity and total traction at the points of the fluid traction.

Fluid pressure, sampler ``fluid_nodal_sharp_v1``:
  samples: every mesh vertex.  The vertices at x = 0 and x = 1 (y = 0 and y = 1) are both
    sampled, as the pressure carries the macroscale gradient and is not periodic.
  pressure: ``p_scale (P_re + dpx (x - 1/2) + dpy (y - 1/2) + lambda + w)`` [Pa] at the vertex,
    signed, without projection or clipping and without contact pressure.
  fluid: the nodal obstruction level set is > 0.  The level set is ``phic_nd_func`` in the
    contact class (``mask_source = "phic_nd_func"``) and otherwise the CG1 projection of
    ``h_nd - h_min_nd`` (``"projected_h"``).
  no fluid vertex: NaN with the status ``no_fluid``.

Fluid traction, sampler ``fluid_quadrature_sharp_v1``:
  samples: the points and weights of the form compiler's ``default`` rule on every cell, at the
    quadrature degree in force.
  traction: ``eta U_s / (H + 1e-30) + (H / 2) grad P`` [Pa] in the cell frame, with ``grad P``
    the cell gradient of the CG1 pressure and ``H`` the film as the forms evaluate it.
  fluid: the nodal level set, interpolated linearly to the point, is > 0.  The traction is zero
    outside the fluid.

Contact pressure, sampler ``fft_nodal_p1_periodic_v1`` (numpy only): the periodic P1 interpolant
of the nodal contact solution (``microcell.contact.ContactSolution``; arrays ``[i, j]``, node
``(i, j)`` at ``(i/nx, j/ny)``, the images at x = 1 and y = 1 not stored).  Each grid square is
split along its 'right' diagonal, as in dolfin's ``UnitSquareMesh``, so the interpolant is the
CG1 function of the same nodal values.  Sample coordinates are fractions of the cell, or metres
with ``cell_size``; points outside the cell are wrapped.  The interpolant returns the nodal value
at a node, is non-negative and has the cell mean ``mean(sol.p)``.  ``statistic`` is
``native_nodal`` when every sample is a node and ``p1_interpolated`` otherwise.

Supports at the samples: ``mech_support`` is ``p_a > 0`` and ``obstruction`` is ``phi_c <= 0``,
the complement of the fluid region of the flow solve.  The two need not agree: the P1 pressure is
positive on a fringe on the fluid side of the obstruction edge.

Asperity and total traction at the common samples:

    t_a = mu p_a U / (|U| + 1e-30),     t_t = t_f + t_a

The asperity traction is zero at zero sliding speed: there is no static friction.

Contact statuses: ``ok``; ``no_support`` (zero contact pressure everywhere, ``p_a = 0`` at every
sample); ``state_invalid`` (no stored solution, or arrays that are not a finite non-negative
grid: NaN).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from microcell.pressure import (FluidPressureExtrema, PRESSURE_EXTREMA_SAMPLER, PHASE_NO_FLUID,
                          PHASE_NONFINITE, PHASE_OK)


MASK_CONTACT = "phic_nd_func"
MASK_FILM = "projected_h"


class FluidPressureSamples(NamedTuple):
    """Sample set of ``fluid_nodal_sharp_v1``, in mesh vertex order."""
    X: np.ndarray          # (n_v, 2) unit-cell coordinates of the vertices
    p: np.ndarray          # (n_v,) signed pressure [Pa]
    phi: np.ndarray        # (n_v,) nodal obstruction level set (non-dimensional)
    fluid: np.ndarray      # (n_v,) bool: phi > 0
    mask_source: str


def _vertex_maps(solver):
    """Returns (X, vertex to Vper dof, vertex to Vfull dof) of the solver's mesh, cached on the
    solver."""
    cache = getattr(solver, "_extrema_vertex_maps", None)
    if cache is not None and cache[0] is solver.mesh_m:
        return cache[1:]
    from dolfin import vertex_to_dof_map
    X = np.array(solver.mesh_m.coordinates(), dtype=float).reshape(-1, 2)
    v_per = np.asarray(vertex_to_dof_map(solver.Vper), dtype=np.intp)
    v_full = np.asarray(vertex_to_dof_map(solver.Vfull), dtype=np.intp)
    solver._extrema_vertex_maps = (solver.mesh_m, X, v_per, v_full)
    return X, v_per, v_full


def fluid_pressure_samples(solver, film_dofs_m=None) -> FluidPressureSamples:
    """Samples of ``fluid_nodal_sharp_v1`` for the current state of the solver.

    film_dofs_m: CG1 projection of the film ``H0 h_nd`` [m] on ``solver.Vfull``, in dof order,
        for the ``projected_h`` mask.  Projected here when None; not used in the contact class.
    Returns a ``FluidPressureSamples``.
    """
    X, v_per, v_full = _vertex_maps(solver)
    w = solver.w.vector().get_local()[v_per]
    p = float(solver.p_scale_const) * (
        float(solver.P_re_nd)
        + float(solver.dpx_nd_const) * (X[:, 0] - 0.5)
        + float(solver.dpy_nd_const) * (X[:, 1] - 0.5)
        + float(solver.lambda_nd)
        + w)
    phic = getattr(solver, "phic_nd_func", None)
    if phic is not None:
        phi = phic.vector().get_local()[v_per]
        source = MASK_CONTACT
    else:
        if film_dofs_m is None:
            from dolfin import Constant, project
            film_dofs_m = project(Constant(float(solver.zeta.H)) * solver.h_nd,
                                  solver.Vfull).vector().get_local()
        phi = (np.asarray(film_dofs_m, dtype=float)[v_full] / float(solver.zeta.H)
               - float(solver.hmin_nd_const))
        source = MASK_FILM
    fluid = phi > 0.0
    # Fluid-region level constraint: a vertex whose basis function holds no fluid quadrature point
    # (``_lc_mf_zero``) carries no fluid pressure and is left out of the fluid samples.
    mf_zero = getattr(solver, "_lc_mf_zero", None)
    if mf_zero is not None and getattr(solver, "_lc_path", "whole_cell") in ("fluid_mean", "no_fluid"):
        excl = np.asarray(mf_zero, dtype=bool)[v_per]
        if excl.any():
            fluid = fluid & ~excl
    return FluidPressureSamples(X=X, p=p, phi=phi, fluid=fluid, mask_source=source)


def quadrature_audit(solver, degree: int, film_dofs_m=None) -> dict:
    """Fluid pressure integrals of the pressure ledger, evaluated point by point.  A diagnostic:
    the solver does not call it.

    degree: quadrature degree of the form compiler's ``default`` rule, applied on every cell.
    film_dofs_m: as for ``fluid_pressure_samples``.
    The pressure and the level set are interpolated linearly from the vertex samples, and a
    point is fluid where the level set is > 0.
    Returns a dict: ``P_f``, ``P_fluid_pos``, ``P_fluid_neg`` per cell area [Pa]; the numbers of
    points, fluid points and fluid points with negative pressure; the lowest fluid-point
    pressure [Pa] and its unit-cell location.
    """
    from ffc.fiatinterface import create_quadrature
    s = fluid_pressure_samples(solver, film_dofs_m)
    cells = np.asarray(solver.mesh_m.cells(), dtype=np.intp)          # (n_c, 3) vertex indices
    pts, wts = create_quadrature("triangle", int(degree), "default")
    pts, wts = np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)
    lam = np.column_stack([1.0 - pts.sum(axis=1), pts])                 # (n_q, 3) barycentric
    V = s.X[cells]                                                      # (n_c, 3, 2)
    det = np.abs((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1])
                 - (V[:, 2, 0] - V[:, 0, 0]) * (V[:, 1, 1] - V[:, 0, 1]))
    W = wts[:, None] * det[None, :]                                     # (n_q, n_c)
    P = lam @ s.p[cells].T
    chi = (lam @ s.phi[cells].T) > 0.0
    area = float(W.sum())
    fluid_neg = chi & (P < 0.0)
    out = dict(degree=int(degree), n_points=int(P.size), n_fluid_points=int(np.count_nonzero(chi)),
               n_fluid_negative_points=int(np.count_nonzero(fluid_neg)),
               P_f=float(np.sum(W * chi * P)) / area,
               P_fluid_pos=float(np.sum(W * chi * np.maximum(P, 0.0))) / area,
               P_fluid_neg=float(np.sum(W * chi * np.minimum(P, 0.0))) / area,
               p_min_fluid_point=float("nan"), p_min_fluid_point_xy=(float("nan"), float("nan")))
    if out["n_fluid_points"]:
        q, c = np.unravel_index(np.argmin(np.where(chi, P, np.inf)), P.shape)
        out["p_min_fluid_point"] = float(P[q, c])
        out["p_min_fluid_point_xy"] = tuple(float(v) for v in lam[q] @ V[c])
    return out


def sample_fluid_pressure_extrema(solver, film_dofs_m=None) -> FluidPressureExtrema:
    """Fluid pressure extrema of the current state of the solver (one phase).

    film_dofs_m: as for ``fluid_pressure_samples``.
    Returns a ``FluidPressureExtrema`` with status ``ok``, ``no_fluid`` or ``nonfinite``.  The
    extrema are NaN unless ``ok``; one non-finite fluid sample gives ``nonfinite``.
    """
    s = fluid_pressure_samples(solver, film_dofs_m)
    n = int(s.p.size)
    n_fluid = int(np.count_nonzero(s.fluid))
    nan = float("nan")
    if n_fluid == 0:
        status, lo, hi = PHASE_NO_FLUID, nan, nan
    else:
        pf = s.p[s.fluid]
        if not np.isfinite(pf).all():
            status, lo, hi = PHASE_NONFINITE, nan, nan
        else:
            status, lo, hi = PHASE_OK, float(pf.min()), float(pf.max())
    return FluidPressureExtrema(p_min_Pa=lo, p_max_Pa=hi, n_fluid_samples=n_fluid, n_samples=n,
                                fluid_frac=n_fluid / n if n else nan, status=status,
                                sampler=PRESSURE_EXTREMA_SAMPLER, mask_source=s.mask_source)


# ---- fluid traction, sampler fluid_quadrature_sharp_v1 ----
FILM_ANALYTIC = "analytic"          # film from the formula of the cosine callback at the points
FILM_CG1 = "cg1_projection"         # film from the CG1 projection of H0 h_nd, interpolated linearly

_RULES: dict = {}


def _rule(degree: int):
    """Returns (barycentric coordinates (n_q, 3), reference weights (n_q,)) of the form
    compiler's ``default`` rule on the triangle, the rule ``assemble`` uses at this degree."""
    r = _RULES.get(int(degree))
    if r is None:
        from ffc.fiatinterface import create_quadrature
        pts, wts = create_quadrature("triangle", int(degree), "default")
        pts, wts = np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)
        r = (np.column_stack([1.0 - pts.sum(axis=1), pts]), wts)
        _RULES[int(degree)] = r
    return r


def _quadrature_degree() -> int:
    """Quadrature degree set in the form compiler parameters.  Raises if none is set."""
    from dolfin import parameters
    q = parameters["form_compiler"]["quadrature_degree"]
    if q is None or int(q) < 0:
        raise RuntimeError("the fluid traction sampler needs the form-compiler quadrature degree pinned "
                           "(microcell/form_compiler.py): it samples at the points of that rule")
    return int(q)


def _cell_geometry(solver):
    """Returns (cells (n_c, 3) vertex indices, Jacobian determinants (n_c,), DG0 dof of each
    cell) of the solver's mesh, cached on the solver."""
    cache = getattr(solver, "_traction_cells", None)
    if cache is not None and cache[0] is solver.mesh_m:
        return cache[1:]
    X, _, _ = _vertex_maps(solver)
    cells = np.asarray(solver.mesh_m.cells(), dtype=np.intp)
    V = X[cells]
    d1, d2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dm0 = solver.Vdg0.dofmap()
    dg0 = np.array([dm0.cell_dofs(c)[0] for c in range(cells.shape[0])], dtype=np.intp)
    solver._traction_cells = (solver.mesh_m, cells, det, dg0)
    return cells, det, dg0


def _cg1_film_parts(solver):
    """CG1 functions (on Vper, non-dimensional) added to the rigid film in ``h_nd``: the ripple
    deformation and the contact deformation, where present."""
    parts = []
    if getattr(solver, "_ripple_on", False) and getattr(solver, "ur_nd_func", None) is not None:
        parts.append(solver.ur_nd_func)
    if getattr(solver, "uc_nd_func", None) is not None:
        parts.append(solver.uc_nd_func)
    return parts


def _film_at_points(solver, lam, cells, film_dofs_m):
    """Film [m] at the quadrature points, as the forms evaluate ``H0 h_nd`` there.

    lam: barycentric coordinates of the points (n_q, 3).  cells: vertex indices (n_c, 3).
    film_dofs_m: as for ``fluid_pressure_samples``.
    Returns (film (n_q, n_c), ``FILM_ANALYTIC`` or ``FILM_CG1``).
    """
    from microcell.film import CosineContactDescriptor
    X, v_per, v_full = _vertex_maps(solver)
    desc = getattr(solver.ht_cb, "contact_cosine", None)
    if isinstance(desc, CosineContactDescriptor):
        Pq = np.einsum("qk,ckd->qcd", lam, X[cells])                    # points, unit-cell coordinates
        rigid = desc.height(Pq[..., 0], Pq[..., 1], solver.settings.xmax, solver.settings.ymax, solver.H0_const,
                            solver.HT_const, solver.T_dim_const, solver._Ux_dim_const, solver._Uy_dim_const)
        nod = np.zeros(X.shape[0])
        for f in _cg1_film_parts(solver):
            nod = nod + f.vector().get_local()[v_per]
        return rigid + lam @ (float(solver.zeta.H) * nod)[cells].T, FILM_ANALYTIC
    if film_dofs_m is None:
        from dolfin import Constant, project
        film_dofs_m = project(Constant(float(solver.zeta.H)) * solver.h_nd, solver.Vfull).vector().get_local()
    Hv = np.asarray(film_dofs_m, dtype=float)[v_full]
    return lam @ Hv[cells].T, FILM_CG1


class FluidTractionSamples(NamedTuple):
    """Sample set of ``fluid_quadrature_sharp_v1``: n = n_q n_c points, with point q of cell c
    at index q n_c + c."""
    t: np.ndarray            # (n, 2) traction (tau_x, tau_y) [Pa], cell frame
    weight: np.ndarray       # (n,) rule weight times Jacobian, in unit-cell area (sum 1)
    fluid: np.ndarray        # (n,) bool: level set at the point > 0
    existing_weight: np.ndarray  # (n,) mask of the solver's ``taust`` integral at the point's cell (DG0 chi, or 1)
    degree: int
    mask_source: str
    film_source: str
    # (n, 2) unit-cell coordinates of the points, in the same order; the contact pressure is
    # sampled at these points
    xy: Optional[np.ndarray] = None


def fluid_traction_samples(solver, film_dofs_m=None, existing_mask=None) -> FluidTractionSamples:
    """Samples of ``fluid_quadrature_sharp_v1`` for the current state of the solver.

    film_dofs_m: as for ``fluid_pressure_samples``.
    existing_mask: DG0 function that weights the solver's ``taust`` integral (``chi_dg0`` of
        the mixed classes); None for the base class, which has no mask.
    Returns a ``FluidTractionSamples``.
    """
    s = fluid_pressure_samples(solver, film_dofs_m)          # nodal level set
    X, v_per, _ = _vertex_maps(solver)
    cells, det, dg0 = _cell_geometry(solver)
    q = _quadrature_degree()
    lam, wts = _rule(q)
    W = wts[:, None] * np.abs(det)[None, :]                   # (n_q, n_c)
    H, film_source = _film_at_points(solver, lam, cells, film_dofs_m)
    # cell gradient of P_nd = P_re + dpx (x - 1/2) + dpy (y - 1/2) + lambda + w in unit-cell
    # coordinates: the P1 gradient of w plus the macroscale gradient
    w = solver.w.vector().get_local()[v_per]
    V = X[cells]
    d1, d2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    f1, f2 = w[cells[:, 1]] - w[cells[:, 0]], w[cells[:, 2]] - w[cells[:, 0]]
    gx = (f1 * d2[:, 1] - f2 * d1[:, 1]) / det + float(solver.dpx_nd_const)
    gy = (d1[:, 0] * f2 - d2[:, 0] * f1) / det + float(solver.dpy_nd_const)
    eta = float(solver.settings.eta0)                            # viscosity [Pa s]
    scale = float(solver.p_scale) * float(solver.invLx)        # grad P [Pa/m] = grad P_nd p_scale / Lx
    us_x = float(solver.Ux_nd_const) * float(solver.Uref)
    us_y = float(solver.Uy_nd_const) * float(solver.Uref)
    tx = eta * us_x / (H + 1e-30) + (H / 2.0) * (gx * scale)[None, :]
    ty = eta * us_y / (H + 1e-30) + (H / 2.0) * (gy * scale)[None, :]
    fluid = (lam @ s.phi[cells].T) > 0.0
    if existing_mask is None:
        m = np.ones_like(W)
    else:
        m = np.broadcast_to(existing_mask.vector().get_local()[dg0][None, :], W.shape)
    xy = np.einsum("qk,ckd->qcd", lam, V).reshape(-1, 2)      # points, in the order of the samples
    return FluidTractionSamples(t=np.column_stack((tx.ravel(), ty.ravel())), weight=W.ravel(),
                                fluid=fluid.ravel(), existing_weight=np.ascontiguousarray(m).ravel(),
                                degree=q, mask_source=s.mask_source, film_source=film_source, xy=xy)


def sample_fluid_traction(solver, film_dofs_m=None, existing_mask=None):
    """Fluid traction statistics of the current state of the solver (one phase), in the cell
    frame.

    film_dofs_m, existing_mask: as for ``fluid_traction_samples``.
    Returns a ``microcell.shear.FluidTractionPhase``.  The sample set is stored on the solver as
    ``fluid_traction_samples``.
    """
    from microcell.shear import SHEAR_SAMPLER, FluidTractionPhase
    from microcell.shear import GLOBAL_X, reduce_traction
    smp = fluid_traction_samples(solver, film_dofs_m, existing_mask)
    # stored for ``sample_contact_traction``, which samples the contact pressure at the same points
    solver.fluid_traction_samples = smp
    valid = np.all(np.isfinite(smp.t), axis=1)
    local = reduce_traction(smp.t, smp.weight, valid, GLOBAL_X, support=smp.fluid)
    total = float(np.sum(smp.weight))
    ew = smp.existing_weight * smp.weight
    sel = ew != 0.0                     # points with weight in the ``taust`` integral (0 in a contact cell)
    existing = tuple(float(v) for v in (ew[sel] @ smp.t[sel]) / total)
    return FluidTractionPhase(mean_local=(float(local.mean[0]), float(local.mean[1])),
                              mean_existing_weights=existing, status=local.status,
                              n_samples=int(local.n_samples), n_support=int(local.n_support),
                              support_fraction=float(local.support_fraction), quadrature_degree=smp.degree,
                              mask_source=smp.mask_source, film_source=smp.film_source,
                              sampler=SHEAR_SAMPLER)


# ---- asperity and total traction at the points of the fluid traction ----
def _local_sliding_direction(solver):
    """``U / (|U| + 1e-30)`` of the sliding velocity of the cell (``Ux_nd_const``,
    ``Uy_nd_const``, the velocity of the Couette term of the fluid traction), in the cell
    frame.  Zero at zero speed."""
    return sliding_direction((float(solver.Ux_nd_const), float(solver.Uy_nd_const), 0.0))


def _grid_matches_mesh(sol, solver) -> bool:
    """True where the contact grid ``sol.p`` is n by n and the mesh has the cell and vertex
    counts of ``UnitSquareMesh(n, n)``."""
    mesh = solver.mesh_m
    p = np.asarray(getattr(sol, "p", None))
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        return False
    n = int(p.shape[0])
    return mesh.num_cells() == 2 * n * n and mesh.num_vertices() == (n + 1) ** 2


def sample_contact_traction(solver):
    """Asperity and total traction statistics of the current phase (contact class).

    Reads ``solver.fluid_traction_samples`` and ``solver.fluid_traction``, set by
    ``sample_fluid_traction``, the stored contact solution ``solver._contact_solution`` and
    ``settings.contact.mu``.  The contact solution is used only if its state key is current; it
    is not refreshed here, as that would change the mask of the flow solve.
    Returns a ``microcell.shear.ContactTractionPhase`` in the cell frame.  Its status is
    ``state_invalid``, ``check_failed`` or ``unavailable`` where the solution is not current, the
    grid does not match the mesh, or ``mu`` is not a finite non-negative number.
    """
    from microcell import shear as cs
    from microcell.shear import GLOBAL_X, VALID_STATUSES as _FLUID_OK
    smp = getattr(solver, "fluid_traction_samples", None)
    fl = getattr(solver, "fluid_traction", None)
    if smp is None or fl is None or smp.xy is None:
        raise RuntimeError("sample_contact_traction needs this phase's fluid samples (sample_fluid_traction "
                           "in spatial_homogenisation)")
    model = cs.FFT_MODEL                                        # the only contact law
    n = int(smp.weight.size)
    u_hat = _local_sliding_direction(solver)
    fluid_mean = tuple(float(v) for v in fl.mean_local)
    fluid_ok = fl.status in _FLUID_OK
    sol = None
    grid_ok, check = True, float("nan")
    if model == cs.FFT_MODEL:
        sol = getattr(solver, "_contact_solution", None)
        current = sol is not None and getattr(solver, "_contact_key", None) == solver._contact_state_key()
        if not current:
            return cs.invalid_phase(cs.STATE_INVALID, n, model, u_hat, fluid_mean)
        grid_ok = _grid_matches_mesh(sol, solver)
        if not grid_ok:
            return cs.invalid_phase(cs.CHECK_FAILED, n, model, u_hat, fluid_mean, solution=sol)
        contact = sample_contact(sol, smp.xy, contact_model=model)
        if contact.valid:
            check = cs.mean_check(contact.p, smp.weight, contact.Pasp_hom)
    else:
        raise ValueError(f"unknown contact model {model!r}")
    mu = float(solver.settings.contact.mu)
    if not (np.isfinite(mu) and mu >= 0.0):
        # no usable friction coefficient: the statistics are reported as unavailable
        return cs.invalid_phase(cs.UNAVAILABLE, n, model, u_hat, fluid_mean, solution=sol)
    if not contact.valid:
        return cs.invalid_phase(cs.STATE_INVALID, n, model, u_hat, fluid_mean, solution=sol)
    fv = np.all(np.isfinite(smp.t), axis=1)
    loc = reduce_contact_and_total(smp.t, smp.fluid, fv, contact, mu, u_hat, smp.weight,
                                   direction=GLOBAL_X, fluid_state_valid=fluid_ok)
    return cs.phase_from_reductions(loc, contact_status=contact.status, contact_model=model,
                                    fluid_state_valid=fluid_ok, mean_check_rel=check, u_hat=u_hat,
                                    fluid_mean=fluid_mean, solution=sol, grid_ok=grid_ok)


def finish_contact_traction(solver, asperity_load: dict):
    """Rechecks the contact traction after ``compute_asperity_load`` of the same phase.

    Returns ``solver.contact_traction``, marked ``state_mismatch`` where the stored contact
    solution was replaced after the samples were taken, or None where the solver has none.
    """
    from microcell import shear as cs
    ph = getattr(solver, "contact_traction", None)
    if ph is None:
        return None
    if ph.solution is not None and getattr(solver, "_contact_solution", None) is not ph.solution:
        return cs.mismatch(ph)
    return ph



from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from microcell.shear import GLOBAL_X, reduce_traction   # numpy only

CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"

from microcell.contact import FFT as FFT_MODEL      # the only contact law

STATUS_OK = "ok"
STATUS_NO_SUPPORT = "no_support"
STATUS_STATE_INVALID = "state_invalid"
CONTACT_VALID_STATUSES = (STATUS_OK, STATUS_NO_SUPPORT)

STATISTIC_NATIVE = "native_nodal"
STATISTIC_INTERPOLATED = "p1_interpolated"

#: a coordinate within this distance of a node, in grid units, is moved to the node
SNAP_TOL = 1e-9
#: regularisation of the sliding direction U / (|U| + eps), in the units of the velocity
ZERO_SPEED_EPS = 1e-30


@dataclass(frozen=True)
class ContactSamples:
    """Contact pressure and supports at N samples."""
    p: np.ndarray                 # (N,) contact pressure [Pa]; NaN unless the status is ok or no_support
    mech_support: np.ndarray      # (N,) bool: p > 0
    obstruction: np.ndarray       # (N,) bool: phi_c <= 0, the contact mask of the flow solve
    at_node: np.ndarray           # (N,) bool: the sample is a grid node (after snapping)
    status: str
    contact_model: str
    grid: tuple                   # (nx, ny) of the solution; (0, 0) when there is none
    native_peak: float            # max(sol.p) [Pa]
    Pasp_hom: float               # mean(sol.p) [Pa]
    statistic: str                # native_nodal | p1_interpolated
    sampler: str = CONTACT_SAMPLER

    @property
    def valid(self) -> bool:
        return self.status in CONTACT_VALID_STATUSES

    def summary(self) -> dict:
        """Counts, peaks [Pa] and labels as plain numbers and strings, without arrays."""
        finite = np.isfinite(self.p)
        return {"status": self.status, "contact_model": self.contact_model, "grid": list(self.grid),
                "n_samples": int(self.p.size), "n_mech_support": int(np.count_nonzero(self.mech_support)),
                "n_obstruction": int(np.count_nonzero(self.obstruction)),
                "n_at_node": int(np.count_nonzero(self.at_node)),
                "sample_peak_Pa": float(np.max(self.p)) if finite.any() else float("nan"),
                "native_peak_Pa": float(self.native_peak), "Pasp_hom_Pa": float(self.Pasp_hom),
                "statistic": self.statistic, "sampler": self.sampler}


# ---- periodic P1 interpolant ----
def _grid_coords(xy, shape, cell_size):
    """Sample coordinates in grid units, snapped to nodes within ``SNAP_TOL``.

    Returns (gx, gy, at_node), each (N,).
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"sample coordinates must have shape (N, 2), got {xy.shape}")
    if not np.all(np.isfinite(xy)):
        raise ValueError("sample coordinates must be finite")
    u = xy.copy()
    if cell_size is not None:
        lx, ly = (float(v) for v in cell_size)
        if not (np.isfinite(lx) and np.isfinite(ly) and lx > 0 and ly > 0):
            raise ValueError(f"cell_size must be two positive lengths, got {cell_size!r}")
        u[:, 0] /= lx
        u[:, 1] /= ly
    nx, ny = shape
    gx, gy = u[:, 0] * nx, u[:, 1] * ny
    rx, ry = np.rint(gx), np.rint(gy)
    sx, sy = np.abs(gx - rx) <= SNAP_TOL, np.abs(gy - ry) <= SNAP_TOL
    gx, gy = np.where(sx, rx, gx), np.where(sy, ry, gy)
    return gx, gy, sx & sy


def periodic_p1(values, xy, *, cell_size=None):
    """Periodic P1 interpolant of nodal values at sample points.

    values: (nx, ny) array ``[i, j]``, node ``(i, j)`` at ``(i/nx, j/ny)`` of the cell.
    xy: (N, 2) points, in fractions of the cell, or in metres with ``cell_size = (Lx, Ly)``.
    Each grid square is split along its 'right' diagonal, with offsets ``(s, t)`` in the square:
        s >= t:  (1-s) p00 + (s-t) p10 + t p11;     t > s:  (1-t) p00 + (t-s) p01 + s p11.
    Returns (v (N,), at_node (N,) bool).
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 2:
        raise ValueError(f"nodal values must be an nx x ny grid with nx, ny >= 2, got shape {v.shape}")
    nx, ny = v.shape
    gx, gy, at_node = _grid_coords(xy, v.shape, cell_size)
    fx, fy = np.floor(gx), np.floor(gy)
    s, t = gx - fx, gy - fy
    i0 = fx.astype(np.int64) % nx
    j0 = fy.astype(np.int64) % ny
    i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
    p00, p11 = v[i0, j0], v[i1, j1]
    lower = s >= t
    out = np.where(lower,
                   (1.0 - s) * p00 + (s - t) * v[i1, j0] + t * p11,
                   (1.0 - t) * p00 + (t - s) * v[i0, j1] + s * p11)
    out = np.where(at_node, p00, out)            # nodal value at a node, without arithmetic
    return out, at_node


# ---- contact pressure sampler ----
def _empty(n, status, model, grid=(0, 0)):
    """``ContactSamples`` of n samples with NaN pressure, empty supports and the given status."""
    nan = np.full(n, np.nan)
    no = np.zeros(n, dtype=bool)
    return ContactSamples(p=nan, mech_support=no, obstruction=no.copy(), at_node=no.copy(), status=status,
                          contact_model=str(model), grid=tuple(int(g) for g in grid),
                          native_peak=float("nan"), Pasp_hom=float("nan"), statistic=STATISTIC_INTERPOLATED)


def sample_contact(solution: Optional[object], xy, *, contact_model: str = FFT_MODEL, cell_size=None) -> ContactSamples:
    """Contact pressure ``p_a`` [Pa] and the supports at the samples ``xy``.

    solution: the stored contact solution of the solver (an object with ``p`` and ``phi_c``
        grids, such as ``microcell.contact.ContactSolution``), or None.
    xy, cell_size: as for ``periodic_p1``.
    contact_model: ``fft_elastic_v1``; any other value raises.
    Returns a ``ContactSamples``; its status is ``state_invalid`` where the solution is None or
    its grids are not finite, non-negative and of one shape.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"sample coordinates must have shape (N, 2), got {xy.shape}")
    n = xy.shape[0]
    model = str(contact_model)
    if model != FFT_MODEL:
        raise ValueError(f"unknown contact model {model!r}")
    if solution is None:
        return _empty(n, STATUS_STATE_INVALID, model)
    p_grid = np.asarray(getattr(solution, "p", None), dtype=float)
    phi_grid = np.asarray(getattr(solution, "phi_c", None), dtype=float)
    if (p_grid.ndim != 2 or phi_grid.shape != p_grid.shape or min(p_grid.shape) < 2
            or not np.all(np.isfinite(p_grid)) or not np.all(np.isfinite(phi_grid)) or np.any(p_grid < 0.0)):
        return _empty(n, STATUS_STATE_INVALID, model, p_grid.shape if p_grid.ndim == 2 else (0, 0))
    p, at_node = periodic_p1(p_grid, xy, cell_size=cell_size)
    phi, _ = periodic_p1(phi_grid, xy, cell_size=cell_size)
    peak = float(np.max(p_grid))
    status = STATUS_OK if peak > 0.0 else STATUS_NO_SUPPORT
    return ContactSamples(p=p, mech_support=p > 0.0, obstruction=phi <= 0.0, at_node=at_node, status=status,
                          contact_model=model, grid=tuple(int(g) for g in p_grid.shape), native_peak=peak,
                          Pasp_hom=float(p_grid.mean()),
                          statistic=STATISTIC_NATIVE if n and bool(np.all(at_node)) else STATISTIC_INTERPOLATED)


# ---- vertex weights ----
def p1_lumped_weights(coords, cells) -> np.ndarray:
    """Lumped P1 weights of the vertices of a triangle mesh.

    coords: (nv, 2) vertex coordinates.  cells: (nc, 3) vertex indices of the triangles.
    Each vertex receives a third of the area of every triangle it belongs to, which integrates
    a function that is linear on each triangle without error.
    Returns (nv,) weights that sum to the mesh area.
    """
    X = np.asarray(coords, dtype=float)
    C = np.asarray(cells, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 2 or C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("p1_lumped_weights needs (nv, 2) coordinates and (nc, 3) triangles")
    a, b, c = X[C[:, 0]], X[C[:, 1]], X[C[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    w = np.zeros(X.shape[0])
    for k in range(3):
        np.add.at(w, C[:, k], area / 3.0)
    return w


# ---- asperity and total traction ----
def sliding_direction(u, eps: float = ZERO_SPEED_EPS) -> np.ndarray:
    """Direction of the asperity traction, ``U / (|U| + eps)``: zero at zero speed.

    u: velocity, (3,) or (N, 3).  Returns an array of the same shape.
    """
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 3:
        raise ValueError(f"velocity must have 3 components, got shape {u.shape}")
    mag = np.sqrt(np.sum(u * u, axis=-1, keepdims=True))
    return u / (mag + float(eps))


def asperity_traction(p, mu_material: float, u_hat) -> np.ndarray:
    """Asperity traction ``t_a = mu_material p_a u_hat``.

    p: contact pressure at N samples [Pa].
    mu_material: friction coefficient, finite and >= 0.
    u_hat: direction from ``sliding_direction``, (3,) for the cell or (N, 3) per sample.
    Returns (N, 3) [Pa].
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    d = np.asarray(u_hat, dtype=float)
    mu = float(mu_material)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu_material must be finite and >= 0, got {mu_material!r}")
    if d.shape == (3,):
        d = np.broadcast_to(d, (p.size, 3))
    if d.shape != (p.size, 3):
        raise ValueError(f"u_hat must have shape (3,) or ({p.size}, 3), got {d.shape}")
    return mu * p[:, None] * d


def _pad3(v, n, what):
    """``v`` as (n, 3): a zero third component is added to an (n, 2) array."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] != n or v.shape[1] not in (2, 3):
        raise ValueError(f"{what} must have shape ({n}, 2) or ({n}, 3), got {v.shape}")
    return v if v.shape[1] == 3 else np.column_stack((v, np.zeros(n)))


def total_traction(t_fluid, fluid_support, t_asp, mech_support) -> np.ndarray:
    """Total traction at the common samples, ``t_t = t_f [fluid] + t_a [mech]``.

    t_fluid, t_asp: (N, 2) or (N, 3) [Pa].  fluid_support, mech_support: (N,) bool.
    Each contribution is zero outside its support, and both are added where the supports overlap.
    Returns (N, 3) [Pa].
    """
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    ms = np.asarray(mech_support, dtype=bool).reshape(-1)
    n = fs.size
    if ms.shape != (n,):
        raise ValueError("fluid and mechanical supports must be on the same samples")
    tf, ta = _pad3(t_fluid, n, "t_fluid"), _pad3(t_asp, n, "t_asp")
    return np.where(fs[:, None], tf, 0.0) + np.where(ms[:, None], ta, 0.0)


def support_report(fluid_support, contact: ContactSamples, weights) -> dict:
    """Agreement of the fluid set, the obstruction set and the mechanical support.

    fluid_support: (N,) bool.  contact: the ``ContactSamples`` at the same points.  weights: (N,).
    Returns a dict of ``{"count", "fraction"}`` per set (fraction of the total sample weight), with
    ``contact_force_share_on_fluid_samples`` and ``status``.  A report only: nothing is corrected.
    """
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = fs.size
    if w.shape != (n,) or contact.p.shape != (n,):
        raise ValueError("fluid support, weights and contact samples must have the same length")
    ms, ob = contact.mech_support, contact.obstruction
    W = float(np.sum(w)) if n else float("nan")

    def frac(m):
        return float(np.sum(w[m]) / W) if n else float("nan")

    sets = {
        "fluid": fs, "mech_support": ms, "obstruction": ob,
        "fluid_and_mech": fs & ms,                          # both tractions present
        "mech_outside_obstruction": ms & ~ob,               # contact pressure outside the obstruction
        "obstruction_without_pressure": ob & ~ms,           # obstructed, no contact traction
        "neither": ~fs & ~ms,                               # no traction
        "fluid_not_complement_of_obstruction": fs == ob,    # fluid set and obstruction set coincide
    }
    out = {k: {"count": int(np.count_nonzero(m)), "fraction": frac(m)} for k, m in sets.items()}
    if contact.valid and np.all(np.isfinite(contact.p)):
        f_all = float(np.sum(w * contact.p))
        f_fluid = float(np.sum(w[fs & ms] * contact.p[fs & ms]))
        out["contact_force_share_on_fluid_samples"] = (f_fluid / f_all) if f_all > 0.0 else 0.0
    else:
        out["contact_force_share_on_fluid_samples"] = float("nan")
    out["status"] = contact.status
    return out


@dataclass(frozen=True)
class ContactTotalReduction:
    """Fluid, asperity and total traction of one state of the cell, reduced on one sample set."""
    fluid: object                  # microcell.shear.ShearReduction
    asperity: object               # microcell.shear.ShearReduction
    total: object                  # microcell.shear.ShearReduction
    #: ``support_report`` and ``ContactSamples.summary`` of the same samples
    support: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)


def reduce_contact_and_total(t_fluid, fluid_support, fluid_valid, contact: ContactSamples, mu_material: float,
                             u_hat, weights, *, direction=GLOBAL_X,
                             fluid_state_valid: bool = True) -> ContactTotalReduction:
    """Fluid, asperity and total traction on the same samples, each reduced by
    ``microcell.shear.reduce_traction``.

    t_fluid: fluid traction, (N, 2) or (N, 3) [Pa], cell frame.
    fluid_support, fluid_valid: (N,) bool, the fluid samples and the samples that were evaluated.
    contact: ``ContactSamples`` at the same points.  mu_material: friction coefficient.
    u_hat: asperity direction in the same frame, (3,) or (N, 3), from ``sliding_direction``.
    weights: (N,) sample weights.  direction: unit vector the samples are projected on.
    The total is formed sample by sample and then reduced.  An invalid contact state leaves the
    fluid reduction unchanged and makes the asperity and total reductions invalid.
    Returns a ``ContactTotalReduction``.
    """
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    fv = np.asarray(fluid_valid, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = fs.size
    tf = _pad3(t_fluid, n, "t_fluid")
    if fv.shape != (n,) or w.shape != (n,) or contact.p.shape != (n,):
        raise ValueError("fluid arrays, weights and contact samples must have the same length")
    fluid = reduce_traction(tf, w, fv, direction, support=fs, state_valid=fluid_state_valid)

    c_ok = contact.valid
    p = np.where(np.isfinite(contact.p), contact.p, 0.0) if c_ok else np.zeros(n)
    ta = asperity_traction(p, mu_material, u_hat)
    a_valid = np.isfinite(contact.p) if c_ok else np.zeros(n, dtype=bool)
    asperity = reduce_traction(ta, w, a_valid, direction, support=contact.mech_support,
                               state_valid=c_ok)

    ms = contact.mech_support
    tt = total_traction(tf, fs, ta, ms)
    t_valid = (fv | ~fs) & (a_valid | ~ms)
    total = reduce_traction(tt, w, t_valid, direction, support=fs | ms,
                            state_valid=bool(fluid_state_valid and c_ok))
    return ContactTotalReduction(fluid=fluid, asperity=asperity, total=total,
                                 support=support_report(fs, contact, w), contact=contact.summary())
