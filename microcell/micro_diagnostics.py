"""Diagnostics of a solved micro cell, shared by every phase-neutral solver class.

[extrema M1, 23 Oct 2026] The fluid pressure extrema (decision D1 of
docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md, section 2.3; the
definition and its status vocabulary are in ``ehl/pressure.py``).  The base, mixed and
contact ``spatial_homogenisation()`` all call :func:`sample_fluid_pressure_extrema`; none
builds a mask of its own for the pressure extrema any more.

Sampler ``fluid_nodal_sharp_v1``:

* samples: every vertex of the micro mesh.  On the periodic cell the vertices at x = 0
  and x = 1 (y = 0 and y = 1) are distinct physical points: the film and the mask are
  periodic, the pressure is not (it carries the macro gradient), so both are sampled.
* pressure: the signed physical pressure ``P_dim`` [Pa] at the vertex, from its parts,
  ``p_scale * (P_re + dpx (x - 1/2) + dpy (y - 1/2) + lambda + w(vertex))``.  Every part is
  linear on a cell or a CG1 nodal value, so this is ``P_dim`` itself at the vertex: no
  projection, no clipping, no asperity or fictitious contact pressure added.
* fluid: the vertex's nodal value of the obstruction level set is > 0 -- the sharp test
  ``chi_fluid = conditional(gt(phi, 0), 1, 0)`` applied to nodal values:
    ``phic_nd_func`` (``fft_elastic_v1``: the complementarity level set, a CG1 Function on
    the pressure's own periodic space) -> ``mask_source = "phic_nd_func"``;
    otherwise ``phi = h_nd - h_min_nd``, an expression: its CG1 projection
    (the caller's projection of ``H0 h_nd`` when it has one) -> ``"projected_h"``.
* no fluid vertex: NaN, phase status ``no_fluid`` -- never the unmasked values, never zero.

Cost: index arithmetic on arrays the solver already holds (plus one CG1 projection of
the film when the caller does not pass one).  The vertex maps are cached on the solver.

[extrema M2b, 23 Oct 2026] The fluid traction (the contract and its definition are in
``ehl/fluid_shear.py``; the reduction is M2a's ``ehl/shear_reduce.py``).  The base, mixed and
contact ``spatial_homogenisation()`` all call :func:`sample_fluid_traction`.

Sampler ``fluid_quadrature_sharp_v1`` (Rob's choice of 23 Oct 2026 over the M1 vertices):

* samples: the points and weights of the form compiler's ``default`` rule of the quadrature
  degree in force on every cell -- the rule ``assemble`` uses for ``taust`` (M1's
  :func:`quadrature_audit` uses the same rule);
* traction: the existing micro shear expression at each point, ``eta U_s / (H + 1e-30) +
  (H / 2) grad P``, local frame, Pa; ``grad P`` the cell's gradient of the CG1 pressure, the
  film as the form evaluates it (an analytic film by its own formula; otherwise the CG1
  projection the class made, exact for a CG1 film);
* fluid: M1's nodal level set interpolated linearly to the point, > 0 (the ``chi_fluid`` test
  there); the traction is zero outside the fluid (t_f = chi_f tau_f);
* reduced by ``reduce_traction``: the local mean over the whole set and, with the task's frame
  (``solver.shear_frame``, the macro's ``R_inv``), the global-x envelope.  The same samples
  weighted by the class's own mask of ``taust`` (DG0 ``chi`` / none) reproduce ``taust``: that
  is the check that the sampler evaluates the existing expression.

Cost: numpy on the cells and the rule's points (the cell geometry cached on the solver).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from microcell.pressure import (FluidPressureExtrema, PRESSURE_EXTREMA_SAMPLER, PHASE_NO_FLUID,
                          PHASE_NONFINITE, PHASE_OK)

__all__ = ("FluidPressureExtrema", "FluidPressureSamples", "FluidTractionSamples", "MASK_CONTACT",
           "MASK_FILM", "fluid_pressure_samples", "fluid_traction_samples", "quadrature_audit",
           "sample_fluid_pressure_extrema", "sample_fluid_traction")

MASK_CONTACT = "phic_nd_func"
MASK_FILM = "projected_h"


class FluidPressureSamples(NamedTuple):
    """The sample set of ``fluid_nodal_sharp_v1``, in mesh vertex order."""
    X: np.ndarray          # (n_v, 2) unit-cell coordinates of the vertices
    p: np.ndarray          # (n_v,) signed P_dim [Pa]
    phi: np.ndarray        # (n_v,) nodal obstruction level set (nondimensional)
    fluid: np.ndarray      # (n_v,) bool: phi > 0
    mask_source: str


def _vertex_maps(solver):
    """(X, vertex -> Vper dof, vertex -> Vfull dof), cached on the solver (its mesh and
    spaces live as long as it does; a reused solver keeps them, ``reset_task``)."""
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
    """Every sample of ``fluid_nodal_sharp_v1`` for the solver's current state.

    ``film_dofs_m``: the caller's CG1 projection of ``H0 h_nd`` [m] on ``solver.Vfull``
    (dof order), reused for the ``projected_h`` mask; projected here when None.  Unused
    for ``phic_nd_func``.
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
    # [fluid-mean WP2, plan rev 3 (m2)] under the fluid-region level constraint the vertices whose basis
    # functions hold no fluid quadrature point (m_f,i = 0, pinned) carry no fluid pressure: they are left out of
    # the samples, whatever this sampler's own fluid test says (the m_f = 0 mask itself, not the ripple's nodal
    # projection; and on the no-fluid path every vertex; code review, 30 Oct).  Off the samples are as before.
    mf_zero = getattr(solver, "_lc_mf_zero", None)
    if mf_zero is not None and getattr(solver, "_lc_path", "whole_cell") in ("fluid_mean", "no_fluid"):
        excl = np.asarray(mf_zero, dtype=bool)[v_per]
        if excl.any():
            fluid = fluid & ~excl
    return FluidPressureSamples(X=X, p=p, phi=phi, fluid=fluid, mask_source=source)


def quadrature_audit(solver, degree: int, film_dofs_m=None) -> dict:
    """The ledger's fluid pressure integrals, point by point (plan 6.1.4; a diagnostic, never
    called by the solver).

    At the points and weights of the form compiler's ``degree`` rule on every cell
    (``ffc.fiatinterface.create_quadrature(..., "default")``, the rule ``assemble`` uses):
    ``P`` and the level set ``phi`` interpolated linearly from this module's vertex values
    -- exactly what ``assemble`` evaluates when both are CG1 on each cell (the contact level
    set; a film that is CG1 on the mesh) -- and ``chi = phi > 0``.  Returns ``P_f``,
    ``P_fluid_pos``, ``P_fluid_neg`` per total cell area [Pa] (at the ledger's degree they
    reproduce ``pressure_ledger``), the point count, the fluid points and the fluid points
    with ``P < 0``, and the lowest fluid-point pressure with its location.
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
    """The fluid pressure extrema of the solver's current state (one phase); see the
    module docstring.  Status ``ok`` / ``no_fluid`` / ``nonfinite`` (a fluid sample that
    is not finite: NaN, never a partial extremum)."""
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


# ======================================================================================
# [extrema M2b, 23 Oct 2026] the fluid traction, sampler fluid_quadrature_sharp_v1
# ======================================================================================
FILM_ANALYTIC = "analytic"          # the callback's own formula at the points (make_ideal_ht)
FILM_CG1 = "cg1_projection"         # the class's CG1 projection of H0 h_nd, interpolated linearly

_RULES: dict = {}


def _rule(degree: int):
    """(barycentric coordinates (n_q, 3), reference weights (n_q,)) of the form compiler's
    ``default`` rule on the triangle -- the rule ``assemble`` uses at this degree."""
    r = _RULES.get(int(degree))
    if r is None:
        from ffc.fiatinterface import create_quadrature
        pts, wts = create_quadrature("triangle", int(degree), "default")
        pts, wts = np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)
        r = (np.column_stack([1.0 - pts.sum(axis=1), pts]), wts)
        _RULES[int(degree)] = r
    return r


def _quadrature_degree() -> int:
    """The form compiler's quadrature degree in force: the one ``taust``'s ``assemble`` used."""
    from dolfin import parameters
    q = parameters["form_compiler"]["quadrature_degree"]
    if q is None or int(q) < 0:
        raise RuntimeError("the fluid traction sampler needs the form-compiler quadrature degree pinned "
                           "(utils/form_compiler.py): it samples at the points of that rule")
    return int(q)


def _cell_geometry(solver):
    """(cells (n_c, 3) vertex indices, Jacobian determinants (n_c,), DG0 dof of each cell),
    cached on the solver for its mesh (as M1's vertex maps)."""
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
    """The CG1 Functions (on Vper, nondimensional) in the film sum ``h_nd`` besides the callback's
    rigid film: the option-H ripple, the fft contact deformation
    (``micro_phase_neutral_class.py`` _setup_film_and_mask; ``_contact.py``)."""
    parts = []
    if getattr(solver, "_ripple_on", False) and getattr(solver, "ur_nd_func", None) is not None:
        parts.append(solver.ur_nd_func)
    if getattr(solver, "uc_nd_func", None) is not None:
        parts.append(solver.uc_nd_func)
    return parts


def _film_at_points(solver, lam, cells, film_dofs_m):
    """(n_q, n_c) dimensional film [m] at the rule's points, as the form evaluates
    ``Constant(H0) * h_nd`` there, and its source label."""
    from microcell.ideal_ht import CosineContactDescriptor
    X, v_per, v_full = _vertex_maps(solver)
    desc = getattr(solver.ht_cb, "contact_cosine", None)
    if isinstance(desc, CosineContactDescriptor):
        Pq = np.einsum("qk,ckd->qcd", lam, X[cells])                    # the points, unit cell
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
    """The sample set of ``fluid_quadrature_sharp_v1``: n = n_q * n_c points, cell-major blocks of the
    rule (point q of cell c at index q * n_c + c)."""
    t: np.ndarray            # (n, 2) local traction [Pa]: (tau_x, tau_y) of the existing expression
    weight: np.ndarray       # (n,) rule weight x Jacobian, unit-cell area (sum = cell area = 1)
    fluid: np.ndarray        # (n,) bool: M1's level set at the point > 0
    existing_weight: np.ndarray  # (n,) the class's own mask of taust at the point's cell (DG0 chi / 1)
    degree: int
    mask_source: str
    film_source: str
    # [extrema M3b] the points themselves, unit-cell coordinates (n, 2), in the same order: the common
    # sample set of the asperity / total traction (the contact sampler evaluates p_a there)
    xy: Optional[np.ndarray] = None


def fluid_traction_samples(solver, film_dofs_m=None, existing_mask=None) -> FluidTractionSamples:
    """Every sample of ``fluid_quadrature_sharp_v1`` for the solver's current state.

    ``film_dofs_m``: the caller's CG1 projection of ``H0 h_nd`` on ``Vfull`` (dof order), as for
    :func:`fluid_pressure_samples`.  ``existing_mask``: the DG0 Function the class weights ``taust``
    with (the mixed classes' ``chi_dg0``), None for the unmasked base class.
    """
    s = fluid_pressure_samples(solver, film_dofs_m)          # M1's nodal level set and vertex maps
    X, v_per, _ = _vertex_maps(solver)
    cells, det, dg0 = _cell_geometry(solver)
    q = _quadrature_degree()
    lam, wts = _rule(q)
    W = wts[:, None] * np.abs(det)[None, :]                   # (n_q, n_c)
    H, film_source = _film_at_points(solver, lam, cells, film_dofs_m)
    # the cell's gradient of P_nd = P_re + dpx (x - 1/2) + dpy (y - 1/2) + lambda + w (unit-cell
    # coordinates), as grad_cell(P_nd): the macro ramp exactly, w's P1 gradient from its dofs
    w = solver.w.vector().get_local()[v_per]
    V = X[cells]
    d1, d2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    f1, f2 = w[cells[:, 1]] - w[cells[:, 0]], w[cells[:, 2]] - w[cells[:, 0]]
    gx = (f1 * d2[:, 1] - f2 * d1[:, 1]) / det + float(solver.dpx_nd_const)
    gy = (d1[:, 0] * f2 - d2[:, 0] * f1) / det + float(solver.dpy_nd_const)
    eta = float(solver.settings.eta0)                            # constant lubricant viscosity [Pa s]
    scale = float(solver.p_scale) * float(solver.invLx)        # gradP_dim = gradP_nd * p_scale / Lx (x and y)
    us_x = float(solver.Ux_nd_const) * float(solver.Uref)
    us_y = float(solver.Uy_nd_const) * float(solver.Uref)
    tx = eta * us_x / (H + 1e-30) + (H / 2.0) * (gx * scale)[None, :]
    ty = eta * us_y / (H + 1e-30) + (H / 2.0) * (gy * scale)[None, :]
    fluid = (lam @ s.phi[cells].T) > 0.0
    if existing_mask is None:
        m = np.ones_like(W)
    else:
        m = np.broadcast_to(existing_mask.vector().get_local()[dg0][None, :], W.shape)
    xy = np.einsum("qk,ckd->qcd", lam, V).reshape(-1, 2)      # [extrema M3b] the points, same order
    return FluidTractionSamples(t=np.column_stack((tx.ravel(), ty.ravel())), weight=W.ravel(),
                                fluid=fluid.ravel(), existing_weight=np.ascontiguousarray(m).ravel(),
                                degree=q, mask_source=s.mask_source, film_source=film_source, xy=xy)


def _frame(frame):
    """The task's 3 x 3 ``R_inv`` (local -> global) or None (absent, malformed or not finite)."""
    if frame is None:
        return None
    r = np.asarray(frame, dtype=float)
    return r if r.shape == (3, 3) and np.all(np.isfinite(r)) else None


def sample_fluid_traction(solver, film_dofs_m=None, existing_mask=None, frame=None):
    """The fluid traction statistics of the solver's current state (one phase); see the module
    docstring and ``ehl/fluid_shear.py``.  ``frame``: the task's ``R_inv`` (3 x 3, the macro's
    convention, applied as given); None gives the local mean only (status ``no_frame``)."""
    from microcell.fluid_shear import SHEAR_NO_FRAME, SHEAR_SAMPLER, FluidTractionPhase
    from microcell.shear_reduce import GLOBAL_X, VALID_STATUSES, reduce_traction
    smp = fluid_traction_samples(solver, film_dofs_m, existing_mask)
    # [extrema M3b] the phase's sample set, kept on the solver for the asperity / total traction of the
    # contact class (sample_contact_traction): one common set, not a second sampling
    solver.fluid_traction_samples = smp
    valid = np.all(np.isfinite(smp.t), axis=1)
    local = reduce_traction(smp.t, smp.weight, valid, GLOBAL_X, support=smp.fluid)
    total = float(np.sum(smp.weight))
    ew = smp.existing_weight * smp.weight
    sel = ew != 0.0                     # the points the existing integral weights (a contact cell's are 0)
    existing = tuple(float(v) for v in (ew[sel] @ smp.t[sel]) / total)
    r = _frame(frame)
    glob = (reduce_traction(smp.t, smp.weight, valid, GLOBAL_X, support=smp.fluid, rotation=r)
            if r is not None else None)
    if local.status not in VALID_STATUSES:
        status = local.status
    elif glob is None:
        status = SHEAR_NO_FRAME
    else:
        status = glob.status
    return FluidTractionPhase(mean_local=(float(local.mean[0]), float(local.mean[1])),
                              mean_existing_weights=existing, reduction=glob, status=status,
                              n_samples=int(local.n_samples), n_support=int(local.n_support),
                              support_fraction=float(local.support_fraction), quadrature_degree=smp.degree,
                              mask_source=smp.mask_source, film_source=smp.film_source,
                              sampler=SHEAR_SAMPLER)


# ======================================================================================
# [extrema M3b, 23 Oct 2026] the asperity and total traction on the fluid sampler's points
# (definition and statuses: ehl/contact_shear.py; sampler and algebra: M3a's contact_sample.py)
# ======================================================================================
def _local_sliding_direction(solver):
    """``U / (|U| + 1e-30)`` of the micro's own sliding velocity at the end of its run, local frame
    (``Ux_nd_const``, ``Uy_nd_const``: the velocity of the Couette term of the fluid traction, so the
    asperity traction has its sign; the macro's law ``calc_asp_friction``, zero at zero speed)."""
    from microcell.contact_sample import sliding_direction
    return sliding_direction((float(solver.Ux_nd_const), float(solver.Uy_nd_const), 0.0))


def _grid_matches_mesh(sol, solver) -> bool:
    """The contact grid is the micro mesh's structured n x n grid (the P1 interpolant's triangulation is
    the mesh's own: ``UnitSquareMesh(n, n)``, checked by ``_periodic_grid`` when the solution was made)."""
    mesh = solver.mesh_m
    p = np.asarray(getattr(sol, "p", None))
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        return False
    n = int(p.shape[0])
    return mesh.num_cells() == 2 * n * n and mesh.num_vertices() == (n + 1) ** 2


def sample_contact_traction(solver):
    """The asperity and total traction statistics of the solver's current phase (contact class, called
    at the end of ``spatial_homogenisation``, on the sample set the fluid sampler of that call kept):

    * ``fft_elastic_v1``: ``p_a`` of the stored solution if its state key is current (no refresh here:
      it would change the mask the fluid was solved with), checked against the grid and the solution's
      own ``Pasp_hom`` (``ehl.contact_shear.MEAN_CHECK_BUDGET``), then M3a's ``reduce_contact_and_total``
      in the local frame (the means) and, with the task's ``R_inv``, along global x (the envelopes).
    """
    from microcell import contact_shear as cs
    from microcell.shear_reduce import GLOBAL_X, VALID_STATUSES as _FLUID_OK
    from microcell.contact_sample import reduce_contact_and_total, sample_contact
    smp = getattr(solver, "fluid_traction_samples", None)
    fl = getattr(solver, "fluid_traction", None)
    if smp is None or fl is None or smp.xy is None:
        raise RuntimeError("sample_contact_traction needs this phase's fluid samples (sample_fluid_traction "
                           "in spatial_homogenisation)")
    model = cs.FFT_MODEL                                        # the one contact law
    n = int(smp.weight.size)
    u_hat = _local_sliding_direction(solver)
    fluid_mean = tuple(float(v) for v in fl.mean_local)
    fluid_ok = fl.status in _FLUID_OK + ("no_frame",)
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
        # a caller that gave no friction coefficient (direct callers of the class, older settings):
        # nothing is published, nothing raised -- the statistics are absent, not wrong
        return cs.invalid_phase(cs.UNAVAILABLE, n, model, u_hat, fluid_mean, solution=sol)
    if not contact.valid:
        return cs.invalid_phase(cs.STATE_INVALID, n, model, u_hat, fluid_mean, solution=sol)
    fv = np.all(np.isfinite(smp.t), axis=1)
    frame = _frame(getattr(solver, "shear_frame", None))
    loc = reduce_contact_and_total(smp.t, smp.fluid, fv, contact, mu, u_hat, smp.weight, rotation=None,
                                   direction=GLOBAL_X, fluid_state_valid=fluid_ok)
    glob = (reduce_contact_and_total(smp.t, smp.fluid, fv, contact, mu, u_hat, smp.weight, rotation=frame,
                                     direction=GLOBAL_X, fluid_state_valid=fluid_ok)
            if frame is not None else None)
    return cs.phase_from_reductions(loc, glob, contact_status=contact.status, contact_model=model,
                                    fluid_state_valid=fluid_ok, mean_check_rel=check, u_hat=u_hat,
                                    fluid_mean=fluid_mean, solution=sol, grid_ok=grid_ok)


def finish_contact_traction(solver, asperity_load: dict):
    """After ``compute_asperity_load`` of the same phase (contact class): recheck the shared state --
    an ``fft_elastic_v1`` extraction that refreshed the stored solution after the samples were taken makes
    the phase ``state_mismatch``.  Returns the phase (None when the phase had none)."""
    from microcell import contact_shear as cs
    ph = getattr(solver, "contact_traction", None)
    if ph is None:
        return None
    if ph.solution is not None and getattr(solver, "_contact_solution", None) is not ph.solution:
        return cs.mismatch(ph)
    return ph
