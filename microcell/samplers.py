"""Samplers of a solved micro cell: the fluid pressure and traction samples, and the contact pressure
and traction at the same sample points.

MicroCell WP5 (11 Nov 2026): ``micro_diagnostics.py`` and ``contact_sample.py`` in one file, in that
order.  The code is moved verbatim; the imports between the two are gone and the others point at
the merged modules (``film``, ``shear``).  Their module docstrings follow.  Where one names one of
the two files, read this file.

``micro_diagnostics.py`` (the parent's ``microscale/src/functions/micro_diagnostics.py``)
------------------------------------------------------------------------------------------
Diagnostics of a solved micro cell, shared by every phase-neutral solver class.

[extrema M1, 23 Oct 2026] The fluid pressure extrema (decision D1 of
docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md, section 2.3; the
definition and its status vocabulary are in ``microcell/pressure.py``).  The base, mixed and
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
``microcell/shear.py``, with M2a's reduction).  The base, mixed and
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
* reduced by ``reduce_traction``: the local mean over the whole set.  The same samples
  weighted by the class's own mask of ``taust`` (DG0 ``chi`` / none) reproduce ``taust``: that
  is the check that the sampler evaluates the existing expression.

Cost: numpy on the cells and the rule's points (the cell geometry cached on the solver).

``contact_sample.py`` (the parent's ``microscale/src/functions/contact_sample.py``)
------------------------------------------------------------------------------------------
Contact pressure of the resolved ``fft_elastic_v1`` solution at cell-local sample points, and the
asperity / total traction on a common sample set (extrema M3a, 23 October 2026).

Pure numpy; nothing from FEniCS is imported or accepted.  ``microscale/src/functions/__init__.py``
imports the FEniCS micro classes, so importing this module through the package imports dolfin; a
FEniCS-free consumer loads the file by path (as ``tests/elastic_contact/test_elastic_contact.py``
loads ``generated_ht_cell.py``).  Plan ``docs/plans/masked_pressure_extrema_and_shear_exports_
2026-10-23.md`` sections 2.2 (contact / total), 3.2 and 6.2.4; review-and-launch note section 2.3
(D5 global x only, D7 the contact pressure source); log ``docs/notes/extrema_m3a_implementation_log_
2026-10-23.md``.

Source (D7)
-----------
The nodal solution the contact class stores, ``MicroPhaseNeutralContactSolver._contact_solution``:
an ``microcell.contact.ContactSolution`` written by ``_refresh_contact`` and re-checked against the
state key by ``_fft_asperity_load`` (``solver_contact.py``).  Its arrays are
indexed ``[i, j]``, ``i`` along x and ``j`` along y; node ``(i, j)`` sits at ``(i/nx, j/ny)`` of the
unit cell; the periodic images at x = 1 and y = 1 are not stored (``contact.py``, the module docstring).
The class writes the grid onto the Vper dofs as ``field[i, j]``, through the map of ``_periodic_grid``
(``solver_base.py``; ``solver_contact.py``).  This module never
goes through dof or CG indices and never reads ``p_asp_field``.  For ``hertz_summit_v1`` that name is
a DG0 force-conserving display over the obstruction mask (``_contact.py`` 351-388), not a pressure
distribution.

Coordinates
-----------
Cell-local, in fractions of the cell: the micro mesh is ``UnitSquareMesh(n, n)``
(``scripts/run/run_microscale_phase_neutral.py`` 372), so its own vertex and quadrature coordinates
can be passed unchanged.  With ``cell_size=(Lx, Ly)`` the coordinates are read in metres instead.
Points outside [0, 1)^2 are wrapped: the cell is periodic.

Interpolant
-----------
The periodic P1 interpolant on the 'right'-diagonal triangulation of the grid.  Each square
``(i, j)``-``(i+1, j+1)`` is split along that diagonal, indices taken mod n:

    lower triangle (i,j),(i+1,j),(i+1,j+1)  for s >= t:  p = (1-s) p00 + (s-t) p10 + t p11
    upper triangle (i,j),(i,j+1),(i+1,j+1)  for t >  s:  p = (1-t) p00 + (t-s) p01 + s p11

with ``(s, t)`` the point's offset inside its square, in grid units.  These are the cells of dolfin's
``UnitSquareMesh(n, n)`` (default diagonal), so the interpolant is what the mesh's CG1 / Vper space
makes of the same nodal values.  ``phic_nd_func``, ``uc_nd_func`` and the fft ``p_asp_field``
(``solver_contact.py``) all use that space, and ``microcell.contact.p1_area`` integrates
the same triangulation.  Properties used by the gates:

* at a grid node it returns the nodal value exactly.  A coordinate within ``SNAP_TOL`` grid units of a
  node is snapped to it, so mesh-vertex coordinates such as 0.34 hit the node;
* ``p_a >= 0`` everywhere, and its maximum over any sample set is at most ``max(sol.p)``: no overshoot;
* it integrates to ``mean(sol.p) = Pasp_hom`` exactly over the cell, because every hat function on a
  uniform periodic grid integrates to 1/(nx ny).  Any positive rule that is exact on each triangle
  therefore reproduces ``Pasp_hom`` to rounding (gate 1; budget in the log, section 1.1);
* the seams are periodic: x and x + 1 give the same value, and the square between node n-1 and the
  image at x = 1 uses node 0.

The native nodal peak ``max(sol.p)``, the peak of this interpolant at points that are not nodes, and
any projected display peak (a DG0 cell mean, say) are three different statistics.  ``statistic`` says
which one a sample set gives: ``native_nodal`` when every sample is a node, ``p1_interpolated``
otherwise.

Supports at the samples
-----------------------
``mech_support``  ``p_a > 0``: where the contact traction is not zero.  Under P1 this is the open star
                  of the active nodes.
``obstruction``   ``phi_c <= 0``, with ``phi_c`` the P1 interpolant of ``sol.phi_c = g - p/k_N``: the
                  complement of the fluid solver's ``chi_fluid = [phi_c > 0]`` (``_contact.py``
                  130-133).  At a vertex it is the class's nodal level set, so M1's fluid samples
                  (``fluid_nodal_sharp_v1``, ``phic_nd_func > 0``) are its complement there.

The two need not agree, and plan section 2.2 says not to assume they do.  Under P1 the pressure is
positive on a fringe on the fluid side of the obstruction edge.  A node can also be obstructed with
p = 0, or carry p > 0 on the fluid side, within the complementarity tolerance.  ``support_report``
measures all of this; nothing here suppresses either contribution.

Traction (plan section 2.2)
---------------------------
    t_a(x) = mu_material p_a(x) U_hat,     t_t(x) = t_f(x) + t_a(x)    at the common samples

Zero-speed law, exactly as the existing code has it (``ehl/macro/friction.py`` 186-191,
``calc_asp_friction``; documented, not changed)::

    tau_asp_i = mu_material * Pc * p0 * U_i / (sqrt(U.U) + 1e-30)

``U`` is the macro's CG1 field ``Rball U0 omega (x_z, 0, -x_x)`` (``ehl/macro/fe_problem.py`` 156-166),
so at a reversal (omega = 0) and on the rotation axis ``U = 0`` and the asperity traction is zero: there
is no static friction.  For ``|U| > 0`` the factor is ``U/|U|`` to within ``1e-30/|U|``, so it is
regularised, not normalised.  The 1e-30 is in the units of that field.  ``sliding_direction``
reproduces the law, bit for bit when it is given the macro's own ``U``.  In the micro's local basis
the micro receives ``R U = (|U|, 0, 0)`` up to rounding (``fe_problem.py`` 836-858), so the local
direction is ``(|U|/(|U| + 1e-30), 0, 0)``.  ``p_a`` is in Pa here.  The macro's ``Pc`` is
nondimensional and is multiplied by ``p0``.

Statuses (strings, never inferred from NaN)
-------------------------------------------
``ok``                 ``fft_elastic_v1``, some contact pressure
``no_support``         ``fft_elastic_v1`` with ``p = 0`` everywhere: a valid state without the
                       contribution; p_a = 0 at every sample
``state_invalid``      no stored solution, or arrays that are not a finite, non-negative grid: NaN
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from microcell.pressure import (FluidPressureExtrema, PRESSURE_EXTREMA_SAMPLER, PHASE_NO_FLUID,
                          PHASE_NONFINITE, PHASE_OK)


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
                           "(microcell/form_compiler.py): it samples at the points of that rule")
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
    (``solver_base.py`` _setup_film_and_mask; ``solver_contact.py``)."""
    parts = []
    if getattr(solver, "_ripple_on", False) and getattr(solver, "ur_nd_func", None) is not None:
        parts.append(solver.ur_nd_func)
    if getattr(solver, "uc_nd_func", None) is not None:
        parts.append(solver.uc_nd_func)
    return parts


def _film_at_points(solver, lam, cells, film_dofs_m):
    """(n_q, n_c) dimensional film [m] at the rule's points, as the form evaluates
    ``Constant(H0) * h_nd`` there, and its source label."""
    from microcell.film import CosineContactDescriptor
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


def sample_fluid_traction(solver, film_dofs_m=None, existing_mask=None):
    """The fluid traction statistics of the solver's current state (one phase), in the micro's
    local frame; see the module docstring and ``shear.py``."""
    from microcell.shear import SHEAR_SAMPLER, FluidTractionPhase
    from microcell.shear import GLOBAL_X, reduce_traction
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
    return FluidTractionPhase(mean_local=(float(local.mean[0]), float(local.mean[1])),
                              mean_existing_weights=existing, status=local.status,
                              n_samples=int(local.n_samples), n_support=int(local.n_support),
                              support_fraction=float(local.support_fraction), quadrature_degree=smp.degree,
                              mask_source=smp.mask_source, film_source=smp.film_source,
                              sampler=SHEAR_SAMPLER)


# ======================================================================================
# [extrema M3b, 23 Oct 2026] the asperity and total traction on the fluid sampler's points
# (definition and statuses: microcell/shear.py; sampler and algebra: M3a's, in this file)
# ======================================================================================
def _local_sliding_direction(solver):
    """``U / (|U| + 1e-30)`` of the micro's own sliding velocity at the end of its run, local frame
    (``Ux_nd_const``, ``Uy_nd_const``: the velocity of the Couette term of the fluid traction, so the
    asperity traction has its sign; the macro's law ``calc_asp_friction``, zero at zero speed)."""
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
      own ``Pasp_hom`` (``shear.MEAN_CHECK_BUDGET``), then M3a's ``reduce_contact_and_total``
      in the local frame (the means).
    """
    from microcell import shear as cs
    from microcell.shear import GLOBAL_X, VALID_STATUSES as _FLUID_OK
    smp = getattr(solver, "fluid_traction_samples", None)
    fl = getattr(solver, "fluid_traction", None)
    if smp is None or fl is None or smp.xy is None:
        raise RuntimeError("sample_contact_traction needs this phase's fluid samples (sample_fluid_traction "
                           "in spatial_homogenisation)")
    model = cs.FFT_MODEL                                        # the one contact law
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
        # a caller that gave no friction coefficient (direct callers of the class, older settings):
        # nothing is published, nothing raised -- the statistics are absent, not wrong
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
    """After ``compute_asperity_load`` of the same phase (contact class): recheck the shared state --
    an ``fft_elastic_v1`` extraction that refreshed the stored solution after the samples were taken makes
    the phase ``state_mismatch``.  Returns the phase (None when the phase had none)."""
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

from microcell.shear import GLOBAL_X, reduce_traction   # pure numpy; ehl/__init__ imports nothing

CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"

from microcell.contact import FFT as FFT_MODEL      # the one contact law (one definition)

STATUS_OK = "ok"
STATUS_NO_SUPPORT = "no_support"
STATUS_STATE_INVALID = "state_invalid"
CONTACT_VALID_STATUSES = (STATUS_OK, STATUS_NO_SUPPORT)

STATISTIC_NATIVE = "native_nodal"
STATISTIC_INTERPOLATED = "p1_interpolated"

#: a coordinate this close to a node, in grid units, is the node
SNAP_TOL = 1e-9
#: the regularisation of the macro's asperity direction, friction.py 188 (units of the velocity field)
ZERO_SPEED_EPS = 1e-30


@dataclass(frozen=True)
class ContactSamples:
    """The resolved contact pressure at N samples.  Pa; boolean masks per sample."""
    p: np.ndarray                 # (N,) contact pressure [Pa]; NaN unless the status is ok / no_support
    mech_support: np.ndarray      # (N,) p > 0
    obstruction: np.ndarray       # (N,) phi_c <= 0: the fluid solver's contact mask
    at_node: np.ndarray           # (N,) the sample is a grid node (after snapping)
    status: str
    contact_model: str
    grid: tuple                   # (nx, ny) of the solution; (0, 0) when there is none
    native_peak: float            # max(sol.p) [Pa] (the class's pasp_max, _contact.py 263)
    Pasp_hom: float               # mean(sol.p) [Pa] (the class's Pasp_hom, _contact.py 260)
    statistic: str                # native_nodal | p1_interpolated
    sampler: str = CONTACT_SAMPLER

    @property
    def valid(self) -> bool:
        return self.status in CONTACT_VALID_STATUSES

    def summary(self) -> dict:
        """Plain numbers and strings (no arrays): picklable, JSON-ready."""
        finite = np.isfinite(self.p)
        return {"status": self.status, "contact_model": self.contact_model, "grid": list(self.grid),
                "n_samples": int(self.p.size), "n_mech_support": int(np.count_nonzero(self.mech_support)),
                "n_obstruction": int(np.count_nonzero(self.obstruction)),
                "n_at_node": int(np.count_nonzero(self.at_node)),
                "sample_peak_Pa": float(np.max(self.p)) if finite.any() else float("nan"),
                "native_peak_Pa": float(self.native_peak), "Pasp_hom_Pa": float(self.Pasp_hom),
                "statistic": self.statistic, "sampler": self.sampler}


# ---------------------------------------------------------------------------
# the interpolant
# ---------------------------------------------------------------------------
def _grid_coords(xy, shape, cell_size):
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
    """The periodic P1 interpolant of nodal ``values`` (``[i, j]``, node at ``(i/nx, j/ny)``) at ``xy``
    (N x 2, cell fractions, or metres with ``cell_size``).  Returns ``(v, at_node)``."""
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
    out = np.where(at_node, p00, out)            # a node is the node's value, bit for bit
    return out, at_node


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
def _empty(n, status, model, grid=(0, 0)):
    nan = np.full(n, np.nan)
    no = np.zeros(n, dtype=bool)
    return ContactSamples(p=nan, mech_support=no, obstruction=no.copy(), at_node=no.copy(), status=status,
                          contact_model=str(model), grid=tuple(int(g) for g in grid),
                          native_peak=float("nan"), Pasp_hom=float("nan"), statistic=STATISTIC_INTERPOLATED)


def sample_contact(solution: Optional[object], xy, *, contact_model: str = FFT_MODEL, cell_size=None) -> ContactSamples:
    """``p_a`` [Pa] and the supports at the samples ``xy``, from the stored contact solution.

    ``solution`` is the class's ``_contact_solution`` (anything with ``p`` and ``phi_c`` grids, as
    ``microcell.contact.ContactSolution``), or None.  ``contact_model`` is the task's identity:
    ``fft_elastic_v1``, the only one in this repo (any other raises)."""
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


# ---------------------------------------------------------------------------
# weights on a sample set (P1-exact)
# ---------------------------------------------------------------------------
def p1_lumped_weights(coords, cells) -> np.ndarray:
    """Lumped P1 weights of the vertices of a triangle mesh: each vertex gets a third of the area of
    every triangle it belongs to.  Exact for every function linear on each triangle, so on the micro
    mesh (``mesh_m.coordinates()``, ``mesh_m.cells()``) the weighted vertex mean of ``p_a`` is
    ``Pasp_hom`` up to rounding.  They sum to the mesh area (1 on the unit cell)."""
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


# ---------------------------------------------------------------------------
# asperity and total traction
# ---------------------------------------------------------------------------
def sliding_direction(u, eps: float = ZERO_SPEED_EPS) -> np.ndarray:
    """``U / (|U| + eps)``: the asperity direction of ``calc_asp_friction`` (friction.py 186-191),
    zero at zero speed.  ``u`` is (3,) or (N, 3); the result has the same shape."""
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 3:
        raise ValueError(f"velocity must have 3 components, got shape {u.shape}")
    mag = np.sqrt(np.sum(u * u, axis=-1, keepdims=True))
    return u / (mag + float(eps))


def asperity_traction(p, mu_material: float, u_hat) -> np.ndarray:
    """``t_a = mu_material p_a U_hat`` (N, 3) [Pa] for p_a [Pa] at N samples and the direction
    ``u_hat`` (3,) for the whole cell or (N, 3) per sample (from ``sliding_direction``)."""
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
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] != n or v.shape[1] not in (2, 3):
        raise ValueError(f"{what} must have shape ({n}, 2) or ({n}, 3), got {v.shape}")
    return v if v.shape[1] == 3 else np.column_stack((v, np.zeros(n)))


def total_traction(t_fluid, fluid_support, t_asp, mech_support) -> np.ndarray:
    """``t_t = t_f [fluid] + t_a [mech]`` at each common sample: each contribution is its physical zero
    outside its own support, and both are kept where the supports overlap."""
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    ms = np.asarray(mech_support, dtype=bool).reshape(-1)
    n = fs.size
    if ms.shape != (n,):
        raise ValueError("fluid and mechanical supports must be on the same samples")
    tf, ta = _pad3(t_fluid, n, "t_fluid"), _pad3(t_asp, n, "t_asp")
    return np.where(fs[:, None], tf, 0.0) + np.where(ms[:, None], ta, 0.0)


def support_report(fluid_support, contact: ContactSamples, weights) -> dict:
    """Where the fluid set, the obstruction set and the mechanical support agree and where they do not,
    by sample weight (fractions of the whole sample set) and count.  Reported, never corrected."""
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
        "fluid_and_mech": fs & ms,                          # both tractions present: the overlap
        "mech_outside_obstruction": ms & ~ob,               # contact pressure where the fluid sees fluid
        "obstruction_without_pressure": ob & ~ms,           # obstructed, no contact traction
        "neither": ~fs & ~ms,                               # no traction contribution at all
        "fluid_not_complement_of_obstruction": fs == ob,    # the caller's fluid set vs the contact level set
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
    """Fluid, asperity and total traction of one micro state, reduced on one common sample set."""
    fluid: object                  # microcell.shear.ShearReduction
    asperity: object
    total: object
    support: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)


def reduce_contact_and_total(t_fluid, fluid_support, fluid_valid, contact: ContactSamples, mu_material: float,
                             u_hat, weights, *, direction=GLOBAL_X,
                             fluid_state_valid: bool = True) -> ContactTotalReduction:
    """The three contributions on the same samples, each reduced by ``microcell.shear.reduce_traction``
    (every sample projected on ``direction``, in the local basis, before anything is reduced).  The total is formed sample by sample first and reduced afterwards; extrema
    of the components are never added.

    ``t_fluid`` (N, 2|3) [Pa] is the fluid traction in the local basis (M2b), ``fluid_support`` its
    samples (``chi_f``), ``fluid_valid`` which were evaluated.  ``u_hat`` is the asperity direction in the
    same basis (``sliding_direction`` of the velocity, (3,) or (N, 3)).  An unsupported or invalid
    contact state leaves the fluid reduction as it is and makes the asperity and total ``state_invalid``.
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
