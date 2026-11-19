"""Full-film micro-cell solver (``MicroPhaseNeutralSolver``), steady and transient.

The Reynolds equation with penalty cavitation is solved on a periodic unit cell.  The
non-dimensional pressure is split as

    P_nd = P_re_nd + phi_c_nd + lambda_nd + w

  - P_re_nd   = p0 / p_scale is the macroscale pressure;
  - phi_c_nd  = dpx (x - 0.5) + dpy (y - 0.5) is the zero-mean ramp of the macroscale gradient;
  - lambda_nd is the scalar pressure-level correction, an unknown;
  - w         is the periodic fluctuation, held at zero mean by the multiplier mu_nd.
The cell-mean pressure is P_re + lambda, as phi_c and w have zero mean.

Closure of lambda:  G = beta lambda_nd - gamma_nd mean(spos(-P_nd)) = 0,
with d = dG/dlambda = beta + gamma_nd mean(spos') >= beta > 0.

No Dirichlet condition is applied.  Each Newton step solves the augmented system

    (i)   J dw   + c dlambda + m dmu = -F_w
    (ii)  g^T dw + d dlambda         = -G
    (iii) m^T dw                     = -H

by one factorisation of the Reynolds Jacobian J and a Schur complement for (dlambda, dmu).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from fenics import *  # legacy FEniCS

from microcell import ripple as _rp   # ripple operator, traction and acceptance checks
from microcell.samplers import sample_fluid_pressure_extrema, sample_fluid_traction

# Quadrature degree 2, unless the process has already set one.
from microcell import form_compiler as _qdeg
_qdeg.set_at_import(2, __name__)
set_log_level(LogLevel.ERROR)
set_log_active(False)

logger = logging.getLogger(__name__)

#: cap of the non-dimensional cavitation smoothing width (``_nondimensionalise``)
_EPSP_ND_CAP = 1.0


# -------------------------------------------------------------------
# Inputs and fixed values
# -------------------------------------------------------------------
# Inputs held by the solver:
#   self.zeta            ``microcell.Zeta``, the macroscale state at t = 0 and its rates
#   self.t_end           time window [s] (0.0 for a steady solve)
#   self.settings        ``microcell.MicroSettings``, with ``.contact`` and ``.ripple``
#   self.velocity_ratio  fraction of the roughness on the moving surface
#   self.pair            True for a stationary + translating pair of surfaces
#   self.h_min           contact onset film [m]: ``settings.h_min`` with contact, 0.0 in full film
#
# Settings read here (defined in ``microcell.api.MicroSettings``):
#   spos_form         smoothed positive part of the cavitation penalty.  "spline": one-sided C1
#                     quadratic, zero for z <= 0.  "sqrt": 0.5 (z + sqrt(z^2 + eps^2)), which
#                     leaves eps^2 / (4 P) at positive pressure.
#   jacobian_reg_eps  shift ``eps * I`` added to the Reynolds Jacobian before it is factorised.
#                     Without cavitation the periodic Jacobian has a constant null mode and the
#                     factorisation fails; the shift moves that eigenvalue off zero.  0.0: no shift.
#   ripple            ``None``: off.  Otherwise the film gains u_r, the zero-mean periodic
#                     half-space response (modulus ``E_star`` [Pa]) to the fluid traction of the
#                     cell, fixed during a pass and updated between passes (``_ripple_run``).
#   level_constraint  "whole_cell": (1/A) int P = P_re + lambda.  "fluid_mean_v1":
#                     (1/A_f) int chi P = P_re + lambda, chi the fluid indicator (mixed class only).

# Fixed values of the Newton solve.

#: adaptive damping of the Newton step (``DampingPolicy``)
ADAPTIVE_DAMPING = True
#: smallest damping factor; None: max(omega / 32, 1e-3), from ``relaxation_parameter``
OMEGA_MIN = None
#: most consecutive halvings of one step
NEWTON_BACKTRACK_MAX = 10
#: a step is rejected when the squared residual grows by more than this factor
NEWTON_BACKTRACK_TOL = 2.0
#: iterations before the growth test applies
NEWTON_BACKTRACK_WARMUP_ITERS = 3
#: growth of the damping factor after an accepted step
OMEGA_RECOVERY_FACTOR = 1.2
#: shrink of the damping ceiling after a rejected step
OMEGA_CAP_SHRINK_FACTOR = 0.85
#: print switches; nothing reads them in the solve
PRINT_NEWTON = False
PRINT_TIMING = False
#: closure parameter beta of G (non-dimensional)
BETA = 1.0
#: fluid-mean level constraint: penalty holding w at zero on the nodes with no fluid in their
#: support (None: on), and its weight relative to the nodal mass
PATCH_PIN = None
PATCH_PIN_KAPPA = 1.0


#: level-constraint names
LEVEL_CONSTRAINT_WHOLE_CELL = "whole_cell"
LEVEL_CONSTRAINT_FLUID_MEAN = "fluid_mean_v1"


#: counters zeroed by ``_reset_timers`` and summed over the passes of one ripple loop
_PASS_COUNTERS = (
    "_n_newton_calls",
    "_n_newton_iters_total", "_n_newton_backtracks_total",
    # Newton solves per level-constraint path
    "_n_lc_whole_cell", "_n_lc_fluid_mean", "_n_lc_no_fluid",
)

#: iteration cap of the per-step ripple of a stationary + translating pair
PAIR_RIPPLE_MAXIT = 30


# -------------------------------------------------------------------
# Periodic boundary
# -------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
        return on_boundary and (
            (near(X[0], 0.0) or near(X[1], 0.0))
            and not (
                (near(X[0], 1.0) and near(X[1], 0.0))
                or (near(X[0], 0.0) and near(X[1], 1.0))
                or (near(X[0], 1.0) and near(X[1], 1.0))
            )
        )

    def map(self, X, Y):
        if near(X[0], 1.0) and near(X[1], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1] - 1.0
        elif near(X[0], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1]
        else:
            Y[0] = X[0]
            Y[1] = X[1] - 1.0


# -------------------------------------------------------------------
# Solver
# -------------------------------------------------------------------

class DampingPolicy:
    """Adaptive damping of the Newton step.

    A step that increases the residual is halved, down to ``omega_min``.  Each halving
    lowers the ceiling to which the damping factor omega may recover in this solve.
    Holds omega and the backtrack count; the Newton loop holds the state and reassembles.
    omega_base: ``settings.relaxation_parameter``.  The other values are the module constants.
    """

    def __init__(self, omega_base: float):
        self.adaptive = bool(ADAPTIVE_DAMPING)
        self.omega_min = (float(OMEGA_MIN) if OMEGA_MIN is not None
                          else max(omega_base / 32.0, 1e-3))
        self.max_backtracks = int(NEWTON_BACKTRACK_MAX)
        self.bt_tol = float(NEWTON_BACKTRACK_TOL)
        self.bt_warmup_iters = int(NEWTON_BACKTRACK_WARMUP_ITERS)
        self.omega_recovery = float(OMEGA_RECOVERY_FACTOR)
        self.omega_cap_shrink = float(OMEGA_CAP_SHRINK_FACTOR)
        self.omega_base = omega_base
        # damping factor of the next step: halved on a rejected step, raised on an accepted one
        self.omega_current = omega_base
        # ceiling of the recovery in this solve: lowered at each backtrack, never raised
        self.omega_cap = omega_base
        self.omega_cap_floor = max(self.omega_min * 2.0, omega_base * 0.2)
        # combined squared residual at the last accepted iterate
        self.prev_res_sq: "float | None" = None
        self.bt_count_seq = 0   # consecutive backtracks at the current iteration
        self.n_bt = 0           # backtracks in this solve
        self.diverged = False   # result of the last divergence test

    def wants_backtrack(self, nit: int, res_sq: float,
                        have_step: bool) -> bool:
        """True when the last step should be halved and retried.

        nit: Newton iteration; res_sq: combined squared residual after the step;
        have_step: a step has been applied.
        A non-finite residual always counts as divergence; the growth test applies from
        iteration ``bt_warmup_iters``.  ``self.diverged`` stays set for ``on_accept``.
        No backtrack is requested once omega is at ``omega_min`` or the budget is spent.
        """
        nonfinite = not np.isfinite(res_sq)
        arm_ratio_check = nit >= self.bt_warmup_iters
        self.diverged = nonfinite or (
            arm_ratio_check
            and self.prev_res_sq is not None
            and res_sq > self.bt_tol * self.prev_res_sq
        )
        return (
            self.adaptive
            and nit > 0
            and have_step
            and self.bt_count_seq < self.max_backtracks
            and self.diverged
            and max(self.omega_current * 0.5, self.omega_min) < self.omega_current
        )

    def on_backtrack(self) -> None:
        """Halve omega and lower the recovery ceiling."""
        self.omega_current = max(self.omega_current * 0.5, self.omega_min)
        # the ceiling is at most twice the halved omega
        self.omega_cap = max(
            min(self.omega_cap * self.omega_cap_shrink,
                self.omega_current * 2.0),
            self.omega_cap_floor,
        )
        self.bt_count_seq += 1
        self.n_bt += 1

    def on_accept(self, nit: int, res_sq: float) -> None:
        """Record an accepted step and raise omega towards the ceiling.

        omega is not raised when the step was accepted while still diverging
        (budget spent or omega at its floor).
        """
        if nit > 0 and not self.diverged:
            self.omega_current = min(self.omega_cap,
                                     self.omega_current * self.omega_recovery)
        self.bt_count_seq = 0
        self.prev_res_sq = res_sq


class MicroPhaseNeutralSolver:
    """Full-film solver of one periodic micro cell.

    Unknowns: the fluctuation ``w`` (periodic CG1), the level ``lambda_nd`` (a scalar
    Constant) and the multiplier ``mu_nd`` (a float) that holds ``w`` at zero mean.

    mesh_m: unit-square mesh.  zeta: ``microcell.Zeta``.  t_end: time window [s].
    settings: ``microcell.MicroSettings``.  ht: film callback of the roughness.
    velocity_ratio: fraction of the roughness on the moving surface.
    pair: True for a stationary + translating pair.  auto_solve: run a steady solve on construction.

    ``run`` or ``run_steady`` solves; ``spatial_homogenisation`` returns the outputs.
    """

    def __init__(
        self,
        mesh_m: Mesh,
        zeta,
        t_end: float,
        settings,
        ht: Callable,
        velocity_ratio: float,
        pair: bool,
        auto_solve: bool = True,
    ) -> None:
        # ---- inputs (described at the top of this module) ----
        self.mesh_m = mesh_m
        self.zeta = zeta
        self.t_end = t_end
        self.settings = settings
        self.velocity_ratio = float(velocity_ratio)
        self.pair = bool(pair)
        self.h_min = self._contact_onset_film()
        # label of the running sub-solve ("steady", "transient-seed", "step k/N"), for messages
        self._phase = "init"
        # cavitation smoothing form: fixed per instance, as it selects the UFL forms
        self._spos_form = str(settings.spos_form).strip().lower()
        if self._spos_form not in ("spline", "sqrt"):
            raise ValueError(
                "spos_form must be 'spline' or 'sqrt', got %r"
                % (self._spos_form,)
            )
        self.ht_cb = ht

        # ---- setup ----
        self._setup_coordinates()
        self._setup_time_controls()
        self._nondimensionalise()
        # spaces before the film: the ripple and contact fields are allocated on ``Vper``
        self._setup_spaces_and_unknowns()
        self._setup_film_and_mask()
        self._setup_pressure_expressions()
        self._setup_mean_vector()
        self._setup_forms()
        self._solver_needs_rebuild: bool = True
        self._build_nonlinear_solver()

        if auto_solve:
            self.run_steady()

    def _contact_onset_film(self) -> float:
        """Contact onset film ``self.h_min`` [m]: 0.0 in full film, which keeps the contact
        mask off.  The mixed class returns ``settings.h_min``."""
        return 0.0

    # -------------------------
    # Reuse of a built solver for another state
    # -------------------------
    def reset_task(self, zeta, t_end: float, velocity_ratio: float) -> None:
        """Point the solver at a new state without rebuilding spaces, forms or factorisations.

        zeta: ``microcell.Zeta``.  t_end: time window [s].  velocity_ratio: as the constructor.
        Every Constant the forms read from the state is reassigned and the solution and
        history are zeroed, as in a new solver.  The settings, the cell size included, are
        not changed.  ``microsolver`` builds a solver per call and does not use this.
        """
        self.zeta = zeta
        self.t_end = t_end
        self.velocity_ratio = float(velocity_ratio)

        # --- as _setup_time_controls ---
        self.T_nd_const.assign(0.0)
        self.T_dim_const.assign(0.0)
        self.H0_const.assign(float(zeta.H))
        self.HT_const.assign(float(zeta.Hdot))
        self.p0_const.assign(float(zeta.P))
        self.PT_const.assign(float(zeta.Pdot))
        self.Tend_dim = float(self.t_end) if self.t_end is not None else 0.0
        _alpha = float(self.velocity_ratio)
        self._Ux_dim_const.assign(_alpha * 2.0 * float(zeta.Ux))
        self._Uy_dim_const.assign(_alpha * 2.0 * float(zeta.Uy))

        # scales and non-dimensional constants
        self._nondim_scalars()
        self.invLx.assign(1.0 / self.Lx)
        self.invLy.assign(1.0 / self.Ly)

        # --- as _setup_film_and_mask ---
        self.hmin_nd_const.assign(self.hmin_nd)
        self.delta_h_nd_const.assign(self.delta_h_nd)
        if self._ripple_on:                 # a new state starts at u_r = 0
            self._ripple_reset()
        self.Ux_nd_const.assign(2.0 * self.Ux_nd)
        self.Uy_nd_const.assign(2.0 * self.Uy_nd)

        # --- as _setup_pressure_expressions ---
        self.dpx_nd_const.assign(self.dpx_nd)
        self.dpy_nd_const.assign(self.dpy_nd)
        self.P_re_nd.assign(float(zeta.P) / self.p_scale)
        self.lambda_nd.assign(0.0)
        self.mu_nd = 0.0
        self.p_scale_const.assign(self.p_scale)
        self.omega_const.assign(self.omega)
        self.epsP_nd_const.assign(self.epsP_nd)

        # --- as _setup_forms and _build_forms ---
        self.dt_nd_const.assign(1.0)
        self.bdf_a0.assign(0.0)
        self.bdf_a1.assign(0.0)
        self.bdf_a2.assign(0.0)
        self.transient_on.assign(0.0)
        self.epsP_nd_spos_const.assign(self.epsP_nd)
        self.gamma_nd_c_const.assign(self.gamma_nd)
        self.gamma_nd_g_const.assign(self.gamma_nd)
        self.gamma_nd_cav_const.assign(self.gamma_nd)
        self._squeeze_mean_nd.assign(0.0)

        # --- solution and history ---
        self.w.vector().zero()
        self.w.vector().apply("insert")
        for _f in (self.h_hist_nm1, self.h_hist_nm2,
                   self.mask_hist_nm1, self.mask_hist_nm2):
            _f.vector().zero()
            _f.vector().apply("insert")
        self._have_two_hist = False

        # --- Constants of the subclasses ---
        self._reset_task_subclass()

    def _reset_task_subclass(self) -> None:
        """Reassigns the state-dependent Constants a subclass adds.  Base: nothing."""
        return None

    # -------------------------
    # Per-run counters
    # -------------------------
    def _reset_timers(self) -> None:
        """Zeroes the Newton counters of one run."""
        self._n_newton_calls = 0
        self._n_newton_iters_total = 0
        # backtracks of the adaptive damping
        self._n_newton_backtracks_total = 0
        # Newton solves per level-constraint path
        self._n_lc_whole_cell = 0
        self._n_lc_fluid_mean = 0
        self._n_lc_no_fluid = 0
        self._newton_call_idx = 0

    # -------------------------
    # Setup helpers
    # -------------------------
    def _setup_coordinates(self) -> None:
        """Unit-cell coordinates ``self.x``, ``self.y`` in 0 to 1."""
        self.x, self.y = SpatialCoordinate(self.mesh_m)

    def _setup_time_controls(self) -> None:
        """Constants of the time-dependent state: time, H, Hdot, P, Pdot from ``zeta`` and the
        roughness velocity ``velocity_ratio * 2 U`` [m/s]."""
        zeta = self.zeta
        self.T_nd_const = Constant(0.0)
        self.T_dim_const = Constant(0.0)
        self.H0_const = Constant(float(zeta.H))
        self.HT_const = Constant(float(zeta.Hdot))
        self.p0_const = Constant(float(zeta.P))
        self.PT_const = Constant(float(zeta.Pdot))
        self.Tend_dim = float(self.t_end) if self.t_end is not None else 0.0
        _alpha = float(self.velocity_ratio)
        self._Ux_dim_const = Constant(_alpha * 2.0 * float(zeta.Ux))
        self._Uy_dim_const = Constant(_alpha * 2.0 * float(zeta.Uy))

    def _nondimensionalise(self) -> None:
        """Scales (``_nondim_scalars``) and the Constants 1/Lx, 1/Ly."""
        self._nondim_scalars()
        self.invLx = Constant(1.0 / self.Lx)
        self.invLy = Constant(1.0 / self.Ly)

    def _nondim_scalars(self) -> None:
        """Scales of the non-dimensional problem, as floats, from ``zeta`` and the settings.

        Lengths by the cell size Lx, Ly [m]; film by H; velocity by ``Uref``, the largest
        |U| over the window [m/s]; time by ``t0 = 2 Lx / Uref`` [s]; pressure by ``p_scale``
        [Pa], the largest of the Reynolds scale 6 eta0 Uref Lx / H^2 and the corner pressures
        at the start and end of the window.  Sets ``dpx_nd``, ``dpy_nd``, ``Ux_nd``, ``Uy_nd``,
        ``hmin_nd``, ``delta_h_nd``, ``epsP_nd``, ``gamma_nd`` and
        ``omega = H^2 p_scale / (6 eta0 Lx Uref)``.  Creates no dolfin objects.
        """
        zeta = self.zeta
        s = self.settings

        self.Lx = float(self.settings.xmax)
        self.Ly = float(self.settings.ymax)

        Ux_end = zeta.Ux + zeta.Uxdot * (float(self.t_end) if self.t_end else 0.0)
        Uy_end = zeta.Uy + zeta.Uydot * (float(self.t_end) if self.t_end else 0.0)
        Uref_candidates = [abs(zeta.Ux), abs(zeta.Uy), abs(Ux_end), abs(Uy_end)]
        Uref_max = max(Uref_candidates)
        self.Uref = Uref_max if Uref_max > 1e-30 else 1.0

        # macroscale pressure at the four cell corners, at the start and end of the window
        dpx_0 = float(zeta.dpdx * self.settings.xmax)
        dpy_0 = float(zeta.dpdy * self.settings.ymax)
        Tend_f = float(self.t_end) if self.t_end else 0.0
        dpdx_end = zeta.dpdx + zeta.dpdx_dot * Tend_f
        dpdy_end = zeta.dpdy + zeta.dpdy_dot * Tend_f
        dpx_end = float(dpdx_end * self.settings.xmax)
        dpy_end = float(dpdy_end * self.settings.ymax)
        dpt = float(self.t_end * zeta.Pdot) if (self.t_end is not None and zeta.Pdot is not None) else 0.0

        corner_vals_t0 = [zeta.P, zeta.P + dpx_0, zeta.P + dpy_0, zeta.P + dpx_0 + dpy_0]
        corner_vals_t1 = [
            zeta.P + dpt,
            zeta.P + dpt + dpx_end,
            zeta.P + dpt + dpy_end,
            zeta.P + dpt + dpx_end + dpy_end,
        ]

        self.t0 = (2.0 * self.Lx) / self.Uref

        # Reynolds pressure scale of the Couette transport: the floor of p_scale, so that omega
        # stays of order 1 where the macroscale pressure is near zero
        p_rey = (6.0 * self.settings.eta0 * self.Uref * self.Lx) / (zeta.H ** 2)
        self.p_scale = max([p_rey] + [abs(v) for v in (corner_vals_t0 + corner_vals_t1)])

        self.dpx_nd = (zeta.dpdx * self.settings.xmax) / self.p_scale
        self.dpy_nd = (zeta.dpdy * self.settings.ymax) / self.p_scale
        self.Ux_nd = zeta.Ux / self.Uref
        self.Uy_nd = zeta.Uy / self.Uref
        self.hmin_nd = self.h_min / zeta.H
        delta_h, eps_smooth = s.delta_h, s.eps_smooth
        self.delta_h_nd = (delta_h / zeta.H) if float(delta_h) != 0.0 else 0.0
        # The smoothing width epsP_nd = eps_smooth / p_scale is capped at ``_EPSP_ND_CAP``.  The
        # cap binds only where p_scale < eps_smooth: no macroscale pressure and near-zero velocity.
        _epsP_nd_cap = _EPSP_ND_CAP
        self.epsP_nd = eps_smooth / self.p_scale
        self._epsP_nd_capped = False
        if self.epsP_nd > _epsP_nd_cap:
            print(f"  [PN-SCALE-GUARD] epsP_nd {self.epsP_nd:.4e} -> "
                  f"{_epsP_nd_cap:.4e} (eps_smooth={eps_smooth:.3e} Pa / "
                  f"p_scale={self.p_scale:.4e} Pa; H0={zeta.H:.4e} m, "
                  f"Uref={self.Uref:.4e} m/s)", flush=True)
            self.epsP_nd = _epsP_nd_cap
            self._epsP_nd_capped = True
        # gamma_nd = penalty_gamma / p_scale is capped, so that the penalty does not dominate the
        # Reynolds terms where p_scale is small (thick film, zero macroscale pressure)
        _GAMMA_ND_CAP = 1e5
        gamma_nd_raw = self.settings.penalty_gamma / self.p_scale
        self.gamma_nd = min(gamma_nd_raw, _GAMMA_ND_CAP)
        self.omega = (zeta.H**2 * self.p_scale) / (6.0 * self.settings.eta0 * self.Lx * self.Uref)

    def grad_phys(self, q):
        """Gradient of ``q`` in physical coordinates [1/m]."""
        return as_vector((Dx(q, 0) * self.invLx, Dx(q, 1) * self.invLy))

    def grad_cell(self, q):
        """Gradient of ``q`` in unit-cell coordinates."""
        return as_vector((Dx(q, 0), Dx(q, 1)))

    def _call_ht(self, x, y):
        """Film [m] from the film callback at unit-cell coordinates ``x``, ``y``.  A callback
        that takes no time or velocity arguments is called with the cell size and H only."""
        try:
            return self.ht_cb(
                x, y, self.settings.xmax, self.settings.ymax,
                self.H0_const, self.HT_const, self.T_dim_const,
                self._Ux_dim_const, self._Uy_dim_const,
            )
        except TypeError:
            return self.ht_cb(x, y, self.settings.xmax, self.settings.ymax, self.H0_const)

    def _setup_film_and_mask(self) -> None:
        """Non-dimensional film ``h_nd = h / H`` (with the ripple u_r / H when it is on), the
        fluid indicator ``chi_fluid`` (1 where h > h_min), the mobility ``mask`` (1 in the
        fluid, ``eps_solid`` in contact) and the velocity ``U_nd = 2 U / Uref``."""
        s = self.settings

        H_dim = self._call_ht(self.x, self.y)
        self.h_base_nd = H_dim / self.H0_const
        self.h_nd = self.h_base_nd
        # ripple u_r / H on Vper, added to the film only when the ripple is on
        self._ripple_setup()
        if self._ripple_on:
            self.h_nd = self.h_nd + self.ur_nd_func

        self.hmin_nd_const = Constant(self.hmin_nd)
        self.delta_h_nd_const = Constant(self.delta_h_nd)
        phi_mask = self.h_nd - self.hmin_nd_const
        self.chi_fluid = conditional(gt(phi_mask, 0.0), Constant(1.0), Constant(0.0))
        self.mask = self.smooth_levelset_mask(
            phi=phi_mask,
            delta=self.delta_h_nd_const,
            mask_core=Constant(s.eps_solid),
            mask_fluid=Constant(1.0),
        )

        self.Ux_nd_const = Constant(2.0 * self.Ux_nd)
        self.Uy_nd_const = Constant(2.0 * self.Uy_nd)
        self.U_nd = as_vector((self.Ux_nd_const, self.Uy_nd_const))

    def _setup_spaces_and_unknowns(self) -> None:
        """Function spaces (periodic CG1 ``Vper``, CG1 ``Vfull``, DG0 ``Vdg0``), the unknown
        ``w`` and the factorised mass matrix of the L2 projection onto ``Vper``."""
        self.pbc = PeriodicBoundary()
        self.Vper = FunctionSpace(self.mesh_m, "CG", 1, constrained_domain=self.pbc)
        self.Vfull = FunctionSpace(self.mesh_m, "CG", 1)
        self.Vdg0 = FunctionSpace(self.mesh_m, "DG", 0)

        # w: the periodic zero-mean fluctuation, the finite element unknown
        self.w = Function(self.Vper, name="w_fluc")
        self.v = TestFunction(self.Vper)
        self.dw = TrialFunction(self.Vper)

        # Mass matrix of Vper, factorised once and reused by ``_project_to_Vper`` for the
        # history snapshots of each time step.
        _u_tr = TrialFunction(self.Vper)
        _v_te = TestFunction(self.Vper)
        self._proj_Vper_v_test = _v_te  # test function of the projection right-hand side
        self._M_Vper = assemble(_u_tr * _v_te * dx(domain=self.mesh_m))
        self._M_Vper_lu = LUSolver(self._M_Vper, "mumps")
        self._M_Vper_lu.parameters["symmetric"] = True
        # the operator is not set again, so the factor is reused

    def _setup_pressure_expressions(self) -> None:
        """Pressure ``P_nd = P_re_nd + phi_c_nd + lambda_nd + w``, the mobility
        ``K_nd = omega h_nd^3 mask``, the advective flux ``F_adv_nd = U_nd h_nd mask`` and
        the cavitation term ``negP_pos = spos(-P_nd)`` with its weight."""
        zeta = self.zeta

        # non-dimensional macroscale gradient, assigned at each time level
        self.dpx_nd_const = Constant(self.dpx_nd)
        self.dpy_nd_const = Constant(self.dpy_nd)

        # linear ramp about the cell centre: zero mean
        self.phi_c_nd = (
            self.dpx_nd_const * (self.x - 0.5)
            + self.dpy_nd_const * (self.y - 0.5)
        )

        # macroscale pressure, assigned at each time level
        self.P_re_nd = Constant(float(zeta.P) / self.p_scale)

        # pressure-level correction, solved by the Schur step
        self.lambda_nd = Constant(0.0)

        # multiplier of the zero-mean constraint on w
        self.mu_nd: float = 0.0

        self.P_nd = self.P_re_nd + self.phi_c_nd + self.lambda_nd + self.w

        # pressure [Pa]
        self.p_scale_const = Constant(self.p_scale)
        self.P_dim = self.p_scale_const * self.P_nd

        # Isoviscous, incompressible lubricant: the viscosity enters through omega only.
        self.omega_const = Constant(self.omega)
        self.K_nd = self.omega_const * self.h_nd**3 * self.mask
        self.F_adv_nd = (self.U_nd * self.h_nd) * self.mask

        # spos(-P_nd) of the cavitation penalty, in the form ``spos_form`` selects
        self.epsP_nd_const = Constant(self.epsP_nd)
        self.negP_pos = self._spos(-self.P_nd, self.epsP_nd_const)

        # weight of the cavitation penalty: the mask here; the mixed class replaces it
        self._cav_weight = self.mask


    def _setup_mean_vector(self) -> None:
        """Mean vector ``m_i = int phi_i dx`` and the cell area (1 on the unit cell).

        m gives the zero-mean constraint H = m^T w / area, the mu term of the residual
        and the Schur vector z3 = J^{-1} m.
        """
        v_test = TestFunction(self.Vper)
        self._mean_vec = assemble(v_test * dx(domain=self.mesh_m))
        self._cell_area = assemble(Constant(1.0) * dx(domain=self.mesh_m))

    def _constraint_vector(self):
        """Constraint vector m of the Newton system.  Base: the whole-cell ``m_i = int phi_i dx``.
        A subclass returns another vector to hold the level over another region."""
        return self._mean_vec

    def _constraint_residual(self, m, area: float) -> float:
        """Constraint residual H.  Base: ``m^T w / area``."""
        return float(m.inner(self.w.vector())) / area

    def _level_constraint_prepare(self) -> str:
        """Name of the constraint path of the next Newton solve.  Base: "whole_cell" for
        either setting, as a full-film cell has no contact region."""
        lc = self.settings.level_constraint
        if lc not in (LEVEL_CONSTRAINT_WHOLE_CELL, LEVEL_CONSTRAINT_FLUID_MEAN):
            raise ValueError(f"level_constraint {lc!r}: expected 'whole_cell' or 'fluid_mean_v1'")
        return "whole_cell"

    def _no_fluid_bypass(self) -> None:
        """Solution of a cell with no fluid region.  Mixed class only."""
        raise RuntimeError("no-fluid bypass requested by a class that has no fluid-mean constraint")

    def _project_to_Vper(self, expr) -> "Function":
        """L2 projection of the UFL expression ``expr`` onto ``Vper``.

        Uses the mass matrix factorised in ``_setup_spaces_and_unknowns``: one right-hand
        side assembly and one back-substitution.  ``interpolate`` does not accept the UFL
        expressions of the film and the mask.
        """
        out = Function(self.Vper)
        b = assemble(expr * self._proj_Vper_v_test * dx(domain=self.mesh_m))
        self._M_Vper_lu.solve(out.vector(), b)
        return out

    def _setup_forms(self) -> None:
        """History functions, time-stepping Constants, the residual and Jacobian
        (``_build_forms``) and the derivative forms of the Schur step."""
        # film at the two previous time levels (BDF history)
        self.h_hist_nm1 = Function(self.Vper, name="h_nm1")
        self.h_hist_nm2 = Function(self.Vper, name="h_nm2")
        # mask history: allocated and left at zero; no form reads it
        self.mask_hist_nm1 = Function(self.Vper, name="mask_nm1")
        self.mask_hist_nm2 = Function(self.Vper, name="mask_nm2")
        self._have_two_hist = False

        # time step and BDF coefficients as Constants, so the forms are built once
        self.dt_nd_const = Constant(1.0)
        self.bdf_a0 = Constant(0.0)
        self.bdf_a1 = Constant(0.0)
        self.bdf_a2 = Constant(0.0)
        # switch of the transient terms; a subclass may have created it already
        if not hasattr(self, "transient_on"):
            self.transient_on = Constant(0.0)

        self._build_forms(include_transient_terms=True, dt_nd=None)

        # --- derivative forms of the Schur step ---
        # spos'(-P_nd):
        #   sqrt   : 0.5 (1 + (-P_nd) / sqrt(P_nd^2 + eps^2))
        #   spline : 0, (-P_nd) / eps or 1
        # The smoothing width has its own Constant here, separate from ``epsP_nd_const``.
        self.epsP_nd_spos_const = Constant(self.epsP_nd)
        self._spos_deriv_nd = self._spos_deriv(
            -self.P_nd, self.epsP_nd_spos_const
        )

        # c_i = dR_cav/dlambda = -gamma cav_weight spos'(-P) phi_i dx
        self.gamma_nd_c_const = Constant(self.gamma_nd)
        self._c_form = (
            -self.gamma_nd_c_const
            * self._cav_weight
            * self._spos_deriv_nd
            * self.v
            * dx
        )

        # g_j = dG/dw_j = +gamma spos'(-P) phi_j dx, divided by the area in the Newton solve.
        # c carries the cavitation weight; g does not.
        self.gamma_nd_g_const = Constant(self.gamma_nd)
        self._g_form_raw = (
            self.gamma_nd_g_const * self._spos_deriv_nd * self.v * dx
        )

    def _build_forms(self, *, include_transient_terms: bool, dt_nd: float | None) -> None:
        """Residual ``R = Reynolds + cavitation - squeeze`` and its Jacobian ``J = dR/dw``.

        include_transient_terms: add the squeeze term, built from the BDF difference of the
        film and switched by ``transient_on``.  dt_nd: not used; the step is ``dt_nd_const``.
        """
        gradP = self.grad_cell(self.P_nd)
        gradv = self.grad_cell(self.v)

        R_reynolds = inner(self.F_adv_nd - self.K_nd * gradP, gradv) * dx
        # cavitation penalty gamma cav_weight spos(-P_nd); the Constant is made at each build
        self.gamma_nd_cav_const = Constant(self.gamma_nd)
        R_cav = self.gamma_nd_cav_const * self._cav_weight * (
            self.negP_pos
        ) * self.v * dx
        R = R_reynolds + R_cav

        if include_transient_terms:
            dh_dt = (
                self.bdf_a0 * self.h_nd
                + self.bdf_a1 * self.h_hist_nm1
                + self.bdf_a2 * self.h_hist_nm2
            ) / self.dt_nd_const
            R -= self._squeeze_residual_terms(dh_dt)

        self.R = R
        self.J = derivative(self.R, self.w, self.dw)
        self._solver_needs_rebuild = True

    def _squeeze_residual_terms(self, dh_dt):
        """Squeeze term of the residual.

        dh_dt: non-dimensional film rate (UFL), from the BDF difference in ``_build_forms``.
        Sets ``_squeeze_source = mask dh_dt`` and its cell mean ``_squeeze_mean_nd``.
        Returns the UFL term subtracted from R: the source less its mean, switched by
        ``transient_on``.  The mean is removed as a periodic problem needs a zero net source.
        The mixed class overrides this.
        """
        self._squeeze_source = self.mask * dh_dt
        self._squeeze_mean_nd = Constant(0.0)
        return self.transient_on * (
            (self._squeeze_source - self._squeeze_mean_nd) * self.v * dx
        )

    def _flux_means(self, q_dim, q_couette_dim, q_poiseuille_dim, area_total):
        """Cell-area means of the flux and of its Couette and Poiseuille parts.

        q_dim, q_couette_dim, q_poiseuille_dim: UFL flux vectors [m^2/s]; area_total: cell area.
        Returns (Qx, Qy, Qx_couette, Qy_couette, Qx_poiseuille, Qy_poiseuille) [m^2/s].
        """
        return (
            float(assemble(q_dim[0] * dx(domain=self.mesh_m)) / area_total),
            float(assemble(q_dim[1] * dx(domain=self.mesh_m)) / area_total),
            float(assemble(q_couette_dim[0] * dx(domain=self.mesh_m)) / area_total),
            float(assemble(q_couette_dim[1] * dx(domain=self.mesh_m)) / area_total),
            float(assemble(q_poiseuille_dim[0] * dx(domain=self.mesh_m)) / area_total),
            float(assemble(q_poiseuille_dim[1] * dx(domain=self.mesh_m)) / area_total),
        )

    def _update_squeeze_mean(self) -> None:
        """Assigns the cell mean of the current squeeze source to ``_squeeze_mean_nd``.
        Called at the seed and at each time step.  The mixed class overrides this."""
        if hasattr(self, "_squeeze_source"):
            S_mean = float(assemble(self._squeeze_source * dx(domain=self.mesh_m)))
            self._squeeze_mean_nd.assign(S_mean)

    # -------------------------
    # Core static helpers
    # -------------------------
    @staticmethod
    def smooth_pos(z, eps):
        """Symmetric smoothed positive part (``spos_form="sqrt"``):
        ``spos(z) = 0.5 (z + sqrt(z^2 + eps^2))``.

        Never zero: ``spos(0) = eps / 2`` and ``spos(z) -> eps^2 / (4 |z|)`` as z -> -inf.
        A cell at positive pressure everywhere therefore keeps a positive lambda from the closure.
        """
        return 0.5 * (z + sqrt(z * z + eps * eps))

    @staticmethod
    def smooth_pos_spline(z, eps):
        """One-sided C1 quadratic-spline positive part (``spos_form="spline"``).

            spos(z) = 0                z <= 0
                    = z^2 / (2 eps)    0 < z < eps
                    = z - eps / 2      z >= eps

        Zero for z <= 0, so a cell at positive pressure everywhere has lambda = 0.
        """
        # Built from ``conditional``, ``gt`` and ``lt``, which this UFL exports.
        # ``eps_s`` keeps the quadratic branch finite at eps = 0, where the result is max(z, 0).
        eps_s = conditional(gt(eps, 0.0), eps, 1.0)
        return conditional(
            gt(z, 0.0),
            conditional(lt(z, eps), z * z / (2.0 * eps_s), z - 0.5 * eps),
            0.0,
        )

    @staticmethod
    def smooth_pos_deriv_sqrt(z, eps):
        """Derivative of ``smooth_pos``: ``0.5 (1 + z / sqrt(z^2 + eps^2))``."""
        return 0.5 * (1.0 + z / sqrt(z * z + eps * eps))

    @staticmethod
    def smooth_pos_deriv_spline(z, eps):
        """Derivative of ``smooth_pos_spline``: 0 for z <= 0, z / eps for 0 < z < eps, 1 above.

        Zero wherever P >= 0, so a cell at positive pressure everywhere leaves the Reynolds
        Jacobian with its constant null mode; ``jacobian_reg_eps`` must then be non-zero.
        """
        eps_s = conditional(gt(eps, 0.0), eps, 1.0)
        return conditional(
            gt(z, 0.0),
            conditional(lt(z, eps), z / eps_s, 1.0),
            0.0,
        )

    def _spos(self, z, eps):
        """Smoothed positive part in the form ``settings.spos_form`` selects."""
        form = getattr(self, "_spos_form", "spline")
        if form == "sqrt":
            return self.smooth_pos(z, eps)
        return self.smooth_pos_spline(z, eps)

    def _spos_deriv(self, z, eps):
        """Derivative of ``_spos`` with respect to z."""
        form = getattr(self, "_spos_form", "spline")
        if form == "sqrt":
            return self.smooth_pos_deriv_sqrt(z, eps)
        return self.smooth_pos_deriv_spline(z, eps)

    @staticmethod
    def smooth_levelset_mask(phi, delta, mask_core, mask_fluid):
        """UFL mask: ``mask_fluid`` where phi > delta, ``mask_core`` where phi < -delta and
        linear between.  delta = 0: a sharp switch at phi = 0."""
        if float(delta) == 0.0:
            return conditional(gt(phi, 0.0), mask_fluid, mask_core)
        t = (phi + delta) / (2.0 * delta)
        mask_transition = mask_core + (mask_fluid - mask_core) * t
        return conditional(
            gt(phi - delta, 0.0),
            mask_fluid,
            conditional(gt((-delta) - phi, 0.0), mask_core, mask_transition),
        )

    def _solve_level(self) -> None:
        """One solve at the current time level: the Newton solve, or, for a surface pair with
        the ripple on, the Newton solve repeated until u_r converges (``_pair_solve``)."""
        if self._pair_active:
            self._pair_solve()
            return
        self.solve()

    def _refresh_geometry(self) -> None:
        """Updates the nodal geometry fields to the current time Constants.

        Called before every solve and before each history projection.  A film callback with
        ``refresh_geometry`` (generated surfaces) refills its nodal field; the cosine film is
        a UFL expression and needs none.  The ripple field is then written for this time.
        The contact class also solves the elastic contact here.
        """
        refresh = getattr(self.ht_cb, "refresh_geometry", None)
        if refresh is not None:
            refresh(self)
        # ripple after the rigid film and before the contact solve of the subclass
        if self._ripple_on:
            self._ripple_write_field()
        return None

    # -------------------------
    # Ripple: deformation due to the fluid pressure of the cell
    # -------------------------
    _ripple_on = False          # class defaults, for subclasses that skip the base film setup
    _pair_active = False

    def _ripple_setup(self) -> None:
        """Checks the ripple settings and allocates ``ur_nd_func`` (u_r / H on ``Vper``).
        Off: nothing is allocated."""
        s = self.settings
        self._ripple_on = s.ripple is not None
        self.ur_nd_func = None
        self.ripple_info = None
        if not self._ripple_on:
            return
        E = s.E_star
        if E is None or not np.isfinite(float(E)) or float(E) <= 0.0:
            raise ValueError("the ripple needs a positive, finite E_star (the two-body modulus)")
        if not (float(s.ripple.tol) > 0.0 and int(s.ripple.maxpass) >= 1
                and 0.0 < float(s.ripple.omega) <= 1.0):
            raise ValueError("ripple settings: need tol > 0, maxpass >= 1 and "
                             f"0 < omega <= 1; got {s.ripple.tol}, {s.ripple.maxpass}, "
                             f"{s.ripple.omega}")
        self.ur_nd_func = Function(self.Vper, name="ur_nd")
        self._ripple_reset()

    def _ripple_reset(self) -> None:
        """Zeroes u_r and the ripple state: a solve starts from u_r = 0."""
        if self.ur_nd_func is not None:
            self.ur_nd_func.vector().zero()
            self.ur_nd_func.vector().apply("insert")
        self._ripple_u = None               # iterate [m] on the grid, end-time frame; None: zero
        self._ripple_u_serial = 0            # incremented when the iterate changes
        self._ripple_shift_end = (0.0, 0.0)
        self._ripple_written = None          # (shift, iterate serial) held in ur_nd_func
        self._ripple_version = 0             # incremented at every write; read by the contact solve
        self._ripple_warm = None
        self._ripple_levels = None
        self._ripple_pass = 0
        self.ripple_info = None
        # per-step ripple of a surface pair
        self._pair_active = False
        self._pair_u = None                  # last converged field [m], lab frame, at _pair_ref_shift
        self._pair_ref_shift = (0.0, 0.0)
        self._pair_written = None            # field held in ur_nd_func [m]
        self._pair_iters = []                # ripple iterations per time level

    def _periodic_grid(self):
        """Map of the ``Vper`` dofs to the periodic n x n grid.

        Returns ``(n, i, j)``: dof ``k`` is node ``(i[k], j[k])``, ``i`` along x.  Raises
        ``ValueError`` unless the mesh is a ``UnitSquareMesh(n, n)``.  Cached.
        """
        g = getattr(self, "_periodic_grid_cache", None)
        if g is not None:
            return g
        mesh = self.mesh_m
        X = self.Vper.tabulate_dof_coordinates().reshape(-1, 2)
        n = int(round(np.sqrt(X.shape[0])))
        i = np.rint(X[:, 0] * n).astype(int) % n
        j = np.rint(X[:, 1] * n).astype(int) % n
        ok = (n * n == X.shape[0] and mesh.num_vertices() == (n + 1) ** 2
              and mesh.num_cells() == 2 * n * n
              and len(set(zip(i.tolist(), j.tolist()))) == X.shape[0]
              and np.allclose(X[:, 0] * n, np.rint(X[:, 0] * n), atol=1e-9)
              and np.allclose(X[:, 1] * n, np.rint(X[:, 1] * n), atol=1e-9))
        if not ok:
            raise ValueError("fft_elastic_v1 needs the micro mesh to be a structured periodic "
                             "n x n UnitSquareMesh")
        g = (n, i, j)
        self._periodic_grid_cache = g
        return g

    def _dofs_to_grid(self, v, dtype=float) -> np.ndarray:
        """``Vper`` dof values as an (n, n) array ``[i_x, j_y]``."""
        n, i, j = self._periodic_grid()
        g = np.empty((n, n), dtype=dtype)
        g[i, j] = v
        return g

    def _grid_to_dofs(self, g) -> np.ndarray:
        """(n, n) array ``[i_x, j_y]`` as ``Vper`` dof values."""
        n, i, j = self._periodic_grid()
        return np.asarray(g)[i, j]

    def _ripple_operator(self):
        """Half-space operator of the ripple on the n x n grid, for the cell size and
        ``E_star``.  Cached."""
        op = getattr(self, "_ripple_op_cache", None)
        if op is None:
            n, _i, _j = self._periodic_grid()
            op = _rp.ripple_operator(n, n, float(self.Lx), float(self.Ly),
                                     float(self.settings.E_star))
            self._ripple_op_cache = op
        return op

    def _ripple_check_task(self) -> None:
        """Raises ``ValueError`` for roughness the ripple cannot take: roughness split between
        the two surfaces (0 < velocity_ratio < 1 or a split cosine), or a generated pair
        without ``self.pair``.  The pass loop needs one frame in which the roughness is at rest."""
        a = float(self.velocity_ratio)
        desc = getattr(self.ht_cb, "contact_cosine", None)
        two = (0.0 < a < 1.0 or (isinstance(desc, tuple) and len(desc) >= 4 and bool(desc[3]))
               or (self._ripple_pair() and not self._pair_branch_on()))
        if two:
            raise ValueError("the ripple supports one rough surface only: "
                             "a two-surface pattern has no single frame")

    def _pair_branch_on(self) -> bool:
        """``self.pair``: the per-step ripple of a surface pair is allowed."""
        return self.pair

    def _ripple_pair(self) -> bool:
        """True when the film callback holds a generated field with both a stationary and a
        translating member."""
        field = getattr(self.ht_cb, "surface_field", None)
        return field is not None and len(getattr(field, "members", {})) > 1

    def _ripple_surface_shift(self) -> tuple:
        """Displacement (sx, sy) [m] of the rough surface at the current time: T times the
        roughness velocity for the cosine film, the motion of the generated surface, or
        zero for a stationary one."""
        field = getattr(self.ht_cb, "surface_field", None)
        if field is not None:
            if not getattr(field, "translating", False):
                return (0.0, 0.0)
            sx, sy = self.ht_cb.surface_motion.displacement(float(self.T_dim_const))
            return (float(sx), float(sy))
        T = float(self.T_dim_const)
        return (T * float(self._Ux_dim_const), T * float(self._Uy_dim_const))

    def _ripple_write_field(self) -> None:
        """Writes u_r / H into ``ur_nd_func`` for the current time.

        The iterate moves with the rough surface: unchanged for a stationary surface,
        Fourier-shifted for a translating one.  For a surface pair the last converged field,
        shifted with the translating member, is written as the first guess of this level.
        """
        if self._pair_active:
            s = self._ripple_surface_shift()
            d = (s[0] - self._pair_ref_shift[0], s[1] - self._pair_ref_shift[1])
            u = self._pair_u if d == (0.0, 0.0) else _rp.pattern_shift(self._pair_u, d[0] / float(self.Lx),
                                                                        d[1] / float(self.Ly))
            self._pair_write(u)
            return
        u = self._ripple_u
        if u is None:
            return                            # the field is zero (pass 1)
        s_now = self._ripple_surface_shift()
        d = (s_now[0] - self._ripple_shift_end[0], s_now[1] - self._ripple_shift_end[1])
        key = (d, self._ripple_u_serial)
        if self._ripple_written == key:
            return
        if d == (0.0, 0.0):
            field = u
        else:
            field = _rp.pattern_shift(u, d[0] / float(self.Lx), d[1] / float(self.Ly))
        _rp.check_zero_mean(field, what="ripple field at a history level")
        self.ur_nd_func.vector().set_local(self._grid_to_dofs(field) / float(self.zeta.H))
        self.ur_nd_func.vector().apply("insert")
        self._ripple_written = key
        self._ripple_version += 1

    def _on_ripple_pass_start(self) -> None:
        """Called at the start of each ripple pass.  Base: nothing; the mixed class resets
        its per-run records."""
        return None

    def _ripple_level_begin(self, level) -> None:
        """Starts the solve of time level ``level`` from the solution of the same level in
        the previous pass, where there is one."""
        if self._pair_active:
            return                            # a pair runs one pass
        warm = (self._ripple_warm or {}).get(level)
        if warm is not None:
            self.w.vector().set_local(warm["w"])
            self.w.vector().apply("insert")
            self.lambda_nd.assign(warm["lam"])
            self.mu_nd = warm["mu"]

    def _ripple_level_end(self, level) -> None:
        """Stores the solution and the snapshot of time level ``level`` for this pass."""
        if self._pair_active:
            return                            # no snapshots for a pair
        snap = {"w": self.w.vector().get_local().copy(), "lam": float(self.lambda_nd),
                "mu": float(self.mu_nd)}
        snap.update(self._ripple_snapshot())
        self._ripple_levels[level] = snap

    def _ripple_traction_fluid(self):
        """Fluid traction ``chi_fluid max(P, 0)`` [Pa] (UFL)."""
        z = Constant(0.0)
        return self.chi_fluid * conditional(gt(self.P_dim, z), self.P_dim, z)

    def _ripple_snapshot(self) -> dict:
        """Current state for the ripple checks: fluid traction ``t`` at the ``Vper`` dofs [Pa],
        contact area fraction ``A_mask`` and the contact entries of the subclass."""
        t = self._project_to_Vper(self._ripple_traction_fluid()).vector().get_local()
        A_mask = 1.0 - float(assemble(self.chi_fluid * dx(domain=self.mesh_m))) / float(self._cell_area)
        snap = {"t": t, "A_mask": A_mask}
        snap.update(self._ripple_contact_snapshot())
        return snap

    def _ripple_contact_snapshot(self) -> dict:
        """Contact set at the nodes (``set``: projected film below h_min) when h_min > 0;
        empty in full film.  The contact class overrides this."""
        if float(self.h_min) <= 0.0:
            return {}
        hv = self._project_to_Vper(self.h_nd).vector().get_local()
        return {"set": self._dofs_to_grid(hv < float(self.hmin_nd), dtype=bool)}

    def _ripple_chi_nodes(self) -> np.ndarray:
        """Fluid indicator at the ``Vper`` dofs (1 where the projected film exceeds h_min)."""
        if float(self.h_min) <= 0.0:
            return np.ones(self.w.vector().local_size())
        hv = self._project_to_Vper(self.h_nd).vector().get_local()
        return (hv > float(self.hmin_nd)).astype(float)

    def _ripple_traction_overlap(self, s_dofs) -> float:
        """Share of the projected traction ``s_dofs`` on contact nodes.  Base: 0; the contact
        class overrides this."""
        return 0.0

    def _ripple_mnorm(self, v) -> float:
        """L2 norm of the ``Vper`` dof vector ``v`` with the consistent mass matrix."""
        f = getattr(self, "_ripple_mnorm_f", None)
        if f is None:
            f = Function(self.Vper)
            self._ripple_mnorm_f = f
        f.vector().set_local(np.asarray(v, dtype=float))
        f.vector().apply("insert")
        Mv = self._M_Vper * f.vector()
        return float(np.sqrt(max(float(f.vector().inner(Mv)), 0.0)))

    def _ripple_target(self) -> dict:
        """Ripple response to the current pressure.

        The traction is s = chi_fluid (P - ramp) where P > 0 and zero elsewhere [Pa],
        projected onto ``Vper``; the ramp is the macroscale gradient part of the pressure.
        Returns ``T`` = H[s] on the grid [m], the relative projection error ``proj_err``,
        the largest traction jump across the cell edges ``seam`` and the contact ``overlap``.
        Raises ``RippleNotConvergedError`` for a non-finite traction.
        """
        P = self.P_dim
        z = Constant(0.0)
        ramp = self.p_scale_const * self.phi_c_nd
        s_expr = self.chi_fluid * conditional(gt(P, z), P - ramp, z)
        s_fun = self._project_to_Vper(s_expr)
        v = s_fun.vector().get_local()
        if not np.all(np.isfinite(v)):
            raise _rp.RippleNotConvergedError("ripple_target_nonfinite: the fluid traction has NaN/inf")
        T = _rp.ripple(self._dofs_to_grid(v), self._ripple_operator())
        dxm = dx(domain=self.mesh_m)
        num = float(assemble((s_expr - s_fun) ** 2 * dxm))
        den = float(assemble(s_expr ** 2 * dxm))
        proj_err = float(np.sqrt(max(num, 0.0) / den)) if den > 0.0 else 0.0
        Q = self.p_scale * (float(self.P_re_nd) + float(self.lambda_nd) + self.w.vector().get_local())
        Gx = float(self.dpx_nd_const) * self.p_scale / float(self.Lx)
        Gy = float(self.dpy_nd_const) * self.p_scale / float(self.Ly)
        seam = _rp.seam_traction_jump(self._dofs_to_grid(Q), self._dofs_to_grid(self._ripple_chi_nodes()),
                                      Gx, Gy, float(self.Lx), float(self.Ly))
        return {"T": T, "proj_err": proj_err, "seam": seam["max"],
                "overlap": float(self._ripple_traction_overlap(v))}

    def _ripple_run(self, body: Callable, later: "Callable | None" = None) -> None:
        """Fixed point of the ripple u_r for one rough surface.

        body: pass 1, a whole run with u_r fixed in the frame of the rough surface.
        later: the pass used from pass 2 on (the end-level pass); None: ``body`` again.
        Between passes u <- u + omega (H[s] - u), with omega secant-relaxed between
        0.1 and 1 times ``ripple.omega``.

        Pass k is accepted (``microcell.ripple.residual_acceptance``) when
        max|H[s_k] - u_k| <= ``ripple.tol`` x H, the residual has at least halved since
        pass k - 1 (from pass 2), and the nodal contact set of the end level is unchanged.
        Pass 1 can be accepted.  On acceptance ``ripple_info`` is set; its ``worst`` entry
        holds the pass-to-pass changes of the levels both passes solved.  The Newton
        counters are summed over the passes.

        Raises ``microcell.ripple.RippleNotConvergedError`` after ``ripple.maxpass`` passes.
        """
        self._ripple_check_task()
        s = self.settings
        H0 = float(self.zeta.H)
        tol = float(s.ripple.tol)
        omega0 = float(s.ripple.omega)
        omega = omega0
        maxpass = int(s.ripple.maxpass)
        n, _i, _j = self._periodic_grid()
        totals = dict.fromkeys(_PASS_COUNTERS, 0)
        u = np.zeros((n, n))
        self._ripple_u = None
        self._ripple_u_serial += 1
        # every run starts from u_r = 0, also on a solver that has already run
        self.ur_nd_func.vector().zero()
        self.ur_nd_func.vector().apply("insert")
        self._ripple_written = None
        self._ripple_version += 1
        self.ripple_info = None
        prev_levels = prev_u = r_old = None
        worst = {}
        k = 0
        counted = True
        try:
            for k in range(1, maxpass + 1):
                self._ripple_pass = k
                self._ripple_warm = prev_levels
                self._ripple_levels = {}
                self._on_ripple_pass_start()
                counted = False
                (later if (later is not None and k >= 2) else body)()
                for name in _PASS_COUNTERS:
                    totals[name] += getattr(self, name, 0)
                counted = True
                self._ripple_shift_end = self._ripple_surface_shift()
                levels = self._ripple_levels
                tgt = self._ripple_target()
                R = tgt["T"] - u
                resid = float(np.max(np.abs(R)))
                if prev_levels is not None:
                    # pass-to-pass changes over the levels both passes solved: diagnostics only
                    common = [lv for lv in levels if lv in prev_levels]
                    _ok134, worst = _rp.transition_checks(
                        {lv: prev_levels[lv] for lv in common}, {lv: levels[lv] for lv in common},
                        resid_max=resid, change_max=float(np.max(np.abs(u - prev_u))), H0=H0,
                        p_scale=float(self.p_scale), n=n, deform_tol=tol, mnorm=self._ripple_mnorm)
                end = next(reversed(levels))   # last level solved
                ok, info = _rp.residual_acceptance(
                    resid, None if r_old is None else float(np.max(np.abs(r_old))),
                    (prev_levels or {}).get(end, {}).get("set"), levels[end].get("set"),
                    H0=H0, deform_tol=tol, n=n)
                worst.update(info)
                if ok:
                    self.ripple_info = {
                        "passes": k, "max": float(np.max(np.abs(u))),
                        "resid": resid / H0,
                        "mean_resid": _rp.zero_mean_residual(u), "seam_mismatch": tgt["seam"],
                        "proj_err": tgt["proj_err"], "traction_overlap": tgt["overlap"],
                        "omega": omega, "worst": worst, "model": _rp.RIPPLE_MODEL}
                    return
                if r_old is not None:
                    dr = R - r_old
                    dn = float(np.sum(dr * dr))
                    if dn > 0.0:
                        omega = -omega * float(np.sum(r_old * dr)) / dn
                    omega = min(max(omega, 0.1 * omega0), omega0)
                u_new = u + omega * R
                _rp.check_zero_mean(u_new, what="ripple iterate")
                prev_levels, prev_u, r_old = levels, u, R
                u = u_new
                self._ripple_u = u
                self._ripple_u_serial += 1
            raise _rp.RippleNotConvergedError(
                f"ripple_not_converged: {maxpass} passes, max|H[s] - u| = "
                f"{float(np.max(np.abs(R))) / H0:.3e} H0 (budget {tol:.1e}); worst "
                + ", ".join(f"{a} {b:.2e}" for a, b in worst.items() if isinstance(b, float)))
        finally:
            # the counters hold the totals over the passes, a failed pass included
            if k > 0:
                for name in _PASS_COUNTERS:
                    if not counted:
                        totals[name] += getattr(self, name, 0)
                    setattr(self, name, totals[name])

    # -------------------------
    # Ripple of a stationary + translating pair: converged at every time level
    # -------------------------
    def _pair_ripple_run(self, body: Callable) -> None:
        """Ripple of a surface pair: one run of ``body`` with u_r converged at every level.

        At the seed, each step and the steady solve, the solve is repeated with
        u_r <- u_r + omega (H[s] - u_r) until max|H[s] - u_r| <= ``ripple.tol`` x H
        (``_pair_solve``).  The contact is solved again after each update and the converged
        field enters the BDF history.  Each level starts from the field of the previous
        level, shifted with the translating member.  Sets ``ripple_info``: ``passes`` is the
        largest iteration count of a level and ``omega`` is NaN.
        """
        self._ripple_check_task()
        n, _i, _j = self._periodic_grid()
        self._ripple_u = None
        self._ripple_u_serial += 1
        self._ripple_written = None
        self.ripple_info = None
        self._ripple_warm = None
        self._ripple_levels = {}
        self._pair_u = np.zeros((n, n))
        self._pair_iters = []
        self._pair_active = True
        motion = getattr(self.ht_cb, "surface_motion", None)
        if motion is not None:
            # surface kinematics, needed for the reference shift before the first film refresh
            motion.set_from_solver(self)
        self._pair_ref_shift = self._ripple_surface_shift()
        self._pair_write(self._pair_u)
        self._on_ripple_pass_start()
        try:
            body()
        finally:
            self._pair_active = False
        tgt = self._ripple_target()
        H0 = float(self.zeta.H)
        u = self._pair_u
        self._ripple_pass = int(max(self._pair_iters)) if self._pair_iters else 0
        self.ripple_info = {"passes": self._ripple_pass, "max": float(np.max(np.abs(u))),
                            "resid": float(np.max(np.abs(tgt["T"] - u))) / H0,
                            "mean_resid": _rp.zero_mean_residual(u), "seam_mismatch": tgt["seam"],
                            "proj_err": tgt["proj_err"], "traction_overlap": tgt["overlap"],
                            "omega": float("nan"), "worst": {}, "model": _rp.RIPPLE_MODEL,
                            "inner_iterations": [int(k) for k in self._pair_iters]}

    def _pair_write(self, u) -> None:
        """Writes the field ``u`` [m] (grid, lab frame, current time) into ``ur_nd_func`` and
        increments ``_ripple_version``."""
        _rp.check_zero_mean(u, what="pair ripple field")
        self.ur_nd_func.vector().set_local(self._grid_to_dofs(u) / float(self.zeta.H))
        self.ur_nd_func.vector().apply("insert")
        self._ripple_version += 1
        self._pair_written = u

    def _pair_resynth_history(self) -> None:
        """Rebuilds the film history of the seed (times T_start - dt and T_start - 2 dt) with
        the current u_r, then restores the time, the geometry and the squeeze mean."""
        T0, dt = float(self.T_start), float(self.dt_dim_run)
        for T, hh in ((T0 - dt, self.h_hist_nm1), (T0 - 2.0 * dt, self.h_hist_nm2)):
            self.T_dim_const.assign(T)
            self._refresh_geometry()
            hh.assign(self._project_to_Vper(self.h_nd))
        self.T_dim_const.assign(T0)
        self._refresh_geometry()
        self._update_squeeze_mean()

    def _pair_solve(self) -> None:
        """Solve of one time level with u_r iterated to ``ripple.tol`` x H, with omega
        secant-relaxed between 0.1 and 1.  Appends the iteration count to ``_pair_iters``.
        Raises ``RippleNotConvergedError`` after ``PAIR_RIPPLE_MAXIT`` iterations."""
        H0 = float(self.zeta.H)
        tol = float(self.settings.ripple.tol) * H0
        u = self._pair_written
        omega, r_old = 1.0, None
        fft = self.settings.contact is not None
        for it in range(1, PAIR_RIPPLE_MAXIT + 1):
            self.solve()
            T = self._ripple_target()["T"]
            R = T - u
            if float(np.max(np.abs(R))) <= tol:
                break
            if r_old is not None:
                d = R - r_old
                dn = float(np.sum(d * d))
                if dn > 0.0:
                    omega = min(max(-omega * float(np.sum(r_old * d)) / dn, 0.1), 1.0)
            u = u + omega * R
            r_old = R
            self._pair_u = u
            self._pair_ref_shift = self._ripple_surface_shift()
            self._pair_write(u)
            if self._phase == "transient-seed":
                # seed: the history is rebuilt with the same field, so u_r does not jump at step 1
                self._pair_resynth_history()
            elif fft:
                self._refresh_contact()            # contact on the film with the new u_r
        else:
            raise _rp.RippleNotConvergedError(
                f"ripple_not_converged: pair level {self._phase}: {PAIR_RIPPLE_MAXIT} inner iterations, "
                f"max|H[s] - u| = {float(np.max(np.abs(R))) / H0:.3e} H0 (budget "
                f"{float(self.settings.ripple.tol):.1e})")
        self._pair_u = u
        self._pair_ref_shift = self._ripple_surface_shift()
        self._pair_iters.append(it)

    def _steady_solve(self) -> None:
        """Solve of ``run_steady``.  Base: the Newton solve; the mixed class uses ``_solve_level``."""
        self.solve()

    def _seed_solve(self) -> None:
        """Seed solve of ``run`` at T_start.  Base: the Newton solve; the mixed class uses
        ``_solve_level``."""
        self.solve()

    # -------------------------
    # Nonlinear solver setup
    # -------------------------
    def _build_nonlinear_solver(self) -> None:
        """Assembler, Jacobian matrix, residual vector, MUMPS LU solver and update vector of
        the Newton solve.  No boundary condition is applied."""
        self._bcs = []
        self._sys_asm = SystemAssembler(self.J, self.R, [])

        self._A_mat = PETScMatrix()
        self._b_vec = PETScVector()
        self._sys_asm.assemble(self._A_mat, self._b_vec)

        self._lu_solver = LUSolver(self._A_mat, "mumps")
        self._lu_solver.parameters["symmetric"] = False

        # Newton update
        self._du = Function(self.Vper)

        self._solver_needs_rebuild = False

    # -------------------------
    # Closure of lambda (overridden by the mixed class)
    # -------------------------

    def _closure_G(self, beta: float, gamma: float, area: float) -> float:
        """Closure residual ``G = beta lambda_nd - gamma mean(spos(-P_nd))`` at the current state.
        beta: ``BETA``; gamma: ``gamma_nd``; area: cell area."""
        spos_mean = float(assemble(self.negP_pos * dx(domain=self.mesh_m))) / area
        return beta * float(self.lambda_nd) - gamma * spos_mean

    def _closure_g_vec(self):
        """Assembled vector g with ``g_j / area = dG/dw_j``."""
        return assemble(self._g_form_raw)

    def _closure_d_scalar(self, beta: float, gamma: float, area: float) -> float:
        """``d = dG/dlambda = beta + gamma mean(spos'(-P_nd))``, at least beta."""
        spos_deriv_mean = (
            float(assemble(self._spos_deriv_nd * dx(domain=self.mesh_m))) / area
        )
        return beta + gamma * spos_deriv_mean

    # -------------------------
    # Schur complement Newton solver
    # -------------------------
    def _apply_jacobian_reg_shift(self) -> None:
        """Adds ``settings.jacobian_reg_eps * I`` to the assembled Jacobian ``_A_mat``.  Applied
        after every assembly of the matrix.

        The periodic Reynolds Jacobian has a constant null mode where no cavitation penalty
        is active, and the factorisation then fails; the shift removes it.  The diffusion
        and penalty terms of the Jacobian are negative, so a negative shift adds to them,
        whereas a positive shift can cancel small negative diagonals in contact.  Zero: no
        shift.  A non-finite value raises ``ValueError``.

        The shift changes the Newton direction and not the residual or the convergence
        test.  However, the converged pressure of a contact cell can depend on its value.
        """
        reg_eps = float(self.settings.jacobian_reg_eps)
        if not np.isfinite(reg_eps):
            raise ValueError("jacobian_reg_eps must be finite; zero disables the shift")
        if reg_eps == 0.0:
            return
        try:
            self._A_mat.mat().shift(reg_eps)
        except Exception as _shift_exc:
            # without the PETSc shift: add eps to the diagonal
            try:
                _diag = Vector()
                self._A_mat.init_vector(_diag, 0)
                self._A_mat.get_diagonal(_diag)
                _diag[:] = _diag.get_local() + reg_eps
                self._A_mat.set_diagonal(_diag)
            except Exception:
                raise RuntimeError(
                    f"jacobian_reg_eps={reg_eps} shift failed "
                    f"(petsc4py path: {_shift_exc}); disable by "
                    f"setting jacobian_reg_eps=0.0"
                ) from _shift_exc

    def _assemble_newton_rhs(self) -> None:
        """Assembles the residual only: ``_b_vec = -(R + mu m)``, the right-hand side of
        ``J dw = -F_w``.  The Jacobian is assembled separately, before each factorisation."""
        self._sys_asm.assemble(self._b_vec)
        # multiplier term, with the constraint vector of the active level constraint
        self._b_vec.axpy(self.mu_nd, self._constraint_vector())
        self._b_vec *= -1.0

    def _schur_step(self, z1, z2, z3, m, beta, gamma, area, G, H, nit):
        """One Newton direction by the Schur complement.

        z1, z2, z3: work Functions on ``Vper``.  m: constraint vector.  G, H: closure and
        constraint residuals.  nit: iteration, for the warning.  The Jacobian ``_A_mat`` is
        assembled by the caller.  Factorises J, solves for z1 = J^{-1} b, z2 = J^{-1} c and
        z3 = J^{-1} m, solves the 2 x 2 system for (dlambda, dmu) and writes
        dw = z1 - dlambda z2 - dmu z3 into ``self._du``.
        Returns ``(d_lambda, d_mu)``.
        """
        # --- vectors that depend on the current pressure ---
        # c_i = dR_cav/dlambda
        c_vec = assemble(self._c_form)
        # g_j = d(area G)/dw_j; d = dG/dlambda
        g_vec = self._closure_g_vec()
        d_scalar = self._closure_d_scalar(beta, gamma, area)

        # --- factorisation and three back-substitutions ---
        self._lu_solver.solve(z1.vector(), self._b_vec)
        self._lu_solver.solve(z2.vector(), c_vec)
        self._lu_solver.solve(z3.vector(), m)

        # --- 2 x 2 Schur complement ---
        # S = [[d - g^T z2 / area,  -g^T z3 / area],
        #      [-m^T z2 / area,     -m^T z3 / area]]
        g_z1 = g_vec.inner(z1.vector()) / area
        g_z2 = g_vec.inner(z2.vector()) / area
        g_z3 = g_vec.inner(z3.vector()) / area
        m_z1 = m.inner(z1.vector()) / area
        m_z2 = m.inner(z2.vector()) / area
        m_z3 = m.inner(z3.vector()) / area

        S = np.array([
            [d_scalar - g_z2, -g_z3],
            [-m_z2,           -m_z3],
        ])
        r_schur = np.array([
            -G - g_z1,
            -H - m_z1,
        ])

        # regularise a near-singular Schur complement
        det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        scale_S = max(abs(S[0, 0]), abs(S[1, 1]), abs(S[0, 1]), 1e-30)
        if abs(det_S) < 1e-12 * scale_S**2:
            logger.warning(
                f"Schur complement near-singular at Newton it {nit}: "
                f"det(S) = {det_S:.2e}. Adding regularisation 1e-10."
            )
            S[0, 0] += 1e-10 * abs(beta)
            S[1, 1] -= 1e-10

        dl_dmu = np.linalg.solve(S, r_schur)
        d_lambda = float(dl_dmu[0])
        d_mu = float(dl_dmu[1])

        # --- field update: dw = z1 - dlambda z2 - dmu z3 ---
        self._du.vector().zero()
        self._du.vector().axpy(1.0, z1.vector())
        self._du.vector().axpy(-d_lambda, z2.vector())
        self._du.vector().axpy(-d_mu, z3.vector())
        self._du.vector().apply("insert")

        return d_lambda, d_mu

    def _newton_solve(self) -> None:
        """Newton solve of the augmented system for ``w``, ``lambda_nd`` and ``mu_nd``.

        Tolerances, iteration cap and base damping come from the settings
        (``abs_tolerance``, ``rel_tolerance``, ``max_iterations``, ``relaxation_parameter``).
        Each iteration:
          1. residuals: the field residual from b = -(R + mu m), and G and H;
          2. damping: where the residual has grown, revert and retry at half omega
             (``DampingPolicy``);
          3. assemble J and compute the direction (``_schur_step``);
          4. update w, lambda and mu by omega times the direction;
          5. reassemble b.
        Cost: one factorisation and three back-substitutions per accepted iteration.
        Raises ``RuntimeError`` when the solve does not converge.
        """
        s = self.settings
        omega = s.relaxation_parameter
        atol = s.abs_tolerance
        rtol = s.rel_tolerance
        max_it = s.max_iterations
        beta = float(BETA)
        gamma = self.gamma_nd
        area = float(self._cell_area)

        pol = DampingPolicy(float(omega))
        omega_base = pol.omega_base

        # constraint vector of the active level constraint
        m = self._constraint_vector()  # PETScVector

        # work vectors of the back-substitutions, allocated once per function space
        if getattr(self, "_z_work_space", None) is not self.Vper:
            self._z1 = Function(self.Vper)
            self._z2 = Function(self.Vper)
            self._z3 = Function(self.Vper)
            self._z_work_space = self.Vper
        z1, z2, z3 = self._z1, self._z2, self._z3

        # --- initial assembly: b = -(R + mu m) ---
        self._assemble_newton_rhs()

        # initial residual norms, for the relative tolerance
        res0_field: float | None = None
        res0_scalar: float | None = None

        # last G and H, for the error message
        G: float = 0.0
        H: float = 0.0

        # State before the last update (w_snap, lam_snap, mu_snap) and the last direction
        # (last_du, last_d_lambda, last_d_mu): a backtrack reapplies the direction at a
        # smaller omega without a new Schur step.
        w_snap: np.ndarray | None = None
        lam_snap: float = 0.0
        mu_snap: float = 0.0
        last_du: "Function | None" = None
        last_d_lambda = 0.0
        last_d_mu = 0.0

        # True when the loop ends at its own convergence check on a finite state; the final
        # check after the loop is then skipped.
        converged_at_check = False

        for nit in range(max_it):
            # --- scalar residuals at the current (w, lambda, mu) ---
            G = self._closure_G(beta, gamma, area)
            H = self._constraint_residual(m, area)

            res_field = self._b_vec.norm("l2")
            res_scalar = max(abs(G), abs(H))

            if res0_field is None:
                res0_field = max(res_field, 1e-30)
                res0_scalar = max(res_scalar, 1e-30)

            # --- adaptive damping: backtrack where the last step increased the residual ---
            res_sq = res_field * res_field + res_scalar * res_scalar
            if pol.wants_backtrack(nit, res_sq,
                                   last_du is not None and w_snap is not None):
                # revert to the state before the step and reapply the direction at half omega
                self.w.vector().set_local(w_snap)
                self.w.vector().apply("insert")
                self.lambda_nd.assign(lam_snap)
                self.mu_nd = mu_snap

                pol.on_backtrack()

                self.w.vector().axpy(pol.omega_current, last_du.vector())
                self.w.vector().apply("insert")
                self.lambda_nd.assign(lam_snap + pol.omega_current * last_d_lambda)
                self.mu_nd = mu_snap + pol.omega_current * last_d_mu

                # residual of the damped state, tested at the top of the loop
                self._assemble_newton_rhs()
                continue

            # step accepted: omega recovers towards its cap
            pol.on_accept(nit, res_sq)

            # --- convergence check ---
            field_ok = (res_field < atol) or (res_field / res0_field < rtol)
            scalar_ok = (res_scalar < atol) or (res_scalar / res0_scalar < rtol)
            if field_ok and scalar_ok:
                # max(|G|, nan) returns |G|, so a non-finite H can pass the test above
                converged_at_check = bool(np.isfinite(G) and np.isfinite(H))
                break

            # --- Jacobian of this iteration, with the shift ---
            self._sys_asm.assemble(self._A_mat)
            self._apply_jacobian_reg_shift()

            # --- Schur direction ---
            d_lambda, d_mu = self._schur_step(
                z1, z2, z3, m, beta, gamma, area, G, H, nit,
            )

            # --- state before the damped update and the direction, for a backtrack ---
            w_snap = self.w.vector().get_local().copy()
            lam_snap = float(self.lambda_nd)
            mu_snap = self.mu_nd
            if last_du is None:
                last_du = Function(self.Vper)
            last_du.vector().zero()
            last_du.vector().axpy(1.0, self._du.vector())
            last_du.vector().apply("insert")
            last_d_lambda = d_lambda
            last_d_mu = d_mu

            # --- damped update at the current omega ---
            self.w.vector().axpy(pol.omega_current, self._du.vector())
            self.w.vector().apply("insert")
            self.lambda_nd.assign(float(self.lambda_nd) + pol.omega_current * d_lambda)
            self.mu_nd += pol.omega_current * d_mu

            # --- residual for the next iteration ---
            self._assemble_newton_rhs()


        # Final check of the returned state, with the initial residuals and the same tests.
        # Needed where the loop did not check it: a last update or backtrack at the
        # iteration limit, or a non-finite G or H.
        if not converged_at_check:
            self._assemble_newton_rhs()
            final_res = self._b_vec.norm("l2")
            G = self._closure_G(beta, gamma, area)
            H = self._constraint_residual(m, area)
            final_scalar = max(abs(G), abs(H))
            field_ok = np.isfinite(final_res) and (
                final_res < atol or final_res / max(res0_field or 0.0, 1e-30) < rtol)
            scalar_ok = np.isfinite(G) and np.isfinite(H) and (
                final_scalar < atol or final_scalar / max(res0_scalar or 0.0, 1e-30) < rtol)
            if not (field_ok and scalar_ok):
                # the iterations of a failed solve are counted too
                if hasattr(self, "_n_newton_iters_total"):
                    self._n_newton_iters_total += max(0, (nit + 1) - pol.n_bt)
                    self._n_newton_backtracks_total += pol.n_bt
                raise RuntimeError(
                    f"MicroPhaseNeutralSolver: Newton (Schur) did not converge after "
                    f"{max_it} iterations.  ||F|| = {final_res:.4e}  "
                    f"G = {G:.4e}  H = {H:.4e}  λ = {float(self.lambda_nd):.4e}"
                )

        # run counters: accepted iterations and backtracks, counted apart
        if hasattr(self, "_n_newton_iters_total"):
            self._n_newton_iters_total += max(0, (nit + 1) - pol.n_bt)
            self._n_newton_backtracks_total += pol.n_bt
        # index of the Newton solve within the run
        self._newton_call_idx += 1


    def solve(self) -> None:
        """One solve at the current state: builds the solver objects where needed, selects the
        level-constraint path and runs the Newton solve, or the bypass for a cell with no fluid."""
        if getattr(self, "_solver_needs_rebuild", True):
            self._build_nonlinear_solver()

        # path of the level constraint for this solve; counted per path
        _path = self._level_constraint_prepare()
        _cnt = "_n_lc_" + _path
        if hasattr(self, _cnt):
            setattr(self, _cnt, getattr(self, _cnt) + 1)
        if _path == "no_fluid":
            self._no_fluid_bypass()
        else:
            self._newton_solve()
        if hasattr(self, "_n_newton_calls"):
            self._n_newton_calls += 1

    def run_steady(self) -> None:
        """Steady solve.  With the ripple on, the solve is repeated as ripple passes
        (``_ripple_run``), or as the pair loop for two rough surfaces; otherwise one solve."""
        if self._ripple_on:
            if self._ripple_pair():
                self._pair_ripple_run(self._run_steady_once)      # two rough surfaces
                return
            self._ripple_run(self._run_steady_once)
            return
        self._run_steady_once()

    def _run_steady_once(self) -> None:
        """One steady solve: squeeze term off, geometry refreshed, then ``_steady_solve``."""
        # counters and the Newton call index, read by _newton_solve
        self._reset_timers()
        self._phase = "steady"
        if hasattr(self, "transient_on"):
            self.transient_on.assign(0.0)
        self._refresh_geometry()         # no action in this class
        if self._ripple_on:
            self._ripple_level_begin("steady")
        self._steady_solve()
        if self._ripple_on:
            self._ripple_level_end("steady")

    @property
    def dP(self) -> float:
        """Pressure correction ``lambda_nd * p_scale`` [Pa]: the level of the cell pressure
        above the macroscale pressure."""
        return float(self.lambda_nd) * self.p_scale

    # -------------------------
    # Transient run
    # -------------------------
    def _compute_cfl_dt(self, Tf: float, cfl_target: float = 1.0,
                        T_start: float = 0.0) -> float:
        """Time step [s] at the CFL number ``cfl_target``: element size over the largest
        surface speed at ``T_start`` and ``Tf``.  ``Tf / 10`` where the speed is zero."""
        zeta = self.zeta
        Ux_candidates = [abs(2.0 * (zeta.Ux + zeta.Uxdot * T_start)),
                         abs(2.0 * (zeta.Ux + zeta.Uxdot * Tf))]
        Uy_candidates = [abs(2.0 * (zeta.Uy + zeta.Uydot * T_start)),
                         abs(2.0 * (zeta.Uy + zeta.Uydot * Tf))]
        Umax = max(Ux_candidates + Uy_candidates)
        n_elem = int(np.sqrt(self.mesh_m.num_cells() / 2) + 0.5)
        h_elem = min(self.Lx, self.Ly) / n_elem
        if Umax > 1e-30 and h_elem > 0:
            return cfl_target * h_elem / Umax
        else:
            return Tf / max(1, 10)

    def _update_P_re_nd(self, t_dim: float) -> None:
        """Sets the macroscale pressure ``P_re_nd = (zeta.P + zeta.Pdot t_dim) / p_scale``.
        t_dim: time [s].  ``P_re_nd`` is a Constant in every form."""
        zeta = self.zeta
        p_re_dim = zeta.P + float(zeta.Pdot) * t_dim
        self.P_re_nd.assign(p_re_dim / self.p_scale)

    def run(self, Tf: float, n_steps: int = 10, cfl_target: float = 1.0,
            dt_override: "float | None" = None,
            step_callback: "Callable | None" = None) -> None:
        """Transient run over the micro window, ending at ``Tf``.

        Tf: end time [s].  ``Tf <= 0`` or ``n_steps < 1`` gives ``run_steady()``, without the
            callback.
        n_steps: number of time steps.
        cfl_target: CFL number that sets the time step where ``dt_override`` is None.
        dt_override: fixed time step [s], or None.
        step_callback: ``step_callback(step, t_dim, dt_dim)``, called after the converged solve
            of each level: step 0 is the seed at the start time, steps 1 to ``n_steps`` end at
            ``Tf``.  The caller reads the solver state in it.

        With the ripple on, the whole run is pass 1 of ``_ripple_run``, with the ripple fixed
        in the frame of the rough surface.  Later passes solve the end level only
        (``_run_end_level_once``); with a callback every pass is a whole run and the callback
        is called in every pass (``self._ripple_pass`` gives the pass).  Two rough surfaces:
        one run, with the ripple converged at every level (``_pair_ripple_run``).
        """
        Tf = float(Tf)
        if Tf <= 0.0 or n_steps < 1:
            self.run_steady()
            return
        if self._ripple_on:
            if self._ripple_pair():
                # two rough surfaces: one run, the ripple converged at every level
                self._pair_ripple_run(lambda: self._run_once(Tf, n_steps, cfl_target, dt_override,
                                                             step_callback))
                return
            later = self._run_end_level_once if step_callback is None else None
            self._ripple_run(lambda: self._run_once(Tf, n_steps, cfl_target, dt_override,
                                                    step_callback), later)
            return
        self._run_once(Tf, n_steps, cfl_target, dt_override, step_callback)

    # -------------------------
    # Time level, BDF order and history: shared by the full run and the end-level pass
    # -------------------------
    def _set_time_level(self, t_dim: float) -> None:
        """Sets every time-dependent input at the time ``t_dim`` [s]: the time Constants, the
        advection velocities, the pattern velocity ``_Ux_dim_const`` / ``_Uy_dim_const``, the
        pressure gradients and the macroscale pressure.

        The film callback multiplies ``_Ux_dim_const`` by T to give the displacement of the
        surface pattern.  With the surface velocity ``2 (Ux + Uxdot t)`` the displacement is
        ``s(T) = 2 Ux T + Uxdot T^2``, so the Constant holds the time-averaged velocity
        ``s(T) / T = 2 Ux + Uxdot T`` and not the instantaneous one.
        """
        zeta = self.zeta
        t_nd = t_dim / self.t0
        self.T_nd_const.assign(t_nd)
        self.T_dim_const.assign(t_dim)

        Ux_t = zeta.Ux + t_dim * zeta.Uxdot
        Uy_t = zeta.Uy + t_dim * zeta.Uydot
        # instantaneous velocity, non-dimensional, for the advection term
        self.Ux_nd_const.assign(2.0 * Ux_t / self.Uref)
        self.Uy_nd_const.assign(2.0 * Uy_t / self.Uref)
        # time-averaged velocity of the surface pattern, scaled by the velocity ratio
        _alpha = float(self.velocity_ratio)
        self._Ux_dim_const.assign(_alpha * (2.0 * zeta.Ux + zeta.Uxdot * t_dim))
        self._Uy_dim_const.assign(_alpha * (2.0 * zeta.Uy + zeta.Uydot * t_dim))

        dpdx_t = zeta.dpdx + t_dim * zeta.dpdx_dot
        dpdy_t = zeta.dpdy + t_dim * zeta.dpdy_dot
        self.dpx_nd_const.assign((dpdx_t * self.Lx) / self.p_scale)
        self.dpy_nd_const.assign((dpdy_t * self.Ly) / self.p_scale)

        self._update_P_re_nd(t_dim=t_dim)

    def _set_bdf(self, step: int) -> None:
        """Sets the BDF coefficients of step ``step``: BDF1 at step 1, BDF2 from step 2.
        The seed sets its own BDF1 coefficients."""
        use_bdf2 = bool(step >= 2)
        self._have_two_hist = use_bdf2
        if use_bdf2:
            self.bdf_a0.assign(1.5)
            self.bdf_a1.assign(-2.0)
            self.bdf_a2.assign(0.5)
        else:
            self.bdf_a0.assign(1.0)
            self.bdf_a1.assign(-1.0)
            self.bdf_a2.assign(0.0)

    def _advance_history(self) -> None:
        """Shifts the film history of the BDF scheme by one level: ``h_hist_nm2`` takes
        ``h_hist_nm1``, and ``h_hist_nm1`` the projection of ``h_nd`` at the current state."""
        h_curr = self._project_to_Vper(self.h_nd)
        self.h_hist_nm2.assign(self.h_hist_nm1)
        self.h_hist_nm1.assign(h_curr)

    def _run_end_level_once(self) -> None:
        """A later ripple pass: the end level of the run only, with the film history of a
        full run.  Uses ``T_start``, ``dt_dim_run`` and ``n_steps_run`` of pass 1.

        The film history of the end level is rebuilt at levels n - 2 and n - 1 (level 0 is
        the seed at ``T_start``; for n = 1, level -1 is at ``T_start - dt`` with the film
        time only moved).  This holds as the film at a level depends on the geometry and the
        time only: the ripple is fixed in the frame of the surface and the film does not
        depend on the pressure.  The solve starts from the end level of the previous pass.
        The last history shift leaves ``h_hist_nm1 = h(T_end)`` and
        ``h_hist_nm2 = h(T_end - dt)``, as a full run does."""
        self._reset_timers()
        T_start, dt_dim, n = float(self.T_start), float(self.dt_dim_run), int(self.n_steps_run)
        for j in (n - 2, n - 1):                  # oldest first, so nm1 ends at level n - 1
            self._set_time_level(T_start + max(j, 0) * dt_dim)
            if j < 0:                             # n = 1: level -1, the film time only
                self.T_dim_const.assign(T_start - dt_dim)
            self._refresh_geometry()              # no action in this class
            self._advance_history()
        t_dim = T_start + n * dt_dim
        self._set_time_level(t_dim)
        self.dt_nd_const.assign(float(dt_dim / self.t0))
        self.transient_on.assign(1.0)
        self._set_bdf(n)
        self._refresh_geometry()
        self._update_squeeze_mean()
        self._phase = f"step {n}/{n} (end level)"
        self._ripple_level_begin(n)
        self._solve_level()
        self._ripple_level_end(n)
        self._advance_history()                   # nm1 = h(T_end)

    def _run_once(self, Tf: float, n_steps: int, cfl_target: float,
                  dt_override: "float | None",
                  step_callback: "Callable | None") -> None:
        """One transient run, for ``Tf > 0`` and ``n_steps >= 1``; arguments as in ``run``.

        The run starts at ``T_start = Tf - n_steps dt`` (at 0 with a longer step where that
        is negative): a seed solve at ``T_start`` with a synthesised film history, then
        ``n_steps`` steps to ``Tf``.  Sets ``T_start``, ``dt_dim_run``, ``n_steps_run`` and
        ``cfl_actual``.
        """
        self._reset_timers()

        if dt_override is not None:
            # fixed time step
            dt_dim = float(dt_override)
            T_start = Tf - n_steps * dt_dim
            if T_start < 0.0:
                T_start = 0.0
                n_steps = max(1, int(np.ceil(Tf / dt_dim)))
                dt_dim = Tf / n_steps
        else:
            # time step from the CFL number, evaluated twice as it depends on T_start
            dt_cfl = self._compute_cfl_dt(Tf, cfl_target, T_start=0.0)
            T_start = Tf - n_steps * dt_cfl
            if T_start < 0.0:
                T_start = 0.0
                n_steps = max(1, int(np.ceil(Tf / dt_cfl)))
                dt_dim = Tf / n_steps
            else:
                dt_cfl = self._compute_cfl_dt(Tf, cfl_target, T_start=T_start)
                T_start = Tf - n_steps * dt_cfl
                if T_start < 0.0:
                    T_start = 0.0
                    n_steps = max(1, int(np.ceil(Tf / dt_cfl)))
                    dt_dim = Tf / n_steps
                else:
                    dt_dim = dt_cfl

        dt_nd = dt_dim / self.t0
        zeta = self.zeta
        n_elem = int(np.sqrt(self.mesh_m.num_cells() / 2) + 0.5)
        h_elem = min(self.Lx, self.Ly) / n_elem
        Umax_candidates = [
            abs(2.0 * (zeta.Ux + zeta.Uxdot * T_start)),
            abs(2.0 * (zeta.Ux + zeta.Uxdot * Tf)),
            abs(2.0 * (zeta.Uy + zeta.Uydot * T_start)),
            abs(2.0 * (zeta.Uy + zeta.Uydot * Tf)),
        ]
        Umax = max(Umax_candidates)
        self.T_start = T_start
        self.dt_dim_run = dt_dim
        self.n_steps_run = n_steps
        self.cfl_actual = Umax * dt_dim / h_elem if h_elem > 0 else 0.0

        # --- Seed at T_start ---
        # One BDF1 solve at T_start with a film history built from the film at earlier
        # times, so the seed pressure balances the squeeze term of the moving surface.

        # 1. every time-dependent input at T_start
        self._set_time_level(T_start)

        # 2. film history at T_start - dt and T_start - 2 dt; the film depends on time
        #    through T_dim_const only
        self.T_dim_const.assign(T_start - dt_dim)
        self._refresh_geometry()         # no action in this class
        h_past_1 = self._project_to_Vper(self.h_nd)

        self.T_dim_const.assign(T_start - 2.0 * dt_dim)
        self._refresh_geometry()
        h_past_2 = self._project_to_Vper(self.h_nd)

        self.T_dim_const.assign(T_start)
        self._refresh_geometry()         # state of the seed

        self.h_hist_nm1.assign(h_past_1)
        self.h_hist_nm2.assign(h_past_2)

        # 3. seed solve: BDF1 with the squeeze term on
        self.dt_nd_const.assign(float(dt_nd))
        self.transient_on.assign(1.0)
        self.bdf_a0.assign(1.0)
        self.bdf_a1.assign(-1.0)
        self.bdf_a2.assign(0.0)

        self._update_squeeze_mean()

        self._phase = "transient-seed"
        if self._ripple_on:
            self._ripple_level_begin("seed")
        self._seed_solve()
        if self._ripple_on:
            self._ripple_level_end("seed")

        # 4. history shift, so step 1 sees h_hist_nm1 = h(T_start) and
        #    h_hist_nm2 = h(T_start - dt)
        self._advance_history()
        self._have_two_hist = True               # two history levels are set


        # callback at the seed state (step 0): every input at T_start, w and lambda_nd from the seed
        if step_callback is not None:
            step_callback(0, T_start, dt_dim)

        # --- time steps ---
        for step in range(1, n_steps + 1):
            self._phase = f"step {step}/{n_steps}"
            t_dim = T_start + step * dt_dim
            self._set_time_level(t_dim)
            self._set_bdf(step)

            self._refresh_geometry()     # no action in this class
            self._update_squeeze_mean()


            # --- solve of the level ---
            if self._ripple_on:
                self._ripple_level_begin(step)
            self._solve_level()
            if self._ripple_on:
                self._ripple_level_end(step)

            # callback at the converged state of the step
            if step_callback is not None:
                step_callback(step, t_dim, dt_dim)

            self._advance_history()          # nm2 <- nm1, nm1 <- h(t_dim)

    # -------------------------
    # Homogenisation
    # -------------------------
    def _record_pressure_ledger(self, Pst, fluid_area, area_total):
        """Sets ``pressure_ledger``, ``P_f``, ``P_patch`` and ``Pst_fluid`` [Pa].

        Pst: cell-mean pressure [Pa].  fluid_area, area_total: fluid and cell areas.
        ``P_f``: signed fluid pressure integrated over the fluid region, per total cell
        area; ``P_fluid_pos`` and ``P_fluid_neg`` are its positive and negative parts.
        ``P_patch = Pst - P_f``.  ``Pst_fluid``: mean over the fluid area, NaN with no fluid.
        Every term uses the same pressure and the same quadrature of the fluid indicator.
        """
        from microcell.pressure import DELTA_P_DEFINITION
        dxm = dx(domain=self.mesh_m)
        P_f = float(assemble(self.chi_fluid * self.P_dim * dxm) / area_total)
        if float(assemble((Constant(1.0) - self.chi_fluid) * dxm)) == 0.0:
            P_f = Pst  # no solid region: P_f is the cell mean
        P_pos = float(assemble(self.chi_fluid * conditional(gt(self.P_dim, 0), self.P_dim, 0) * dxm) / area_total)
        P_neg = float(assemble(self.chi_fluid * conditional(gt(self.P_dim, 0), 0, self.P_dim) * dxm) / area_total)
        self.pressure_ledger = {
            "delta_P_definition": DELTA_P_DEFINITION, "Pst": Pst,
            "P_f": P_f, "P_patch": Pst - P_f,
            # no fluid region: the mean over the fluid area is undefined
            "Pst_fluid": P_f * area_total / fluid_area if fluid_area > 0.0 else float("nan"),
            "P_fluid_pos": P_pos, "P_fluid_neg": P_neg,
        }

        self.P_f = P_f
        self.P_patch = Pst - P_f
        self.Pst_fluid = self.pressure_ledger["Pst_fluid"]

    def spatial_homogenisation(self):
        """Cell means and extrema of the solved state.

        Returns the 14-tuple ``(Qx, Qy, Pst, Fst, taust_x, taust_y, p_max, p_min, h_max,
        h_min, qymax, qymin, fmax, fmin)``, the same as the mixed class:
        Qx, Qy: mean flux [m^2/s].  Pst: mean pressure [Pa], equal to
        ``p_scale (P_re_nd + lambda_nd)`` as phi_c_nd and w have zero mean.
        Fst: fluid area fraction.  taust_x, taust_y: mean shear stress [Pa].
        p_max, p_min: fluid pressure extrema [Pa], NaN with no fluid.  h_max, h_min: film
        extrema [m].  qymax, qymin: extrema of the y flux.  fmax, fmin: 1 or 0, and 0.
        Also sets ``pressure_extrema``, ``fluid_traction``, the pressure ledger and the
        Couette and Poiseuille parts of the flux.
        """
        area_total = float(assemble(Constant(1.0) * dx(domain=self.mesh_m)))

        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.zeta.H) * self.h_nd, self.Vfull)

        h_arr = Hfull.vector().get_local()

        # p_max, p_min: signed pressure at the vertices the nodal level-set test classes as
        # fluid; NaN where there are none.  h_max, h_min are taken over the whole cell:
        # inside a contact patch the film is the overclosure.
        self.pressure_extrema = sample_fluid_pressure_extrema(self, film_dofs_m=h_arr)
        p_max = self.pressure_extrema.p_max_Pa
        p_min = self.pressure_extrema.p_min_Pa
        h_max = float(np.nanmax(h_arr))
        h_min = float(np.nanmin(h_arr))

        # cell-mean pressure [Pa]
        Pst = float(assemble(Pfull * dx(domain=self.mesh_m)) / area_total)

        # fluid area fraction
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = assemble(chi_dg0 * dx(domain=self.mesh_m))
        Fst = float(fluid_area / area_total)

        self._record_pressure_ledger(Pst, fluid_area, area_total)

        fmax = 1.0 if fluid_area > 0.0 else 0.0
        fmin = 0.0

        # --- fluxes: Couette plus Poiseuille, scaled by Uref H / 2 ---
        gradP_nd = self.grad_cell(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd
        q_couette_nd = self.F_adv_nd
        q_poiseuille_nd = -self.K_nd * gradP_nd

        dim_scale = Constant(self.Uref * self.zeta.H / 2)
        q_dim = dim_scale * q_nd
        q_couette_dim = dim_scale * q_couette_nd
        q_poiseuille_dim = dim_scale * q_poiseuille_nd

        # cell means of the fluxes; a subclass can sample them differently
        (Qx, Qy, Qx_couette, Qy_couette,
         Qx_poiseuille, Qy_poiseuille) = self._flux_means(
            q_dim, q_couette_dim, q_poiseuille_dim, area_total)

        qy_full = project(q_dim[1], self.Vfull)
        qy_arr = qy_full.vector().get_local()
        qymax = float(np.nanmax(qy_arr))
        qymin = float(np.nanmin(qy_arr))

        # --- shear stress: tau = eta U / h + (h / 2) grad p ---
        Ux_surf = float(self.Ux_nd_const) * self.Uref
        Uy_surf = float(self.Uy_nd_const) * self.Uref

        gradP_dim = gradP_nd * Constant(self.p_scale * self.invLx)
        eta_dim_expr = Constant(float(self.settings.eta0))      # viscosity [Pa s]
        H_dim_expr = Constant(self.zeta.H) * self.h_nd
        H_floor_expr = H_dim_expr + Constant(1e-30)

        tau_x_field = (
            (eta_dim_expr * Constant(Ux_surf) / H_floor_expr)
            + (H_dim_expr / 2) * gradP_dim[0]
        )
        tau_y_field = (
            (eta_dim_expr * Constant(Uy_surf) / H_floor_expr)
            + (H_dim_expr / 2) * gradP_dim[1]
        )

        taust_x = float(assemble(tau_x_field * dx(domain=self.mesh_m)) / area_total)
        taust_y = float(assemble(tau_y_field * dx(domain=self.mesh_m)) / area_total)

        # fluid traction at the quadrature points of the integrals above, zero where the level-set
        # test says solid.  taust is the integral over the whole cell.
        self.fluid_traction = sample_fluid_traction(self, film_dofs_m=h_arr, existing_mask=None)

        self.Qx_couette = Qx_couette
        self.Qy_couette = Qy_couette
        self.Qx_poiseuille = Qx_poiseuille
        self.Qy_poiseuille = Qy_poiseuille

        return (
            Qx, Qy, Pst, Fst, taust_x, taust_y,
            p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin,
        )
