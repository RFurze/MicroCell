"""
Phase-Neutral Microscale Solver (MicroPhaseNeutralSolver)
=========================================================

This module implements the phase-neutral pressure decomposition described in
``docs/phase_neutral_solver_plan.md``, Stage 1 (steady-state).

Mathematical background
-----------------------
The existing solver pins the nondimensional pressure w at the corner (0,0),
creating a phase-dependent gauge.  This solver instead decomposes:

    P_nd = P_re_nd + phi_c_nd + lambda_nd + w_fluc

where:
  - P_re_nd   = p0 / p_scale  (macroscale reference, scalar Constant)
  - phi_c_nd  = dpx*(x-0.5) + dpy*(y-0.5)  (zero-mean linear ramp)
  - lambda_nd = scalar pressure-level correction, SOLVED as part of the system
  - w_fluc    = zero-mean periodic fluctuation field (enforced exactly via
                the Lagrange multiplier mu_nd)

The cell-mean pressure is P_re + lambda (phi_c and w_fluc are both zero-mean).
The pressure correction returned to the macroscale is simply dP = lambda_dim.

Closure equation:  G = β·λ_nd − γ_nd·⟨spos(−P_nd)⟩ = 0
    At equilibrium: λ = γ·⟨spos(−P_nd)⟩/β ≥ 0.
    d = ∂G/∂λ = β + γ·⟨spos'⟩ ≥ β > 0 always (positive-definite).

Algorithmic change: Schur complement Newton
-------------------------------------------
Instead of a Dirichlet BC at the corner, the solver uses a Schur complement
extension of the N×N Newton system to an (N+2)-dimensional augmented system:

    (i)  J dw  + c dλ  + m dμ  = -F_w
    (ii) g^T dw + d dλ          = -G
    (iii) m^T dw                = -H

The Schur complement eliminates (dλ, dμ) via two extra MUMPS back-substitutions
(negligible cost vs. the factorisation).  See plan for full derivation.

Stage 1 scope: steady-state only.  Transient infrastructure is carried over
from the template but is not tested here (Stage 2).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
from fenics import *  # legacy FEniCS

from microcell import ripple as _rp   # [option H] numpy-only operator, traction and checks
# [extrema M1] the shared fluid pressure extrema sampler (base, mixed and contact classes);
# [extrema M2b] the fluid traction sampler (the same classes)
from microcell.samplers import sample_fluid_pressure_extrema, sample_fluid_traction

# [phase D1] pinned per process by the driver / the pool worker
# (utils/form_compiler.py, QDEG_MICRO_PHASE_NEUTRAL = 2); this import-time 2
# stands only where nothing pinned it, exactly as before.
from microcell import form_compiler as _qdeg
_qdeg.set_at_import(2, __name__)  # was: parameters["form_compiler"]["quadrature_degree"] = 2  # was 4; qdeg=2
# verified equivalent for dP / Qx / Qy on this problem (see
# quadrature_diag_summary.csv) and yields ~15-20% speedup at stiff
# Ah/H.  Override locally before solver construction if higher
# accuracy is needed (e.g. high-order elements, stiffer physics).
set_log_level(LogLevel.ERROR)
set_log_active(False)

# -------------------------------------------------------------------
# Logging / utilities
# -------------------------------------------------------------------
# The parent disabled logging for the whole process at this point (MICRO_DEBUG_LOG unset).  A
# library must not do that to its caller: this module only owns its own logger.
logger = logging.getLogger(__name__)

#: cap of the cavitation smoothing width in nondimensional units (the parent's MICRO_EPSP_ND_CAP
#: default; see ``_nondimensionalise``)
_EPSP_ND_CAP = 1.0


# -------------------------------------------------------------------
# Inputs and fixed values
# -------------------------------------------------------------------
# The solver reads its inputs where the caller wrote them (MicroCell WP4); there is no
# solver-side settings or parameters class, and every default is ``MicroSettings``'s:
#   self.zeta            a ``microcell.Zeta`` (``microsolver`` zeroes the rates of a steady solve)
#   self.t_end           the time window [s] (0.0 for a steady solve)
#   self.settings        the ``microcell.MicroSettings`` (with ``.contact`` and ``.ripple``)
#   self.velocity_ratio  fraction of the roughness on the moving surface
#   self.pair            True for a stationary + translating pair (the pair branch)
#   self.h_min           the contact onset film [m]: ``settings.h_min`` with contact on, 0.0 in
#                        full film (0.0 keeps the mixed masking off)
#
# Notes on the settings read here (definitions and defaults: ``microcell.api.MicroSettings``)
# --------------------------------------------------------------------------------------
# ``spos_form`` [WP-B 2026-09-15]: which smoothed positive part the cavitation penalty (and its
# derivative) uses.
#   "spline" -- one-sided C1 quadratic spline, spos(z) == 0 for z <= 0.  NO closure bias at
#               positive pressure.
#   "sqrt"   -- legacy symmetric 0.5*(z + sqrt(z^2+eps^2)).  Its eps^2/(4P) residue at P > 0
#               biases the phase-neutral closure, which with eps_smooth = 1e5 Pa put
#               dP = 15.8 MPa on every low-pressure node of Wang2008-Mixed2 (fix plan
#               2026-09-15, fact F3).
#
# ``beta``: the normalised closure parameter of the scalar equation
#     G = beta * lambda_nd - gamma_nd * mean(spos(-P_nd)) = 0.
# At equilibrium lambda_nd = gamma_nd * mean(spos(-P_nd)) / beta >= 0, and
# d = dG/dlambda = beta + gamma*mean(spos') >= beta > 0 always (positive-definite).  With the
# normalised form beta is dimensionless; values in [0.1, 10] are typically stable.
#
# ``jacobian_reg_eps``: Tikhonov shift ``eps * I`` applied to the assembled Reynolds Jacobian
# ``_A_mat`` before MUMPS factorises it.  The Schur complement Newton solver factorises the
# N x N Reynolds block alone (the lambda, mu rows are eliminated externally).  On a periodic
# mesh with no active cavitation penalty (``spos'(-P) ~ 0``) this block carries a constant-mode
# null space and MUMPS aborts with DIVERGED_PC_FAILED / zero pivot.  A tiny shift lifts that
# single eigenvalue above MUMPS's pivot threshold: the Newton iteration count was identical
# across ``eps in [1e-12, 1e-4]`` in the parent's regularise sweep (the shift acts on the null
# eigenvalue, not on any data-bearing mode).  0.0 disables it (and reproduces the zero-pivot
# failure on non-cavitating periodic cases).  Since 2026-10-15 the shift may be negative
# (``_apply_jacobian_reg_shift`` says why): the parent's production runs take +1e-10
# historically and -1e-7 with the ``pressure-accounting`` recipe.
#
# ``ripple`` [option H, 17 Oct 2026], the micro fluid deformation ``ripple_halfspace``
# (parent docs/notes/ripple_halfspace_plan_2026-10-17.md, decisions O1-O10).  ``None`` leaves
# every UFL graph and code path as without it.  On: the film gains u_r = H[s], the zero-mean
# periodic two-body half-space response to the cell's own fluid traction (combined plan 18.15),
# held fixed during each micro run ("pass") in the rough surface's frame and updated between
# passes until the residual acceptance holds (O6; the acceptance of
# docs/notes/ripple_fast_loop_plan_2026-10-19.md section 4.6: residual within
# ``ripple.tol`` x H, contracted by half since the previous pass, end-level contact set
# unchanged).  Pass 1 is a whole run; later passes solve the end level only where the caller
# allows it (``_ripple_run``, section 4.7).  ``MicroSettings.E_star`` is the resolved two-body modulus
# [Pa]; ``ripple.tol`` the deformation budget relative to H; ``ripple.maxpass`` the pass cap (a
# pass can be accepted from pass 1); ``ripple.omega`` the initial and largest relaxation (floor
# 0.1x).
#
# ``level_constraint`` [fluid-mean WP2, 30 Oct 2026], the pressure-level constraint (parent
# docs/notes/micro_fluid_mean_constraint_plan_2026-10-30.md rev 3, decisions D1-D14).
# "whole_cell": (1/A) int P = P_re + lambda.  "fluid_mean_v1": (1/A_f) int chi P = P_re + lambda,
# chi the ledger's sharp chi_fluid (D2, D3); implemented by the mixed class (the full-film
# solver has no patch, so it always takes the whole-cell path).  ``patch_pin`` (None = on with
# fluid_mean_v1, D4): the nodal penalty w_i -> 0 at the nodes with no fluid quadrature point in
# their support, weight ``patch_pin_kappa`` x m_i (nondimensional).

# Fixed values.  The parent's settings dataclasses carried these as fields; the API never set
# them and every reference stores exactly these values (``tests/test_parity.py`` checks that).

#: [Phase 5.5] adaptive damping of the Schur Newton solver (``DampingPolicy``; the April-2026
#: damping-policy comment in ``_newton_solve`` gives the rationale of each number)
ADAPTIVE_DAMPING = True
#: None -> the dynamic floor max(omega/32, 1e-3), computed at solve time from
#: ``relaxation_parameter`` (cannot be a static value)
OMEGA_MIN = None
NEWTON_BACKTRACK_MAX = 10
NEWTON_BACKTRACK_TOL = 2.0
NEWTON_BACKTRACK_WARMUP_ITERS = 3
OMEGA_RECOVERY_FACTOR = 1.2
OMEGA_CAP_SHRINK_FACTOR = 0.85
#: the parent's print switches.  The prints they guarded are removed (WP5); the parity gate still
#: checks that a reference was made with both off
PRINT_NEWTON = False
PRINT_TIMING = False
#: the closure parameter of the Newton solve
BETA = 1.0
#: the fluid-mean level constraint's pin of the patch degrees of freedom (None: on) and its weight
PATCH_PIN = None
PATCH_PIN_KAPPA = 1.0


#: [fluid-mean WP2] the level-constraint formulations (the tag; ehl.pressure.LEVEL_CONSTRAINTS)
LEVEL_CONSTRAINT_WHOLE_CELL = "whole_cell"
LEVEL_CONSTRAINT_FLUID_MEAN = "fluid_mean_v1"


#: [option H] the per-run counters ``_reset_timers`` zeroes; the pass loop sums
#: them over the passes of one task so callers read task totals, as before.
_PASS_COUNTERS = (
    "_t_newton_solve", "_t_history_update",
    "_t_diagnostics", "_t_step_overhead", "_t_init", "_n_newton_calls",
    "_t_asm_total", "_t_lin_total",
    "_t_upd_total", "_n_newton_iters_total", "_n_newton_backtracks_total",
    # [fluid-mean WP2 (S6)] Newton solves per constraint path
    "_n_lc_whole_cell", "_n_lc_fluid_mean", "_n_lc_no_fluid",
)

#: [phase-table WP3, 4 Nov 2026] the inner-iteration cap of the converged per-step ripple of a two-member
#: (stationary + translating) surface field: the study's ``S3_MAXIT`` (tests/ripple/ripple_s3.py,
#: tests/micro_averaging/study_solver.py ``converged_class``), a module constant, not a setting (plan D5,
#: review F11).  ``MICRO_RIPPLE_MAXPASS`` counts passes of the one-member loop and does not apply here.
PAIR_RIPPLE_MAXIT = 30


# -------------------------------------------------------------------
# Periodic BC
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
    """[Phase 6.4] Adaptive damping policy for the Schur Newton solver.

    Extracted verbatim from ``_newton_solve`` (the April-2026 policy:
    backtrack-halving with a per-solve recovery ceiling that ratchets
    down on every confirmed overshoot).  Owns the omega state and the
    backtrack budget; the Newton loop owns state snapshots, step
    reapplication and reassembly. Backtracking stops once the minimum step
    has been evaluated; retrying that identical candidate cannot help. See the extensive knob
    documentation at the fixed values of this module ([Phase 5.5]): adaptive_damping,
    omega_min (None sentinel => dynamic max(omega/32, 1e-3) floor),
    newton_backtrack_max / _tol / _warmup_iters, omega_recovery_factor,
    omega_cap_shrink_factor.
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
        # "next step size"; shrinks on revert, grows on accept
        self.omega_current = omega_base
        # Per-solve cap on omega recovery: ratcheted DOWN each time a
        # backtrack fires, never climbs back up (April-2026 policy core).
        self.omega_cap = omega_base
        self.omega_cap_floor = max(self.omega_min * 2.0, omega_base * 0.2)
        # prev_res_sq: combined residual at the previous accepted iterate.
        self.prev_res_sq: "float | None" = None
        self.bt_count_seq = 0   # consecutive backtracks at current iter
        self.n_bt = 0           # total reverts this solve (reported)
        self.diverged = False   # last wants_backtrack() divergence verdict

    def wants_backtrack(self, nit: int, res_sq: float,
                        have_step: bool) -> bool:
        """Request a backtrack only when it produces a smaller candidate step.

        NaN/Inf residuals always count as divergence regardless of the
        warmup window; the ratio test arms at ``bt_warmup_iters``.
        ``self.diverged`` is left set for the accept path (an exhausted
        budget or reached floor with diverged=True suppresses omega recovery).
        The existing exhausted-search acceptance policy is unchanged.
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
        """Halve omega and ratchet the recovery ceiling down (verbatim)."""
        self.omega_current = max(self.omega_current * 0.5, self.omega_min)
        # Cap tracks omega_current*2 as an upper bound: we know
        # omega_current works, so recovery may reach at most 2x that,
        # even if the geometric ratchet would permit more.
        self.omega_cap = max(
            min(self.omega_cap * self.omega_cap_shrink,
                self.omega_current * 2.0),
            self.omega_cap_floor,
        )
        self.bt_count_seq += 1
        self.n_bt += 1

    def on_accept(self, nit: int, res_sq: float) -> None:
        """Accepted step: recover omega toward the per-solve cap (verbatim).

        If the backtrack budget was exhausted (``diverged`` still True),
        omega stays at its reduced value — the solve is struggling and a
        small step is safer.
        """
        if nit > 0 and not self.diverged:
            self.omega_current = min(self.omega_cap,
                                     self.omega_current * self.omega_recovery)
        self.bt_count_seq = 0
        self.prev_res_sq = res_sq


class MicroPhaseNeutralSolver:
    """Phase-neutral microscale EHL solver (steady-state, Stage 1).

    Drop-in compatible with ``MicroMixedSolver``.  Accepts the same inputs
    and returns the same 14-tuple from ``spatial_homogenisation()``.

    The key difference is that no corner Dirichlet BC is applied.  Instead,
    ``lambda_nd`` (a scalar Constant) absorbs the cell-mean pressure
    correction and ``mu_nd`` (a plain float) enforces the zero-mean
    constraint on ``w_fluc``.  Both are solved as part of the Schur
    complement Newton iteration.
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
        # ---- persistent inputs (the header of this module says what each is) ----
        self.mesh_m = mesh_m
        self.zeta = zeta
        self.t_end = t_end
        self.settings = settings
        self.velocity_ratio = float(velocity_ratio)
        self.pair = bool(pair)
        self.h_min = self._contact_onset_film()
        # [WP-G 2026-09-15] Which sub-solve is running, for the worker's
        # [MICRO-FAIL-PN] report: "steady" | "transient-seed" |
        # "step k/N".  A
        # label only -- nothing branches on it.
        self._phase = "init"
        # [WP-B 2026-09-15] Cavitation-smoothing form, resolved ONCE per
        # solver instance (it selects the UFL graph, so it must be fixed
        # before _setup_pressure_expressions / _setup_forms run and must
        # not change under reset_task).
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
        # spaces BEFORE film/mask: the ripple and the fft contact allocate
        # their nodal fields on ``Vper`` inside _setup_film_and_mask.
        self._setup_spaces_and_unknowns()
        self._setup_film_and_mask()
        self._setup_pressure_expressions()
        self._setup_mean_vector()
        self._setup_forms()
        # No point-constraint BC — Schur complement handles the null mode.
        self._solver_needs_rebuild: bool = True
        self._build_nonlinear_solver()

        if auto_solve:
            self.run_steady()

    def _contact_onset_film(self) -> float:
        """``self.h_min`` [m].  Full film: 0.0, which keeps the mixed masking off whatever
        ``settings.h_min`` holds (the mixed class returns the setting)."""
        return 0.0

    # -------------------------
    # [Micro-perf P2.3c] Task re-targeting for per-worker solver reuse
    # -------------------------
    def reset_task(self, zeta, t_end: float, velocity_ratio: float) -> None:
        """Re-target this solver at a new task WITHOUT rebuilding
        spaces / periodic constraint / mass-matrix LU / forms / assembler.

        Transcribes the scalar+Constant part of ``__init__`` exactly; a
        reused solver after ``reset_task`` must be indistinguishable from
        a freshly constructed one (property test:
        ``microscale/tests/test_solver_reuse_parity.py`` asserts bitwise
        equality of the full task output tuple).  The persistent forms
        consume the new task ONLY through the Constants assigned here —
        the P2.3a audit (_refactor_log/P2.3a_constants_audit.md) is the
        completeness argument, and any Constant missed here is a silent
        freeze of the previous task's scaling, which is exactly what the
        property test exists to catch.

        The cell size, like every other setting, is the solver's own (``self.settings``): a task
        with another cell size needs a fresh solver.  MicroCell WP4: ``microsolver`` builds a solver
        per call and does not use this; it takes the task state the constructor takes.
        """
        self.zeta = zeta
        self.t_end = t_end
        self.velocity_ratio = float(velocity_ratio)

        # --- mirror _setup_time_controls (assign, not recreate) ---
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

        # --- mirror _nondimensionalise (floats recomputed) ---
        self._nondim_scalars()
        self.invLx.assign(1.0 / self.Lx)
        self.invLy.assign(1.0 / self.Ly)

        # --- mirror _setup_film_and_mask ---
        self.hmin_nd_const.assign(self.hmin_nd)
        self.delta_h_nd_const.assign(self.delta_h_nd)
        if self._ripple_on:                 # [option H] a task starts at u_r = 0 (local cell)
            self._ripple_reset()
        self.Ux_nd_const.assign(2.0 * self.Ux_nd)
        self.Uy_nd_const.assign(2.0 * self.Uy_nd)

        # --- mirror _setup_pressure_expressions ---
        self.dpx_nd_const.assign(self.dpx_nd)
        self.dpy_nd_const.assign(self.dpy_nd)
        self.P_re_nd.assign(float(zeta.P) / self.p_scale)
        self.lambda_nd.assign(0.0)
        self.mu_nd = 0.0
        self.p_scale_const.assign(self.p_scale)
        self.omega_const.assign(self.omega)
        self.epsP_nd_const.assign(self.epsP_nd)

        # --- mirror _setup_forms (incl. _build_forms constants) ---
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

        # --- solution / history state (fresh-Function equivalents) ---
        self.w.vector().zero()
        self.w.vector().apply("insert")
        for _f in (self.h_hist_nm1, self.h_hist_nm2,
                   self.mask_hist_nm1, self.mask_hist_nm2):
            _f.vector().zero()
            _f.vector().apply("insert")
        self._have_two_hist = False

        # --- subclass extras (mixed pattern-flux constants etc.) ---
        self._reset_task_subclass()

    def _reset_task_subclass(self) -> None:
        """[Micro-perf P2.3c] Hook for variant subclasses with additional
        task-varying form Constants.  Base: no-op."""
        return None

    # -------------------------
    # Timing infrastructure
    # -------------------------
    def _reset_timers(self) -> None:
        self._t_newton_solve = 0.0
        self._t_history_update = 0.0
        self._t_diagnostics = 0.0
        self._t_step_overhead = 0.0
        self._t_init = 0.0
        self._n_newton_calls = 0
        self._t_asm_total = 0.0
        self._t_lin_total = 0.0
        self._t_upd_total = 0.0
        self._n_newton_iters_total = 0
        # Number of backtracking reverts triggered by adaptive damping
        # (see _newton_solve).  Reported by the diagnostic harness so we
        # can correlate iteration-count drops with backtracks actually
        # happening.
        self._n_newton_backtracks_total = 0
        # [fluid-mean WP2 (S6)] how many Newton solves took each level-constraint path
        self._n_lc_whole_cell = 0
        self._n_lc_fluid_mean = 0
        self._n_lc_no_fluid = 0
        self._newton_call_idx = 0

    # -------------------------
    # Setup helpers
    # -------------------------
    def _setup_coordinates(self) -> None:
        self.x, self.y = SpatialCoordinate(self.mesh_m)

    def _setup_time_controls(self) -> None:
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
        # [Micro-perf P2.3c] Split: float-only core in _nondim_scalars()
        # (re-runnable by reset_task without touching UFL), Constants here.
        self._nondim_scalars()
        self.invLx = Constant(1.0 / self.Lx)
        self.invLy = Constant(1.0 / self.Ly)

    def _nondim_scalars(self) -> None:
        """Float-only nondimensionalisation (verbatim ex-_nondimensionalise).

        Creates NO dolfin objects — safe to re-run on a cached solver.
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

        # [WP-F 2026-09-15] The corner-pressure block moved ABOVE t0 (it
        # depends on neither t0 nor Uref) so the optional Uref floor below
        # can see the macroscale pressure scale.  Values unchanged.
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

        # Reynolds-equation pressure scale: the natural pressure the
        # thin-film flow generates from Couette transport alone.  Used as
        # the floor instead of 1.0 so that when macroscale pressure inputs
        # are near zero (cavitation boundary / cavitation region), the
        # nondimensionalisation keeps omega ~ O(1) and gamma_nd properly
        # scaled against the Reynolds transport rather than being enormous.
        p_rey = (6.0 * self.settings.eta0 * self.Uref * self.Lx) / (zeta.H ** 2)
        self.p_scale = max([p_rey] + [abs(v) for v in (corner_vals_t0 + corner_vals_t1)])

        self.dpx_nd = (zeta.dpdx * self.settings.xmax) / self.p_scale
        self.dpy_nd = (zeta.dpdy * self.settings.ymax) / self.p_scale
        self.Ux_nd = zeta.Ux / self.Uref
        self.Uy_nd = zeta.Uy / self.Uref
        self.hmin_nd = self.h_min / zeta.H
        delta_h, eps_smooth = s.delta_h, s.eps_smooth
        self.delta_h_nd = (delta_h / zeta.H) if float(delta_h) != 0.0 else 0.0
        # [WP-F 2026-09-15] Cap the SMOOTHING width in nondimensional
        # units.  epsP_nd = eps_smooth / p_scale, and p_scale collapses to
        # p_rey on a cell with no macroscale pressure; at a reversal
        # (Uref ~ 1e-18) p_rey goes to ~0 and epsP_nd explodes — with the
        # legacy symmetric spos that alone puts lambda = gamma*eps/2 into
        # the closure and the Newton solve returns NaN.  Those are the
        # four corner tasks (447/492/539/589, H = c, P = 0) that failed in
        # EVERY batch of Wang2008-Mixed2 (fix plan F5e / WP-F).
        # A smoothing width WIDER than the pressure scale it smooths is
        # meaningless in any case, so the cap is 1.0 (``_EPSP_ND_CAP``); it can
        # only bind where p_scale < eps_smooth.
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
        # Cap gamma_nd to prevent blowup when p_scale is very small
        # (thick film + zero macroscale pressure).  The penalty must be
        # strong enough to enforce cavitation (> O(100)) but not so stiff
        # that it dominates the Reynolds transport and distorts the
        # pressure field / causes Newton non-convergence.
        _GAMMA_ND_CAP = 1e5
        gamma_nd_raw = self.settings.penalty_gamma / self.p_scale
        self.gamma_nd = min(gamma_nd_raw, _GAMMA_ND_CAP)
        self.omega = (zeta.H**2 * self.p_scale) / (6.0 * self.settings.eta0 * self.Lx * self.Uref)

    def grad_phys(self, q):
        return as_vector((Dx(q, 0) * self.invLx, Dx(q, 1) * self.invLy))

    def grad_cell(self, q):
        return as_vector((Dx(q, 0), Dx(q, 1)))

    def _call_ht(self, x, y):
        try:
            return self.ht_cb(
                x, y, self.settings.xmax, self.settings.ymax,
                self.H0_const, self.HT_const, self.T_dim_const,
                self._Ux_dim_const, self._Uy_dim_const,
            )
        except TypeError:
            return self.ht_cb(x, y, self.settings.xmax, self.settings.ymax, self.H0_const)

    def _setup_film_and_mask(self) -> None:
        s = self.settings

        H_dim = self._call_ht(self.x, self.y)
        self.h_base_nd = H_dim / self.H0_const
        self.h_nd = self.h_base_nd
        # [option H, 17 Oct 2026] the ripple u_r / H0 on Vper, in the film only when the
        # option is on: the OFF graph is the one above, unchanged.
        self._ripple_setup()
        if self._ripple_on:
            self.h_nd = self.h_nd + self.ur_nd_func

        # [Micro-perf P2.3b] hmin_nd / delta_h_nd promoted from anonymous
        # inline Constants to stored attributes so a future per-worker
        # solver cache can .assign() the per-task nondimensionalisation.
        # ONE attribute per former inline site — the UFL graph (number
        # and position of distinct Constant leaves) is unchanged.
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
        self.pbc = PeriodicBoundary()
        self.Vper = FunctionSpace(self.mesh_m, "CG", 1, constrained_domain=self.pbc)
        self.Vfull = FunctionSpace(self.mesh_m, "CG", 1)
        self.Vdg0 = FunctionSpace(self.mesh_m, "DG", 0)

        # w = w_fluc: zero-mean periodic fluctuation (the primary FEM unknown)
        self.w = Function(self.Vper, name="w_fluc")
        self.v = TestFunction(self.Vper)
        self.dw = TrialFunction(self.Vper)

        # --- Cached L^2 projector onto Vper for BDF history updates ------
        # The Vper mass matrix depends only on the mesh and function space,
        # so we assemble and LU-factorise it ONCE here and reuse the factor
        # for every history snapshot taken during run() (see
        # _project_to_Vper).  This replaces per-call project() which would
        # otherwise re-assemble + re-factorise on every time step.
        # Switched from interpolate() because legacy DOLFIN's interpolate
        # does not accept raw UFL Sum/Conditional expressions (AttributeError
        # on _cpp_object when fed self.h_nd or self.mask).
        _u_tr = TrialFunction(self.Vper)
        _v_te = TestFunction(self.Vper)
        self._proj_Vper_v_test = _v_te  # reused for RHS assembly
        self._M_Vper = assemble(_u_tr * _v_te * dx(domain=self.mesh_m))
        self._M_Vper_lu = LUSolver(self._M_Vper, "mumps")
        self._M_Vper_lu.parameters["symmetric"] = True
        # Do NOT call set_operator() again on this solver — the MUMPS
        # factor will be reused across calls as long as the operator
        # is untouched.

    def _setup_pressure_expressions(self) -> None:
        """Set up the phase-neutral pressure decomposition.

        P_nd = P_re_nd + phi_c_nd + lambda_nd + w_fluc

        phi_c_nd is zero-mean (centred ramp); lambda_nd is solved as a scalar
        Constant; w_fluc is zero-mean (enforced by mu_nd Lagrange multiplier).
        """
        zeta = self.zeta

        # Nondimensional gradient constants (updated each transient step)
        self.dpx_nd_const = Constant(self.dpx_nd)
        self.dpy_nd_const = Constant(self.dpy_nd)

        # Zero-mean linear ramp: ∫(x-0.5)dx = 0 on [0,1], so phi_c is zero-mean
        self.phi_c_nd = (
            self.dpx_nd_const * (self.x - 0.5)
            + self.dpy_nd_const * (self.y - 0.5)
        )

        # Macroscale reference pressure level (scalar, updated each transient step)
        self.P_re_nd = Constant(float(zeta.P) / self.p_scale)

        # Scalar pressure-level correction (primary new unknown, solved via Schur)
        self.lambda_nd = Constant(0.0)

        # Lagrange multiplier for zero-mean constraint on w_fluc (plain float)
        self.mu_nd: float = 0.0

        # Full nondimensional pressure
        self.P_nd = self.P_re_nd + self.phi_c_nd + self.lambda_nd + self.w

        # Dimensional pressure (Pa)
        # [Micro-perf P2.3b] p_scale / omega / epsP_nd promoted to stored
        # attributes (task-varying nondimensionalisation baked into
        # persistent forms).  Graph unchanged — see _setup_film_and_mask.
        self.p_scale_const = Constant(self.p_scale)
        self.P_dim = self.p_scale_const * self.P_nd

        # Isoviscous, incompressible lubricant: eta/eta0 and rho/rho0 are one.  The dimensional
        # viscosity enters omega and the dimensional shear; reset_task updates omega.
        # Masked transport coefficients
        self.omega_const = Constant(self.omega)
        self.K_nd = self.omega_const * self.h_nd**3 * self.mask
        self.F_adv_nd = (self.U_nd * self.h_nd) * self.mask

        # spos(-P_nd) for cavitation penalty
        # [WP-B 2026-09-15] form-dispatched: one-sided spline by default,
        # legacy symmetric sqrt under spos_form="sqrt".
        self.epsP_nd_const = Constant(self.epsP_nd)
        self.negP_pos = self._spos(-self.P_nd, self.epsP_nd_const)

        # [Phase 2.4 seam S2] Cavitation-penalty weight.  Base: the fluid
        # mask.  Variant subclasses (mixed contact model) replace this with
        # a chi-blended weight in their _setup_pressure_expressions.
        self._cav_weight = self.mask


    def _setup_mean_vector(self) -> None:
        """Precompute the mean vector m_i = ∫ φ_i dx and cell area.

        This vector is constant for a given mesh.  It is the key ingredient for:
          - the zero-mean constraint H = m^T w / area
          - the mu contribution to the residual: mu * m
          - the Schur complement back-substitution z3 = J^{-1} m
        """
        v_test = TestFunction(self.Vper)
        self._mean_vec = assemble(v_test * dx(domain=self.mesh_m))
        self._cell_area = assemble(Constant(1.0) * dx(domain=self.mesh_m))

    # [fluid-mean WP1, 30 Oct 2026; plan docs/notes/micro_fluid_mean_constraint_plan_2026-10-30.md
    # rev 3 section 5] The pressure-level constraint as two hooks, so a subclass can hold the level
    # over another region.  The defaults are today's whole-cell constraint, bit for bit: the same
    # vector object and the same arithmetic as the inline code they replace.
    def _constraint_vector(self):
        """The constraint vector of the Newton system: the residual's mu term, the Schur vector
        ``z3 = J^{-1} m`` and the constraint residual.  Default: the whole-cell ``m_i = int phi_i``."""
        return self._mean_vec

    def _constraint_residual(self, m, area: float) -> float:
        """The constraint residual H (normalised by the cell area).  Default: ``m^T w / area``."""
        return float(m.inner(self.w.vector())) / area

    # [fluid-mean WP2] which constraint the next Newton solve uses.  Base: the whole cell always
    # (the full-film solver has no contact patch); ``fluid_mean_v1`` is accepted and is the whole
    # cell here by construction.  The mixed class holds the fluid mean (and the no-fluid bypass).
    def _level_constraint_prepare(self) -> str:
        lc = self.settings.level_constraint
        if lc not in (LEVEL_CONSTRAINT_WHOLE_CELL, LEVEL_CONSTRAINT_FLUID_MEAN):
            raise ValueError(f"level_constraint {lc!r}: expected 'whole_cell' or 'fluid_mean_v1'")
        return "whole_cell"

    def _no_fluid_bypass(self) -> None:
        raise RuntimeError("no-fluid bypass requested by a class that has no fluid-mean constraint")

    def _project_to_Vper(self, expr) -> "Function":
        """Fast L^2 projection of a UFL expression onto ``Vper``.

        Equivalent in effect to ``project(expr, self.Vper)`` but reuses the
        mass matrix assembled and LU-factorised once in
        :meth:`_setup_spaces_and_unknowns`.  The per-call cost collapses to
        a single RHS assembly plus one back-substitution — roughly an order
        of magnitude cheaper than ``project()`` which re-factorises the
        mass matrix every call.

        Used by the BDF history update on every time step and during the
        transient seed; these are the sites where item (2) of the
        MicroPhaseNeutral_Speedup_Investigation_2026-04-15 report measured
        ~0.34 s of waste on the 60×60 benchmark.

        Note: legacy DOLFIN's ``interpolate()`` cannot stand in here — it
        rejects raw UFL ``Sum``/``Conditional`` nodes (as ``self.h_nd``
        and ``self.mask`` both are) with an ``AttributeError`` on
        ``_cpp_object``.
        """
        out = Function(self.Vper)
        b = assemble(expr * self._proj_Vper_v_test * dx(domain=self.mesh_m))
        self._M_Vper_lu.solve(out.vector(), b)
        return out

    def _setup_forms(self) -> None:
        """Allocate transient history holders and UFL forms."""
        # Transient BDF history (carried from template for Stage 2 compatibility)
        self.h_hist_nm1 = Function(self.Vper, name="h_nm1")
        self.h_hist_nm2 = Function(self.Vper, name="h_nm2")
        # [S1b, ripple fast loop plan 2026-10-19, WP7 (D1: yes, 20 Oct)] The mask
        # history is allocated as before but no longer filled: no residual form of
        # the phase-neutral classes reads mask_hist_* (the BDF squeeze term uses the
        # film history only), and the six projections per run were pure cost.  The
        # Functions stay zero; tests/roughness/test_roughness_fenics.py (G5) asserts
        # that nothing depends on them.
        self.mask_hist_nm1 = Function(self.Vper, name="mask_nm1")
        self.mask_hist_nm2 = Function(self.Vper, name="mask_nm2")
        self._have_two_hist = False

        # Transient controls as Constants (so forms need no rebuild per step)
        self.dt_nd_const = Constant(1.0)
        self.bdf_a0 = Constant(0.0)
        self.bdf_a1 = Constant(0.0)
        self.bdf_a2 = Constant(0.0)
        # [Phase 2.4 seam S3] Guarded: a variant subclass may have created
        # transient_on earlier (its _setup_pressure_expressions bakes the
        # reference into F_adv_nd); re-creating it here would orphan that
        # reference.  Base behaviour unchanged (nothing creates it earlier).
        if not hasattr(self, "transient_on"):
            self.transient_on = Constant(0.0)

        # Build baseline residual/Jacobian
        self._build_forms(include_transient_terms=True, dt_nd=None)

        # --- Schur complement augmentation forms ---
        # spos'(-P_nd), in whichever form this solver was built with:
        #   sqrt   : 0.5*(1 + (-P_nd)/sqrt(P_nd^2 + eps^2))   (legacy;
        #            NUMERICALLY identical to the pre-WP-B expression --
        #            z*z == P_nd**2 and eps*eps == eps**2 exactly in IEEE
        #            -- and it reuses the SAME epsP_nd_spos_const object,
        #            which is what the P2.3b note below is about.  The UFL
        #            nodes are Product where they used to be Power.)
        #   spline : 0 | (-P_nd)/eps | 1                      (WP-B default)
        # [Micro-perf P2.3b] Own stored Constant for this site (NOT shared
        # with self.epsP_nd_const): the pre-P2.3b code had two distinct
        # Constant objects, and sharing one would change the UFL graph.
        self.epsP_nd_spos_const = Constant(self.epsP_nd)
        self._spos_deriv_nd = self._spos_deriv(
            -self.P_nd, self.epsP_nd_spos_const
        )

        # c_i = ∂R_cav/∂λ = -gamma * cav_weight * spos'(-P) * φ_i dx
        # (derivative of the cavitation penalty w.r.t. lambda_nd)
        # [Phase 2.4 seam S2] _cav_weight == mask in the base.
        # [Micro-perf P2.3b] gamma promoted per site (graph unchanged).
        self.gamma_nd_c_const = Constant(self.gamma_nd)
        self._c_form = (
            -self.gamma_nd_c_const
            * self._cav_weight
            * self._spos_deriv_nd
            * self.v
            * dx
        )

        # g_j = ∂G/∂w_j = +gamma * spos'(-P) * φ_j dx  (no mask; /area applied in Newton)
        # Closure: G = β·λ − γ·⟨spos(−P)⟩
        #   ∂G/∂w_j = −γ/area · ∫ spos'(−P) · ∂(−P)/∂w_j dx
        #           = −γ/area · ∫ spos'(−P) · (−φ_j) dx
        #           = +γ/area · ∫ spos'(−P) · φ_j dx
        # c (from R_cav) has gamma AND mask; g has gamma but NO mask.
        self.gamma_nd_g_const = Constant(self.gamma_nd)  # [Micro-perf P2.3b]
        self._g_form_raw = (
            self.gamma_nd_g_const * self._spos_deriv_nd * self.v * dx
        )

    def _build_forms(self, *, include_transient_terms: bool, dt_nd: float | None) -> None:
        """(Re)build residual/Jacobian UFL forms."""
        gradP = self.grad_cell(self.P_nd)
        gradv = self.grad_cell(self.v)

        R_reynolds = inner(self.F_adv_nd - self.K_nd * gradP, gradv) * dx
        # Cavitation penalty: penalise spos(-P_nd) in the field residual.
        # [Phase 2.4 seam S2] _cav_weight == mask in the base.
        # [Micro-perf P2.3b] Stored attribute, but (re)created on EVERY
        # _build_forms call — exactly the pre-P2.3b lifetime (a rebuild
        # made a fresh Constant).  A future reset_task() must therefore
        # update the FLOAT self.gamma_nd as well as assigning this
        # Constant, so a later rebuild picks up the new task's value.
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
        """[Phase 2.4 seam S4] Squeeze source term for the transient residual.

        Sets ``self._squeeze_source`` / ``self._squeeze_mean_nd`` and returns
        the UFL term to SUBTRACT from R.  Base: mask-weighted source with the
        mean subtracted for solvability.  The mixed contact variant overrides
        this with a pattern/legacy two-branch version.
        """
        # Squeeze source (periodic fluctuation only; mean subtracted for solvability)
        self._squeeze_source = self.mask * dh_dt
        self._squeeze_mean_nd = Constant(0.0)
        return self.transient_on * (
            (self._squeeze_source - self._squeeze_mean_nd) * self.v * dx
        )

    def _flux_means(self, q_dim, q_couette_dim, q_poiseuille_dim, area_total):
        """[Phase 2.4 seam] Homogenised flux sampling for spatial_homogenisation.

        Returns (Qx, Qy, Qx_couette, Qy_couette, Qx_poiseuille, Qy_poiseuille).
        Base: cell-area means.  The bound variant overrides with
        boundary-edge line integrals.
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
        """[Phase 2.4 seam S1] Refresh the squeeze-mean Constant from the
        current squeeze source.  Extracted verbatim from the call
        sites in run() (seed, per-step); the mixed
        contact variant overrides this with a pattern/legacy two-branch
        version."""
        if hasattr(self, "_squeeze_source"):
            S_mean = float(assemble(self._squeeze_source * dx(domain=self.mesh_m)))
            self._squeeze_mean_nd.assign(S_mean)

    # -------------------------
    # Core static helpers
    # -------------------------
    @staticmethod
    def smooth_pos(z, eps):
        """LEGACY symmetric smooth positive part (``spos_form="sqrt"``).

        ``spos(z) = 0.5*(z + sqrt(z^2 + eps^2))``.

        C-infinity, but two-sided: ``spos(0) = eps/2`` and
        ``spos(z) -> eps^2/(4z)`` as ``z -> +inf``, so it is never zero.
        In the phase-neutral closure ``beta*lambda = gamma*<spos(-P)>``
        that residue is a BIAS: a fully pressurised cell (P > 0
        everywhere, no cavitation anywhere) settles at

            lambda = 0.5*(-P + sqrt(P^2 + gamma_nd*epsP_nd^2))  >  0

        instead of 0.  At ``eps_smooth = 1e5`` Pa that is 15.8 MPa on any
        node with P <~ 0.1 MPa, which is what destroyed
        Wang2008-Mixed2-hmin2.0e-08-Ah10 (fix plan 2026-09-15, F3/F4).
        Retained for A/B only -- prefer :meth:`smooth_pos_spline`.
        """
        return 0.5 * (z + sqrt(z * z + eps * eps))

    @staticmethod
    def smooth_pos_spline(z, eps):
        """One-sided C1 quadratic-spline positive part [WP-B 2026-09-15].

            spos(z) = 0              z <= 0
                    = z^2/(2*eps)    0 < z < eps
                    = z - eps/2      z >= eps

        Exactly zero for ``z <= 0`` -- so a fully pressurised cell drives
        the closure to ``lambda = 0`` with no residue, at ANY eps -- and
        within ``eps/2`` of ``max(z, 0)`` for ``z >= eps``.  C1 at both
        knots, which is all the Newton solve ever needed from the
        smoothing (the cure for the chatter was a CONTINUOUS ``spos'``,
        not the symmetric tail).
        """
        # NOTE: built from ``conditional`` / ``gt`` / ``lt`` ONLY.  This
        # legacy dolfin's ``from fenics import *`` (ufl_legacy) does not
        # export ``max_value`` or ``le`` -- caught by
        # tests/test_spos_form_unit.py::test_ufl_matches_numpy_mirror,
        # which is why that test evaluates the real UFL.
        #
        # ``eps_s`` exists only so the (unselected) quadratic branch
        # cannot divide by zero when eps == 0.  Its guard still tests the
        # ORIGINAL eps, so at eps == 0 the branch is unreachable and the
        # function degenerates to max(z, 0) exactly.
        eps_s = conditional(gt(eps, 0.0), eps, 1.0)
        return conditional(
            gt(z, 0.0),
            conditional(lt(z, eps), z * z / (2.0 * eps_s), z - 0.5 * eps),
            0.0,
        )

    @staticmethod
    def smooth_pos_deriv_sqrt(z, eps):
        """d/dz of :meth:`smooth_pos`: ``0.5*(1 + z/sqrt(z^2 + eps^2))``."""
        return 0.5 * (1.0 + z / sqrt(z * z + eps * eps))

    @staticmethod
    def smooth_pos_deriv_spline(z, eps):
        """d/dz of :meth:`smooth_pos_spline`: ``0 | z/eps | 1``.

        Continuous at both knots and monotone in [0, 1], so the closure
        scalar ``d = beta + gamma*<spos'>`` stays in
        ``[beta, beta + gamma]`` exactly as with the sqrt form.  Note it
        is exactly 0 wherever P >= eps, so a fully pressurised cell
        leaves the Reynolds block with its constant-mode null space --
        that is what ``jacobian_reg_eps`` (Tikhonov shift) is for, and it
        must stay armed (> 0) with this form.
        """
        # conditional / gt / lt only -- see smooth_pos_spline.
        eps_s = conditional(gt(eps, 0.0), eps, 1.0)
        return conditional(
            gt(z, 0.0),
            conditional(lt(z, eps), z / eps_s, 1.0),
            0.0,
        )

    # ------------------------------------------------------------------
    # spos form dispatch (instance level -- honours settings.spos_form,
    # resolved once in __init__ as self._spos_form).
    # ------------------------------------------------------------------
    def _spos(self, z, eps):
        """Smoothed positive part in this solver's configured form."""
        form = getattr(self, "_spos_form", "spline")
        if form == "sqrt":
            return self.smooth_pos(z, eps)
        return self.smooth_pos_spline(z, eps)

    def _spos_deriv(self, z, eps):
        """d(spos)/dz in this solver's configured form."""
        form = getattr(self, "_spos_form", "spline")
        if form == "sqrt":
            return self.smooth_pos_deriv_sqrt(z, eps)
        return self.smooth_pos_deriv_spline(z, eps)

    @staticmethod
    def smooth_levelset_mask(phi, delta, mask_core, mask_fluid):
        """UFL smooth mask transitioning between mask_core and mask_fluid."""
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
        """One solve at the current level: the Newton solve, or, in the pair branch of the ripple
        ([phase-table WP3]), that solve repeated with u_r iterated to the ripple tolerance
        (:meth:`_pair_solve`).  Call sites: the step loop, the end-level pass, and the mixed
        subclass's seed and steady solves; a one-member run never enters the branch."""
        if self._pair_active:
            self._pair_solve()
            return
        self.solve()

    def _refresh_geometry(self) -> None:
        """[fft_elastic_v1 seam, 16 Oct 2026] Re-evaluate state-dependent
        geometry fields for the CURRENT time Constants.

        Called by run_steady() before the steady solve, by run() at each
        synthesised BDF history time (before that time's projections, the
        current time restored afterwards), before the transient seed and
        before every step's solve (after all of that time's forcings are
        assigned, before the squeeze mean).  Base: no-op -- the film is a
        pure UFL expression of the Constants.  The contact subclass refreshes
        the nodal elastic contact solution here for ``fft_elastic_v1``; the
        Case API's translating generated surfaces (plan section 7.2) are the
        other intended consumer (a nodal refill per time level).

        [R5a, 16 Oct 2026] A film callback that carries ``refresh_geometry``
        (generated surfaces, ``microscale/src/functions/surface_ht.py``) is
        refilled here for the current ``T_dim_const``.  The idealised cosine
        callback has no such attribute: nothing changes for it.
        """
        refresh = getattr(self.ht_cb, "refresh_geometry", None)
        if refresh is not None:
            refresh(self)
        # [option H] the ripple at this time, after the rigid film and before the
        # contact subclass's solve (order of plan O6)
        if self._ripple_on:
            self._ripple_write_field()
        return None

    # -------------------------
    # [option H, 17 Oct 2026] micro fluid deformation ``ripple_halfspace``
    # (docs/notes/ripple_halfspace_plan_2026-10-17.md, decisions O1-O10)
    # -------------------------
    _ripple_on = False          # class default: subclasses that never run the base film setup
    _pair_active = False        # [phase-table WP3] class default: the pair branch is never entered off

    def _ripple_setup(self) -> None:
        """Validate the setting and allocate ``ur_nd_func`` (from ``_setup_film_and_mask``;
        ``Vper`` exists).  Off: nothing is allocated and no graph changes."""
        s = self.settings
        self._ripple_on = s.ripple is not None
        self.ur_nd_func = None
        self.ripple_info = None
        if not self._ripple_on:
            return
        E = s.E_star
        if E is None or not np.isfinite(float(E)) or float(E) <= 0.0:
            raise ValueError("the ripple needs a positive, finite E_star (the resolved "
                             "two-body modulus, plan O3)")
        if not (float(s.ripple.tol) > 0.0 and int(s.ripple.maxpass) >= 1
                and 0.0 < float(s.ripple.omega) <= 1.0):
            raise ValueError("ripple settings: need tol > 0, maxpass >= 1 and "
                             f"0 < omega <= 1; got {s.ripple.tol}, {s.ripple.maxpass}, "
                             f"{s.ripple.omega}")
        self.ur_nd_func = Function(self.Vper, name="ur_nd")
        self._ripple_reset()

    def _ripple_reset(self) -> None:
        """A task starts at u_r = 0: nothing is carried between tasks (O6)."""
        if self.ur_nd_func is not None:
            self.ur_nd_func.vector().zero()
            self.ur_nd_func.vector().apply("insert")
        self._ripple_u = None               # the iterate [m] on the grid, end-time frame; None = 0
        self._ripple_u_serial = 0            # bumped whenever the iterate changes
        self._ripple_shift_end = (0.0, 0.0)
        self._ripple_written = None          # (shift, iterate id) now in ur_nd_func
        self._ripple_version = 0             # bumped on every write (contact staleness key)
        self._ripple_warm = None
        self._ripple_levels = None
        self._ripple_pass = 0
        self.ripple_info = None
        # [phase-table WP3] the converged per-step branch of a two-member field (inactive between runs)
        self._pair_active = False
        self._pair_u = None                  # the last converged field [m], lab frame, at _pair_ref_shift
        self._pair_ref_shift = (0.0, 0.0)
        self._pair_written = None            # the field now in ur_nd_func [m]
        self._pair_iters = []                # inner iterations per level

    def _periodic_grid(self):
        """``(n, i, j)``: Vper dof ``k`` is node ``(i[k], j[k])`` of the structured
        periodic n x n grid (``i`` along x).  Checked to be a bijection on a
        ``UnitSquareMesh(n, n)``; shared by option H and ``fft_elastic_v1``."""
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
        n, i, j = self._periodic_grid()
        g = np.empty((n, n), dtype=dtype)
        g[i, j] = v
        return g

    def _grid_to_dofs(self, g) -> np.ndarray:
        n, i, j = self._periodic_grid()
        return np.asarray(g)[i, j]

    def _ripple_operator(self):
        op = getattr(self, "_ripple_op_cache", None)
        if op is None:
            n, _i, _j = self._periodic_grid()
            op = _rp.ripple_operator(n, n, float(self.Lx), float(self.Ly),
                                     float(self.settings.E_star))
            self._ripple_op_cache = op
        return op

    def _ripple_check_task(self) -> None:
        """Per-task refusals (configuration errors, not micro failures)."""
        a = float(self.velocity_ratio)
        desc = getattr(self.ht_cb, "contact_cosine", None)
        # [phase-table WP3, 4 Nov 2026] R-e: a two-member generated field (stationary + translating) is
        # accepted only with ``self.pair`` on (plan D5): it then runs the
        # converged per-step branch (_pair_ripple_run), which needs no single frame.  Every other
        # refusal stays: 0 < ratio < 1 and the split cosine descriptor (one pattern velocity per member).
        two = (0.0 < a < 1.0 or (isinstance(desc, tuple) and len(desc) >= 4 and bool(desc[3]))
               or (self._ripple_pair() and not self._pair_branch_on()))
        if two:
            raise ValueError("the ripple supports one rough surface only in "
                             "stage 1 (plan O1): a two-surface pattern has no single frame (O6)")

    def _pair_branch_on(self) -> bool:
        """[phase-table WP3] ``self.pair``.  The parent keyed this on
        ``Asperity.phase_average == "contact_table_v1"``; the contact
        table itself is not in this repo (MicroCell WP2)."""
        return self.pair

    def _ripple_pair(self) -> bool:
        """[phase-table WP3] True for a two-member generated surface field (``ht.surface_field`` with both a
        stationary and a translating member): the ripple then runs the converged per-step branch.  A
        one-member field or the idealised cosine callback: False (the one-member pass loop, unchanged)."""
        field = getattr(self.ht_cb, "surface_field", None)
        return field is not None and len(getattr(field, "members", {})) > 1

    def _ripple_surface_shift(self) -> tuple:
        """Pattern displacement [m] of the rough surface at the current time Constants,
        by the same rule as the rigid film: ``T * _Ux_dim`` for the idealised callback,
        ``motion.displacement(T)`` for a generated surface (R5a), 0 when static."""
        field = getattr(self.ht_cb, "surface_field", None)
        if field is not None:
            if not getattr(field, "translating", False):
                return (0.0, 0.0)
            sx, sy = self.ht_cb.surface_motion.displacement(float(self.T_dim_const))
            return (float(sx), float(sy))
        T = float(self.T_dim_const)
        return (T * float(self._Ux_dim_const), T * float(self._Uy_dim_const))

    def _ripple_write_field(self) -> None:
        """``ur_nd_func`` <- the iterate carried to the current time in the rough surface's
        frame (O6): unchanged for a static surface, Fourier-shifted for a translating one.

        [phase-table WP3] In the pair branch: the last converged field, carried with the translating
        member since it converged, as the starting guess of this level's iteration only (the level then
        converges the field at its own time in the lab frame; ripple_s3 ``_ripple_write_field``)."""
        if self._pair_active:
            s = self._ripple_surface_shift()
            d = (s[0] - self._pair_ref_shift[0], s[1] - self._pair_ref_shift[1])
            u = self._pair_u if d == (0.0, 0.0) else _rp.pattern_shift(self._pair_u, d[0] / float(self.Lx),
                                                                        d[1] / float(self.Ly))
            self._pair_write(u)
            return
        u = self._ripple_u
        if u is None:
            return                            # the field is zero (task start, pass 1)
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
        """Hook: subclasses reset per-run records (the mixed subclass's homog_history)."""
        return None

    def _ripple_level_begin(self, level) -> None:
        """Warm start each sub-solve from the same time level of the previous pass (O5)."""
        if self._pair_active:
            return                            # [phase-table WP3] one pass: no previous-pass levels
        warm = (self._ripple_warm or {}).get(level)
        if warm is not None:
            self.w.vector().set_local(warm["w"])
            self.w.vector().apply("insert")
            self.lambda_nd.assign(warm["lam"])
            self.mu_nd = warm["mu"]

    def _ripple_level_end(self, level) -> None:
        if self._pair_active:
            return                            # [phase-table WP3] no 13.4 snapshots in the pair branch
        snap = {"w": self.w.vector().get_local().copy(), "lam": float(self.lambda_nd),
                "mu": float(self.mu_nd)}
        snap.update(self._ripple_snapshot())
        self._ripple_levels[level] = snap

    def _ripple_traction_fluid(self):
        """t = chi_fluid * max(P, 0), the physical fluid traction (13.4 field check)."""
        z = Constant(0.0)
        return self.chi_fluid * conditional(gt(self.P_dim, z), self.P_dim, z)

    def _ripple_snapshot(self) -> dict:
        """The 13.4 quantities of the current state: fluid traction (Vper dofs), mask area
        (cell fraction) and, from the hook, the contact quantities."""
        t = self._project_to_Vper(self._ripple_traction_fluid()).vector().get_local()
        A_mask = 1.0 - float(assemble(self.chi_fluid * dx(domain=self.mesh_m))) / float(self._cell_area)
        snap = {"t": t, "A_mask": A_mask}
        snap.update(self._ripple_contact_snapshot())
        return snap

    def _ripple_contact_snapshot(self) -> dict:
        """Base: the obstruction set at the nodes from the projected film when contact
        is on (h_min > 0); none in full film.  ``fft_elastic_v1`` overrides."""
        if float(self.h_min) <= 0.0:
            return {}
        hv = self._project_to_Vper(self.h_nd).vector().get_local()
        return {"set": self._dofs_to_grid(hv < float(self.hmin_nd), dtype=bool)}

    def _ripple_chi_nodes(self) -> np.ndarray:
        """The fluid indicator at the dofs (for the seam diagnostic)."""
        if float(self.h_min) <= 0.0:
            return np.ones(self.w.vector().local_size())
        hv = self._project_to_Vper(self.h_nd).vector().get_local()
        return (hv > float(self.hmin_nd)).astype(float)

    def _ripple_traction_overlap(self, s_dofs) -> float:
        """Share of the projected traction on contact nodes (fft only; base 0)."""
        return 0.0

    def _ripple_mnorm(self, v) -> float:
        """Consistent-mass L2 norm of a Vper dof vector (unit-square mesh)."""
        f = getattr(self, "_ripple_mnorm_f", None)
        if f is None:
            f = Function(self.Vper)
            self._ripple_mnorm_f = f
        f.vector().set_local(np.asarray(v, dtype=float))
        f.vector().apply("insert")
        Mv = self._M_Vper * f.vector()
        return float(np.sqrt(max(float(f.vector().inner(Mv)), 0.0)))

    def _ripple_target(self) -> dict:
        """End state of a pass: the traction of 18.13.3 (L2-projected on Vper), its
        ripple T = H[s] [m] on the grid, and the diagnostics of 18.15 item 2."""
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
        """The option H fixed point: ``body`` is pass 1 (a whole run) with the field
        held fixed in the rough surface's frame; from pass 2 on, ``later`` (the
        end-level pass :meth:`_run_end_level_once`) is the body where the caller
        allows it, else ``body`` again.  Between passes u <- u + omega (H[s] - u),
        secant-relaxed in [0.1, 1] x omega_0, signed and without clips.

        Acceptance (plan ``docs/notes/ripple_fast_loop_plan_2026-10-19.md`` section
        4.6, :func:`ehl.ripple.residual_acceptance`; Rob's decision of 19 October):
        pass k is accepted when its own deformation residual max|H[s_k] - u_k| is
        within ``ripple_tol`` x H0, from pass 2 the residual shrank by at least half
        since pass k-1 (``CONTRACTION_MAX``), and the end level's nodal contact set
        did not change.  Pass 1 can be accepted (u = 0 stands when the ripple itself
        is below the budget).  The 13.4 checks of O5 (two transitions at every level:
        field, area, contact pressure, force, set) are no longer the acceptance; they
        are still computed over the levels both passes hold and reported in
        ``ripple_info["worst"]`` as diagnostics.

        Why the end-level pass is exact (section 4.4): a film at a level depends
        only on the geometry and the time (the ripple is held fixed in the surface
        frame, the film does not depend on the pressure,
        the contact solve sees the film only), so rebuilding the two levels before
        the end reproduces the BDF history a full run gives the end level, and the
        end solve is warm-started from the previous pass's end level as every level
        of a full pass is.  The cap raises :class:`ehl.ripple.RippleNotConvergedError`
        (a recorded micro failure)."""
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
        # every run starts from u_r = 0, also on a solver that already ran (determinism)
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
                    # the 13.4 checks of this transition, over the levels both passes
                    # hold (every level after full passes; the end level only after an
                    # end-level pass): diagnostics, not the acceptance
                    common = [lv for lv in levels if lv in prev_levels]
                    _ok134, worst = _rp.transition_checks(
                        {lv: prev_levels[lv] for lv in common}, {lv: levels[lv] for lv in common},
                        resid_max=resid, change_max=float(np.max(np.abs(u - prev_u))), H0=H0,
                        p_scale=float(self.p_scale), n=n, deform_tol=tol, mnorm=self._ripple_mnorm)
                end = next(reversed(levels))   # the last level solved; dicts keep insertion order
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
            # callers read task totals (the worker's Newton count); a pass that raised
            # still adds what it did
            if k > 0:
                for name in _PASS_COUNTERS:
                    if not counted:
                        totals[name] += getattr(self, name, 0)
                    setattr(self, name, totals[name])

    # -------------------------
    # [phase-table WP3, 4 Nov 2026] The converged per-step ripple of a two-member surface field
    # (plan docs/plans/phase_average_route_a_production_plan_2026-11-04.md D5, review F11; ported by
    # copy from the study's tests/ripple/ripple_s3.py ``full_history_class`` 53-175 and
    # tests/micro_averaging/study_solver.py ``converged_class`` 246-376).  Selected by ``run`` /
    # ``run_steady`` when ``_ripple_pair()``; a one-member field runs the pass loop above, untouched.
    #
    # One run, not passes: at every level (the seed, every step, the steady solve) the sub-solve is
    # repeated with u_r <- u_r + omega (H[s] - u_r) until max|H[s] - u_r| <= ripple_tol H0, omega
    # secant-relaxed in [0.1, 1], the contact re-solved on h_r + u_r after each write; the converged
    # field enters the BDF history (the transient seed's synthesised history is re-built with it, so
    # u_r does not jump in at the first step).  The next level starts from the previous level's
    # converged field carried with the translating member (``_ripple_write_field``); no frame
    # assumption is made (the O6 question of a pair is answered as the study answered it).  A level
    # that does not converge in PAIR_RIPPLE_MAXIT inner iterations raises RippleNotConvergedError (a
    # recorded micro failure, kNN-filled as any other; review F12).  ``ripple_info["passes"]`` is the
    # largest inner-iteration count of a level and ``omega`` is NaN (ripple_s3 90-100).
    # -------------------------
    def _pair_ripple_run(self, body: Callable) -> None:
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
            # the kinematics the film refresh would set, needed for the reference shift before the
            # first refresh (study_solver 339-343)
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
        """``ur_nd_func`` <- ``u`` [m] (lab frame, the current time); every write bumps the contact
        staleness key."""
        _rp.check_zero_mean(u, what="pair ripple field")
        self.ur_nd_func.vector().set_local(self._grid_to_dofs(u) / float(self.zeta.H))
        self.ur_nd_func.vector().apply("insert")
        self._ripple_version += 1
        self._pair_written = u

    def _pair_resynth_history(self) -> None:
        """The transient seed's history levels (T_start - dt, - 2 dt) again with the current field, as
        ``_run_once`` builds them; the current time and the contact restored (ripple_s3 131-143)."""
        T0, dt = float(self.T_start), float(self.dt_dim_run)
        for T, hh in ((T0 - dt, self.h_hist_nm1), (T0 - 2.0 * dt, self.h_hist_nm2)):
            self.T_dim_const.assign(T)
            self._refresh_geometry()
            hh.assign(self._project_to_Vper(self.h_nd))
        self.T_dim_const.assign(T0)
        self._refresh_geometry()
        self._update_squeeze_mean()

    def _pair_solve(self) -> None:
        """The level's sub-solve with u_r converged (ripple_s3 145-175)."""
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
                # the rolling-steady seed: its synthesised history carries the same field (no
                # artificial jump of u_r at the start: a quasi-static initial state)
                self._pair_resynth_history()
            elif fft:
                self._refresh_contact()            # the contact on h_r + the new u_r
        else:
            raise _rp.RippleNotConvergedError(
                f"ripple_not_converged: pair level {self._phase}: {PAIR_RIPPLE_MAXIT} inner iterations, "
                f"max|H[s] - u| = {float(np.max(np.abs(R))) / H0:.3e} H0 (budget "
                f"{float(self.settings.ripple.tol):.1e})")
        self._pair_u = u
        self._pair_ref_shift = self._ripple_surface_shift()
        self._pair_iters.append(it)

    def _steady_solve(self) -> None:
        """[Mixed P1.0 seam] Solve site used by run_steady().

        Base: raw Newton solve.  The mixed subclass overrides this to
        _solve_level(), which enters the pair branch of the ripple.
        """
        self.solve()

    def _seed_solve(self) -> None:
        """[Mixed P1.0 seam] Solve site for the transient rolling-steady
        seed at T_start in run().

        Base: raw Newton solve.  The mixed subclass overrides this to
        _solve_level(), which enters the pair branch of the ripple.
        """
        self.solve()

    # -------------------------
    # Nonlinear solver setup
    # -------------------------
    def _build_nonlinear_solver(self) -> None:
        """Set up persistent objects for the Schur complement Newton solver.

        No Dirichlet BCs are applied — the null mode is handled by the
        scalar Lagrange multiplier (mu_nd) in the augmented Newton system.
        """
        # No point-constraint BC
        self._bcs = []

        # SystemAssembler with empty BC list
        self._sys_asm = SystemAssembler(self.J, self.R, [])

        # Persistent PETSc matrix/vector
        self._A_mat = PETScMatrix()
        self._b_vec = PETScVector()
        self._sys_asm.assemble(self._A_mat, self._b_vec)

        # Persistent LU solver (MUMPS)
        self._lu_solver = LUSolver(self._A_mat, "mumps")
        self._lu_solver.parameters["symmetric"] = False

        # Newton update vector
        self._du = Function(self.Vper)

        self._solver_needs_rebuild = False

    # -------------------------
    # Closure hooks (overridable by variant subclasses)
    # -------------------------
    # [Phase 2.2] These three methods are the ONLY seams a closure variant
    # (e.g. the LC "load_consistent" solver) needs to override.  They were
    # cut verbatim out of _newton_solve; base behaviour is unchanged.

    def _closure_G(self, beta: float, gamma: float, area: float) -> float:
        """Scalar closure residual G at the current (w, lambda) state.

        Base (reform) closure: G = beta*lambda - gamma*mean(spos(-P_nd)),
        which should -> 0; at equilibrium lambda >= 0 (dP >= 0).
        """
        spos_mean = float(assemble(self.negP_pos * dx(domain=self.mesh_m))) / area
        return beta * float(self.lambda_nd) - gamma * spos_mean

    def _closure_g_vec(self):
        """dG/dw Schur augmentation vector (assembled N-vector)."""
        return assemble(self._g_form_raw)

    def _closure_d_scalar(self, beta: float, gamma: float, area: float) -> float:
        """d = dG/dlambda.  Base: beta + gamma*mean(spos'(-P)) >= beta > 0."""
        spos_deriv_mean = (
            float(assemble(self._spos_deriv_nd * dx(domain=self.mesh_m))) / area
        )
        return beta + gamma * spos_deriv_mean

    # -------------------------
    # Schur complement Newton solver
    # -------------------------
    def _apply_jacobian_reg_shift(self) -> None:
        """Add ``jacobian_reg_eps * I`` to the assembled field Jacobian.  Must be
        applied after EVERY assembly of ``_A_mat``.

        Why: the Schur-complement Newton factorises the periodic Reynolds block
        alone; with no active cavitation penalty that block has a constant-mode
        null space and MUMPS stops on a zero pivot (``jacobian_reg_eps``
        docstring).  A small shift lifts it.

        Sign: the block's diffusion and cavitation-penalty derivatives are
        negative (residual ``-K grad P`` plus a positive penalty on negative P),
        so a negative shift reinforces them, whereas a positive one can cancel
        small negative diagonals in contact cores.  On the 2026-10-15 A2 batches
        -1e-7 converged every cell that +1e-10 failed on
        (``docs/notes/A2_jacobian_only_stabilization_2026-10-15.md``,
        ``A2_stabilization_batch_and_conditioning_2026-10-15.md``).  Zero
        disables the shift; a nonfinite value is refused.

        The shift changes the Newton direction, not the residual or its
        convergence test -- but it is not answer-neutral in practice: contact
        cells converged at -1e-8 and -1e-7 differ in P_f by up to ~2 %, and
        more against +1e-10 (same notes).
        """
        reg_eps = float(self.settings.jacobian_reg_eps)
        if not np.isfinite(reg_eps):
            raise ValueError("jacobian_reg_eps must be finite; zero disables the shift")
        if reg_eps == 0.0:
            return
        try:
            self._A_mat.mat().shift(reg_eps)
        except Exception as _shift_exc:
            # Fallback: add eps to diagonal directly
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

    def _assemble_newton_rhs(self) -> float:
        """[Phase 6.4-U7.1] Residual-only assembly: b = -(R + mu*m).

        Uses the SAME SystemAssembler as the full (A, b) assembly — the
        single-vector overload runs the identical cell kernels in the
        identical order, so b is bitwise-identical to the pair assembly.
        The Jacobian is now assembled separately in the Schur step, just
        before factorisation — so the final (converged) iteration and
        every backtrack re-check skip the matrix work entirely.
        Returns elapsed seconds (caller accumulates into _t_asm).
        """
        _ta = time.perf_counter()
        self._sys_asm.assemble(self._b_vec)
        # Add Lagrange multiplier contribution to residual vector
        # [fluid-mean WP1, plan micro_fluid_mean_constraint_plan_2026-10-30.md rev 3 (m6)] the
        # constraint vector comes from the hook; ``_mean_vec`` itself is never overwritten.
        self._b_vec.axpy(self.mu_nd, self._constraint_vector())
        # Negate: b = -F_w  (FEniCS convention: solve J dw = -R)
        self._b_vec *= -1.0
        return time.perf_counter() - _ta

    def _schur_step(self, z1, z2, z3, m, beta, gamma, area, G, H, nit):
        """[Phase 6.4] One Schur-complement Newton direction (verbatim math).

        Assembles the augmentation vectors, factorises J (assembled by the
        caller immediately beforehand), does the three back-substitutions,
        solves the 2x2 Schur complement and recovers dw into ``self._du``.
        Returns ``(d_lambda, d_mu, t_lin, t_upd_recover)``.

        [Phase 6.4-U7.2] set_operator() is NOT re-called per iteration:
        the LUSolver was constructed bound to ``self._A_mat`` and in-place
        reassembly bumps the PETSc object state, triggering
        refactorisation — the same mechanism the cached ``_M_Vper_lu``
        projector relies on (see _build_nonlinear_solver's "Do NOT call
        set_operator() again" note).
        """
        # --- Assemble Schur augmentation vectors (depend on current P_nd) ---
        # c_i = dR_cav/dlambda  (N-vector)
        c_vec = assemble(self._c_form)
        # g_j_raw = d(area*G)/dw_j  (N-vector); d = dG/dlambda >= beta > 0.
        # [Phase 2.2] closure hooks.
        g_vec = self._closure_g_vec()
        d_scalar = self._closure_d_scalar(beta, gamma, area)

        # --- MUMPS factorisation + 3 back-substitutions ---
        _tl = time.perf_counter()
        # z1 = J^{-1} . b  (standard Newton step for w)
        self._lu_solver.solve(z1.vector(), self._b_vec)
        # z2 = J^{-1} . c  (Schur augmentation for lambda)
        self._lu_solver.solve(z2.vector(), c_vec)
        # z3 = J^{-1} . m  (Schur augmentation for mu)
        self._lu_solver.solve(z3.vector(), m)
        _t_lin = time.perf_counter() - _tl

        # --- 2x2 Schur complement ---
        # S = [[d - g^T z2/area,  -g^T z3/area ],
        #      [-m^T z2/area,     -m^T z3/area ]]
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

        # Regularise if Schur complement is near-singular (R1 mitigation).
        # d = beta + gamma*mean(spos') >= beta > 0 always.
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

        # --- Recover full field update: dw = z1 - dlambda.z2 - dmu.z3 ---
        _tu = time.perf_counter()
        self._du.vector().zero()
        self._du.vector().axpy(1.0, z1.vector())
        self._du.vector().axpy(-d_lambda, z2.vector())
        self._du.vector().axpy(-d_mu, z3.vector())
        self._du.vector().apply("insert")
        _t_upd = time.perf_counter() - _tu

        return d_lambda, d_mu, _t_lin, _t_upd

    def _newton_solve(self) -> None:
        """Schur complement Newton solver for the augmented (N+2)-dimensional system.

        Solves simultaneously for:
          - w_fluc  : zero-mean periodic field (N DOFs)
          - lambda_nd : scalar pressure-level correction (1 DOF)
          - mu_nd     : Lagrange multiplier for zero-mean constraint (1 DOF)

        Algorithm (per iteration):
          1. Residual check from b = -(R + mu.m)   [b-only assembly, U7.1]
          2. Adaptive damping: revert + halve omega on overshoot
             (policy state in DampingPolicy — [Phase 6.4])
          3. Assemble J, factor via MUMPS, three back-subs, 2x2 Schur
             solve, recover dw   (_schur_step)
          4. Damped update: w+=w.dw, lambda+=w.dl, mu+=w.dmu
          5. Reassemble b for the next convergence check

        Cost: 1 factorisation + 3 back-subs per accepted Newton iteration.
        [Phase 6.4-U7] vs the pre-6.4 solver: the Jacobian is no longer
        assembled on the final (converged) iteration nor on backtrack
        re-checks; set_operator is not re-called per iteration; the
        back-substitution work vectors persist across calls.  Each change
        is value-identical (same kernels, same state, fully-overwritten
        vectors); only cost changes.
        """
        s = self.settings
        omega = s.relaxation_parameter
        atol = s.abs_tolerance
        rtol = s.rel_tolerance
        max_it = s.max_iterations
        # beta: closure parameter
        beta = float(BETA)
        gamma = self.gamma_nd
        area = float(self._cell_area)

        # [Phase 6.4] Adaptive damping policy (April-2026 semantics,
        # extracted verbatim into DampingPolicy; knob docs there and on
        # at the fixed values of this module).
        pol = DampingPolicy(float(omega))
        omega_base = pol.omega_base

        # [fluid-mean WP1] the constraint vector through the hook (default: the whole-cell
        # m_i = integral(phi_i dx), ``_mean_vec``; a subclass may hold the level over another region)
        m = self._constraint_vector()  # PETScVector

        # [Phase 6.4-U7.3] Persistent back-substitution work vectors —
        # allocated once per function space and reused across calls.
        # Safe: the direct solves fully overwrite them each iteration.
        if getattr(self, "_z_work_space", None) is not self.Vper:
            self._z1 = Function(self.Vper)
            self._z2 = Function(self.Vper)
            self._z3 = Function(self.Vper)
            self._z_work_space = self.Vper
        z1, z2, z3 = self._z1, self._z2, self._z3

        # Timing sub-accumulators
        _t_asm = 0.0
        _t_lin = 0.0
        _t_upd = 0.0

        # --- Initial assembly: b = -(R + mu.m)  [b-only, U7.1] ---
        _t_asm += self._assemble_newton_rhs()

        # Initial residual norms (for relative tolerance)
        res0_field: float | None = None
        res0_scalar: float | None = None

        # Track G and H for error reporting if loop exhausted
        G: float = 0.0
        H: float = 0.0

        # Snapshot/step-direction state for backtracking (see DampingPolicy):
        #   w_snap/lam_snap/mu_snap: state BEFORE the last update was applied.
        #   last_du / last_d_lambda / last_d_mu: last step direction, so a
        #   backtrack can reapply at a smaller omega without redoing the
        #   Schur back-substitutions.
        w_snap: np.ndarray | None = None
        lam_snap: float = 0.0
        mu_snap: float = 0.0
        last_du: "Function | None" = None
        last_d_lambda = 0.0
        last_d_mu = 0.0

        # [S1a, plan ripple_fast_loop_plan_2026-10-19 section 4.2] True when the loop
        # left at its own convergence check on a finite state: the final
        # validation below would then re-assemble and re-test the very state the
        # check just passed (nothing mutates between the last assembly of b and
        # the break), so it is skipped.  Bitwise: only the cost changes.
        converged_at_check = False

        for nit in range(max_it):
            # --- Scalar residuals from current (w, lambda, mu) ---
            # [Phase 2.2] closure hook — see _closure_G for the formula.
            G = self._closure_G(beta, gamma, area)

            # H = mean(w_fluc) = m^T w / area  [should -> 0]  ([fluid-mean WP1] the hook)
            H = self._constraint_residual(m, area)

            # Field residual norm
            res_field = self._b_vec.norm("l2")
            res_scalar = max(abs(G), abs(H))

            # Store initial residuals for relative convergence test
            if res0_field is None:
                res0_field = max(res_field, 1e-30)
                res0_scalar = max(res_scalar, 1e-30)

            # --- Adaptive damping: backtrack if last step made things worse
            # (decision + omega state in DampingPolicy; conditions verbatim).
            res_sq = res_field * res_field + res_scalar * res_scalar
            if pol.wants_backtrack(nit, res_sq,
                                   last_du is not None and w_snap is not None):
                # Revert (w, lambda, mu) to the pre-step snapshot and
                # reapply the SAME direction at a halved omega.  No new
                # Schur solve is needed.
                self.w.vector().set_local(w_snap)
                self.w.vector().apply("insert")
                self.lambda_nd.assign(lam_snap)
                self.mu_nd = mu_snap

                pol.on_backtrack()

                self.w.vector().axpy(pol.omega_current, last_du.vector())
                self.w.vector().apply("insert")
                self.lambda_nd.assign(lam_snap + pol.omega_current * last_d_lambda)
                self.mu_nd = mu_snap + pol.omega_current * last_d_mu

                # Reassemble so the next convergence check sees the new
                # state  [b-only, U7.1].
                _t_asm += self._assemble_newton_rhs()

                # Re-enter the loop top to re-evaluate the damped state
                continue

            # Step accepted: omega recovery toward the per-solve cap
            # (suppressed while diverged with an exhausted budget).
            pol.on_accept(nit, res_sq)

            # --- Convergence check ---
            field_ok = (res_field < atol) or (res_field / res0_field < rtol)
            scalar_ok = (res_scalar < atol) or (res_scalar / res0_scalar < rtol)
            if field_ok and scalar_ok:
                # the finiteness guard keeps the one case the old re-check could
                # still raise on: H not finite while |G| passes (Python's
                # max(|G|, nan) returns |G|)
                converged_at_check = bool(np.isfinite(G) and np.isfinite(H))
                break

            # --- Assemble J for THIS iteration's factorisation ---
            # [Phase 6.4-U7.1] moved from the post-update reassembly: same
            # state (nothing mutates between the residual check and here),
            # so J is value-identical — but the final iteration and
            # backtrack re-checks no longer pay for it.  The Tikhonov
            # shift must follow every J assembly (see helper docstring).
            _ta = time.perf_counter()
            self._sys_asm.assemble(self._A_mat)
            self._apply_jacobian_reg_shift()
            _t_asm += time.perf_counter() - _ta

            # --- Schur direction (factor + back-subs + 2x2 + recover) ---
            d_lambda, d_mu, _dt_lin, _dt_upd = self._schur_step(
                z1, z2, z3, m, beta, gamma, area, G, H, nit,
            )
            _t_lin += _dt_lin
            _t_upd += _dt_upd

            # --- Snapshot state BEFORE the damped update, so that a
            # subsequent backtrack can cleanly revert and reapply the
            # same step direction at a smaller omega.  Also cache the
            # step direction itself (_du is reused next iteration).
            _tu = time.perf_counter()
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

            # --- Damped update (omega_current may have been reduced by a
            # previous backtrack; it recovers toward the — possibly
            # ratcheted — cap on each accepted step).
            self.w.vector().axpy(pol.omega_current, self._du.vector())
            self.w.vector().apply("insert")
            self.lambda_nd.assign(float(self.lambda_nd) + pol.omega_current * d_lambda)
            self.mu_nd += pol.omega_current * d_mu
            _t_upd += time.perf_counter() - _tu

            # --- Reassemble residual for next iteration [b-only, U7.1] ---
            _t_asm += self._assemble_newton_rhs()


        # Validate the actual returned state, including a final update or
        # backtrack at the iteration limit. G/H above belong to the state
        # BEFORE that update. Do this before any caller advances BDF history;
        # use the original residual and the same absolute-or-relative tests.
        # [S1a] Skipped when the loop broke at its convergence check: the
        # state is the one that check saw (the closure G and H at the loop top
        # only read; a backtrack `continue`s; on_accept and the print touch
        # policy state only), so the re-assembly would reproduce the values
        # that just passed.  The block stays for the states the loop did not
        # check: a final update or backtrack at the iteration limit.
        if not converged_at_check:
            _t_asm += self._assemble_newton_rhs()
            final_res = self._b_vec.norm("l2")
            G = self._closure_G(beta, gamma, area)
            H = self._constraint_residual(m, area)
            final_scalar = max(abs(G), abs(H))
            field_ok = np.isfinite(final_res) and (
                final_res < atol or final_res / max(res0_field or 0.0, 1e-30) < rtol)
            scalar_ok = np.isfinite(G) and np.isfinite(H) and (
                final_scalar < atol or final_scalar / max(res0_scalar or 0.0, 1e-30) < rtol)
            if not (field_ok and scalar_ok):
                # count the iterations of the failed solve too, so a failed result reports them
                if hasattr(self, "_t_asm_total"):
                    self._n_newton_iters_total += max(0, (nit + 1) - pol.n_bt)
                    self._n_newton_backtracks_total += pol.n_bt
                raise RuntimeError(
                    f"MicroPhaseNeutralSolver: Newton (Schur) did not converge after "
                    f"{max_it} iterations.  ||F|| = {final_res:.4e}  "
                    f"G = {G:.4e}  H = {H:.4e}  λ = {float(self.lambda_nd):.4e}"
                )

        # Accumulate timing into run-level accumulators.
        # Backtrack passes through the loop are counted separately so that
        # _n_newton_iters_total stays comparable to pre-adaptive-damping
        # numbers (it counts accepted steps, not reverts).
        if hasattr(self, "_t_asm_total"):
            self._t_asm_total += _t_asm
            self._t_lin_total += _t_lin
            self._t_upd_total += _t_upd
            self._n_newton_iters_total += max(0, (nit + 1) - pol.n_bt)
            self._n_newton_backtracks_total += pol.n_bt
        # Advance the per-call trace index regardless of whether timers
        # are active, so (newton_call_idx, nit) tuples in the trace are
        # unambiguous even under repeated run()/run_steady() invocations.
        self._newton_call_idx += 1


    def solve(self) -> None:
        """Run one Newton solve (rebuilds persistent solver objects if needed)."""
        if getattr(self, "_solver_needs_rebuild", True):
            self._build_nonlinear_solver()

        _t0 = time.perf_counter()
        # [fluid-mean WP2] the level-constraint path of this solve (base: always the whole cell)
        _path = self._level_constraint_prepare()
        _cnt = "_n_lc_" + _path
        if hasattr(self, _cnt):
            setattr(self, _cnt, getattr(self, _cnt) + 1)
        if _path == "no_fluid":
            self._no_fluid_bypass()
        else:
            self._newton_solve()
        _dt = time.perf_counter() - _t0
        if hasattr(self, "_t_newton_solve"):
            self._t_newton_solve += _dt
            self._n_newton_calls += 1

    def run_steady(self) -> None:
        """Run a single steady-state solve.

        [option H] With ``fluid_deformation="ripple_halfspace"`` the solve is
        repeated as passes of :meth:`_ripple_run` (O5); otherwise exactly one."""
        if self._ripple_on:
            if self._ripple_pair():
                self._pair_ripple_run(self._run_steady_once)      # [phase-table WP3] two members
                return
            self._ripple_run(self._run_steady_once)
            return
        self._run_steady_once()

    def _run_steady_once(self) -> None:
        """The steady solve (the body ``run_steady`` always had)."""
        # Initialise timers / newton-trace state. Must happen before
        # self.solve() — ``_newton_solve`` reads ``_newton_call_idx`` and
        # increments it, which is set up in ``_reset_timers()``.
        # (``run()`` already does this at line 1444; ``run_steady()`` used
        # to skip it, which caused AttributeError: '_newton_call_idx'.)
        self._reset_timers()
        self._phase = "steady"
        if hasattr(self, "transient_on"):
            self.transient_on.assign(0.0)
        self._refresh_geometry()         # [fft_elastic_v1 seam] base: no-op
        # [Mixed P1.0 seam] base: raw solve; the mixed subclass overrides to
        # _solve_level() (the pair branch of the ripple).
        if self._ripple_on:
            self._ripple_level_begin("steady")
        self._steady_solve()
        if self._ripple_on:
            self._ripple_level_end("steady")

    # -------------------------
    # Convenience property
    # -------------------------
    @property
    def dP(self) -> float:
        """Pressure correction (Pa) — the primary microscale output.

        In the phase-neutral formulation, the cell-mean pressure is
        P_re + lambda, so the correction relative to the macroscale reference
        is simply lambda_dim = lambda_nd * p_scale.
        """
        return float(self.lambda_nd) * self.p_scale

    # -------------------------
    # Transient run (Stage 2)
    # -------------------------
    def _compute_cfl_dt(self, Tf: float, cfl_target: float = 1.0,
                        T_start: float = 0.0) -> float:
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
        """Update the macroscale reference pressure for the current time.

        Replaces the old _update_pinned_bc() call.  P_re_nd is a Constant
        embedded in the UFL graph for P_nd, so updating it here automatically
        propagates to all forms (residual, Jacobian, Schur vectors).
        """
        zeta = self.zeta
        p_re_dim = zeta.P + float(zeta.Pdot) * t_dim
        self.P_re_nd.assign(p_re_dim / self.p_scale)

    def run(self, Tf: float, n_steps: int = 10, cfl_target: float = 1.0,
            dt_override: "float | None" = None,
            step_callback: "Callable | None" = None) -> None:
        """Run a transient simulation (Stage 2 — carries over from template).

        Parameters
        ----------
        Tf : float
            End time (seconds).
        n_steps : int
            Number of time steps to perform.
        cfl_target : float
            Target CFL number for automatic dt computation.
        dt_override : float or None
            If provided, use this fixed time step instead of computing from
            CFL.  Useful for verification studies that need identical dt
            across runs with different n_steps.
        step_callback : callable or None
            Optional per-step hook ``step_callback(step, t_dim, dt_dim)``,
            invoked AFTER the converged solve at each time level so the
            caller can read solver state (e.g. ``spatial_homogenisation()``)
            at that instant.  It is called once for the rolling-steady seed
            state at ``T_start`` (``step == 0``) and once per main-loop step
            (``step == 1 .. n_steps``, with ``t_dim`` running up to ``Tf``).
            This enables time-averages of the homogenised outputs over the
            micro window without re-running the solve.  Default ``None``
            leaves production behaviour unchanged.  Note: if ``Tf <= 0`` or
            ``n_steps < 1`` the method delegates to ``run_steady()`` and the
            callback is NOT invoked.

        NOTE: Stage 1 tests use ``run_steady()`` only.  This method is
        provided for forward compatibility with Stage 2.

        [option H] With ``fluid_deformation="ripple_halfspace"`` the whole run
        (history synthesis, seed, steps) is pass 1 of :meth:`_ripple_run`, the
        ripple held fixed in the rough surface's frame during it (O6).  Later
        passes solve the end level only (:meth:`_run_end_level_once`, plan
        ripple_fast_loop_plan_2026-10-19 section 4.4) unless a ``step_callback``
        is given (every step must be visited: the mixed solver's ``record_homog``,
        any caller's hook): then every pass is a whole run and the callback fires in
        every pass (``self._ripple_pass`` says which).
        """
        Tf = float(Tf)
        if Tf <= 0.0 or n_steps < 1:
            self.run_steady()
            return
        if self._ripple_on:
            if self._ripple_pair():
                # [phase-table WP3, plan D5] two members: the converged per-step branch replaces the
                # pass loop (one run, u_r converged at every level); no end-level pass
                self._pair_ripple_run(lambda: self._run_once(Tf, n_steps, cfl_target, dt_override,
                                                             step_callback))
                return
            later = self._run_end_level_once if step_callback is None else None
            self._ripple_run(lambda: self._run_once(Tf, n_steps, cfl_target, dt_override,
                                                    step_callback), later)
            return
        self._run_once(Tf, n_steps, cfl_target, dt_override, step_callback)

    # -------------------------
    # [WP2 of docs/notes/ripple_fast_loop_plan_2026-10-19.md, section 4.3] Three
    # helpers extracted verbatim from _run_once (the seed block, the step loop's
    # time block, its BDF block and the two identical history shifts), so that
    # the ripple's end-level pass (_run_end_level_once) sets a time level, the
    # BDF order and the history exactly as a full run does.  Bitwise.
    # -------------------------
    def _set_time_level(self, t_dim: float) -> None:
        """Set every time-dependent forcing at the dimensional time ``t_dim``: the
        time Constants, the advection (half-)velocities, the surface-pattern
        velocity ``_Ux_dim_const`` / ``_Uy_dim_const``, the pressure gradients and
        the macroscale reference pressure ``P_re``.

        The seed of ``_run_once`` calls it at ``T_start`` and the step loop at
        ``T_start + step * dt`` (the same expressions in the same order: the seed's
        former ``zeta.Ux + T_start * zeta.Uxdot`` is the loop's ``zeta.Ux + t_dim * zeta.Uxdot``).

        ``_Ux_dim_const`` feeds ideal_ht, where it is multiplied by T to give the
        *surface pattern displacement*  s(T) = T * _Ux_dim_const.
        With a time-varying surface velocity  u_full(t) = 2*(Ux + UxT*t),
        the true displacement from 0 to T is
          s(T) = integral_0^T 2*(Ux + UxT*t) dt = 2*Ux*T + UxT*T^2
        so the "effective velocity" that reproduces this via T * _Ux_dim_const
        is the time-AVERAGED (not instantaneous) full velocity:
          _Ux_dim_const = s(T)/T = 2*Ux + UxT*T.
        A naive "2*(Ux + UxT*T)" double-counts the acceleration contribution
        and causes the pattern to reverse direction mid-run when u(t) never
        changes sign (reproduced symptom: film translates left, halts, then
        drifts right even though u(t) stays negative throughout).
        """
        zeta = self.zeta
        t_nd = t_dim / self.t0
        self.T_nd_const.assign(t_nd)
        self.T_dim_const.assign(t_dim)

        Ux_t = zeta.Ux + t_dim * zeta.Uxdot
        Uy_t = zeta.Uy + t_dim * zeta.Uydot
        # Instantaneous (half-)velocity, nondim, for advection in Reynolds.
        self.Ux_nd_const.assign(2.0 * Ux_t / self.Uref)
        self.Uy_nd_const.assign(2.0 * Uy_t / self.Uref)
        # Time-AVERAGED full velocity for ideal_ht's "T * Ux" surface-pattern
        # displacement term (derivation in the docstring).
        _alpha = float(self.velocity_ratio)
        self._Ux_dim_const.assign(_alpha * (2.0 * zeta.Ux + zeta.Uxdot * t_dim))
        self._Uy_dim_const.assign(_alpha * (2.0 * zeta.Uy + zeta.Uydot * t_dim))

        dpdx_t = zeta.dpdx + t_dim * zeta.dpdx_dot
        dpdy_t = zeta.dpdy + t_dim * zeta.dpdy_dot
        self.dpx_nd_const.assign((dpdx_t * self.Lx) / self.p_scale)
        self.dpy_nd_const.assign((dpdy_t * self.Ly) / self.p_scale)

        # Update macroscale reference pressure (no BC pin — uses Constant assign)
        self._update_P_re_nd(t_dim=t_dim)

    def _set_bdf(self, step: int) -> None:
        """The step loop's BDF order: BDF1 at step 1, BDF2 from step 2 (the seed
        sets its own BDF1 coefficients in place, without ``_have_two_hist``)."""
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
        """Shift the BDF film history by one level: project h_nd at the current
        state, nm2 <- nm1, nm1 <- the projection.  Cached L^2 projector instead of
        project() (which would re-factorise the mass matrix on every step): see
        _project_to_Vper and the seed-history block of _run_once.  [S1b] The mask
        history is no longer projected (no form reads it; see _setup_forms)."""
        h_curr = self._project_to_Vper(self.h_nd)
        self.h_hist_nm2.assign(self.h_hist_nm1)
        self.h_hist_nm1.assign(h_curr)

    def _run_end_level_once(self) -> None:
        """[option H pass loop] A later pass of :meth:`_ripple_run`: only the run's
        end level, with the BDF history a full run gives it (plan
        ``docs/notes/ripple_fast_loop_plan_2026-10-19.md`` section 4.4).  Uses
        ``T_start``, ``dt_dim_run`` and ``n_steps_run`` of the task's pass 1.

        Production's step-n solve sees nm1 = the film projected after level n - 1
        and nm2 = after level n - 2 (level 0 is the seed at T_start; for n = 1,
        level -1 is the synthesised T_start - dt with only T_dim moved, as
        ``_run_once`` builds it; BDF1 gives it a zero weight but it is filled the
        same way so the state matches).  A film at a level depends only on the
        geometry and the time: the ripple is held fixed in the surface frame,
the film does not depend on the pressure, and the contact solve sees
        the film only.  Rebuilding those two levels therefore reproduces that
        history; the end-level solve is then warm-started from the previous pass's
        end level, as every level of a full pass is.  Level times use the step
        loop's own expression ``T_start + j * dt`` through ``_set_time_level``, so
        every Constant is assigned the value the full run assigned.  The premise
        fails with a pressure-dependent film (the spring removed at WP2 was one):
        a change that adds one must revisit this method.

        Cost: two history levels (2 contact solves, 4 projections), one solve, and
        the final history shift that leaves the solver as a full run leaves it
        (nm1 = h(T_end), nm2 = h(T_end - dt)).  ``run`` offers this pass only
        without a ``step_callback``."""
        self._reset_timers()
        _t0 = time.perf_counter()
        T_start, dt_dim, n = float(self.T_start), float(self.dt_dim_run), int(self.n_steps_run)
        for j in (n - 2, n - 1):                  # oldest first, so nm1 ends at level n - 1
            self._set_time_level(T_start + max(j, 0) * dt_dim)
            if j < 0:                             # n = 1: _run_once's synthesised level (T_dim only)
                self.T_dim_const.assign(T_start - dt_dim)
            self._refresh_geometry()              # [fft_elastic_v1 seam] base: no-op
            self._advance_history()
        t_dim = T_start + n * dt_dim              # the step loop's own expression
        self._set_time_level(t_dim)
        self.dt_nd_const.assign(float(dt_dim / self.t0))
        self.transient_on.assign(1.0)
        self._set_bdf(n)
        self._refresh_geometry()
        self._update_squeeze_mean()
        self._t_init = time.perf_counter() - _t0
        self._phase = f"step {n}/{n} (end level)"
        self._ripple_level_begin(n)
        self._solve_level()
        self._ripple_level_end(n)
        _t0 = time.perf_counter()
        self._advance_history()                   # leave the state a full run leaves: nm1 = h(T_end)
        self._t_history_update += time.perf_counter() - _t0

    def _run_once(self, Tf: float, n_steps: int, cfl_target: float,
                  dt_override: "float | None",
                  step_callback: "Callable | None") -> None:
        """One transient run (the body ``run`` always had); ``Tf > 0``, ``n_steps >= 1``."""
        self._reset_timers()

        if dt_override is not None:
            # Fixed dt: bypass CFL computation entirely
            dt_dim = float(dt_override)
            T_start = Tf - n_steps * dt_dim
            if T_start < 0.0:
                T_start = 0.0
                n_steps = max(1, int(np.ceil(Tf / dt_dim)))
                dt_dim = Tf / n_steps
        else:
            # CFL-based time step and fast-forward start time
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

        # --- Initialise at T_start (rolling-steady fast-forward seed) -----
        # We synthesise one dt of prior surface history analytically from the
        # time-dependent ht expression, then run a single BDF1 transient
        # solve at T_start.  The resulting pressure is in balance with the
        # moving-surface dh/dt term, so no step-1 transient spike occurs
        # when the main loop begins.
        _t0 = time.perf_counter()

        # 1. Set all time-dependent forcings at T_start.  [WP2 of the ripple fast
        #    loop plan] The block that stood here is _set_time_level, shared
        #    with the step loop below (same expressions, same order: e.g.
        #    zeta.Ux + T_start * zeta.Uxdot is the loop's zeta.Ux + t_dim * zeta.Uxdot).
        self._set_time_level(T_start)

        # 2. Pre-populate BDF history by evaluating h_nd at past times.
        #    Only T_dim_const enters the surface expression in ideal_ht, so
        #    temporarily reassigning it is sufficient.  We populate two
        #    slots (T_start - dt, T_start - 2*dt) for robustness even though
        #    the BDF1 seed only consumes one.
        # NOTE: project() here would re-assemble AND re-factorise the
        # Vper mass matrix on every call.  self._project_to_Vper uses
        # the cached LU factor set up in _setup_spaces_and_unknowns, so
        # only the RHS is reassembled and a single back-substitution is
        # done.  Legacy DOLFIN's interpolate() is unusable here — it
        # rejects raw UFL Sum/Conditional nodes (self.h_nd / self.mask).
        # See MicroPhaseNeutral_Speedup_Investigation_2026-04-15.md item 2.
        #    [S1b] The mask history is not projected (no form reads it, _setup_forms).
        self.T_dim_const.assign(T_start - dt_dim)
        self._refresh_geometry()         # [fft_elastic_v1 seam] base: no-op
        h_past_1 = self._project_to_Vper(self.h_nd)

        self.T_dim_const.assign(T_start - 2.0 * dt_dim)
        self._refresh_geometry()
        h_past_2 = self._project_to_Vper(self.h_nd)

        # Restore T_dim_const before the seed solve.
        self.T_dim_const.assign(T_start)
        self._refresh_geometry()         # the seed's state (current time restored)

        self.h_hist_nm1.assign(h_past_1)
        self.h_hist_nm2.assign(h_past_2)

        # 3. Rolling-steady seed at T_start: one BDF1 transient solve with
        #    dt_nd and the synthesised history active.  transient_on=1
        #    ensures the dh/dt term participates in the residual; the
        #    solution is therefore the pressure that already balances the
        #    surface-motion squeeze term at T_start, not a static snapshot.
        self.dt_nd_const.assign(float(dt_nd))
        self.transient_on.assign(1.0)
        self.bdf_a0.assign(1.0)
        self.bdf_a1.assign(-1.0)
        self.bdf_a2.assign(0.0)

        self._update_squeeze_mean()

        # [Mixed P1.0 seam] base: raw solve; the mixed subclass overrides to
        # _solve_level().
        self._phase = "transient-seed"
        if self._ripple_on:
            self._ripple_level_begin("seed")
        self._seed_solve()
        if self._ripple_on:
            self._ripple_level_end("seed")

        # 4. Shift history forward by one step so that when the main loop
        #    runs step 1 (t = T_start + dt, BDF1, a1 = -1), it sees
        #      h_hist_nm1 = h(T_start)     <- correct "previous" snapshot
        #      h_hist_nm2 = h(T_start - dt)
        #    i.e. the BDF1 derivative at step 1 is (h(T+dt) - h(T))/dt, which
        #    is the genuine dh/dt -- no artificial first-step jump.
        # Cached-projector instead of project(): see rationale at the
        # seed-history block above.
        self._advance_history()                  # T_dim_const == T_start here:
        #                                          nm2 <- h(T_start - dt), nm1 <- h(T_start)
        self._have_two_hist = True               # two valid snapshots now exist

        self._t_init = time.perf_counter() - _t0

        # Optional per-step hook: emit the rolling-steady seed state at
        # T_start (step 0).  At this point all time-dependent forcings
        # (velocity, P_re, pressure ramp, T_dim_const) are set to their
        # T_start values and (w, lambda_nd) hold the seed solve solution,
        # so spatial_homogenisation() here returns the homogenised outputs
        # at the START of the fast-forward window.  Lets callers include the
        # window start in a time-average over [T_start, Tf].
        if step_callback is not None:
            step_callback(0, T_start, dt_dim)

        # --- Time-stepping loop ---
        for step in range(1, n_steps + 1):
            self._phase = f"step {step}/{n_steps}"
            _t0 = time.perf_counter()
            t_dim = T_start + step * dt_dim
            self._set_time_level(t_dim)      # [WP2] time Constants, velocities, gradients, P_re
            self._set_bdf(step)              # [WP2] BDF1 at step 1, BDF2 from step 2

            self._refresh_geometry()     # [fft_elastic_v1 seam] base: no-op
            self._update_squeeze_mean()

            self._t_step_overhead += time.perf_counter() - _t0

            # --- Solve (the pair branch of the ripple iterates u_r) ---
            if self._ripple_on:
                self._ripple_level_begin(step)
            self._solve_level()
            if self._ripple_on:
                self._ripple_level_end(step)

            # Optional per-step hook: emit the converged state for this step
            # (t_dim = T_start + step*dt_dim, up to Tf at step == n_steps) so
            # callers can accumulate per-step homogenised outputs, e.g. a
            # time-average of dQ over the micro window.
            if step_callback is not None:
                step_callback(step, t_dim, dt_dim)

            # [Phase 1a] Removed MICRO_DIAGNOSTICS block: it projected
            # phi/mask every step but its only consumers were commented-out
            # prints — pure cost, no output.

            _t0 = time.perf_counter()
            # Cached L^2 projector replaces project() (which would
            # re-factorise the mass matrix on every step).  See
            # _project_to_Vper and the seed-history block for rationale.
            self._advance_history()          # [WP2] nm2 <- nm1, nm1 <- h(t_dim), mask alike
            self._t_history_update += time.perf_counter() - _t0

    # -------------------------
    # Homogenisation
    # -------------------------
    def _record_pressure_ledger(self, Pst, fluid_area, area_total):
        # Signed fluid pressure per TOTAL cell area. Use the same physical
        # pressure and mask quadrature for all ledger terms, not nodal clipping.
        from microcell.pressure import DELTA_P_DEFINITION
        dxm = dx(domain=self.mesh_m)
        P_f = float(assemble(self.chi_fluid * self.P_dim * dxm) / area_total)
        if float(assemble((Constant(1.0) - self.chi_fluid) * dxm)) == 0.0:
            P_f = Pst  # exact full-fluid identity with the historical diagnostic
        P_pos = float(assemble(self.chi_fluid * conditional(gt(self.P_dim, 0), self.P_dim, 0) * dxm) / area_total)
        P_neg = float(assemble(self.chi_fluid * conditional(gt(self.P_dim, 0), 0, self.P_dim) * dxm) / area_total)
        self.pressure_ledger = {
            "delta_P_definition": DELTA_P_DEFINITION, "Pst": Pst,
            "P_f": P_f, "P_patch": Pst - P_f,
            # Empty fluid region has an undefined fluid-area mean.
            "Pst_fluid": P_f * area_total / fluid_area if fluid_area > 0.0 else float("nan"),
            "P_fluid_pos": P_pos, "P_fluid_neg": P_neg,
        }

        self.P_f = P_f
        self.P_patch = Pst - P_f
        self.Pst_fluid = self.pressure_ledger["Pst_fluid"]

    def spatial_homogenisation(self):
        """Return homogenised quantities (same 14-tuple as MicroMixedSolver).

        Since phi_c_nd and w_fluc are both zero-mean, the cell-averaged
        pressure is P_re + lambda, so Pst = p_scale*(P_re_nd + lambda_nd)
        = p0 + lambda_dim.  This is correct by construction — no gauge
        ambiguity, no phase-dependent artefact.

        Returns
        -------
        (Qx, Qy, Pst, Fst, taust_x, taust_y,
         p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin)
        """
        area_total = float(assemble(Constant(1.0) * dx(domain=self.mesh_m)))

        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.zeta.H) * self.h_nd, self.Vfull)

        h_arr = Hfull.vector().get_local()

        # [extrema M1, 23 Oct 2026] p_max / p_min: the fluid pressure extrema of the
        # shared sampler (micro_diagnostics.py; definition in ehl/pressure.py) -- signed
        # P_dim at the vertices that the sharp nodal level-set test classes as fluid; NaN
        # (phase status no_fluid) when there is none, never the unmasked values.  The
        # projected film above is the sampler's h_nd - h_min level set.  h_max / h_min
        # stay unmasked: inside a patch the film is the overclosure h_min reports.
        self.pressure_extrema = sample_fluid_pressure_extrema(self, film_dofs_m=h_arr)
        p_max = self.pressure_extrema.p_max_Pa
        p_min = self.pressure_extrema.p_min_Pa
        h_max = float(np.nanmax(h_arr))
        h_min = float(np.nanmin(h_arr))

        # Cell-averaged pressure (Pa). By construction:
        # ∫ P_dim dx = p_scale * (P_re_nd + lambda_nd) * area  (phi_c and w_fluc integrate to 0)
        Pst = float(assemble(Pfull * dx(domain=self.mesh_m)) / area_total)

        # Fluid fraction
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = assemble(chi_dg0 * dx(domain=self.mesh_m))
        Fst = float(fluid_area / area_total)

        self._record_pressure_ledger(Pst, fluid_area, area_total)

        fmax = 1.0 if fluid_area > 0.0 else 0.0
        fmin = 0.0

        # --- Fluxes ---
        gradP_nd = self.grad_cell(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd
        q_couette_nd = self.F_adv_nd
        q_poiseuille_nd = -self.K_nd * gradP_nd

        dim_scale = Constant(self.Uref * self.zeta.H / 2)
        q_dim = dim_scale * q_nd
        q_couette_dim = dim_scale * q_couette_nd
        q_poiseuille_dim = dim_scale * q_poiseuille_nd

        # [Phase 2.4 seam] Overridable flux sampling (bound variant uses
        # boundary-edge line integrals instead of cell-area means).
        (Qx, Qy, Qx_couette, Qy_couette,
         Qx_poiseuille, Qy_poiseuille) = self._flux_means(
            q_dim, q_couette_dim, q_poiseuille_dim, area_total)

        qy_full = project(q_dim[1], self.Vfull)
        qy_arr = qy_full.vector().get_local()
        qymax = float(np.nanmax(qy_arr))
        qymin = float(np.nanmin(qy_arr))

        # --- Shear stress (same as template) ---
        Ux_surf = float(self.Ux_nd_const) * self.Uref
        Uy_surf = float(self.Uy_nd_const) * self.Uref

        gradP_dim = gradP_nd * Constant(self.p_scale * self.invLx)
        eta_dim_expr = Constant(float(self.settings.eta0))      # constant lubricant viscosity [Pa s]
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

        # [extrema M2b, 23 Oct 2026] The fluid traction at the points of the quadrature rule of the
        # integrals above (micro_diagnostics.py; definition in ehl/fluid_shear.py): zero where the
        # sharp level-set test says solid (every point is fluid when h_min = 0).  taust above stays
        # the unmasked integral; the samples unweighted by any mask reproduce it (the sampler's check).
        self.fluid_traction = sample_fluid_traction(self, film_dofs_m=h_arr, existing_mask=None)

        # Store decomposition for diagnostics
        self.Qx_couette = Qx_couette
        self.Qy_couette = Qy_couette
        self.Qx_poiseuille = Qx_poiseuille
        self.Qy_poiseuille = Qy_poiseuille

        return (
            Qx, Qy, Pst, Fst, taust_x, taust_y,
            p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin,
        )
