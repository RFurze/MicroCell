"""Phase-neutral mixed solver with explicit local contact-law identities.

The inherited fluid obstruction mask and h_min datum are unchanged. Legacy
``legacy_patch_v1`` retains its footprint-radius surrogate. ``hertz_summit_v1``
uses analytic summit curvature, assumed overclosure and explicit two-body E*
for force, mechanical area, mean pressure and tangent. Neither law resolves
asperity interactions or displacement in neighbouring fluid regions.

``contact_frac`` is the integrated obstruction indicator; ``A_patch_frac`` is
its thresholded cell-mask area. Neither is mechanical area. The corrected law
provides ``A_mech_frac`` separately. Its DG0 pressure field is a force-conserving
mask display, not a resolved pressure distribution. Homogenised force is per
TOTAL cell area (Pa), additive to the existing macro pressure channel.

Serial-only: one micro task per worker. Periodic connected components describe
the obstruction mask; analytic summit counts describe corrected mechanics.

``fft_elastic_v1`` (16 Oct 2026, plan docs/notes/fft_elastic_contact_plan_
2026-10-16.md) solves one periodic elastic contact problem per cell on the
nodal grid (``ehl.elastic_contact``), refreshed at every time level through
the base class's ``_refresh_geometry`` seam.  The fluid sees the deformed film
``h_rigid + u_c`` (``uc_nd_func``, zero-mean ripple; the macro keeps the cell
mean, D3) and its obstruction mask is the contact set of the same solution,
through the complementarity level set ``phi_c = g - p/k_N`` (``phic_nd_func``,
D2).  Load, mechanical area, peak, tangent and mask all come from one nodal
solution.  The other two identities' UFL graphs are untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from fenics import *  # noqa: F401,F403 — UFL/assembly symbols used below
from microcell.contact import LEGACY, HERTZ, FFT, CONTACT_MODELS, cosine_summits, hertz_response, ContactResolutionError
from microcell import elastic_contact as _ec

from microcell.micro_phase_neutral_class_mixed import (  # noqa: F401
    MicroPhaseNeutralMixedSolver as _MixedSolver,
    MicroPhysicalParameters,        # re-export (worker convenience)
    PhaseNeutralSettings as _MixedPhaseNeutralSettings,
)


@dataclass(frozen=True)
class PhaseNeutralContactSettings(_MixedPhaseNeutralSettings):
    """Mixed settings and explicit contact-law parameters.

    The closure needs material data that the micro solver otherwise never
    sees.  They live on the settings dataclass (the Phase-2 subclass
    pattern: variant-specific knobs extend the settings, the shared
    ``MicroPhysicalParameters`` tuple stays stable); the worker fills
    them from ``config.material`` / ``config.mixed_lubrication`` (P1.5).

    ``contact_E`` MUST be set > 0 for any run that can contact —
    ``_asperity_pressure`` fails fast otherwise (no silent zero-load).
    """
    contact_E: float = 0.0            # Pa; Young's modulus (paper Eq. 23)
    contact_nu: float = 0.3           # Poisson ratio (only used if reduced modulus)
    use_reduced_modulus: bool = False  # legacy only: plain E or E/(1-nu^2)
    overclosure_measure: str = "patch_max"   # or "patch_mean" (plan sensitivity)
    sigma_yield: float = 250e6        # Pa; elastic-validity FLAG threshold (never clamps)
    # [WP-1 2026-09-03] Tangent of the closure, k_asp = -dP_asp/dH [Pa/m],
    # for the Newton-like macro coupling (implementation_plan_newton_
    # coupling_2026-09-02.md).  "off" (default) emits 0.0 -> the macro
    # behaves as today; "closure" emits the analytic derivative
    #   k_patch = k_asp_exponent * F_asp / delta
    # summed over patches per TOTAL cell area, exponent 1.75 = 1.5 (Hertz,
    # delta^1.5) + 0.25 (sqrt(r_c) with A ~ delta for a paraboloid trough;
    # the dense-grid geometric derivative on the P1 gate points is
    # 1.74-1.76).  1.5 = frozen-area (Hertz-only) lower bound; 2.0 =
    # elastic-foundation-like.  Never negative (patch merges can give a
    # locally negative true slope; the macro must never see one).
    k_asp_mode: str = "off"           # "off" | "closure"
    k_asp_exponent: float = 1.75
    contact_model: str = LEGACY
    contact_E_star: float | None = None  # explicit combined modulus; never reduce twice
    # [fft_elastic_v1] Polonsky-Keer stopping rule sum|dp|/sum p and iteration
    # budget (plan 3.2); the tangent's CG uses the same tolerance.
    contact_tol: float = 1e-8
    contact_maxit: int = 500
    # [extrema M3b, 23 Oct 2026] the dry asperity friction coefficient of the macro's asperity traction
    # (mixed_lubrication.mu_material, calc_asp_friction): the diagnostic asperity / total traction only
    # (micro_diagnostics.sample_contact_traction); no solver channel reads it.  NaN = not given.
    mu_material: float = float("nan")
    # [phase-table WP3/WP4, 4 Nov 2026] Asperity.phase_average as the worker fills it from
    # MicroCase.phase_average ("off" | "contact_table_v1"; plumbing of the one Case field, Rob 4 Nov):
    # "contact_table_v1" lifts the two-member refusal of the ripple (R-e, the converged per-step branch)
    # and makes the worker compose P_asp from the contact table.  Default "off": the settings of every
    # existing case compare equal to what they were (the solver cache key is unchanged).
    phase_average: str = "off"


class MicroPhaseNeutralContactSolver(_MixedSolver):
    """Mixed micro solver + asperity contact load channel (P1.2/P1.3)."""

    # [fft_elastic_v1] per-task contact bookkeeping (zeroed by reset_task)
    _contact_solution = None
    _contact_key = None
    _contact_p_warm = None
    _contact_solve_s = 0.0
    _contact_solves = 0
    _contact_iterations = 0
    _contact_restarts = 0                   # [phase-table D-B2] the warm-start rescues of the task (field on only)
    _contact_restart_kinds = None
    # [extrema M3b] this phase's asperity / total traction statistics (ehl/contact_shear.py)
    contact_traction = None

    # ------------------------------------------------------------------
    # [extrema M3b, 23 Oct 2026] asperity and total traction on the fluid sampler's points
    # ------------------------------------------------------------------
    def spatial_homogenisation(self):
        """The mixed class's homogenisation (every output unchanged), then the asperity and total
        traction of this phase on the sample set its fluid traction sampler kept
        (``micro_diagnostics.sample_contact_traction``: the stored ``fft_elastic_v1`` solution of this
        state, its key checked, never refreshed here; ``unsupported_model`` for the other identities,
        whose means :meth:`compute_asperity_load` completes)."""
        out = super().spatial_homogenisation()
        from microcell.micro_diagnostics import sample_contact_traction
        self.contact_traction = sample_contact_traction(self)
        return out

    def _spring_enabled(self):
        # This milestone has prescribed geometry, including for the historical
        # 1e20 Pa/m 'off' convention. No tiny residual spring iteration.
        if self.settings.contact_model == HERTZ:
            self._summit_geometry()  # reject unsupported configs before solving
            return False
        if self.settings.contact_model == FFT:
            self._fft_validate()     # micro fluid deformation is off in stage 1 (D6)
            return False
        return super()._spring_enabled()

    # ------------------------------------------------------------------
    # fft_elastic_v1: film and mask graph, grid map, refresh (plan WP3)
    # ------------------------------------------------------------------
    def _setup_film_and_mask(self) -> None:
        super()._setup_film_and_mask()
        if self._ripple_on and self.settings.contact_model == HERTZ:
            # [option H, plan O1] hertz's indentation comes from the inputs, not the film
            raise ValueError("hertz_summit_v1 cannot be used with fluid_deformation='ripple_halfspace' "
                             "(its overclosure ignores the film; plan O1)")
        if self.settings.contact_model != FFT:
            return          # legacy / hertz: the base graph, unchanged
        s = self.settings
        if self._ripple_on and s.contact_E_star is not None:
            # [option H, WP8] one body pair: u_f and u_c are the same operator (DV5 (i))
            Ef, Ec = float(s.fluid_E_star), float(s.contact_E_star)
            if not abs(Ef - Ec) <= 1e-12 * Ec:
                raise ValueError(f"fluid_E_star {Ef!r} != contact_E_star {Ec!r}: fft_elastic_v1 with "
                                 "fluid_deformation='ripple_halfspace' needs one body pair (plan O3)")
        # Nodal fields on Vper (allocated before this call: __init__ order).
        # h_nd gains the zero-mean contact deformation; the obstruction level
        # set is phi_c / H0 instead of h_nd - h_min (D2).  Zero until the first
        # refresh, exactly as after reset_task.
        self.uc_nd_func = Function(self.Vper, name="uc_nd")
        self.phic_nd_func = Function(self.Vper, name="phic_nd")
        self.h_nd = self.h_nd + self.uc_nd_func
        phi_mask = self.phic_nd_func
        self.chi_fluid = conditional(gt(phi_mask, 0.0), Constant(1.0), Constant(0.0))
        self.mask = self.smooth_levelset_mask(
            phi=phi_mask,
            delta=self.delta_h_nd_const,
            mask_core=Constant(s.eps_solid),
            mask_fluid=Constant(1.0),
        )

    def _reset_task_subclass(self) -> None:
        super()._reset_task_subclass()
        self.contact_traction = None        # [extrema M3b] never a previous task's
        if self.settings.contact_model != FFT:
            return
        for f in (self.uc_nd_func, self.phic_nd_func):
            f.vector().zero()
            f.vector().apply("insert")
        self._contact_solution = None
        self._contact_key = None
        self._contact_p_warm = None
        self._contact_solve_s = 0.0
        self._contact_solves = 0
        self._contact_iterations = 0
        self._contact_restarts = 0          # [phase-table D-B2] warm-start rescues of this task (field on only)
        self._contact_restart_kinds = {"collapse": 0, "stall": 0}

    def _fft_validate(self):
        s = self.settings
        if s.contact_E_star is None or not np.isfinite(s.contact_E_star) or s.contact_E_star <= 0:
            raise ValueError("fft_elastic_v1 requires an explicit positive two-body contact_E_star")
        if s.use_reduced_modulus or s.overclosure_measure != "patch_max":
            raise ValueError("fft_elastic_v1 uses E_star and the nodal contact solve; legacy "
                             "modulus/mean-overlap switches are incompatible")
        if 0 < self.params.k_spring < 1e19:
            raise ValueError("fft_elastic_v1 stage 1 requires micro fluid deformation off (D6)")
        if s.k_asp_mode not in ("off", "closure"):
            raise ValueError("k_asp_mode must be 'off' or 'closure'")
        descriptor = getattr(self.ht_cb, "contact_cosine", None)
        if not callable(getattr(descriptor, "height", None)):
            raise ValueError("fft_elastic_v1 needs a numpy rigid-film evaluator: the canonical "
                             "make_ideal_ht callback (ht.contact_cosine.height)")
        return descriptor

    def _contact_grid(self):
        """(n, i, j, Xg, Yg, op): the structured periodic n x n grid of the Vper dofs
        (the base class's checked map, shared with option H) and the contact operator."""
        g = getattr(self, "_contact_grid_cache", None)
        if g is not None:
            return g
        n, i, j = self._periodic_grid()
        t = np.arange(n) / n
        Xg, Yg = np.meshgrid(t, t, indexing="ij")
        op = _ec.PeriodicHalfSpace(n, n, float(self.Lx), float(self.Ly), float(self.settings.contact_E_star))
        g = (n, i, j, Xg, Yg, op)
        self._contact_grid_cache = g
        return g

    def _contact_state_key(self):
        descriptor = self.ht_cb.contact_cosine
        vals = (self.H0_const, self.HT_const, self.T_dim_const, self._Ux_dim_const, self._Uy_dim_const,
                descriptor.phi_x, descriptor.phi_y, descriptor.dphi_x, descriptor.dphi_y,
                self.params.h_min, self.params.xmax, self.params.ymax)
        key = tuple(float(v) for v in vals)
        if self._ripple_on:                 # [option H] the rigid gap includes u_f
            key = key + (float(self._ripple_version),)
        motion = getattr(self.ht_cb, "surface_motion", None)
        if motion is not None:              # [phase-table WP4, D3] the task's starting displacement
            key = key + (float(motion.s0[0]), float(motion.s0[1]))
        return key

    def _solve_contact_rescued(self, h_r, h_min, op, p0):
        """[phase-table D-B2, Rob 4 Nov 2026: "port the rescue, field-on only"] ``solve_contact`` with the study's
        warm-start rescue (``tests/micro_averaging/cell.py`` ``install_contact_restart``, 48-91 of the study tree) under
        ``Asperity.phase_average = "contact_table_v1"`` only: a WARM-started solve that collapses ("relative change
        inf / nan": the first CG step clears every warm-start node when the pattern has moved) or stalls
        ("contact_complementarity": converged on the stale active set) is retried once from the solver's own cold
        start, which converges in a few iterations on the same problem; a solve that never fails is unchanged bit
        for bit; a cold start that fails, and any other failure, is raised as before.  Counted per task
        (``_contact_restarts``, by kind) and printed by the worker.  With the field off the call is exactly the one
        it always was (no try, no rescue: the off path is bitwise)."""
        tol, maxit = float(self.settings.contact_tol), int(self.settings.contact_maxit)
        if not self._phase_table_on():
            return _ec.solve_contact(h_r, h_min, op, p0=p0, tol=tol, maxit=maxit)
        try:
            return _ec.solve_contact(h_r, h_min, op, p0=p0, tol=tol, maxit=maxit)
        except _ec.ContactNotConvergedError as exc:
            msg = str(exc)
            collapse = (msg.startswith("contact_not_converged: relative change inf")
                        or msg.startswith("contact_not_converged: relative change nan"))
            stall = msg.startswith("contact_complementarity")
            if p0 is None or not (collapse or stall):
                raise
            self._contact_restarts += 1
            if self._contact_restart_kinds is None:        # a solver before its first reset_task
                self._contact_restart_kinds = {"collapse": 0, "stall": 0}
            self._contact_restart_kinds["collapse" if collapse else "stall"] += 1
            return _ec.solve_contact(h_r, h_min, op, p0=None, tol=tol, maxit=maxit)

    def _rigid_film_grid(self):
        """Dimensional rigid film [m] on the contact grid, from the SAME
        description as the UFL ht (CosineContactDescriptor.height, gate G2a)."""
        n, i, j, Xg, Yg, op = self._contact_grid()
        p = self.params
        return self.ht_cb.contact_cosine.height(
            Xg, Yg, p.xmax, p.ymax, self.H0_const, self.HT_const, self.T_dim_const,
            self._Ux_dim_const, self._Uy_dim_const)

    def _refresh_geometry(self) -> None:
        super()._refresh_geometry()
        if self.settings.contact_model == FFT:
            self._refresh_contact()

    def _refresh_contact(self) -> None:
        """Rigid film -> contact solve (warm) -> uc_nd_func, phic_nd_func.

        D8: a mean rigid gap at or below h_min raises ContactInfeasibleError
        here, before any fluid solve; the worker records it as a micro failure
        (kNN-filled).  h_min <= 0 is the explicit contact-off convention: no
        deformation and the plain h - h_min level set.
        """
        self._fft_validate()
        _t0 = time.perf_counter()
        n, i, j, Xg, Yg, op = self._contact_grid()
        H0 = float(self.params.H0)
        h_min = float(self.params.h_min)
        h_r = self._rigid_film_grid()
        if self._ripple_on:
            # [option H, WP8] the contact is solved on the rigid film plus the fluid
            # ripple at this time (written just before by the base refresh, plan O2)
            h_r = h_r + self._dofs_to_grid(self.ur_nd_func.vector().get_local()) * H0
        if h_min <= 0.0:
            zero = np.zeros_like(h_r)
            sol = _ec.ContactSolution(p=zero, u=zero.copy(), g=h_r - h_min, phi_c=h_r - h_min,
                                      active=np.zeros(h_r.shape, bool), iterations=0, residual=0.0,
                                      change=0.0, k_N=op.k_nyquist)
        else:
            sol = self._solve_contact_rescued(h_r, h_min, op, self._contact_p_warm)
        for f, field in ((self.uc_nd_func, sol.u), (self.phic_nd_func, sol.phi_c)):
            f.vector().set_local(field[i, j] / H0)
            f.vector().apply("insert")
        self._contact_p_warm = sol.p
        self._contact_solution = sol
        self._contact_key = self._contact_state_key()
        self._contact_solves += 1
        self._contact_iterations += int(sol.iterations)
        self._contact_solve_s += time.perf_counter() - _t0

    def contact_only_load(self) -> dict:
        """[phase-table WP4, 4 Nov 2026; plan D4] The ripple-off contact solve on the SAME final rigid
        film as the stored coupled solution (``_rigid_film_grid()`` at the current time, without
        ``u_r``), warm-started from the coupled solution's pressure: the contact-only load P_co of this
        offset, from which the worker's additive correction c = P_cpl - P_co follows.  Pure: the stored
        solution, the warm start, the nodal fields and the counters are left as they are (the extrema
        and shear samplers and ``compute_asperity_load`` have read them already).  Returns
        ``Pasp`` [Pa], ``A_mech`` (``p1_area(phi_c)``), ``active_frac``, ``iterations`` and the solve
        time [s].  With the ripple off the stored solution IS the contact-only one and is returned
        without a solve (so P_asp = P̄ + 0 exactly, D1's degenerate)."""
        self._fft_validate()
        sol = self._contact_solution
        if sol is None or self._contact_key != self._contact_state_key():
            raise RuntimeError("contact_only_load: no stored contact solution of the current state "
                               "(call compute_asperity_load first)")
        n, i, j, Xg, Yg, op = self._contact_grid()
        if not self._ripple_on:
            return {"Pasp": float(sol.P_asp), "A_mech": float(_ec.p1_area(sol.phi_c)),
                    "active_frac": float(np.count_nonzero(sol.active)) / float(sol.active.size),
                    "iterations": 0, "solve_s": 0.0, "resolved": False}
        _t0 = time.perf_counter()
        h_min = float(self.params.h_min)
        h_r = self._rigid_film_grid()
        if h_min <= 0.0:
            return {"Pasp": 0.0, "A_mech": 0.0, "active_frac": 0.0, "iterations": 0, "solve_s": 0.0,
                    "resolved": False}
        co = self._solve_contact_rescued(h_r, h_min, op, sol.p)
        return {"Pasp": float(co.P_asp), "A_mech": float(_ec.p1_area(co.phi_c)),
                "active_frac": float(np.count_nonzero(co.active)) / float(co.active.size),
                "iterations": int(co.iterations), "solve_s": time.perf_counter() - _t0, "resolved": True}

    def _fft_asperity_load(self) -> dict:
        """compute_asperity_load for fft_elastic_v1: every output from the
        stored nodal solution of the current state (refreshed if stale)."""
        s = self.settings
        self._fft_validate()
        if self._contact_solution is None or self._contact_key != self._contact_state_key():
            self._refresh_contact()
        sol = self._contact_solution
        n, i, j, Xg, Yg, op = self._contact_grid()
        mesh = self.mesh_m
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        self._chi_dg0_last = chi_dg0
        self.contact_frac = 1.0 - float(assemble(chi_dg0 * dx(domain=mesh))) / float(self._cell_area)
        self.Pasp_hom = sol.P_asp
        self.A_mech_frac = _ec.p1_area(sol.phi_c)
        self.A_patch_frac = self.contact_frac          # the obstruction (chi) integral
        self.pasp_max = float(sol.p.max())             # nodal peak, resolution-dependent
        self.yield_flag = int(self.pasp_max > float(s.sigma_yield))
        if s.k_asp_mode == "closure":
            self.k_asp_hom = -_ec.contact_tangent(op, sol.active, tol=float(s.contact_tol),
                                                  maxit=int(s.contact_maxit))
        else:
            self.k_asp_hom = 0.0
        n_comp, min_nodes = _ec.periodic_components(sol.active)
        self.n_patches_loaded = n_comp
        self.contact_patches = []
        self.p_asp_field = Function(self.Vper, name="p_asp_nodal")
        self.p_asp_field.vector().set_local(sol.p[i, j])
        self.p_asp_field.vector().apply("insert")
        A_mech = self.A_mech_frac
        self.contact_indicators = {
            "contact_geometry_valid": True,             # no geometric screen for the nodal solve
            "contact_iterations": int(self._contact_iterations),
            "contact_solves": int(self._contact_solves),
            "contact_solve_s": float(self._contact_solve_s),
            "contact_residual_m": float(sol.residual),
            "contact_min_patch_nodes": int(min_nodes),
            "contact_area_mismatch": (abs(self.A_patch_frac - A_mech) / A_mech) if A_mech > 0 else 0.0,
            "contact_grid_n": int(n),
        }
        return {
            "Pasp_hom": float(self.Pasp_hom),
            "contact_frac": float(self.contact_frac),
            "pasp_max": float(self.pasp_max),
            "yield_flag": int(self.yield_flag),
            "n_patches": int(n_comp),
            "A_patch_frac": float(self.A_patch_frac),
            "n_patches_loaded": int(n_comp),
            "k_asp_hom": float(self.k_asp_hom),
            "A_mech_frac": float(A_mech),
            "contact_model": FFT,
            "contact_pressure_statistic": "resolved_nodal_peak",
            **self.contact_indicators,
        }

    # ------------------------------------------------------------------
    # [option H] fft_elastic_v1 quantities for the 13.4 pass checks (plan O5)
    # ------------------------------------------------------------------
    def _ripple_contact_snapshot(self) -> dict:
        if self.settings.contact_model != FFT:
            return super()._ripple_contact_snapshot()
        sol = self._contact_solution
        if sol is None:
            return {}
        return {"p": sol.p.copy(), "P_asp": float(sol.P_asp), "A_mech": float(_ec.p1_area(sol.phi_c)),
                "set": np.asarray(sol.active, dtype=bool).copy()}

    def _ripple_chi_nodes(self) -> np.ndarray:
        if self.settings.contact_model != FFT:
            return super()._ripple_chi_nodes()
        return (self.phic_nd_func.vector().get_local() > 0.0).astype(float)

    def _ripple_traction_overlap(self, s_dofs) -> float:
        """Share of the projected fluid traction |s| on contact nodes: the support the
        L2 projection smears into the contact set (a discretisation overlap, plan O2)."""
        if self.settings.contact_model != FFT or self._contact_solution is None:
            return 0.0
        a = np.abs(np.asarray(s_dofs, dtype=float))
        tot = float(a.sum())
        if tot <= 0.0:
            return 0.0
        act = self._grid_to_dofs(self._contact_solution.active)
        return float(a[act].sum()) / tot

    def _summit_geometry(self):
        s = self.settings
        if s.contact_model not in CONTACT_MODELS:
            raise ValueError(f"unknown contact model {s.contact_model!r}")
        if s.contact_model == LEGACY:
            return None
        if s.contact_E_star is None or not np.isfinite(s.contact_E_star) or s.contact_E_star <= 0:
            raise ValueError("hertz_summit_v1 requires an explicit positive two-body contact_E_star")
        if s.use_reduced_modulus or s.overclosure_measure != "patch_max":
            raise ValueError("Hertz uses E_star and summit indentation; legacy modulus/mean-overlap switches are incompatible")
        if 0 < self.params.k_spring < 1e19:
            raise ValueError("hertz_summit_v1 milestone requires micro fluid deformation off")
        descriptor = getattr(self.ht_cb, "contact_cosine", None)
        if descriptor is None or descriptor[3]:
            raise ValueError("hertz_summit_v1 requires the canonical single-surface cosine callback")
        return cosine_summits(*descriptor[:3], float(self.Lx), float(self.Ly))

    def _summit_pressure(self, patches, geometry):
        """Analytic reaction with the existing obstruction mask kept separate.

        p_asp_field is ONLY a force-conserving display over the obstruction
        cells. Its support and maximum are not the mechanical Hertz footprint
        or pressure. A_mech_frac and pasp_max describe the scalar Hertz patch.
        """
        delta = (float(self.params.h_min) - float(self.H0_const)
                 - float(self.HT_const) * float(self.T_dim_const) + geometry.amplitude)
        if self.params.h_min <= 0:
            delta = 0.0  # same explicit contact-off convention as the fluid mask
        F, A, mean, stiffness = hertz_response(geometry.radius, delta, self.settings.contact_E_star)
        total_area = float(self.Lx) * float(self.Ly)
        n = geometry.count if F > 0 else 0
        self.Pasp_hom = n * F / total_area
        self.A_mech_frac = n * A / total_area
        self.pasp_max = mean
        self.k_asp_hom = n * stiffness / total_area if self.settings.k_asp_mode == "closure" else 0.0
        if self.settings.k_asp_mode not in ("off", "closure"):
            raise ValueError("k_asp_mode must be 'off' or 'closure'")
        self.n_patches_loaded = n
        self.yield_flag = int(mean > self.settings.sigma_yield)
        self.contact_indicators = geometry.indicators(delta)
        self.contact_indicators["curvature_radius_m"] = geometry.radius
        self.contact_indicators["indentation_m"] = max(delta, 0.0)
        mask_area = sum(p["area_dim"] for p in patches)
        self.A_patch_frac = mask_area / total_area
        # Do not quietly discard an analytic load when the fluid mesh has no
        # supporting cells. It is a recorded micro failure under the existing
        # worker policy, eligible for filling but not direct validation.
        if F > 0 and mask_area <= 0:
            raise ContactResolutionError("contact_mask_unresolved: positive analytic summit load without obstruction cells")
        self.p_asp_field = Function(self.Vdg0, name="p_asp_mask_display")
        vec = self.p_asp_field.vector().get_local()
        for patch in patches:
            pressure = n * F / mask_area if mask_area else 0.0
            patch["p_asp"] = pressure
            patch["F_asp"] = pressure * patch["area_dim"]
            vec[self._contact_cell_dofs[patch["cells"]]] = pressure
        self.p_asp_field.vector().set_local(vec)
        self.p_asp_field.vector().apply("insert")

    # ------------------------------------------------------------------
    # P1.2 — contact patch identification
    # ------------------------------------------------------------------
    def _find_contact_patches(self) -> list:
        """Identify contact patches on the DG0 cell indicator.

        Returns a list of patch dicts:
            cells      : np.ndarray of cell indices
            area_mesh  : patch area in mesh (unit-square) coordinates
            area_dim   : dimensional area [m^2]  (= area_mesh * lx * ly)
            over_max   : patch-max  overclosure max(h_min − h, 0) [m]
            over_mean  : patch-mean overclosure (area-weighted)   [m]
            wrapped    : True if the patch was merged across the periodic
                         boundary

        Also stores ``self._chi_dg0_last`` (the projected indicator) and
        ``self.contact_frac`` (integral definition — see module docstring).

        Periodic wrap [plan P1.2.2]: contact cells with an exterior facet
        on x=0 are merged with contact cells on x=lx whose facet
        y-intervals overlap (same in y).  This matters — the asperity is
        cell-centred only for the idealised cosine; arbitrary phase
        shifts move the patch across the wrap.
        """
        mesh = self.mesh_m
        if MPI.size(mesh.mpi_comm()) != 1:
            raise RuntimeError(
                "_find_contact_patches is serial-only (micro tasks run "
                "one-per-worker); got a parallel mesh communicator.")

        _t0 = time.perf_counter()

        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        self._chi_dg0_last = chi_dg0
        # Integral contact fraction (exact complement of the Fst slot).
        area_total = float(self._cell_area)
        self.contact_frac = 1.0 - float(
            assemble(chi_dg0 * dx(domain=mesh))) / area_total

        dm0 = self.Vdg0.dofmap()
        chi_vals = chi_dg0.vector().get_local()
        ncells = mesh.num_cells()
        if not hasattr(self, "_contact_cell_dofs"):
            self._contact_cell_dofs = np.array([dm0.cell_dofs(c)[0] for c in range(ncells)])
        cell_chi = chi_vals[self._contact_cell_dofs]
        contact = cell_chi < 0.5
        if not bool(contact.any()):
            self.contact_patches = []
            if hasattr(self, "_t_diagnostics"):
                self._t_diagnostics += time.perf_counter() - _t0
            return []

        # --- Union-find over facet-adjacent contact cells -------------
        parent = np.arange(ncells)

        def find(a: int) -> int:
            root = a
            while parent[root] != root:
                root = parent[root]
            while parent[a] != root:          # path compression
                parent[a], a = root, parent[a]
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        mesh.init(1, 2)   # facet -> cell connectivity
        mesh.init(1, 0)   # facet -> vertex (for boundary intervals)
        left, right, bottom, top = [], [], [], []   # (lo, hi, cell) intervals
        coords = mesh.coordinates()
        tol = 1e-10
        for f in facets(mesh):
            cells_of_f = f.entities(2)
            if len(cells_of_f) == 2:
                c0, c1 = int(cells_of_f[0]), int(cells_of_f[1])
                if contact[c0] and contact[c1]:
                    union(c0, c1)
            elif len(cells_of_f) == 1:
                # Exterior facet: collect boundary intervals of contact
                # cells for the periodic wrap merge below.
                c0 = int(cells_of_f[0])
                if not contact[c0]:
                    continue
                vts = coords[f.entities(0)]
                xlo, xhi = float(vts[:, 0].min()), float(vts[:, 0].max())
                ylo, yhi = float(vts[:, 1].min()), float(vts[:, 1].max())
                if xhi < tol:
                    left.append((ylo, yhi, c0))
                elif xlo > 1.0 - tol:
                    right.append((ylo, yhi, c0))
                if yhi < tol:
                    bottom.append((xlo, xhi, c0))
                elif ylo > 1.0 - tol:
                    top.append((xlo, xhi, c0))

        # --- Merge across the periodic wrap (interval overlap) --------
        for side_a, side_b in ((left, right), (bottom, top)):
            for lo_a, hi_a, ca in side_a:
                for lo_b, hi_b, cb in side_b:
                    if min(hi_a, hi_b) - max(lo_a, lo_b) > -tol:
                        union(ca, cb)

        # --- Gather patches -------------------------------------------
        groups: dict = {}
        for c in np.nonzero(contact)[0]:
            groups.setdefault(find(int(c)), []).append(int(c))

        wrapped_roots = set()
        for side_a, side_b in ((left, right), (bottom, top)):
            for lo_a, hi_a, ca in side_a:
                for lo_b, hi_b, cb in side_b:
                    if min(hi_a, hi_b) - max(lo_a, lo_b) > -tol:
                        wrapped_roots.add(find(ca))

        # Nodal dimensional film for overclosure (CG1 vertex values via
        # cell dofs — convention-proof: uses the assembled h_nd itself,
        # including any spring deformation).
        Hfull = project(Constant(self.params.H0) * self.h_nd, self.Vfull)
        h_vals = Hfull.vector().get_local()
        dmV = self.Vfull.dofmap()
        h_min_dim = float(self.params.h_min)
        lx_ly = float(self.Lx) * float(self.Ly)

        patches = []
        for root, cell_list in groups.items():
            cells_arr = np.asarray(sorted(cell_list), dtype=int)
            area_mesh = 0.0
            over_max = 0.0
            mean_num = 0.0
            for c in cells_arr:
                a_c = Cell(mesh, int(c)).volume()
                area_mesh += a_c
                h_cell = h_vals[dmV.cell_dofs(int(c))]
                over_cell = np.maximum(h_min_dim - h_cell, 0.0)
                over_max = max(over_max, float(over_cell.max()))
                mean_num += a_c * float(over_cell.mean())
            patches.append({
                "cells": cells_arr,
                "area_mesh": area_mesh,
                "area_dim": area_mesh * lx_ly,
                "over_max": over_max,
                "over_mean": mean_num / max(area_mesh, 1e-300),
                "wrapped": root in wrapped_roots,
            })

        # Deterministic ordering (largest first) for reproducible logs.
        patches.sort(key=lambda p: -p["area_mesh"])
        self.contact_patches = patches
        if hasattr(self, "_t_diagnostics"):
            self._t_diagnostics += time.perf_counter() - _t0
        return patches

    # ------------------------------------------------------------------
    # P1.3 — Hertz/GT patch closure
    # ------------------------------------------------------------------
    def _asperity_pressure(self, patches: list) -> None:
        """Historical footprint-radius surrogate, preserved for reproduction.

        Sets the DG0 mask pressure, homogenised force per total cell area,
        mask-area fraction, patch count, mean pressure and yield flag. The
        thresholded mask area has no ordering invariant against the integrated
        obstruction fraction: partial cells can increase or decrease either.
        This method supplies no independent mechanical contact area.
        """
        s = self.settings
        self.p_asp_field = Function(self.Vdg0, name="p_asp")
        self.Pasp_hom = 0.0
        self.pasp_max = 0.0
        self.yield_flag = 0
        self.A_patch_frac = 0.0
        self.n_patches_loaded = 0
        self.k_asp_hom = 0.0     # [WP-1] Pa/m, per TOTAL cell area
        if not patches:
            return

        E = float(s.contact_E)
        if E <= 0.0:
            raise ValueError(
                "PhaseNeutralContactSettings.contact_E must be > 0 when "
                "contact occurs (got %r) — refusing to emit silent zero "
                "asperity load." % E)
        if s.use_reduced_modulus:
            E_eff = E / (1.0 - float(s.contact_nu) ** 2)
        else:
            E_eff = E   # paper-faithful plain E (Eq. 23)

        measure = str(s.overclosure_measure)
        if measure not in ("patch_max", "patch_mean"):
            raise ValueError(
                "overclosure_measure must be 'patch_max' or 'patch_mean', "
                "got %r" % measure)
        k_mode = str(getattr(s, "k_asp_mode", "off"))
        if k_mode not in ("off", "closure"):
            raise ValueError(
                "k_asp_mode must be 'off' or 'closure', got %r" % k_mode)
        k_exp = float(getattr(s, "k_asp_exponent", 1.75))

        vec = self.p_asp_field.vector().get_local()
        dm0 = self.Vdg0.dofmap()
        F_total = 0.0
        A_total = 0.0          # [G3] sum of load-bearing patch areas [m^2]
        k_total = 0.0          # [WP-1] sum of patch tangents dF/d(-H) [N/m]
        for patch in patches:
            A_asp = float(patch["area_dim"])
            if A_asp <= 0.0:
                continue
            delta = float(patch["over_max"] if measure == "patch_max"
                          else patch["over_mean"])
            r_contact = np.sqrt(A_asp / np.pi)
            F_asp = (4.0 / 3.0) * E_eff * np.sqrt(r_contact) * delta ** 1.5
            p_asp = F_asp / A_asp
            patch["F_asp"] = F_asp
            patch["p_asp"] = p_asp
            F_total += F_asp
            A_total += A_asp
            # [WP-1] closure tangent; delta > 0 whenever F_asp > 0.
            if k_mode == "closure" and delta > 0.0:
                k_patch = k_exp * F_asp / delta
                patch["k_patch"] = k_patch
                k_total += max(k_patch, 0.0)
            self.n_patches_loaded += 1
            self.pasp_max = max(self.pasp_max, p_asp)
            for c in patch["cells"]:
                vec[dm0.cell_dofs(int(c))[0]] = p_asp
        self.p_asp_field.vector().set_local(vec)
        self.p_asp_field.vector().apply("insert")

        _cell_area = float(self.Lx) * float(self.Ly)
        self.Pasp_hom = F_total / _cell_area
        self.A_patch_frac = A_total / _cell_area
        self.k_asp_hom = k_total / _cell_area        # [WP-1] Pa/m (0 when off)
        if self.pasp_max > float(s.sigma_yield):
            self.yield_flag = 1

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def compute_asperity_load(self) -> dict:
        """Identify patches, apply the closure, return the summary dict.

        Call AFTER a converged solve (quasi-static, memory-free — plan
        §1: re-evaluated per step; the worker calls this once per task,
        P1.5).  Keys: ``Pasp_hom`` [Pa], ``contact_frac`` [-],
        ``pasp_max`` [Pa], ``yield_flag`` {0,1}, ``n_patches``.

        ``A_patch_frac`` describes the obstruction mask. ``A_mech_frac``
        describes mechanical Hertz area only for the corrected law (undefined
        for legacy). ``n_patches`` counts mask components; ``n_patches_loaded``
        counts analytic summits in the corrected law and loaded mask components
        in legacy. ``contact_indicators`` records geometric limits explicitly.

        ``fft_elastic_v1``: see :meth:`_fft_asperity_load` (``pasp_max`` is
        the nodal peak of the resolved pressure, resolution-dependent).

        [extrema M3b] Afterwards the phase's asperity / total traction statistics are rechecked
        against the extraction (``micro_diagnostics.finish_contact_traction``: a refreshed solution
        makes them ``state_mismatch``; an unsupported model's means come from ``Pasp_hom``).
        """
        from microcell.micro_diagnostics import finish_contact_traction
        if self.settings.contact_model == FFT:
            out = self._fft_asperity_load()
            self.contact_traction = finish_contact_traction(self, out)
            return out
        geometry = self._summit_geometry()
        patches = self._find_contact_patches()
        self.contact_indicators = {}
        if geometry is None:
            self._asperity_pressure(patches)
            self.A_mech_frac = float("nan")  # legacy has no independent mechanical area
        else:
            self._summit_pressure(patches, geometry)
        self.contact_traction = finish_contact_traction(self, {"Pasp_hom": float(self.Pasp_hom)})
        return {
            "Pasp_hom": float(self.Pasp_hom),
            "contact_frac": float(self.contact_frac),
            "pasp_max": float(self.pasp_max),
            "yield_flag": int(self.yield_flag),
            "n_patches": len(patches),
            "A_patch_frac": float(self.A_patch_frac),
            "n_patches_loaded": int(self.n_patches_loaded),
            # [WP-1] closure tangent k_asp = -dP_asp/dH [Pa/m]; 0.0 when
            # k_asp_mode == "off" or no contact.
            "k_asp_hom": float(self.k_asp_hom),
            "A_mech_frac": float(self.A_mech_frac),
            "contact_model": self.settings.contact_model,
            "contact_pressure_statistic": "mechanical_patch_mean" if geometry else "obstruction_patch_mean",
            **self.contact_indicators,
        }
