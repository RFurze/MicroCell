"""
Phase-Neutral Microscale Solver — MIXED-LUBRICATION variant
===========================================================

[Phase 2.4] Rewritten as a SUBCLASS of ``MicroPhaseNeutralSolver``
(was a 4,825-line isolated copy; the functional delta is the June 2026
contact model below, everything else is inherited).

Delta vs the base solver (all validated by tests/diagnose_contact_blowup.py):

* ``pattern_flux_contact`` setting (default True): inside the contact patch
  the Couette transport blends to the PATTERN velocity (the patch carries
  its trapped film) instead of being truncated in the lab frame; the
  squeeze source is the full unmasked dh/dt with uniform mean
  compensation; and the cavitation penalty acts at FULL strength inside
  the patch (w_cav = chi*mask + (1-chi)), pinning the fictitious patch
  pressure at the cavitation floor at any eps_solid.
  CAVEAT: requires a single rigidly-translating roughness pattern — set
  False for two-surface (standing-wave) roughness.
* Mask-weighted closure (MIXED FIX): G, g and d are mask-weighted so
  fictitious contact-region pressures cannot drive the pressure level.
* Mask-weighted squeeze compensation in the legacy branch (MIXED FIX):
  removes the spurious O(1) dead-zone source that caused the uniform
  ~6e7 Pa post-contact blow-up.
* ``chi_fluid``-restricted spring load (contact pressure must not load
  the spring) and fluid-masked homogenisation outputs (incl. Pst_fluid).
* ``record_homog`` / ``homog_history`` per-step homogenisation recording.
* [Mixed P1.0, July 2026 port] spring coupling wraps the STEADY solve and
  the transient SEED (``_steady_solve``/``_seed_solve`` overrides →
  ``_solve_with_spring``).

All reductions hold: legacy masked model for h_min = 0; masked transport
for static textures (v_p = 0).

Diagnostics/sweeps: run ``microscale/tests/pn_diagnostics.py`` with
``SOLVER_CLS = MicroPhaseNeutralMixedSolver`` (note: force
``pattern_flux_contact=False`` for two_surface modes).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from fenics import *  # noqa: F401,F403 — UFL/assembly symbols used below

from microcell.micro_phase_neutral_class import (  # noqa: F401
    MicroPhaseNeutralSolver as _BaseSolver,
    MicroPhysicalParameters,        # re-export (back-compat)
    MicroSolverSettings,            # re-export (back-compat)
    PhaseNeutralSettings as _BasePhaseNeutralSettings,
    PeriodicBoundary,               # re-export (back-compat)
)
# [extrema M1] the shared fluid pressure extrema sampler; [extrema M2b] the fluid traction sampler
from microcell.micro_diagnostics import sample_fluid_pressure_extrema, sample_fluid_traction


@dataclass(frozen=True)
class PhaseNeutralSettings(_BasePhaseNeutralSettings):
    """Base phase-neutral settings + the mixed contact-model switch.

    ``pattern_flux_contact`` (June 2026 contact model, default True):
    see module docstring.  Set False for two-surface roughness.
    """
    pattern_flux_contact: bool = True


class MicroPhaseNeutralMixedSolver(_BaseSolver):
    """Phase-neutral micro solver + mixed-lubrication contact model.

    NOTE: ``record_homog`` is a CLASS attribute because the base
    ``__init__`` may call ``run_steady()`` (auto_solve=True) before any
    subclass ``__init__`` body could run.  Set it on the instance after
    construction to enable per-step homogenisation recording into
    ``homog_history``.
    """

    record_homog: bool = False
    homog_history: list = []          # shadowed per-instance by run()/run_steady()

    # ------------------------------------------------------------------
    # Setup extensions
    # ------------------------------------------------------------------
    def _setup_pressure_expressions(self) -> None:
        super()._setup_pressure_expressions()
        p = self.params
        # --- Contact model (June 2026): pattern-frame flux blend + pinned
        # patch pressure.  See PhaseNeutralSettings.pattern_flux_contact
        # docstring and tests/diagnose_contact_blowup.py for the validation
        # history.  Three coupled pieces (the other two live in
        # _build_forms / _setup_forms / _update_squeeze_mean):
        #
        # (1) TRANSPORT: the mask model enforces "no flow through contact"
        #     in the LAB frame (flux -> ~0), which is only correct for a
        #     static texture.  A translating patch carries its trapped
        #     film, so the lab flux must blend to v_pattern*h:
        #         F_adv = mask*U*h + (1-mask)*V_p*h
        #     In the patch frame the through-flux is then zero, and the
        #     edge jump is the physical RELATIVE flux (U - V_p)*h_min —
        #     correct damming orientation (fixes the inverted edge dipole).
        #
        #     V_p must be the INSTANTANEOUS pattern velocity: the ht
        #     displacement is s(t) = alpha*(2*Ux*t + UxT*t^2), so
        #     v_p(t) = ds/dt = alpha*2*(Ux + UxT*t), and the doubled-
        #     nondim coefficient is 4*alpha*(Ux + UxT*t)/Uref (the code's
        #     F_adv carries doubled transport, paired with the /6
        #     viscosity scaling).  Built on T_dim_const so run() updates
        #     it every step.  Do NOT use _Ux_dim_const here — it holds
        #     the displacement-AVERAGED velocity (exact only for UxT=0).
        #
        # (2) PENALTY (pin-patch): full strength inside the patch,
        #         w_cav = chi*mask + (1 - chi),
        #     pinning the physically-meaningless patch pressure at the
        #     cavitation floor at ANY eps_solid.  Without this, the patch
        #     truncation residual produces P_patch ~ -resid/(gamma*eps),
        #     and the whole-cell <w>=0 constraint converts it into a
        #     uniform fluid lift that suppresses cavitation and makes
        #     lambda eps- and path-dependent (cold-restart ratios 0.81 /
        #     0.30 at eps 1e-6/1e-7 before the pin; 1.00 after).
        self._pattern_flux_on = bool(
            getattr(self.settings, "pattern_flux_contact", True))
        if self._pattern_flux_on:
            _alpha_pf = float(p.roughness_velocity_ratio)
            if 0.0 < _alpha_pf < 1.0:
                # rvr in (0,1) is the signature of two-surface runs
                # (default 0.5).  The blend is only valid for a SINGLE
                # rigidly-translating pattern at alpha*2*U; standing-wave
                # two-surface films have no single pattern velocity.
                warnings.warn(
                    "pattern_flux_contact=True with "
                    f"roughness_velocity_ratio={_alpha_pf}: the pattern-flux "
                    "contact model assumes a single rigidly-translating "
                    "roughness pattern. For two-surface (standing-wave) "
                    "films set pattern_flux_contact=False.",
                    stacklevel=2,
                )
            # [Micro-perf P2.3c] pattern-flux constants promoted to stored
            # attributes (task-varying: Uref scaling + velocities) so
            # reset_task()/_reset_task_subclass() can re-target them.
            # One attribute per former anonymous site — UFL graph unchanged.
            self._cf_pf_const = Constant(4.0 * _alpha_pf / self.Uref)
            self._pf_Ux_const = Constant(float(p.Ux))
            self._pf_UxT_const = Constant(float(p.UxT))
            self._pf_Uy_const = Constant(float(p.Uy))
            self._pf_UyT_const = Constant(float(p.UyT))
            _cf_pf = self._cf_pf_const
            self._V_p_nd = as_vector((
                _cf_pf * (self._pf_Ux_const
                          + self._pf_UxT_const * self.T_dim_const),
                _cf_pf * (self._pf_Uy_const
                          + self._pf_UyT_const * self.T_dim_const),
            ))
            # The pattern term exists only while time is running: a STEADY
            # solve freezes the pattern, and a frozen pattern is a static
            # texture, for which the masked transport IS the correct
            # contact model.  Without this transient_on gating, a steady
            # solve at a contacted state carries an uncancelled O(1)
            # source div(V_p*h) in the patch (its transient counterpart
            # dh/dt is switched off by transient_on) and Newton diverges —
            # observed in the A/B squeeze-isolation harness (steady
            # re-solve at the final transient state, ball mode).
            if not hasattr(self, "transient_on"):
                self.transient_on = Constant(0.0)
            self.F_adv_nd = (
                self.mask * self.U_nd
                + self.transient_on
                * (Constant(1.0) - self.mask) * self._V_p_nd
            ) * self.h_nd
            self._cav_weight = (
                self.chi_fluid * self.mask
                + (Constant(1.0) - self.chi_fluid)
            )
        else:
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
        super()._setup_forms()
        # g_j = dG/dw_j, MASK-WEIGHTED (consistent with the masked closure
        # G and with c): keeps the fictitious contact-region pressures out
        # of the Jacobian of (R, G).  Identical to the base when h_min = 0.
        # [Micro-perf P2.3c] own stored gamma for the mask-weighted g-form
        # (the base's gamma_nd_g_const belongs to the discarded base form).
        self.gamma_nd_gmix_const = Constant(self.gamma_nd)
        self._g_form_raw = (
            self.gamma_nd_gmix_const * self.mask * self._spos_deriv_nd * self.v * dx
        )

    def _reset_task_subclass(self) -> None:
        """[Micro-perf P2.3c] Re-target the mixed-specific task-varying
        form Constants (pattern-flux transport + mask-weighted g-form).

        Called from the base ``reset_task`` AFTER ``_nondim_scalars()``,
        so ``self.Uref`` / ``self.gamma_nd`` already hold the NEW task's
        values.  Mirrors _setup_pressure_expressions / _setup_forms.
        """
        p = self.params
        if self._pattern_flux_on:
            _alpha_pf = float(p.roughness_velocity_ratio)
            self._cf_pf_const.assign(4.0 * _alpha_pf / self.Uref)
            self._pf_Ux_const.assign(float(p.Ux))
            self._pf_UxT_const.assign(float(p.UxT))
            self._pf_Uy_const.assign(float(p.Uy))
            self._pf_UyT_const.assign(float(p.UyT))
        self.gamma_nd_gmix_const.assign(self.gamma_nd)
        # [fluid-mean WP2] a task starts with no constraint state (a fresh solver's)
        self._lc_reset()

    # ------------------------------------------------------------------
    # [fluid-mean WP2, 30 Oct 2026] the pressure level held over the fluid region
    # docs/notes/micro_fluid_mean_constraint_plan_2026-10-30.md rev 3, sections 1.2, 3.5, 5 (WP2);
    # decisions D1-D4, D7.  Off (``level_constraint="whole_cell"``) nothing below runs and the
    # solve is today's, bit for bit.  On (``fluid_mean_v1``), at every Newton solve:
    #   * full film (int (1 - chi) dx == 0 exactly, the ledger's own rule): today's whole-cell
    #     path (m, c_r = 0) -- bit for bit (V1);
    #   * no fluid (A_f = 0): no constraint; the solve is bypassed: w = 0, lambda = 0, mu = 0, so
    #     P_f = 0 and dP = -P_re (D7);
    #   * otherwise the constraint int chi (phi_c + w) dx = 0, i.e. m_f^T w + c_r = 0, with
    #     m_f,i = int chi phi_i dx, c_r = int chi phi_c dx = dpx M_x + dpy M_y,
    #     M_x = int chi (x - 1/2) dx, M_y = int chi (y - 1/2) dx; H = (m_f^T w + c_r) / A.
    # m_f, A_f, M_x and M_y are re-assembled at every Newton solve (m5's second option: chi moves
    # with every time level, history level and ripple pass in the runs this serves, so a cache keyed
    # on chi's inputs would save little and could go stale); c_r is formed from the current ramp
    # Constants at every solve (m4).  ``_mean_vec`` (the whole-cell m) is never overwritten (m6).
    # The pin (D4, m2, m3): I = {i : m_f,i = 0}; R_i -> R_i - kappa m_i w_i, J_ii -> J_ii - kappa m_i
    # (m_i the whole-cell weight); a node that joins I at a refresh starts at w_i = 0.
    # ------------------------------------------------------------------
    _lc_path: str = "whole_cell"
    _lc_mf = None
    _lc_cr: float = 0.0
    _lc_Af: float = float("nan")
    _lc_pin_idx = np.zeros(0, dtype=np.intp)
    _lc_pin_w = np.zeros(0)
    _lc_pin_prev = frozenset()
    _lc_no_support = None              # bool per Vper dof: nodal fluid with no fluid quadrature point
    _lc_mf_zero = None                 # bool per Vper dof: m_f,i = 0 (pinned; the sampler leaves them out)
    _lc_n_fluid_no_support: int = 0

    def _lc_reset(self) -> None:
        self._lc_path = "whole_cell"
        self._lc_mf = None
        self._lc_cr = 0.0
        self._lc_Af = float("nan")
        self._lc_pin_idx = np.zeros(0, dtype=np.intp)
        self._lc_pin_w = np.zeros(0)
        self._lc_pin_prev = frozenset()
        self._lc_no_support = None
        self._lc_mf_zero = None
        self._lc_n_fluid_no_support = 0

    def _lc_pin_on(self) -> bool:
        pin = getattr(self.settings, "patch_pin", None)
        return True if pin is None else bool(pin)

    def _level_constraint_prepare(self) -> str:
        lc = super()._level_constraint_prepare()          # validates the setting
        if getattr(self.settings, "level_constraint", "whole_cell") != "fluid_mean_v1":
            return lc                                     # today's path; no state touched
        dxm = dx(domain=self.mesh_m)
        chi = self.chi_fluid
        self._lc_mf = None
        self._lc_cr = 0.0
        self._lc_no_support = None
        self._lc_mf_zero = None
        self._lc_n_fluid_no_support = 0
        if float(assemble((Constant(1.0) - chi) * dxm)) == 0.0:
            # full film: the whole-cell path, bit for bit; no node is pinned
            self._lc_path = "whole_cell"
            self._lc_Af = float(self._cell_area)
            self._lc_pin_idx = np.zeros(0, dtype=np.intp)
            self._lc_pin_w = np.zeros(0)
            self._lc_pin_prev = frozenset()
            return "whole_cell"
        A_f = float(assemble(chi * dxm))
        self._lc_Af = A_f
        n = self.w.vector().local_size()
        if not A_f > 0.0:
            self._lc_path = "no_fluid"
            self._lc_mf_zero = np.ones(n, dtype=bool)     # every m_f,i = 0: no vertex carries fluid pressure
            self._lc_pin_idx = np.arange(n, dtype=np.intp)
            self._lc_pin_w = np.zeros(0)
            self._lc_pin_prev = frozenset(range(n))
            return "no_fluid"
        m_f = assemble(chi * self.v * dxm)
        M_x = float(assemble(chi * (self.x - 0.5) * dxm))
        M_y = float(assemble(chi * (self.y - 0.5) * dxm))
        self._lc_mf = m_f
        self._lc_cr = float(self.dpx_nd_const) * M_x + float(self.dpy_nd_const) * M_y
        self._lc_path = "fluid_mean"
        mf = m_f.get_local()
        none = mf == 0.0                                  # I = {i : m_f,i = 0} (m2)
        self._lc_mf_zero = none                           # the sampler's exclusion (code review, 30 Oct)
        nodal_fluid = np.asarray(self._ripple_chi_nodes(), dtype=float) > 0.5
        self._lc_no_support = nodal_fluid & none
        self._lc_n_fluid_no_support = int(np.count_nonzero(self._lc_no_support))
        if self._lc_pin_on():
            idx = np.flatnonzero(none).astype(np.intp)
            joined = np.array(sorted(set(idx.tolist()) - set(self._lc_pin_prev)), dtype=np.intp)
            if joined.size:                               # m3: a new pinned node starts at w = 0
                wv = self.w.vector().get_local()
                wv[joined] = 0.0
                self.w.vector().set_local(wv)
                self.w.vector().apply("insert")
            kappa = float(getattr(self.settings, "patch_pin_kappa", 1.0))
            self._lc_pin_idx = idx
            self._lc_pin_w = kappa * self._mean_vec.get_local()[idx]
            self._lc_pin_prev = frozenset(idx.tolist())
        else:
            self._lc_pin_idx = np.zeros(0, dtype=np.intp)
            self._lc_pin_w = np.zeros(0)
            self._lc_pin_prev = frozenset()
        return "fluid_mean"

    def _constraint_vector(self):
        if self._lc_path == "fluid_mean" and self._lc_mf is not None:
            return self._lc_mf
        return self._mean_vec

    def _constraint_residual(self, m, area: float) -> float:
        if self._lc_path == "fluid_mean" and self._lc_mf is not None:
            return (float(m.inner(self.w.vector())) + self._lc_cr) / area
        return float(m.inner(self.w.vector())) / area

    def _no_fluid_bypass(self) -> None:
        """D7: no fluid in the cell -- no constraint, no solve; the cell carries no fluid force."""
        self.w.vector().zero()
        self.w.vector().apply("insert")
        self.lambda_nd.assign(0.0)
        self.mu_nd = 0.0
        self._newton_call_idx = getattr(self, "_newton_call_idx", 0) + 1

    def _assemble_newton_rhs(self) -> float:
        t = super()._assemble_newton_rhs()                # b = -(R + mu m)
        idx = self._lc_pin_idx
        if self._lc_path == "fluid_mean" and idx.size:
            b = self._b_vec.get_local()
            b[idx] += self._lc_pin_w * self.w.vector().get_local()[idx]   # R_i -> R_i - k m_i w_i
            self._b_vec.set_local(b)
            self._b_vec.apply("insert")
        return t

    def _apply_jacobian_reg_shift(self) -> None:
        super()._apply_jacobian_reg_shift()
        idx = self._lc_pin_idx
        if self._lc_path == "fluid_mean" and idx.size:
            A = self._A_mat.mat()
            d = A.getDiagonal()
            arr = d.getArray()
            arr[idx] -= self._lc_pin_w                    # J_ii -> J_ii - k m_i
            d.setArray(arr)
            A.setDiagonal(d)

    def _squeeze_residual_terms(self, dh_dt):
        """Two-branch squeeze source (see base seam S4).

        Pattern-flux branch: full unmasked dh_dt with uniform mean
        compensation.  Legacy branch: mask-weighted source AND
        mask-weighted compensation (MIXED FIX, June 2026 — the base's
        uniform compensation left the dead zone a spurious O(1) source).
        """
        # Squeeze source (periodic fluctuation only; mean subtracted for
        # solvability).
        #
        # MIXED FIX (June 2026): the compensation constant is MASK-
        # WEIGHTED — the subtracted term is mask*S_mean' with
        # S_mean' = ∫ mask*dh_dt dx / ∫ mask dx (see
        # _update_squeeze_mean).  The previous uniform subtraction
        # (S_mean = ∫ mask*dh_dt dx, compensation not mask-weighted)
        # left the contact dead zone with a spurious O(1) source that
        # only the eps_solid-weighted cavitation penalty could absorb,
        # equilibrating at the fictitious pressure
        #     spos(-P_dead) ≈ -S_mean / (gamma_nd * eps_solid)
        # (≈ -1.4e9 Pa at the single-case defaults; verified to 0.7%
        # by tests/diagnose_contact_blowup.py).  The whole-cell
        # zero-mean constraint on w then lifted the ENTIRE fluid
        # region by ≈ f*|w_dead|/(1-f) — the observed uniform ~6e7 Pa
        # "blow-up" after contact onset — and suppressed all fluid
        # cavitation (lambda/dP collapsed).  With the mask-weighted
        # compensation, ∫(source) dx = 0 still holds exactly and every
        # term in the dead zone carries the same eps_solid weight.
        # Reduces to the previous form when h_min = 0 (mask ≡ 1,
        # ∫mask = area).
        self._squeeze_mean_nd = Constant(0.0)
        if getattr(self, "_pattern_flux_on", False):
            # Pattern-flux contact model: the patch carries its film,
            # so the squeeze source is the FULL (unmasked) dh_dt with a
            # plain uniform mean compensation.  Inside the patch
            # dh_dt + div(V_p*h) = HT-rate = S_mean for a rigidly
            # translating pattern, so the patch interior balances
            # identically — uniform compensation is safe here (unlike
            # the legacy masked-transport model, where the patch had
            # no physics left to absorb it).
            self._squeeze_source = dh_dt
            return self.transient_on * (
                (self._squeeze_source - self._squeeze_mean_nd)
                * self.v * dx
            )
        else:
            self._squeeze_source = self.mask * dh_dt
            return self.transient_on * (
                (self._squeeze_source - self.mask * self._squeeze_mean_nd)
                * self.v * dx
            )

    def _update_squeeze_mean(self) -> None:
        """Assign the mask-weighted squeeze mean for the current state.

        S_mean' = ∫ mask*dh_dt dx / ∫ mask dx, paired with the mask-weighted
        compensation term ``mask*S_mean'`` in :meth:`_build_forms`, so that
        ∫ (mask*dh_dt − mask*S_mean') dx = 0 exactly (global solvability)
        while the masked dead zone carries no spurious source.  ∫mask is
        reassembled each call because the mask follows h_nd(T).  With
        h_min = 0 this is identical to the previous behaviour
        (mask ≡ 1, ∫mask = cell area).
        """
        if not hasattr(self, "_squeeze_source"):
            return
        num = float(assemble(self._squeeze_source * dx(domain=self.mesh_m)))
        if getattr(self, "_pattern_flux_on", False):
            # Full-source model: uniform compensation = cell mean of dh_dt.
            den = float(self._cell_area)
        else:
            den = max(float(assemble(self.mask * dx(domain=self.mesh_m))),
                      1e-30)
        self._squeeze_mean_nd.assign(num / den)

    # ------------------------------------------------------------------
    # Mask-weighted closure (MIXED FIX)
    # ------------------------------------------------------------------
    def _closure_G(self, beta: float, gamma: float, area: float) -> float:
        # G = beta*lambda - gamma*mean(mask*spos(-P_nd)): contact regions
        # carry fictitious pressure (Reynolds is switched off there) and
        # must not drive the pressure level.  Reduces to the unmasked
        # closure when h_min = 0 (mask == 1).
        spos_mean = float(
            assemble(self.mask * self.negP_pos * dx(domain=self.mesh_m))
        ) / area
        return beta * float(self.lambda_nd) - gamma * spos_mean

    # _closure_g_vec: INHERITED — assembles the mask-weighted _g_form_raw
    # reassigned in _setup_forms above.

    def _closure_d_scalar(self, beta: float, gamma: float, area: float) -> float:
        # d = beta + gamma*mean(mask*spos'(-P)) >= beta > 0 (mask-weighted,
        # consistent with G and g).
        spos_deriv_mean = (
            float(
                assemble(
                    self.mask * self._spos_deriv_nd * dx(domain=self.mesh_m)
                )
            ) / area
        )
        return beta + gamma * spos_deriv_mean

    # ------------------------------------------------------------------
    # Spring load: fluid pressure only (MIXED FIX)
    # ------------------------------------------------------------------
    def _cell_avg_positive_pressure_dim(self) -> float:
        """Cell-averaged positive FLUID pressure (Pa).

        The integrand is restricted to the fluid region via ``chi_fluid``
        — contact-region pressures are fictitious and must not load the
        spring.  Denominator stays the TOTAL cell area.  Identical to the
        base when h_min = 0 (chi_fluid == 1).
        """
        Ppos = conditional(gt(self.P_dim, Constant(0.0)), self.P_dim, Constant(0.0))
        area = getattr(self, "_area_total", 1.0)
        return float(assemble(self.chi_fluid * Ppos * dx(domain=self.mesh_m)) / area)

    # ------------------------------------------------------------------
    # [Mixed P1.0] July-2026 spring coverage (ported from the root-tree
    # monolith): the spring-film coupling wraps the STEADY solve and the
    # transient rolling-steady SEED too — previously (and still, in the
    # base class) the spring acted only on per-step transient solves.
    # Steady coverage means run_steady() at k_spring > 0 converges the
    # hd <-> P fixed point instead of solving at hd = 0; the seed
    # coverage avoids the hd-from-zero jump at step 1 of a transient.
    # Kept OUT of the base class deliberately: the production full-film
    # worker instantiates the base with k_spring = 1e20, so a base-level
    # port would shift the flag-OFF baselines (locked --exact gate).
    # ------------------------------------------------------------------
    def _steady_solve(self) -> None:
        self._solve_with_spring()

    def _seed_solve(self) -> None:
        self._solve_with_spring()

    # ------------------------------------------------------------------
    # Runs: clear/record homogenisation history around the base loops
    # ------------------------------------------------------------------
    def run_steady(self) -> None:
        self.homog_history = []
        super().run_steady()
        if self.record_homog:
            self._record_homog_step(t_dim=0.0)

    def run(self, Tf: float, n_steps: int = 10, cfl_target: float = 1.0,
            dt_override: "float | None" = None,
            step_callback: "Callable | None" = None) -> None:
        self.homog_history = []
        cb = step_callback
        if self.record_homog:
            def cb(step, t_dim, dt_dim, _user_cb=step_callback):  # noqa: E306
                self._record_homog_step(t_dim=t_dim)
                if _user_cb is not None:
                    _user_cb(step, t_dim, dt_dim)
        return super().run(Tf, n_steps=n_steps, cfl_target=cfl_target,
                           dt_override=dt_override, step_callback=cb)

    def _on_ripple_pass_start(self) -> None:
        """[option H] Each pass of the ripple loop is a whole run: keep the final
        pass's per-step record only (the base calls this before every pass)."""
        self.homog_history = []

    def _record_homog_step(self, t_dim: float) -> None:
        """Append masked homogenised quantities for the CURRENT solver state
        to ``self.homog_history`` (one dict per call, keyed by name).

        Opt-in via ``self.record_homog``; called from run()/run_steady()
        after each accepted solve.  Cost is several projections per call
        (see spatial_homogenisation), booked under the diagnostics timer.
        """
        _t0 = time.perf_counter()
        (
            Qx, Qy, Pst, Fst, taust_x, taust_y,
            p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin,
        ) = self.spatial_homogenisation()
        self.homog_history.append({
            "t": float(t_dim),
            "Qx": Qx, "Qy": Qy,
            "Pst": Pst,
            "Pst_fluid": self.Pst_fluid,
            "Fst": Fst,
            "taust_x": taust_x, "taust_y": taust_y,
            "p_max": p_max, "p_min": p_min,
            "h_max": h_max, "h_min": h_min,
            "qymax": qymax, "qymin": qymin,
            "fmax": fmax, "fmin": fmin,
            "Qx_couette": self.Qx_couette, "Qy_couette": self.Qy_couette,
            "Qx_poiseuille": self.Qx_poiseuille,
            "Qy_poiseuille": self.Qy_poiseuille,
            "lambda_dim": self.dP,
        })
        if hasattr(self, "_t_diagnostics"):
            self._t_diagnostics += time.perf_counter() - _t0

    def spatial_homogenisation(self):
        """Return homogenised quantities (same 14-tuple as MicroMixedSolver).

        Since phi_c_nd and w_fluc are both zero-mean, the cell-averaged
        pressure is P_re + lambda, so Pst = p_scale*(P_re_nd + lambda_nd)
        = p0 + lambda_dim.  This is correct by construction — no gauge
        ambiguity, no phase-dependent artefact.

        Mixed-lubrication masking (h_min > 0): pressure and flux extrema
        (p_max/p_min, qymax/qymin) are taken over the fluid region only;
        shear averages exclude the contact region (fluid contribution per
        unit TOTAL cell area); fluxes Qx/Qy need no extra treatment (the
        mask is already inside K_nd/F_adv_nd).  Pst remains the whole-cell
        mean for tuple compatibility; the fluid-only mean is exposed as
        ``self.Pst_fluid``.  All of this reduces to the unmasked template
        behaviour when h_min = 0.

        Returns
        -------
        (Qx, Qy, Pst, Fst, taust_x, taust_y,
         p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin)
        """
        area_total = float(assemble(Constant(1.0) * dx(domain=self.mesh_m)))

        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.params.H0) * self.h_nd, self.Vfull)

        h_arr = Hfull.vector().get_local()

        # --- Fluid indicator (computed FIRST so the shear averages below
        # can be restricted to the fluid region; MIXED FIX) ---
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = float(assemble(chi_dg0 * dx(domain=self.mesh_m)))
        Fst = float(fluid_area / area_total)

        # Nodal indicator of the FLUX extrema qymax / qymin below only (the
        # worker discards them; outside the extrema M1 scope, unchanged).
        chi_nodal = interpolate(chi_dg0, self.Vfull)
        chi_arr = chi_nodal.vector().get_local()
        fluid_nodes = chi_arr > 0.5

        # [extrema M1, 23 Oct 2026] Pressure extrema over the FLUID region only,
        # from the shared sampler (micro_diagnostics.py; definition in
        # ehl/pressure.py): signed P_dim at the vertices the sharp nodal level-set
        # test classes as fluid (phic_nd_func for fft_elastic_v1, else the projected
        # h_nd - h_min above) -- no projected or interpolated chi, no 0.5 threshold.
        # NaN (phase status no_fluid) if the whole cell is in contact.  Every vertex
        # is fluid when h_min = 0.
        self.pressure_extrema = sample_fluid_pressure_extrema(self, film_dofs_m=h_arr)
        p_max = self.pressure_extrema.p_max_Pa
        p_min = self.pressure_extrema.p_min_Pa

        # Film thickness extrema stay UNMASKED on purpose: h is geometric
        # (defined everywhere), and h_min is precisely the quantity that
        # signals contact downstream.
        h_max = float(np.nanmax(h_arr))
        h_min = float(np.nanmin(h_arr))

        # Cell-averaged pressure (Pa) over the WHOLE cell. By construction:
        # ∫ P_dim dx = p_scale * (P_re_nd + lambda_nd) * area  (phi_c and w_fluc integrate to 0)
        # Kept whole-cell for backward compatibility of the 14-tuple.
        Pst = float(assemble(Pfull * dx(domain=self.mesh_m)) / area_total)

        self._record_pressure_ledger(Pst, fluid_area, area_total)

        fmax = 1.0 if fluid_area > 0.0 else 0.0
        fmin = 0.0

        # --- Fluxes ---
        gradP_nd = self.grad_cell(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd
        q_couette_nd = self.F_adv_nd
        q_poiseuille_nd = -self.K_nd * gradP_nd

        dim_scale = Constant(self.Uref * self.params.H0 / 2)
        q_dim = dim_scale * q_nd
        q_couette_dim = dim_scale * q_couette_nd
        q_poiseuille_dim = dim_scale * q_poiseuille_nd

        Qx = float(assemble(q_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy = float(assemble(q_dim[1] * dx(domain=self.mesh_m)) / area_total)

        Qx_couette = float(assemble(q_couette_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy_couette = float(assemble(q_couette_dim[1] * dx(domain=self.mesh_m)) / area_total)
        Qx_poiseuille = float(assemble(q_poiseuille_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy_poiseuille = float(assemble(q_poiseuille_dim[1] * dx(domain=self.mesh_m)) / area_total)

        # Flux extrema over the fluid region only (MIXED FIX) — in contact
        # the masked flux is O(eps_solid), and counting those near-zeros as
        # extrema is misleading.  NaN if the whole cell is in contact.
        qy_full = project(q_dim[1], self.Vfull)
        qy_arr = qy_full.vector().get_local()
        if np.any(fluid_nodes):
            qymax = float(np.nanmax(qy_arr[fluid_nodes]))
            qymin = float(np.nanmin(qy_arr[fluid_nodes]))
        else:
            qymax = float("nan")
            qymin = float("nan")

        # --- Shear stress ---
        Ux_surf = float(self.Ux_nd_const) * self.Uref
        Uy_surf = float(self.Uy_nd_const) * self.Uref

        gradP_dim = gradP_nd * Constant(self.p_scale * self.invLx)
        eta_dim_expr = Constant(float(self.params.eta0))      # constant lubricant viscosity [Pa s]
        H_dim_expr = Constant(self.params.H0) * self.h_nd
        H_floor_expr = H_dim_expr + Constant(1e-30)

        tau_x_field = (
            (eta_dim_expr * Constant(Ux_surf) / H_floor_expr)
            + (H_dim_expr / 2) * gradP_dim[0]
        )
        tau_y_field = (
            (eta_dim_expr * Constant(Uy_surf) / H_floor_expr)
            + (H_dim_expr / 2) * gradP_dim[1]
        )

        # MIXED FIX: integrands restricted to the fluid region.  The
        # Couette term eta*U/H diverges as H -> h_min inside contact, and
        # the pressure there is fictitious; excluding contact prevents
        # both from polluting the average.  Denominator stays the TOTAL
        # cell area: this is the fluid contribution to the cell-averaged
        # traction (a contact friction term would be added separately).
        # Identical to the template when h_min = 0 (chi_fluid ≡ 1).
        taust_x = float(
            assemble(chi_dg0 * tau_x_field * dx(domain=self.mesh_m)) / area_total
        )
        taust_y = float(
            assemble(chi_dg0 * tau_y_field * dx(domain=self.mesh_m)) / area_total
        )

        # [extrema M2b, 23 Oct 2026] The fluid traction t_f = chi_f tau_f at the points of the
        # quadrature rule of the integrals above, fluid by the sharp level-set test there
        # (micro_diagnostics.py; definition in ehl/fluid_shear.py): the sampled mean, and with the
        # task's frame (self.shear_frame, the macro's R_inv) the global-x envelope.  taust above is
        # unchanged (micro_raw/mean_existing_integration); the samples weighted by chi_dg0
        # reproduce it, which is the sampler's check.
        self.fluid_traction = sample_fluid_traction(
            self, film_dofs_m=h_arr, existing_mask=chi_dg0, frame=getattr(self, "shear_frame", None))

        # Store decomposition for diagnostics
        self.Qx_couette = Qx_couette
        self.Qy_couette = Qy_couette
        self.Qx_poiseuille = Qx_poiseuille
        self.Qy_poiseuille = Qy_poiseuille

        return (
            Qx, Qy, Pst, Fst, taust_x, taust_y,
            p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin,
        )


# ---------------------------------------------------------------------------
# Back-compat alias (module previously exported the base class name)
# ---------------------------------------------------------------------------
MicroPhaseNeutralSolver = MicroPhaseNeutralMixedSolver
