"""Mixed-lubrication micro solver: ``MicroPhaseNeutralSolver`` with a contact mask.

Changes to the base problem:

* ``pattern_flux_contact`` (default True).  Inside the contact patch the Couette transport
  blends to the pattern velocity, so the patch carries its film.  The squeeze source is the
  full unmasked dh/dt with a uniform mean subtracted.  The cavitation penalty acts at full
  strength inside the patch (``w_cav = chi*mask + (1 - chi)``), which holds the fictitious
  patch pressure at the cavitation floor for any ``eps_solid``.  The model requires a single
  rigidly translating roughness pattern and is set False for roughness on both surfaces.
* With it off, the transport is masked and the squeeze source and its mean are mask weighted.
* The closure (G, g and d) is mask weighted, so the fictitious pressure of the contact region
  does not drive the pressure level.
* The homogenised outputs are restricted to the fluid region (``Pst_fluid`` included).
* ``record_homog`` records the homogenised outputs of each step in ``homog_history``.
* The steady solve and the transient seed go through ``_solve_level``, so the pair branch of
  the ripple covers them.
* ``level_constraint="fluid_mean_v1"`` holds the pressure level over the fluid region.

With ``h_min = 0`` the mask is 1 everywhere and the base problem is recovered.  For a static
texture (pattern velocity 0) the pattern flux reduces to the masked transport.
"""

from __future__ import annotations

import warnings

import numpy as np
from fenics import *  # noqa: F401,F403

from microcell.solver_base import (  # noqa: F401
    MicroPhaseNeutralSolver as _BaseSolver,
    PATCH_PIN,
    PATCH_PIN_KAPPA,
)
from microcell.samplers import sample_fluid_pressure_extrema, sample_fluid_traction


class MicroPhaseNeutralMixedSolver(_BaseSolver):
    """Micro solver with the mixed-lubrication contact mask.

    ``record_homog`` is a class attribute, as the base ``__init__`` can run a solve
    (``auto_solve=True``) before the body of a subclass ``__init__``.  Set on the instance
    after construction, it records the homogenised outputs of each step in ``homog_history``.
    """

    record_homog: bool = False
    homog_history: list = []          # replaced on the instance by run() and run_steady()

    def _contact_onset_film(self) -> float:
        """Contact onset film [m], from ``MicroSettings.h_min``."""
        return float(self.settings.h_min)

    # ---- Set-up ----
    def _setup_pressure_expressions(self) -> None:
        """Base expressions, then the pattern-flux transport and the cavitation weight."""
        super()._setup_pressure_expressions()
        zeta = self.zeta
        # Pattern-flux contact model (``MicroSettings.pattern_flux_contact``).
        #
        # Transport.  The mask alone sets the flux through contact to about zero in the frame
        # of the cell, which is correct only for a static texture.  A translating patch
        # carries its film, so
        #     F_adv = mask*U*h + (1 - mask)*V_p*h.
        # In the patch frame the through-flux is zero and the edge jump is the relative flux
        # (U - V_p)*h_min.  V_p is the instantaneous pattern velocity: the displacement is
        # s(t) = alpha*(2*Ux*t + UxT*t^2), so v_p(t) = alpha*2*(Ux + UxT*t), and the coefficient
        # of the doubled non-dimensional transport is 4*alpha*(Ux + UxT*t)/Uref.  It is built
        # on T_dim_const and follows the time level.  _Ux_dim_const is not used here, as it
        # holds the displacement-averaged velocity.
        #
        # Penalty.  Full strength inside the patch, w_cav = chi*mask + (1 - chi), which holds
        # the fictitious patch pressure at the cavitation floor for any eps_solid.  Otherwise
        # the patch residual gives P_patch ~ -resid/(gamma*eps), and the zero-mean constraint
        # on w turns it into a uniform lift of the fluid pressure.
        self._pattern_flux_on = bool(self.settings.pattern_flux_contact)
        if self._pattern_flux_on:
            _alpha_pf = float(self.velocity_ratio)
            if 0.0 < _alpha_pf < 1.0:
                # A ratio in (0, 1) means roughness on both surfaces, which has no single
                # pattern velocity.  The blend is valid for one pattern translating at alpha*2*U.
                warnings.warn(
                    "pattern_flux_contact=True with "
                    f"roughness_velocity_ratio={_alpha_pf}: the pattern-flux "
                    "contact model assumes a single rigidly-translating "
                    "roughness pattern. For two-surface (standing-wave) "
                    "films set pattern_flux_contact=False.",
                    stacklevel=2,
                )
            # Stored Constants (Uref scaling and velocities), reassigned by _reset_task_subclass.
            self._cf_pf_const = Constant(4.0 * _alpha_pf / self.Uref)
            self._pf_Ux_const = Constant(float(zeta.Ux))
            self._pf_UxT_const = Constant(float(zeta.Uxdot))
            self._pf_Uy_const = Constant(float(zeta.Uy))
            self._pf_UyT_const = Constant(float(zeta.Uydot))
            _cf_pf = self._cf_pf_const
            self._V_p_nd = as_vector((
                _cf_pf * (self._pf_Ux_const
                          + self._pf_UxT_const * self.T_dim_const),
                _cf_pf * (self._pf_Uy_const
                          + self._pf_UyT_const * self.T_dim_const),
            ))
            # The pattern term is on in a transient solve only.  A steady solve has a fixed
            # pattern, for which the masked transport is the contact model, and the source
            # div(V_p*h) in the patch would have no dh/dt to balance it.
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
        """Assembles the mean vector ``m_i = int phi_i dx`` and the cell area.

        The vector is constant for a mesh.  It enters the zero-mean constraint
        ``H = m^T w / area``, the ``mu * m`` term of the residual and the Schur step
        (``z3 = J^-1 m``).
        """
        v_test = TestFunction(self.Vper)
        self._mean_vec = assemble(v_test * dx(domain=self.mesh_m))
        self._cell_area = assemble(Constant(1.0) * dx(domain=self.mesh_m))

    def _project_to_Vper(self, expr) -> "Function":
        """L2 projection of a UFL expression onto ``Vper``.  Returns a ``Function``.

        Uses the mass matrix factorised once in ``_setup_spaces_and_unknowns``: one right-hand
        side assembly and one back-substitution per call.  The BDF history update and the
        transient seed call it.  ``interpolate()`` cannot be used, as it rejects UFL
        expressions such as ``h_nd`` and ``mask``.
        """
        out = Function(self.Vper)
        b = assemble(expr * self._proj_Vper_v_test * dx(domain=self.mesh_m))
        self._M_Vper_lu.solve(out.vector(), b)
        return out

    def _setup_forms(self) -> None:
        """Base forms, then the mask-weighted ``g_j = dG/dw_j``."""
        super()._setup_forms()
        # g is mask weighted, as the closure G and c are: the fictitious pressure of the
        # contact region stays out of the Jacobian of (R, G).  Equal to the base form when
        # h_min = 0.  gamma_nd_gmix_const is the stored Constant of this form.
        self.gamma_nd_gmix_const = Constant(self.gamma_nd)
        self._g_form_raw = (
            self.gamma_nd_gmix_const * self.mask * self._spos_deriv_nd * self.v * dx
        )

    def _reset_task_subclass(self) -> None:
        """Reassigns the task-dependent Constants of the mixed class: the pattern-flux
        transport and the mask-weighted g form.

        Called by the base ``reset_task`` after ``_nondim_scalars()``, so ``Uref`` and
        ``gamma_nd`` hold the values of the new task.
        """
        zeta = self.zeta
        if self._pattern_flux_on:
            _alpha_pf = float(self.velocity_ratio)
            self._cf_pf_const.assign(4.0 * _alpha_pf / self.Uref)
            self._pf_Ux_const.assign(float(zeta.Ux))
            self._pf_UxT_const.assign(float(zeta.Uxdot))
            self._pf_Uy_const.assign(float(zeta.Uy))
            self._pf_UyT_const.assign(float(zeta.Uydot))
        self.gamma_nd_gmix_const.assign(self.gamma_nd)
        # a task starts with no level-constraint state
        self._lc_reset()

    # ---- Pressure level held over the fluid region (``level_constraint="fluid_mean_v1"``) ----
    # With "whole_cell" nothing below runs.  With "fluid_mean_v1", at every Newton solve:
    #   * full film (int (1 - chi) dx == 0): the whole-cell path (m, c_r = 0);
    #   * no fluid (A_f = 0): no constraint and no solve: w = 0, lambda = 0, mu = 0, so
    #     P_f = 0 and dP = -P_re;
    #   * otherwise the constraint int chi (phi_c + w) dx = 0, that is m_f^T w + c_r = 0, with
    #     m_f,i = int chi phi_i dx, c_r = int chi phi_c dx = dpx M_x + dpy M_y,
    #     M_x = int chi (x - 1/2) dx, M_y = int chi (y - 1/2) dx, and H = (m_f^T w + c_r) / A.
    # m_f, A_f, M_x and M_y are assembled at every Newton solve, as chi moves with the time
    # level and the ripple pass.  c_r is formed from the current ramp Constants.  ``_mean_vec``
    # (the whole-cell m) is not overwritten.
    # Pin: I = {i : m_f,i = 0}; R_i -> R_i - kappa m_i w_i and J_ii -> J_ii - kappa m_i, with m_i
    # the whole-cell weight.  A node that joins I starts at w_i = 0.
    _lc_path: str = "whole_cell"
    _lc_mf = None
    _lc_cr: float = 0.0
    _lc_Af: float = float("nan")
    _lc_pin_idx = np.zeros(0, dtype=np.intp)
    _lc_pin_w = np.zeros(0)
    _lc_pin_prev = frozenset()
    _lc_no_support = None              # bool per Vper dof: fluid at the node, no fluid quadrature point
    _lc_mf_zero = None                 # bool per Vper dof: m_f,i = 0 (pinned; left out by the sampler)
    _lc_n_fluid_no_support: int = 0

    def _lc_reset(self) -> None:
        """Clears the level-constraint state."""
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
        """True when the nodes with ``m_f,i = 0`` are pinned (``PATCH_PIN``; None is on)."""
        return True if PATCH_PIN is None else bool(PATCH_PIN)

    def _level_constraint_prepare(self) -> str:
        """Prepares the level constraint of a Newton solve.

        Returns the path: ``"whole_cell"``, ``"fluid_mean"`` or ``"no_fluid"``.  On the
        fluid-mean path it assembles ``m_f`` and ``c_r`` and sets the pinned nodes.
        """
        lc = super()._level_constraint_prepare()          # validates the setting
        if self.settings.level_constraint != "fluid_mean_v1":
            return lc                                     # whole-cell path; no state touched
        dxm = dx(domain=self.mesh_m)
        chi = self.chi_fluid
        self._lc_mf = None
        self._lc_cr = 0.0
        self._lc_no_support = None
        self._lc_mf_zero = None
        self._lc_n_fluid_no_support = 0
        if float(assemble((Constant(1.0) - chi) * dxm)) == 0.0:
            # full film: the whole-cell path; no node is pinned
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
        none = mf == 0.0                                  # I = {i : m_f,i = 0}
        self._lc_mf_zero = none                           # nodes the sampler leaves out
        nodal_fluid = np.asarray(self._ripple_chi_nodes(), dtype=float) > 0.5
        self._lc_no_support = nodal_fluid & none
        self._lc_n_fluid_no_support = int(np.count_nonzero(self._lc_no_support))
        if self._lc_pin_on():
            idx = np.flatnonzero(none).astype(np.intp)
            joined = np.array(sorted(set(idx.tolist()) - set(self._lc_pin_prev)), dtype=np.intp)
            if joined.size:                               # a new pinned node starts at w = 0
                wv = self.w.vector().get_local()
                wv[joined] = 0.0
                self.w.vector().set_local(wv)
                self.w.vector().apply("insert")
            kappa = float(PATCH_PIN_KAPPA)
            self._lc_pin_idx = idx
            self._lc_pin_w = kappa * self._mean_vec.get_local()[idx]
            self._lc_pin_prev = frozenset(idx.tolist())
        else:
            self._lc_pin_idx = np.zeros(0, dtype=np.intp)
            self._lc_pin_w = np.zeros(0)
            self._lc_pin_prev = frozenset()
        return "fluid_mean"

    def _constraint_vector(self):
        """Constraint vector of the Schur step: ``m_f`` on the fluid-mean path, otherwise the
        whole-cell ``m``."""
        if self._lc_path == "fluid_mean" and self._lc_mf is not None:
            return self._lc_mf
        return self._mean_vec

    def _constraint_residual(self, m, area: float) -> float:
        """Constraint residual ``H``: ``(m^T w + c_r) / area`` on the fluid-mean path, otherwise
        ``m^T w / area``."""
        if self._lc_path == "fluid_mean" and self._lc_mf is not None:
            return (float(m.inner(self.w.vector())) + self._lc_cr) / area
        return float(m.inner(self.w.vector())) / area

    def _no_fluid_bypass(self) -> None:
        """No fluid in the cell: no constraint and no solve.  Sets w = 0, lambda = 0, mu = 0."""
        self.w.vector().zero()
        self.w.vector().apply("insert")
        self.lambda_nd.assign(0.0)
        self.mu_nd = 0.0
        self._newton_call_idx = getattr(self, "_newton_call_idx", 0) + 1

    def _assemble_newton_rhs(self) -> None:
        """Base right-hand side, plus the pin term at the pinned nodes."""
        super()._assemble_newton_rhs()                    # b = -(R + mu m)
        idx = self._lc_pin_idx
        if self._lc_path == "fluid_mean" and idx.size:
            b = self._b_vec.get_local()
            b[idx] += self._lc_pin_w * self.w.vector().get_local()[idx]   # R_i -> R_i - kappa m_i w_i
            self._b_vec.set_local(b)
            self._b_vec.apply("insert")

    def _apply_jacobian_reg_shift(self) -> None:
        """Base shift, plus the pin term on the diagonal at the pinned nodes."""
        super()._apply_jacobian_reg_shift()
        idx = self._lc_pin_idx
        if self._lc_path == "fluid_mean" and idx.size:
            A = self._A_mat.mat()
            d = A.getDiagonal()
            arr = d.getArray()
            arr[idx] -= self._lc_pin_w                    # J_ii -> J_ii - kappa m_i
            d.setArray(arr)
            A.setDiagonal(d)

    def _squeeze_residual_terms(self, dh_dt):
        """Squeeze term of the residual, with its mean subtracted for solvability.

        dh_dt: non-dimensional film rate (UFL).
        Pattern flux on: the full unmasked ``dh_dt`` with a uniform mean.
        Pattern flux off: the source ``mask*dh_dt`` with the compensation ``mask*S_mean``.
        Returns the UFL form, multiplied by ``transient_on``.
        """
        # Pattern flux off.  The subtracted term is mask*S_mean, with
        # S_mean = int mask*dh_dt dx / int mask dx (``_update_squeeze_mean``).  The source
        # integrates to zero and every term in the contact region carries the eps_solid weight.
        # A uniform compensation would leave the contact region a source that only the
        # eps_solid-weighted penalty could absorb, at the fictitious pressure
        #     spos(-P) ~ -S_mean / (gamma_nd * eps_solid).
        self._squeeze_mean_nd = Constant(0.0)
        if getattr(self, "_pattern_flux_on", False):
            # The patch carries its film, so the source is the full dh_dt with a uniform mean.
            # Inside the patch dh_dt + div(V_p*h) equals the film rate, which is S_mean, for a
            # rigidly translating pattern, so the patch interior balances.
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
        """Assigns the squeeze mean of the current state.

        Pattern flux on: the cell mean of ``dh_dt``.
        Pattern flux off: ``S_mean = int mask*dh_dt dx / int mask dx``, so that
        ``int (mask*dh_dt - mask*S_mean) dx = 0``.  ``int mask dx`` is assembled at each call,
        as the mask follows the film.  With ``h_min = 0`` the mask is 1 and it is the cell area.
        """
        if not hasattr(self, "_squeeze_source"):
            return
        num = float(assemble(self._squeeze_source * dx(domain=self.mesh_m)))
        if getattr(self, "_pattern_flux_on", False):
            # uniform compensation: the cell mean of dh_dt
            den = float(self._cell_area)
        else:
            den = max(float(assemble(self.mask * dx(domain=self.mesh_m))),
                      1e-30)
        self._squeeze_mean_nd.assign(num / den)

    # ---- Mask-weighted closure ----
    def _closure_G(self, beta: float, gamma: float, area: float) -> float:
        """Closure ``G = beta*lambda - gamma*mean(mask*spos(-P_nd))``.

        The mask keeps the fictitious pressure of the contact region out of the pressure
        level.  With ``h_min = 0`` the mask is 1 and it is the base closure.
        """
        spos_mean = float(
            assemble(self.mask * self.negP_pos * dx(domain=self.mesh_m))
        ) / area
        return beta * float(self.lambda_nd) - gamma * spos_mean

    # _closure_g_vec is inherited: it assembles the mask-weighted _g_form_raw of _setup_forms.

    def _closure_d_scalar(self, beta: float, gamma: float, area: float) -> float:
        """Closure derivative ``d = beta + gamma*mean(mask*spos'(-P_nd)) >= beta > 0``."""
        spos_deriv_mean = (
            float(
                assemble(
                    self.mask * self._spos_deriv_nd * dx(domain=self.mesh_m)
                )
            ) / area
        )
        return beta + gamma * spos_deriv_mean

    # ---- Steady solve and transient seed: through _solve_level(), which enters the pair
    # branch of the ripple ----
    def _steady_solve(self) -> None:
        """Steady solve of the current level."""
        self._solve_level()

    def _seed_solve(self) -> None:
        """Steady solve that seeds the transient march."""
        self._solve_level()

    # ---- Runs: homog_history is cleared and recorded around the base loops ----
    def run_steady(self) -> None:
        """Steady solve.  Records one entry at t = 0 when ``record_homog`` is set."""
        self.homog_history = []
        super().run_steady()
        if self.record_homog:
            self._record_homog_step(t_dim=0.0)

    def run(self, Tf: float, n_steps: int = 10, cfl_target: float = 1.0,
            dt_override: "float | None" = None,
            step_callback: "Callable | None" = None) -> None:
        """Transient march of the base class to time ``Tf`` [s].

        The arguments are those of the base ``run``.  When ``record_homog`` is set, one
        entry is recorded at each level (the seed and every step), before
        ``step_callback(step, t_dim, dt_dim)``.
        """
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
        """Clears ``homog_history`` before each pass of the ripple loop, so the record is that
        of the last pass."""
        self.homog_history = []

    def _record_homog_step(self, t_dim: float) -> None:
        """Appends the homogenised outputs of the current state to ``homog_history``.

        t_dim: time of the state [s].  One dict per call, keyed by name; the values come
        from ``spatial_homogenisation``.
        """
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

    def spatial_homogenisation(self):
        """Homogenised outputs of the current state.

        With contact (``h_min > 0``) the pressure and flux extrema are over the fluid region
        and the shear means exclude contact, per total cell area.  ``Pst`` is the whole-cell
        mean; with the whole-cell constraint it is ``p0 + lambda_dim``, as ``phi_c`` and ``w``
        have zero mean.  The fluid-region mean is ``self.Pst_fluid``.

        Returns ``(Qx, Qy, Pst, Fst, taust_x, taust_y, p_max, p_min, h_max, h_min, qymax,
        qymin, fmax, fmin)``: mean fluxes [m^2/s], mean pressure [Pa], fluid area fraction,
        mean fluid shear [Pa], pressure extrema [Pa], film extrema [m], extrema of the y flux
        [m^2/s] and of the fluid indicator.
        """
        area_total = float(assemble(Constant(1.0) * dx(domain=self.mesh_m)))

        Pfull = project(self.P_dim, self.Vfull)
        Hfull = project(Constant(self.zeta.H) * self.h_nd, self.Vfull)

        h_arr = Hfull.vector().get_local()

        # Fluid indicator, used by the shear means below.
        chi_dg0 = project(self.chi_fluid, self.Vdg0)
        fluid_area = float(assemble(chi_dg0 * dx(domain=self.mesh_m)))
        Fst = float(fluid_area / area_total)

        # Nodal indicator, used by the flux extrema qymax and qymin only.
        chi_nodal = interpolate(chi_dg0, self.Vfull)
        chi_arr = chi_nodal.vector().get_local()
        fluid_nodes = chi_arr > 0.5

        # Pressure extrema over the fluid region (samplers.py): signed P_dim at the vertices
        # that the nodal level-set test classes as fluid (phic_nd_func for fft_elastic_v1,
        # otherwise the projected h_nd - h_min).  NaN (status no_fluid) when the whole cell is
        # in contact.  Every vertex is fluid when h_min = 0.
        self.pressure_extrema = sample_fluid_pressure_extrema(self, film_dofs_m=h_arr)
        p_max = self.pressure_extrema.p_max_Pa
        p_min = self.pressure_extrema.p_min_Pa

        # Film extrema are over the whole cell, as the film is defined everywhere.
        h_max = float(np.nanmax(h_arr))
        h_min = float(np.nanmin(h_arr))

        # Mean pressure [Pa] over the whole cell.
        Pst = float(assemble(Pfull * dx(domain=self.mesh_m)) / area_total)

        self._record_pressure_ledger(Pst, fluid_area, area_total)

        fmax = 1.0 if fluid_area > 0.0 else 0.0
        fmin = 0.0

        # Fluxes: total, Couette and Poiseuille parts.
        gradP_nd = self.grad_cell(self.P_nd)
        q_nd = self.F_adv_nd - self.K_nd * gradP_nd
        q_couette_nd = self.F_adv_nd
        q_poiseuille_nd = -self.K_nd * gradP_nd

        dim_scale = Constant(self.Uref * self.zeta.H / 2)
        q_dim = dim_scale * q_nd
        q_couette_dim = dim_scale * q_couette_nd
        q_poiseuille_dim = dim_scale * q_poiseuille_nd

        Qx = float(assemble(q_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy = float(assemble(q_dim[1] * dx(domain=self.mesh_m)) / area_total)

        Qx_couette = float(assemble(q_couette_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy_couette = float(assemble(q_couette_dim[1] * dx(domain=self.mesh_m)) / area_total)
        Qx_poiseuille = float(assemble(q_poiseuille_dim[0] * dx(domain=self.mesh_m)) / area_total)
        Qy_poiseuille = float(assemble(q_poiseuille_dim[1] * dx(domain=self.mesh_m)) / area_total)

        # Flux extrema over the fluid region, as the masked flux in contact is O(eps_solid).
        # NaN when the whole cell is in contact.
        qy_full = project(q_dim[1], self.Vfull)
        qy_arr = qy_full.vector().get_local()
        if np.any(fluid_nodes):
            qymax = float(np.nanmax(qy_arr[fluid_nodes]))
            qymin = float(np.nanmin(qy_arr[fluid_nodes]))
        else:
            qymax = float("nan")
            qymin = float("nan")

        # Shear stress [Pa]: Couette term eta*U/H and Poiseuille term (H/2)*grad P.
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

        # The integrands are restricted to the fluid region, as the Couette term grows as
        # H -> h_min in contact and the pressure there is fictitious.  The denominator is the
        # total cell area: the fluid part of the cell-averaged traction.
        taust_x = float(
            assemble(chi_dg0 * tau_x_field * dx(domain=self.mesh_m)) / area_total
        )
        taust_y = float(
            assemble(chi_dg0 * tau_y_field * dx(domain=self.mesh_m)) / area_total
        )

        # Fluid traction t_f = chi_f tau_f sampled at the quadrature points of the integrals
        # above, with the fluid points set by the level-set test there (samplers.py).  The
        # samples weighted by chi_dg0 reproduce taust, which is the check of the sampler.
        self.fluid_traction = sample_fluid_traction(self, film_dofs_m=h_arr, existing_mask=chi_dg0)

        # Couette and Poiseuille parts of the mean flux [m^2/s]
        self.Qx_couette = Qx_couette
        self.Qy_couette = Qy_couette
        self.Qx_poiseuille = Qx_poiseuille
        self.Qy_poiseuille = Qy_poiseuille

        return (
            Qx, Qy, Pst, Fst, taust_x, taust_y,
            p_max, p_min, h_max, h_min, qymax, qymin, fmax, fmin,
        )
