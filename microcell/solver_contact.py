"""Phase-neutral mixed solver with the resolved elastic contact law ``fft_elastic_v1``.

``contact_frac`` and ``A_patch_frac`` are the integrated obstruction indicator, not
mechanical area; ``A_mech_frac`` is the mechanical contact area of the nodal solution.
Homogenised force is per TOTAL cell area (Pa), additive to the macro pressure channel.

Serial-only: one micro task per worker.

The parent's ``legacy_patch_v1`` and ``hertz_summit_v1`` laws are not in this repo
(MicroCell WP2, 11 Nov 2026).

``fft_elastic_v1`` (16 Oct 2026, plan docs/notes/fft_elastic_contact_plan_
2026-10-16.md) solves one periodic elastic contact problem per cell on the
nodal grid (``microcell.contact``), refreshed at every time level through
the base class's ``_refresh_geometry`` seam.  The fluid sees the deformed film
``h_rigid + u_c`` (``uc_nd_func``, zero-mean ripple; the macro keeps the cell
mean, D3) and its obstruction mask is the contact set of the same solution,
through the complementarity level set ``phi_c = g - p/k_N`` (``phic_nd_func``,
D2).  Load, mechanical area, peak, tangent and mask all come from one nodal
solution.
"""

from __future__ import annotations

import time

import numpy as np
from fenics import *  # noqa: F401,F403 — UFL/assembly symbols used below
from microcell.contact import FFT
from microcell import contact as _ec

from microcell.solver_mixed import MicroPhaseNeutralMixedSolver as _MixedSolver

# The contact-law parameters are ``MicroSettings.contact`` (a ``microcell.ContactSettings``):
# the combined modulus is ``MicroSettings.E_star`` (explicit, never reduced twice; one body pair, so the
# ripple uses the same one); ``sigma_yield`` the elastic-validity
# FLAG threshold [Pa] (never clamps); ``k_asp_mode`` [WP-1 2026-09-03] the tangent of the closure,
# k_asp = -dP_asp/dH [Pa/m], for a Newton-like macro coupling ("off" emits 0.0, "closure" the
# tangent of the nodal contact solution, ``contact.contact_tangent``); ``tol`` and ``maxit``
# the Polonsky-Keer stopping rule sum|dp|/sum p and iteration budget (plan 3.2; the tangent's CG
# uses the same tolerance); ``mu`` [extrema M3b, 23 Oct 2026] the dry asperity friction coefficient
# of the diagnostic asperity / total traction only (``samplers.sample_contact_traction``;
# no solver channel reads it).  ``self.pair`` [phase-table WP3, 4 Nov 2026; MicroCell WP2] is True for
# a stationary + translating surface pair: it lifts the two-member refusal of the ripple (R-e, the
# converged per-step branch) and turns on the warm-start contact rescue.


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
    # [extrema M3b] this phase's asperity / total traction statistics (shear.py)
    contact_traction = None

    # ------------------------------------------------------------------
    # [extrema M3b, 23 Oct 2026] asperity and total traction on the fluid sampler's points
    # ------------------------------------------------------------------
    def spatial_homogenisation(self):
        """The mixed class's homogenisation (every output unchanged), then the asperity and total
        traction of this phase on the sample set its fluid traction sampler kept
        (``samplers.sample_contact_traction``: the stored ``fft_elastic_v1`` solution of this
        state, its key checked, never refreshed here; ``unsupported_model`` for the other identities,
        whose means :meth:`compute_asperity_load` completes)."""
        out = super().spatial_homogenisation()
        from microcell.samplers import sample_contact_traction
        self.contact_traction = sample_contact_traction(self)
        return out

    # ------------------------------------------------------------------
    # fft_elastic_v1: film and mask graph, grid map, refresh (plan WP3)
    # ------------------------------------------------------------------
    def _setup_film_and_mask(self) -> None:
        super()._setup_film_and_mask()
        s = self.settings
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
        E = s.E_star
        if E is None or not np.isfinite(E) or E <= 0:
            raise ValueError("fft_elastic_v1 requires an explicit positive two-body E_star")
        if s.contact.k_asp_mode not in ("off", "closure"):
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
        op = _ec.PeriodicHalfSpace(n, n, float(self.Lx), float(self.Ly), float(self.settings.E_star))
        g = (n, i, j, Xg, Yg, op)
        self._contact_grid_cache = g
        return g

    def _contact_state_key(self):
        descriptor = self.ht_cb.contact_cosine
        vals = (self.H0_const, self.HT_const, self.T_dim_const, self._Ux_dim_const, self._Uy_dim_const,
                descriptor.phi_x, descriptor.phi_y, descriptor.dphi_x, descriptor.dphi_y,
                self.h_min, self.settings.xmax, self.settings.ymax)
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
        ``pair_branch`` only: a WARM-started solve that collapses ("relative change
        inf / nan": the first CG step clears every warm-start node when the pattern has moved) or stalls
        ("contact_complementarity": converged on the stale active set) is retried once from the solver's own cold
        start, which converges in a few iterations on the same problem; a solve that never fails is unchanged bit
        for bit; a cold start that fails, and any other failure, is raised as before.  Counted per task
        (``_contact_restarts``, by kind) and printed by the worker.  With the field off the call is exactly the one
        it always was (no try, no rescue: the off path is bitwise)."""
        tol, maxit = float(self.settings.contact.tol), int(self.settings.contact.maxit)
        if not self._pair_branch_on():
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
        return self.ht_cb.contact_cosine.height(
            Xg, Yg, self.settings.xmax, self.settings.ymax, self.H0_const, self.HT_const, self.T_dim_const,
            self._Ux_dim_const, self._Uy_dim_const)

    def _refresh_geometry(self) -> None:
        super()._refresh_geometry()
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
        H0 = float(self.zeta.H)
        h_min = float(self.h_min)
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
        self.yield_flag = int(self.pasp_max > float(s.contact.sigma_yield))
        if s.contact.k_asp_mode == "closure":
            self.k_asp_hom = -_ec.contact_tangent(op, sol.active, tol=float(s.contact.tol),
                                                  maxit=int(s.contact.maxit))
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
        sol = self._contact_solution
        if sol is None:
            return {}
        return {"p": sol.p.copy(), "P_asp": float(sol.P_asp), "A_mech": float(_ec.p1_area(sol.phi_c)),
                "set": np.asarray(sol.active, dtype=bool).copy()}

    def _ripple_chi_nodes(self) -> np.ndarray:
        return (self.phic_nd_func.vector().get_local() > 0.0).astype(float)

    def _ripple_traction_overlap(self, s_dofs) -> float:
        """Share of the projected fluid traction |s| on contact nodes: the support the
        L2 projection smears into the contact set (a discretisation overlap, plan O2)."""
        if self._contact_solution is None:
            return 0.0
        a = np.abs(np.asarray(s_dofs, dtype=float))
        tot = float(a.sum())
        if tot <= 0.0:
            return 0.0
        act = self._grid_to_dofs(self._contact_solution.active)
        return float(a[act].sum()) / tot

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def compute_asperity_load(self) -> dict:
        """The asperity load summary of the current state (:meth:`_fft_asperity_load`).

        Call AFTER a converged solve (quasi-static, memory-free: re-evaluated per step; the
        caller reads it once per task).  Keys: ``Pasp_hom`` [Pa], ``contact_frac`` [-],
        ``pasp_max`` [Pa] (the nodal peak of the resolved pressure, resolution-dependent),
        ``yield_flag`` {0,1}, ``n_patches``, ``A_patch_frac`` (the obstruction mask),
        ``A_mech_frac``, ``k_asp_hom`` and the ``contact_indicators``.

        [extrema M3b] Afterwards the phase's asperity / total traction statistics are rechecked
        against the extraction (``samplers.finish_contact_traction``: a refreshed solution
        makes them ``state_mismatch``).
        """
        from microcell.samplers import finish_contact_traction
        out = self._fft_asperity_load()
        self.contact_traction = finish_contact_traction(self, out)
        return out
