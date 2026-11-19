"""Mixed solver with the elastic contact law ``fft_elastic_v1``.

One periodic elastic contact problem is solved per cell on the nodal grid (``microcell.contact``)
and refreshed at every time level.  The fluid sees the deformed film ``h_rigid + u_c``
(``uc_nd_func``; zero mean, as the macroscale keeps the cell mean).  Its obstruction mask is the
contact set of the same solution, through the level set ``phi_c = g - p/k_N`` (``phic_nd_func``).
Load, mechanical area, peak, tangent and mask all come from that one nodal solution.

``contact_frac`` and ``A_patch_frac`` are the integrated obstruction indicator.  ``A_mech_frac``
is the mechanical contact area of the nodal solution.  The homogenised load is per total cell
area [Pa].  Serial only.
"""

from __future__ import annotations

import time

import numpy as np
from fenics import *  # noqa: F401,F403
from microcell.contact import FFT
from microcell import contact as _ec

from microcell.solver_mixed import MicroPhaseNeutralMixedSolver as _MixedSolver

# Settings read (``MicroSettings.contact``, a ``microcell.ContactSettings``):
#   ``E_star``          combined two-body modulus [Pa], from ``MicroSettings``; the ripple uses the same
#   ``sigma_yield``     threshold of ``yield_flag`` [Pa]; it does not limit the pressure
#   ``k_asp_mode``      "closure": k_asp = -dP_asp/dH [Pa/m] from ``contact.contact_tangent``; "off": 0.0
#   ``tol``, ``maxit``  stopping rule sum|dp|/sum p and iteration cap of the contact solve; the
#                       tangent solve uses the same tolerance
#   ``mu``              asperity friction coefficient, read by the sampled asperity and total
#                       traction only (``samplers.sample_contact_traction``)
# ``self.pair`` is True for a stationary + translating pair.  It allows the ripple with two
# surfaces and turns on the cold-start retry of the contact solve.


class MicroPhaseNeutralContactSolver(_MixedSolver):
    """Mixed micro solver with the elastic asperity contact solve.

    ``compute_asperity_load()`` returns the load summary after a converged solve.
    ``contact_traction`` holds the asperity and total traction statistics.
    """

    # contact bookkeeping of the current task, zeroed by reset_task
    _contact_solution = None
    _contact_key = None
    _contact_p_warm = None
    _contact_solve_s = 0.0
    _contact_solves = 0
    _contact_iterations = 0
    _contact_restarts = 0                   # cold-start retries of the contact solve (pairs only)
    _contact_restart_kinds = None
    # asperity and total traction statistics of the current phase (shear.py)
    contact_traction = None

    # ---- Asperity and total traction ----
    def spatial_homogenisation(self):
        """Homogenisation of the mixed class, then the asperity and total traction on the sample
        points of the fluid traction sampler (``samplers.sample_contact_traction``).

        The stored contact solution of this state is used and is not refreshed here.
        Returns the outputs of the mixed class.
        """
        out = super().spatial_homogenisation()
        from microcell.samplers import sample_contact_traction
        self.contact_traction = sample_contact_traction(self)
        return out

    # ---- Film, mask, grid map and contact refresh ----
    def _setup_film_and_mask(self) -> None:
        """Adds the contact deformation to the film and builds the mask from the contact level set."""
        super()._setup_film_and_mask()
        s = self.settings
        # Nodal fields on Vper, zero until the first contact refresh.  h_nd gains the zero-mean
        # contact deformation; the obstruction level set is phi_c / H0, not h_nd - h_min.
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
        """Zeroes the contact fields and the contact bookkeeping of the task."""
        super()._reset_task_subclass()
        self.contact_traction = None        # statistics of the current task only
        for f in (self.uc_nd_func, self.phic_nd_func):
            f.vector().zero()
            f.vector().apply("insert")
        self._contact_solution = None
        self._contact_key = None
        self._contact_p_warm = None
        self._contact_solve_s = 0.0
        self._contact_solves = 0
        self._contact_iterations = 0
        self._contact_restarts = 0          # cold-start retries of this task
        self._contact_restart_kinds = {"collapse": 0, "stall": 0}

    def _fft_validate(self):
        """Checks ``E_star``, ``k_asp_mode`` and the film callback.

        Returns the ``contact_cosine`` descriptor of the callback, which evaluates the rigid
        film with numpy.
        """
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
        """Returns ``(n, i, j, Xg, Yg, op)``, cached.

        n, i, j: the periodic n x n grid of the Vper dofs and its index maps, from the base class.
        Xg, Yg: unit-cell coordinates of the grid.  op: the contact operator.
        """
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
        """Tuple of the state the contact solution depends on.

        Holds the film, its rate, the time, the velocities, the phases, ``h_min``, the cell
        size, and where present the ripple version and the starting displacement.  A stored
        solution is current when its key matches.
        """
        descriptor = self.ht_cb.contact_cosine
        vals = (self.H0_const, self.HT_const, self.T_dim_const, self._Ux_dim_const, self._Uy_dim_const,
                descriptor.phi_x, descriptor.phi_y, descriptor.dphi_x, descriptor.dphi_y,
                self.h_min, self.settings.xmax, self.settings.ymax)
        key = tuple(float(v) for v in vals)
        if self._ripple_on:                 # the contact is solved on the film with the ripple
            key = key + (float(self._ripple_version),)
        motion = getattr(self.ht_cb, "surface_motion", None)
        if motion is not None:              # starting displacement of the translating surface
            key = key + (float(motion.s0[0]), float(motion.s0[1]))
        return key

    def _solve_contact_rescued(self, h_r, h_min, op, p0):
        """``solve_contact`` with one cold-start retry, for pairs only.

        A warm-started solve can collapse (relative change inf or nan: the first step clears
        every warm-start node when the pattern has moved) or stall on a stale active set
        (``contact_complementarity``).  It is then retried once from a cold start.  A failed
        cold start, and any other failure, is raised.  Retries are counted in
        ``_contact_restarts`` and, by kind, in ``_contact_restart_kinds``.

        h_r: rigid film on the contact grid [m].  h_min: contact onset film [m].
        op: contact operator.  p0: warm-start contact pressure, or None for a cold start.
        Returns the ``ContactSolution``.
        """
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
            if self._contact_restart_kinds is None:        # before the first reset_task
                self._contact_restart_kinds = {"collapse": 0, "stall": 0}
            self._contact_restart_kinds["collapse" if collapse else "stall"] += 1
            return _ec.solve_contact(h_r, h_min, op, p0=None, tol=tol, maxit=maxit)

    def _rigid_film_grid(self):
        """Rigid film [m] on the contact grid, from the same description as the UFL film
        (``contact_cosine.height``)."""
        n, i, j, Xg, Yg, op = self._contact_grid()
        return self.ht_cb.contact_cosine.height(
            Xg, Yg, self.settings.xmax, self.settings.ymax, self.H0_const, self.HT_const, self.T_dim_const,
            self._Ux_dim_const, self._Uy_dim_const)

    def _refresh_geometry(self) -> None:
        """Refresh of the base class, then the contact solve."""
        super()._refresh_geometry()
        self._refresh_contact()

    def _refresh_contact(self) -> None:
        """Solves the contact on the rigid film and writes ``uc_nd_func`` and ``phic_nd_func``.

        The solve is warm-started from the previous contact pressure.  A mean rigid film at or
        below ``h_min`` raises ``ContactInfeasibleError``, before any fluid solve.  ``h_min <= 0``
        is contact off: no deformation and the level set ``h - h_min``.
        """
        self._fft_validate()
        _t0 = time.perf_counter()
        n, i, j, Xg, Yg, op = self._contact_grid()
        H0 = float(self.zeta.H)
        h_min = float(self.h_min)
        h_r = self._rigid_film_grid()
        if self._ripple_on:
            # the contact is solved on the rigid film plus the ripple of this time level,
            # which the base refresh has written
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
        """Asperity load summary from the stored nodal solution of the current state, which is
        refreshed if stale.  Returns the dict of ``compute_asperity_load``."""
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
        self.A_patch_frac = self.contact_frac          # obstruction (chi) integral
        self.pasp_max = float(sol.p.max())             # nodal peak; depends on the resolution
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

    # ---- Contact quantities for the ripple pass checks ----
    def _ripple_contact_snapshot(self) -> dict:
        """Contact pressure, load, mechanical area and contact set of the stored solution.
        Empty without a solution."""
        sol = self._contact_solution
        if sol is None:
            return {}
        return {"p": sol.p.copy(), "P_asp": float(sol.P_asp), "A_mech": float(_ec.p1_area(sol.phi_c)),
                "set": np.asarray(sol.active, dtype=bool).copy()}

    def _ripple_chi_nodes(self) -> np.ndarray:
        """Fluid indicator at the Vper dofs: 1.0 where ``phi_c > 0``."""
        return (self.phic_nd_func.vector().get_local() > 0.0).astype(float)

    def _ripple_traction_overlap(self, s_dofs) -> float:
        """Share of the projected fluid traction ``|s|`` (at the Vper dofs) on contact nodes.
        The L2 projection spreads some traction into the contact set."""
        if self._contact_solution is None:
            return 0.0
        a = np.abs(np.asarray(s_dofs, dtype=float))
        tot = float(a.sum())
        if tot <= 0.0:
            return 0.0
        act = self._grid_to_dofs(self._contact_solution.active)
        return float(a[act].sum()) / tot

    # ---- Public entry ----
    def compute_asperity_load(self) -> dict:
        """Asperity load summary of the current state.  Call after a converged solve.

        Returns a dict: ``Pasp_hom`` [Pa], ``contact_frac`` [-], ``pasp_max`` [Pa] (nodal peak,
        depends on the resolution), ``yield_flag`` (0 or 1), ``n_patches``, ``A_patch_frac``
        (the obstruction mask), ``A_mech_frac``, ``k_asp_hom`` [Pa/m] and the
        ``contact_indicators``.

        ``contact_traction`` is then rechecked against this solution
        (``samplers.finish_contact_traction``): a solution refreshed here gives the status
        ``state_mismatch``.
        """
        from microcell.samplers import finish_contact_traction
        out = self._fft_asperity_load()
        self.contact_traction = finish_contact_traction(self, out)
        return out
