"""The ripple in the micro solver, in the full-film and mixed solver classes.

The cells are cosine films built by ``make_base`` and ``make_contact`` through
``tests/parent_fixtures.py``.  Checked:

* DV2  a prescribed one-mode fluid pressure: the ripple target equals the analytic half-space
       response to a relative 1e-10, with a projection error <= 1e-10 and no seam mismatch;
* DV1  no roughness, uniform pressure: no ripple, the cell-mean film is H0 to 1e-12 and the
       loop accepts at pass 1;
* DV3  the cell tiled 2 x 2 gives the tiled target to a relative 1e-10.  A cavitation boundary
       that crosses the seam gives a seam mismatch and a projection error, and none without it;
* the history levels carry the ripple in the frame of the rough surface: the same field for a
  static surface, the shifted field for a translating cosine (relative 1e-9);
* the pass loop ``_ripple_run`` on a manufactured map ``T(u) = T0 + g u``: 2 passes at g = 0,
  3 at a negative gain, 8 at g = 0.25 and the cap at g = 1.6;
* the pass loop on a rippled cell: accepted after at least 2 passes, the cap raises
  ``RippleNotConvergedError`` (a ``RuntimeError``), the counters are task totals and the
  per-step record of the mixed class holds the final pass only;
* a reused solver after ``reset_task`` equals a new one bitwise;
* refusals: no modulus, ``maxpass = 0``, a film that is not one rough surface.

Run from the repository root::

    python -m pytest tests/test_ripple_micro_fenics.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

dolfin = pytest.importorskip("dolfin")

from parent_fixtures import (effective_modulus, ensure_process, parent_ripple,  # noqa: E402
                             parent_settings, parent_zeta)

ensure_process()

from microcell import ripple as rp  # noqa: E402

E_STAR = effective_modulus(380e9, 0.26, 380e9, 0.26)
L = 5e-6
RIPPLE = "ripple_halfspace"


# ---- cells ----
def _zeta(H=47e-9, Ux=0.022, dpdx=0.0, p0=3e7, UxT=0.0):
    """Macro state of the test cells, through ``parent_zeta``."""
    return parent_zeta(Ux=Ux, Uy=0., p0=p0, dpdx=dpdx, dpdy=0., H0=H, HT=0., PT=0., UxT=UxT)


_COMMON = dict(relaxation_parameter=1., max_iterations=100, abs_tolerance=1e-6, rel_tolerance=1e-6,
               delta_h=0., eps_solid=1e-8, eps_smooth=1e-3, jacobian_reg_eps=-1e-7)


def _settings(mode=RIPPLE, h_min=0.0, Lx=L, common=_COMMON, **over):
    """Settings of the test cells, through ``parent_settings``.

    mode: ``RIPPLE`` for the ripple with the modulus ``E_STAR``, "off" for neither.
    over: replaces fields (``ripple=...``, ``E_star=...``).
    """
    kw = dict(common, eta0=1e-3, penalty_gamma=1e10, xmax=Lx, ymax=Lx, h_min=h_min,
              ripple=parent_ripple() if mode != "off" else None,
              E_star=E_STAR if mode != "off" else None)
    kw.update(over)
    return parent_settings(**kw)


def make_base(mode=RIPPLE, n=60, Ah=20e-9, ht=None, mesh=None, kx=1, ky=1, settings=None, alpha=0.0,
              h_min=0.0, Tend=0.0, Lx=L, **kw):
    """``MicroPhaseNeutralSolver`` on a cosine film (or ``ht``), not yet solved.

    settings: fields passed to ``_settings``.  kw: arguments of ``_zeta``.
    """
    from dolfin import Constant, UnitSquareMesh
    from microcell.film import make_ideal_ht
    from microcell.solver_base import MicroPhaseNeutralSolver
    s = _settings(mode=mode, h_min=h_min, Lx=Lx, **(settings or {}))
    mesh = mesh or UnitSquareMesh(n, n)
    ht = ht or make_ideal_ht(Ah, kx, ky, Constant(0.3), Constant(0.0))
    return MicroPhaseNeutralSolver(mesh_m=mesh, zeta=_zeta(**kw), t_end=Tend, settings=s, ht=ht,
                                   velocity_ratio=alpha, pair=False, auto_solve=False)


def make_contact(mode=RIPPLE, n=60, Ah=20e-9, H=58e-9, h_min=40e-9, mixed_only=False, alpha=0.0,
                 Tend=0.0, **kw):
    """``MicroPhaseNeutralMixedSolver`` on a cosine film with the ripple on, not yet solved.
    ``mixed_only`` must be True.  kw: arguments of ``_zeta``."""
    from dolfin import Constant, UnitSquareMesh
    from microcell.film import make_ideal_ht
    from microcell.solver_mixed import MicroPhaseNeutralMixedSolver
    assert mixed_only, "only the mixed class branch of the parent's make_contact is here"
    mesh = UnitSquareMesh(n, n)
    ht = make_ideal_ht(Ah, 1, 1, Constant(0.337), Constant(0.337))
    common = dict(_COMMON, delta_h=.5e-9, eps_solid=1e-6)
    s = _settings(mode=RIPPLE, h_min=h_min, common=common, E_star=E_STAR)
    return MicroPhaseNeutralMixedSolver(mesh_m=mesh, zeta=_zeta(H=H, p0=1e7, Ux=0.01, **kw),
                                        t_end=Tend, settings=s, ht=ht, velocity_ratio=alpha,
                                        pair=False, auto_solve=False)


def _set_pressure(sv, w_expr_str, amp_nd, **params):
    """Sets the fluid pressure: ``w = amp_nd * <expr>`` (zero mean, on ``Vper``) and ``lambda = 0``."""
    from dolfin import Expression, interpolate
    w = interpolate(Expression(w_expr_str, degree=2, A=amp_nd, pi=np.pi, **params), sv.Vper)
    sv.w.vector().set_local(w.vector().get_local())
    sv.w.vector().apply("insert")
    sv.lambda_nd.assign(0.0)


def _grid_xy(sv):
    """Returns the size ``n`` and the coordinates ``X``, ``Y`` [m] of the periodic grid."""
    n, i, j = sv._periodic_grid()
    x = np.arange(n) / n * float(sv.Lx)
    X, Y = np.meshgrid(x, np.arange(n) / n * float(sv.Ly), indexing="ij")
    return n, X, Y


# ---- DV2: one prescribed mode ----
def test_DV2_one_mode_target_is_the_analytic_response():
    sv = make_base(n=60)
    for kx, ky in ((1, 0), (2, 3)):
        amp = 0.2                                              # amplitude 0.2 p_scale, with P0 = p_scale
        _set_pressure(sv, "A*cos(2*pi*(kx*x[0] + ky*x[1]))", amp, kx=kx, ky=ky)
        tgt = sv._ripple_target()
        n, X, Y = _grid_xy(sv)
        q = 2 * np.pi * np.hypot(kx / L, ky / L)
        exact = 2.0 * amp * sv.p_scale / (E_STAR * q) * np.cos(2 * np.pi * (kx * X + ky * Y) / L)
        assert np.max(np.abs(tgt["T"] - exact)) <= 1e-10 * np.max(np.abs(exact)), (kx, ky)
        assert tgt["proj_err"] <= 1e-10 and tgt["seam"] == 0.0


# ---- DV1: no roughness, uniform pressure ----
def test_DV1_smooth_cell_has_no_ripple_and_the_mean_film_is_H0():
    from dolfin import assemble, dx
    for transient in (False, True):
        sv = make_base(Ah=0.0, Tend=2e-6 if transient else 0.0)
        sv.run(Tf=2e-6, n_steps=3, dt_override=2e-6 / 3) if transient else sv.run_steady()
        info = sv.ripple_info
        # accepted at pass 1: the residual at u = 0 is within the tolerance
        assert info["passes"] == 1, info
        assert info["max"] == 0.0 and info["resid"] <= 1e-6
        mean_h = float(assemble(sv.h_nd * dx(domain=sv.mesh_m)))
        assert abs(mean_h - 1.0) <= 1e-12


# ---- the pass loop on a rippled cell: acceptance, zero mean, counters, reuse ----
def _rippled(**kw):
    """Cell with a ripple: small film, high velocity."""
    return make_base(H=32e-9, Ux=0.5, **kw)


def test_rippled_cell_accepts_after_at_least_two_passes_with_zero_mean():
    """A ripple is not accepted at pass 1, where u = 0 and the residual is the whole ripple.
    Checked: at least 2 passes, residual <= 1e-6, contraction <= ``CONTRACTION_MAX`` and a mean
    of ``u_r`` below 1e-10 of its maximum."""
    sv = _rippled(Tend=1e-7)
    sv.run(Tf=1e-7, n_steps=3)
    info = sv.ripple_info
    assert info["passes"] >= 2, info
    assert info["max"] > 1e-13                                 # the ripple is not zero
    assert info["resid"] <= 1e-6 and info["worst"]["contraction"] <= rp.CONTRACTION_MAX
    u = sv.ur_nd_func.vector().get_local()
    assert abs(u.mean()) <= 1e-10 * np.abs(u).max()
    assert sv._n_newton_iters_total > 0
    print(f"rippled cell: passes {info['passes']}, max|u_r| {info['max'] * 1e9:.4f} nm, "
          f"Newton iterations (task total) {sv._n_newton_iters_total}, omega {info['omega']:.3f}")


def test_ripple_changes_the_solution_and_off_is_the_plain_solver():
    """The ripple changes the first six homogenised outputs; with ``mode="off"`` the solver holds
    no ripple function."""
    on, off = _rippled(), _rippled(mode="off")
    assert off._ripple_on is False and off.ur_nd_func is None
    on.run_steady()
    off.run_steady()
    a, b = np.array(on.spatial_homogenisation()[:6]), np.array(off.spatial_homogenisation()[:6])
    assert np.any(a != b)
    print("on - off (Qx, Qy, Pst, Fst, tau_x, tau_y):", (a - b).tolist())


def test_cap_raises_a_recorded_micro_failure_with_task_counters():
    # maxpass = 2: the residual of pass 1 is the whole ripple and that of pass 2 about g times
    # it, both above the tolerance, so the cap raises
    sv = _rippled(settings=dict(ripple=parent_ripple(ripple_maxpass=2)))
    with pytest.raises(rp.RippleNotConvergedError, match="ripple_not_converged: 2 passes") as exc:
        sv.run_steady()
    assert isinstance(exc.value, RuntimeError)
    single = _rippled(mode="off")
    single.run_steady()
    assert sv._n_newton_iters_total > single._n_newton_iters_total   # two passes summed


class _Contraction:
    """Stand-in solver for ``MicroPhaseNeutralSolver._ripple_run`` on a manufactured map.

    The target at the end of a pass is ``T(u) = T0 + g u`` and every level snapshot is the same,
    so only the deformation decides acceptance.  The loop is the method of the solver class.
    """

    def __init__(self, g, maxpass=12, n=16, H0=20e-9, amp=0.3e-9):
        from types import SimpleNamespace
        from dolfin import Function, FunctionSpace, UnitSquareMesh
        from microcell import solver_base as M
        self._ripple_run = M.MicroPhaseNeutralSolver._ripple_run.__get__(self)
        self.g, self.n = g, n
        x = np.arange(n) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.T0 = amp * np.cos(2 * np.pi * X) * np.cos(4 * np.pi * Y)
        self.settings = SimpleNamespace(ripple=SimpleNamespace(tol=1e-6, omega=1.0, maxpass=maxpass))
        self.zeta = SimpleNamespace(H=H0)
        self.p_scale = 1e7
        self.ur_nd_func = Function(FunctionSpace(UnitSquareMesh(2, 2), "CG", 1))
        self._ripple_u_serial = self._ripple_version = 0
        self.passes = 0
        for name in M._PASS_COUNTERS:
            setattr(self, name, 0)

    # solver methods that the loop calls
    def _ripple_check_task(self):
        pass

    def _periodic_grid(self):
        return self.n, None, None

    def _on_ripple_pass_start(self):
        pass

    def _ripple_surface_shift(self):
        return (0.0, 0.0)

    def _ripple_mnorm(self, v):
        return float(np.sqrt(np.mean(np.square(v))))

    def body(self):
        self.passes += 1
        self._n_newton_iters_total = 1
        self._ripple_levels = {"steady": {"t": np.full(4, 3e7), "A_mask": 0.0}}

    def _ripple_target(self):
        u = self._ripple_u if self._ripple_u is not None else 0.0 * self.T0
        return {"T": self.T0 + self.g * u, "proj_err": 0.0, "seam": 0.0, "overlap": 0.0}


@pytest.mark.parametrize("g, passes", [(0.0, 2), (-0.5, 3), (-2.0, 3)])
def test_manufactured_contraction_pass_counts(g, passes):
    """Pass counts of the residual rule, with ``u`` against ``T0 / (1 - g)`` to 1e-6 H0.

    g = 0: pass 1 at u = 0 has residual T0 and pass 2 at u = T0 has residual 0, so the loop
    accepts at 2.  A negative gain: the secant step of pass 2 gives the fixed point
    ``T0 / (1 - g)`` (omega = 1 / (1 - g), inside the clamp), so the loop accepts at 3.
    """
    c = _Contraction(g)
    c._ripple_run(c.body)
    info = c.ripple_info
    assert info["passes"] == passes == c.passes, info
    np.testing.assert_allclose(c._ripple_u, c.T0 / (1.0 - g), rtol=0, atol=1e-6 * 20e-9)
    assert c._n_newton_iters_total == passes                        # task total over the passes


def test_manufactured_positive_gain_converges_at_the_clamp_and_gain_above_one_hits_the_cap():
    """g in (0, 1): the secant omega 1 / (1 - g) > 1 is clamped to 1, so the error falls by g per
    pass and the residual tolerance 1e-6 H0 sets the count.  g > 1: the secant omega is negative
    and clamped to 0.1, the iteration diverges and the cap raises."""
    g = 0.25
    c = _Contraction(g)
    c._ripple_run(c.body)
    assert c.ripple_info["omega"] == 1.0
    # with omega = 1, pass k runs at u_k = T0 (1 + g + ... + g^(k-2)), so its residual is
    # R_k = g^(k-1) T0: accepted at the first k with g^(k-1) max|T0| <= tol H0, which is 8.
    # The field is then R_k / (1 - g) from the fixed point, within 2 tol H0
    a = np.max(np.abs(c.T0))
    k = next(k for k in range(1, 60) if g ** (k - 1) * a <= 1e-6 * 20e-9)
    assert c.ripple_info["passes"] == k == 8, (c.ripple_info["passes"], k)
    assert c.ripple_info["resid"] <= 1e-6 and c.ripple_info["worst"]["contraction"] <= rp.CONTRACTION_MAX
    np.testing.assert_allclose(c._ripple_u, c.T0 / (1.0 - g), rtol=0, atol=2e-6 * 20e-9)
    d = _Contraction(1.6, maxpass=12)
    with pytest.raises(rp.RippleNotConvergedError, match="12 passes"):
        d._ripple_run(d.body)
    assert d.passes == 12 and d._n_newton_iters_total == 12


def test_reused_solver_equals_fresh_bitwise():
    a = _rippled(Tend=1e-7)
    a.run(Tf=1e-7, n_steps=3)
    a.reset_task(_zeta(H=34e-9, Ux=0.4), 1e-7, 0.0)
    a.run(Tf=1e-7, n_steps=3)
    b = _rippled(Tend=1e-7, mesh=a.mesh_m)
    b.reset_task(_zeta(H=34e-9, Ux=0.4), 1e-7, 0.0)
    b.run(Tf=1e-7, n_steps=3)
    assert a.spatial_homogenisation() == b.spatial_homogenisation()
    np.testing.assert_array_equal(a.ur_nd_func.vector().get_local(), b.ur_nd_func.vector().get_local())
    assert a.ripple_info["passes"] == b.ripple_info["passes"]
    # a second run of the same task without a reset starts the ripple from u_r = 0 and the
    # pressure from the last state, so the agreement is to the Newton tolerance
    c = _rippled(Tend=1e-7, mesh=a.mesh_m)
    c.reset_task(_zeta(H=34e-9, Ux=0.4), 1e-7, 0.0)
    c.run(Tf=1e-7, n_steps=3)
    c.run(Tf=1e-7, n_steps=3)
    np.testing.assert_allclose(c.spatial_homogenisation()[:6], b.spatial_homogenisation()[:6], rtol=1e-8)
    assert c.ripple_info["passes"] == b.ripple_info["passes"]


# ---- the history levels ----
def _rigid_nd_at(sv, T):
    """Rigid film / H0 projected on ``Vper`` at time T [s].  The time constant is set back
    afterwards."""
    T_end = float(sv.T_dim_const)
    sv.T_dim_const.assign(T)
    v = sv._project_to_Vper(sv.h_base_nd).vector().get_local()
    sv.T_dim_const.assign(T_end)
    return v


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_history_levels_carry_the_ripple_in_the_surface_frame(alpha):
    dt = 2e-8
    sv = _rippled(alpha=alpha, Tend=3 * dt)
    sv.run(Tf=3 * dt, n_steps=3, dt_override=dt)
    H0 = float(sv.zeta.H)
    u_end = sv._dofs_to_grid(sv.ur_nd_func.vector().get_local()) * H0
    T_end = float(sv.T_dim_const)
    # after the last step: nm1 = h(T_end), nm2 = h(T_end - dt)
    for hist, T in ((sv.h_hist_nm1, T_end), (sv.h_hist_nm2, T_end - dt)):
        got = (hist.vector().get_local() - _rigid_nd_at(sv, T)) * H0
        shift = alpha * 2.0 * 0.5 * (T - T_end)               # s(T) - s(T_end), UxT = 0
        want = sv._grid_to_dofs(rp.pattern_shift(u_end, shift / L, 0.0))
        assert np.max(np.abs(got - want)) <= 1e-9 * np.max(np.abs(want)), (alpha, T)
    if alpha == 0.0:   # static surface: the same field at both levels
        np.testing.assert_allclose(sv.h_hist_nm2.vector().get_local() - _rigid_nd_at(sv, T_end - dt),
                                   sv.h_hist_nm1.vector().get_local() - _rigid_nd_at(sv, T_end),
                                   rtol=0, atol=1e-9 * np.max(np.abs(u_end)) / H0)


# ---- the per-step record of the mixed class ----
def test_mixed_per_step_record_holds_the_final_pass_only():
    sv = make_contact(mixed_only=True, Tend=1e-6)      # H = 58 nm: a contact patch, so a ripple
    sv.record_homog = True
    sv.run(Tf=1e-6, n_steps=3, dt_override=1e-6 / 3)
    # at least 2 passes, so the final pass is not the only one; with record_homog every pass
    # is a whole run
    assert sv.ripple_info["passes"] >= 2 and sv.ripple_info["resid"] <= 1e-6
    assert len(sv.homog_history) == 4                           # seed + 3 steps, recorded once


# ---- DV3: tiling, seam and projection ----
def test_DV3_tiled_fe_cell_gives_the_tiled_target():
    one = make_base(n=40)
    two = make_base(n=80, Lx=2 * L)
    expr = "A*(cos(2*pi*kx*x[0]) + 0.5*sin(2*pi*kx*x[1]) + 0.3*cos(2*pi*kx*(x[0] + 2*x[1])))"
    _set_pressure(one, expr, 0.1, kx=1)
    _set_pressure(two, expr, 0.1, kx=2)                         # two periods per doubled cell
    t1 = one._ripple_target()["T"]
    t2 = two._ripple_target()["T"]
    assert np.max(np.abs(t2 - np.tile(t1, (2, 2)))) <= 1e-10 * np.max(np.abs(t1))


def test_DV3_seam_crossing_cavitation_is_reported():
    # no gradient and no cavitation: no seam mismatch
    sv = make_base(n=60)
    sv.run_steady()
    assert sv.ripple_info["seam_mismatch"] == 0.0
    # a large gradient and a low cell pressure: the cavitation boundary crosses the seam
    sv = make_base(n=60, p0=8e6, dpdx=-4e12, Ux=0.1)
    sv.run_steady()
    info = sv.ripple_info
    print(f"seam mismatch {info['seam_mismatch']:.4e} Pa, projection error {info['proj_err']:.3e}")
    assert info["seam_mismatch"] > 0.0 and info["proj_err"] > 0.0
    u = sv.ur_nd_func.vector().get_local()
    assert abs(u.mean()) <= 1e-10 * np.abs(u).max()


# ---- refusals ----
def test_refusals():
    from dolfin import Constant
    from microcell.film import make_ideal_ht
    with pytest.raises(ValueError, match="E_star"):                    # no modulus
        make_base(settings=dict(E_star=None))
    with pytest.raises(ValueError, match="maxpass >= 1"):              # 0 is refused, 1 is allowed
        make_base(settings=dict(ripple=parent_ripple(ripple_maxpass=0)))
    make_base(settings=dict(ripple=parent_ripple(ripple_maxpass=1)))
    with pytest.raises(ValueError, match="one rough surface"):
        make_base(alpha=0.5).run_steady()
    two = make_ideal_ht(20e-9, 1, 1, Constant(0.0), Constant(0.0), two_surface=True)
    with pytest.raises(ValueError, match="one rough surface"):
        make_base(ht=two).run_steady()
