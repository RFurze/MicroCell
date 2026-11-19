"""The elastic (FFT) contact solve inside the micro solver (the ``G2b`` tests; under a minute).

The cell is a cosine of amplitude 20 nm on a 5 um cell, built by ``make_solver`` through
``tests/parent_fixtures.py``.  The reference is an independent numpy contact solve of the same
state (``numpy_solution``).  Checked:

(a) the area of ``{phi_c < 0}`` from the cells of the dolfin mesh equals the numpy ``p1_area``
    to a relative 1e-10, which checks the grid map and the triangle orientation; the quadrature
    integral of chi (``contact_frac``) agrees with it within 2 % for n >= 100 and 5 % at n = 40;
(b) ``h_hist`` holds ``u_c`` at the history times;
(c) steady and transient solves reproduce the numpy ``P_asp`` to a relative 1e-7;
(d) a translated state is solved again, and a solver reused after ``reset_task`` is bitwise
    equal to a fresh one;
(e) a mean rigid film below ``h_min`` raises before any fluid solve.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

dolfin = pytest.importorskip("dolfin")

from parent_fixtures import (effective_modulus, ensure_process, parent_contact,  # noqa: E402
                             parent_settings, parent_zeta)

ensure_process()

from microcell import contact as ec                                          # noqa: E402
from microcell.contact import ContactResolutionError                                 # noqa: E402

E_STAR = effective_modulus(380e9, 0.26, 380e9, 0.26)
L = 5e-6


# ---- solver and reference ----
def make_solver(n=40, H=58e-9, h_min=40e-9, alpha=0.0, phase=0.337, Ux=0.01, mesh=None):
    """``MicroPhaseNeutralContactSolver`` on the 20 nm cosine, not yet solved.

    n: cells per side.  H: mean film [m].  h_min: contact onset film [m].  alpha: velocity ratio
    of the roughness.  phase: cosine phase in x and y [rad].  Ux: mean velocity [m/s].
    With the defaults the rigid film falls 2 nm below ``h_min``.
    """
    from dolfin import Constant, UnitSquareMesh
    from microcell.film import make_ideal_ht
    from microcell.solver_contact import MicroPhaseNeutralContactSolver
    mesh = mesh or UnitSquareMesh(n, n)
    z = parent_zeta(Ux=Ux, Uy=0., p0=1e7, dpdx=0., dpdy=0., H0=H, HT=0., PT=0.)
    s = parent_settings(eta0=.001, penalty_gamma=1e10, xmax=L, ymax=L, h_min=h_min,
                        relaxation_parameter=1., max_iterations=100, abs_tolerance=1e-6,
                        rel_tolerance=1e-6, delta_h=.5e-9, eps_solid=1e-6, eps_smooth=1e-3,
                        jacobian_reg_eps=-1e-7, E_star=E_STAR,
                        contact=parent_contact(k_asp_mode="closure"))
    ht = make_ideal_ht(20e-9, 1, 1, Constant(phase), Constant(phase))
    return MicroPhaseNeutralContactSolver(mesh_m=mesh, zeta=z, t_end=0., settings=s, ht=ht,
                                          velocity_ratio=alpha, pair=False, auto_solve=False)


def numpy_solution(solver, T=None):
    """Independent numpy contact solve of the current state of ``solver``, or of its state at
    time ``T`` [s].  Returns the solution of ``ec.solve_contact``."""
    n, i, j, Xg, Yg, op = solver._contact_grid()
    p = solver.settings
    Tv = float(solver.T_dim_const) if T is None else T
    hr = solver.ht_cb.contact_cosine.height(Xg, Yg, p.xmax, p.ymax, solver.H0_const, solver.HT_const, Tv,
                                            solver._Ux_dim_const, solver._Uy_dim_const)
    return ec.solve_contact(hr, float(p.h_min), ec.PeriodicHalfSpace(n, n, p.xmax, p.ymax, E_STAR), tol=1e-12)


def cut_area_from_mesh(solver, phi_vper):
    """Area fraction of ``{phi < 0}`` of the CG1 interpolant of ``phi_vper``, summed over the
    triangles of the dolfin mesh.  Independent of the numpy grid map and of the diagonal."""
    from dolfin import FunctionSpace, interpolate, vertex_to_dof_map
    mesh = solver.mesh_m
    Vf = FunctionSpace(mesh, "CG", 1)
    f = interpolate(phi_vper, Vf)
    vals = f.vector().get_local()[vertex_to_dof_map(Vf)]       # per mesh vertex
    cells = mesh.cells()
    xy = mesh.coordinates()
    F = vals[cells].T                                          # 3 x ncells
    P = xy[cells]                                              # ncells x 3 x 2
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    neg = (F < 0).sum(0)
    S = np.sort(F, axis=0)
    frac = np.where(neg == 3, 1.0, 0.0)
    one, two = neg == 1, neg == 2
    frac[one] = S[0, one] ** 2 / ((S[0, one] - S[1, one]) * (S[0, one] - S[2, one]))
    frac[two] = 1 - S[2, two] ** 2 / ((S[2, two] - S[0, two]) * (S[2, two] - S[1, two]))
    return float(np.sum(frac * area) / np.sum(area))


# ---- (a) contact area from the mesh cells; quadrature integral of chi ----
@pytest.mark.parametrize("n,H,h_min,phase", [(40, 58e-9, 40e-9, 0.337), (100, 20e-9, 1e-9, 0.337),
                                             (100, 19.5e-9, 1e-9, 1.9)])
def test_G2b_mask_area_mesh_cells_equal_numpy_p1(n, H, h_min, phase):
    s = make_solver(n=n, H=H, h_min=h_min, phase=phase)
    out = s.compute_asperity_load()            # updates the contact state; no fluid solve
    sol = s._contact_solution
    a_mesh = cut_area_from_mesh(s, s.phic_nd_func)
    a_np = ec.p1_area(sol.phi_c)
    assert a_np > 0
    assert abs(a_mesh - a_np) <= 1e-10 * a_np
    assert out["A_mech_frac"] == a_np
    # quadrature integral of chi against the P1 area
    mism = abs(out["contact_frac"] - a_np) / a_np
    print(f"G2b(a) n={n} H={H*1e9:.1f} nm: P1 area {a_np:.6e}, chi integral {out['contact_frac']:.6e}, "
          f"rel {mism:.3e}")
    assert mism <= (0.02 if n >= 100 else 0.05)
    # u_c and phi_c on Vper equal the numpy fields at the mapped nodes
    n_, i, j, *_ = s._contact_grid()
    np.testing.assert_array_equal(s.uc_nd_func.vector().get_local(), sol.u[i, j] / H)
    np.testing.assert_array_equal(s.phic_nd_func.vector().get_local(), sol.phi_c[i, j] / H)
    assert abs(sol.u.mean()) <= 1e-12 * np.abs(sol.u).max()


# ---- (b) the BDF history holds u_c at the history times ----
def test_G2b_bdf_history_contains_uc_at_history_times():
    s = make_solver(alpha=1.0)
    seen = {}

    def cb(step, t_dim, dt_dim):
        if step == 0:   # after the seed: nm1 = h(T_start), nm2 = h(T_start - dt)
            seen["nm1"] = s.h_hist_nm1.vector().get_local().copy()
            seen["nm2"] = s.h_hist_nm2.vector().get_local().copy()
            seen["T0"], seen["dt"] = t_dim, dt_dim

    s.run(Tf=4e-5, n_steps=2, cfl_target=1., step_callback=cb)
    T0, dt = seen["T0"], seen["dt"]
    assert dt > 0 and T0 > dt                                   # the window starts after t = 0
    n, i, j, *_ = s._contact_grid()
    H0 = float(s.zeta.H)
    for key, T in (("nm1", T0), ("nm2", T0 - dt)):
        # h_hist minus the projected rigid film is the nodal u_c(T) / H0: the projection of a
        # CG1 field onto CG1 is the field itself
        s.T_dim_const.assign(T)
        base = s._project_to_Vper(s.h_base_nd).vector().get_local()
        uc = numpy_solution(s, T=T).u[i, j] / H0
        assert np.abs(uc).max() > 1e-4                          # the deformation is not negligible
        # agreement to the tolerance of the contact solves in the run (1e-8, warm started)
        assert np.abs(seen[key] - base - uc).max() <= 1e-6 * np.abs(uc).max()
        other = numpy_solution(s, T=T + (dt if key == "nm2" else -dt)).u[i, j] / H0
        assert np.abs(seen[key] - base - other).max() > 1e-3 * np.abs(uc).max()   # not u_c of the next level
    # after the run: nm1 is the final time level
    s.T_dim_const.assign(4e-5)
    base = s._project_to_Vper(s.h_base_nd).vector().get_local()
    uc = s.uc_nd_func.vector().get_local()
    assert np.abs(s.h_hist_nm1.vector().get_local() - base - uc).max() <= 1e-12 * np.abs(uc).max() + 1e-14
    # contact solves: 2 history levels, the seed and 2 steps
    assert s._contact_solves == 5


# ---- (c) steady and transient solves reproduce the numpy P_asp ----
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_G2b_steady_and_transient_reproduce_numpy(alpha):
    s = make_solver(alpha=alpha)
    s.run_steady()
    out = s.compute_asperity_load()
    ref = numpy_solution(s)
    assert out["Pasp_hom"] == pytest.approx(ref.P_asp, rel=1e-7)
    assert out["A_mech_frac"] == pytest.approx(ec.p1_area(ref.phi_c), rel=1e-6)
    assert out["k_asp_hom"] == pytest.approx(-ec.contact_tangent(s._contact_grid()[5], ref.active), rel=1e-6)
    assert out["pasp_max"] == pytest.approx(ref.p.max(), rel=1e-6)
    s.run(Tf=1e-5, n_steps=2, cfl_target=1.)
    s.spatial_homogenisation()
    tr = s.compute_asperity_load()
    assert float(s.T_dim_const) == pytest.approx(1e-5)
    ref = numpy_solution(s)
    assert tr["Pasp_hom"] == pytest.approx(ref.P_asp, rel=1e-7)
    assert np.isfinite(s.pressure_ledger["P_f"])
    # the fluid obstruction is the contact set of this state
    assert tr["contact_frac"] > 0 and tr["A_patch_frac"] == tr["contact_frac"]
    if alpha == 0.0:                                            # stationary roughness: no change in time
        assert tr["Pasp_hom"] == pytest.approx(out["Pasp_hom"], rel=1e-7)


# ---- (d) translation and reset_task; (e) infeasible state ----
def test_G2b_translation_refreshes_the_contact_state():
    s = make_solver(alpha=1.0)
    before = s.compute_asperity_load()
    k_before = s._contact_key
    uc_before = s.uc_nd_func.vector().get_local().copy()
    s.T_dim_const.assign(3.7e-4)                                # a move off the grid, across the periodic seam
    moved = s.compute_asperity_load()
    assert s._contact_key != k_before and s._contact_solves == 2
    ref = numpy_solution(s)
    assert moved["Pasp_hom"] == pytest.approx(ref.P_asp, rel=1e-7)
    # same summits: the load differs only by the spread due to the grid position
    assert moved["Pasp_hom"] == pytest.approx(before["Pasp_hom"], rel=0.02)
    assert np.abs(s.uc_nd_func.vector().get_local() - uc_before).max() > 0.1 * np.abs(uc_before).max()
    again = s.compute_asperity_load()                           # same state: no new solve
    assert s._contact_solves == 2 and again["Pasp_hom"] == moved["Pasp_hom"]


def test_G2b_reset_task_zeroes_and_matches_fresh_bitwise():
    a = make_solver()
    a.run_steady()
    ra = a.spatial_homogenisation(), a.compute_asperity_load()
    a.reset_task(replace(a.zeta, H=58.5e-9), a.t_end, a.velocity_ratio)
    for f in (a.uc_nd_func, a.phic_nd_func):
        assert not f.vector().get_local().any()
    assert a._contact_solution is None and a._contact_p_warm is None and a._contact_solves == 0
    a.run_steady()
    reused = (a.spatial_homogenisation(), a.compute_asperity_load(), a.w.vector().get_local().copy())
    b = make_solver(H=58.5e-9, mesh=a.mesh_m)
    b.ht_cb = a.ht_cb
    b.run_steady()
    fresh = (b.spatial_homogenisation(), b.compute_asperity_load(), b.w.vector().get_local().copy())
    assert reused[0] == fresh[0]
    assert {k: v for k, v in reused[1].items() if k != "contact_solve_s"} == \
           {k: v for k, v in fresh[1].items() if k != "contact_solve_s"}
    np.testing.assert_array_equal(reused[2], fresh[2])
    assert ra[1]["Pasp_hom"] > reused[1]["Pasp_hom"] > 0      # larger film: less load


def test_G2b_infeasible_mean_gap_is_a_contact_failure():
    s = make_solver(H=39e-9)                                    # mean rigid film below h_min
    with pytest.raises(ec.ContactInfeasibleError) as err:
        s.run_steady()
    assert isinstance(err.value, ContactResolutionError) and isinstance(err.value, RuntimeError)
    assert s._n_newton_iters_total == 0                         # raised before any fluid solve
