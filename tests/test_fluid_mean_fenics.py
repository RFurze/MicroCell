"""V2 and the FEniCS unit tests of the fluid-region level constraint (FEniCS; seconds per cell).

Copied in WP5 from the parent ``MultiscaleMixedBallinSocket`` at 2aaccad,
``tests/micro_fluid_mean/test_fluid_mean_fenics.py``.  Expected numbers and tolerances are the
parent's.  Changed: the imports; ``make`` builds a ``Zeta`` and a ``MicroSettings`` from the parent's
``PARAMS`` / ``SETTINGS`` names (``tests/parent_fixtures.py``); ``s.params.<x>`` is read from
``s.zeta``; ``beta`` and ``patch_pin_kappa`` are read from the solver module's fixed constants
(they were settings in the parent, with these values).  ``PARAMS`` no longer carries ``rho0`` (never
read) and ``k_spring`` (0; the spring is removed); ``SETTINGS`` no longer carries ``beta`` (1.0, now
fixed), ``print_newton``, ``print_timing``, ``gauge_lift`` and ``spring_pointwise`` (all False, now
fixed or removed).
Left out: ``test_patch_free_keeps_dP`` (``patch_pin=False``; the pin is not a setting here).
In part: ``test_unknown_setting_refused_and_lc_class_takes_the_whole_cell`` keeps the refusal; the
LC class is not in this package.

Manufactured cells: the mixed class with the level set chi = [h - h_min > 0] on an idealised cosine film
(``ideal_ht``), the masked branch (``pattern_flux_contact=False``, D6), steady solves.

Items (V2 (i)-(vii) and section 8.1 of the parent's plan):

* (i)   the fluid mean holds: (1/A_f) int chi P = P_re + lambda, to 1e-12 relative (at a tight tolerance);
* (ii)  a flooded cell: dP = -A_p P_re to 1e-12 (lambda = 0);
* (iii) a cavitating cell: dP = (1 - A_p) lambda - A_p P_re;
* (iv)  A_f = 0: the bypass (dP = -P_re, lambda = 0, Pst_fluid NaN, n_no_fluid counted); a one-element fluid
        region converges with no Schur-regularisation warning, |H| <= 1e-12, fluid mean to 1e-9;
* (v)   a one-node and an L/4 roll of the film (with a pressure ramp): dP to 1e-9 relative;
* (vi)  an interior-patch perturbation (+1 MPa) re-solved warm: dP to 1e-12 relative;
* (vii) the pin: interior |w| <= 1e-10 in steady cells;
* the switched-off and the full-film path: bitwise equal (V1's mechanism on one cell);
* m_f is re-assembled when chi changes; an unknown setting is refused.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

pytest.importorskip("dolfin")

from parent_fixtures import ensure_process, parent_settings, parent_zeta  # noqa: E402

ensure_process()

from microcell import solver_base as base                    # noqa: E402

LX = 5e-6
PARAMS = dict(Ux=2.5e-2, Uy=0.0, H0=4.3e-7, p0=1e7, dpdx=0.0, dpdy=0.0, HT=0.0, PT=0.0, Tend=2.0e-2,
              UxT=0.0, UyT=0.0, dpdx_dot=0.0, dpdy_dot=0.0, eta0=1e-2, penalty_gamma=1e10,
              xmax=LX, ymax=LX, h_min=5e-8, roughness_velocity_ratio=1.0)
#: [WP8, 2 Nov 2026] ``level_constraint="whole_cell"`` is stated: these tests compare the whole cell ("off", the
#: formulation before 2 Nov) with the fluid mean (``FM``) explicitly, and the settings' own default is the run
#: default, fluid_mean_v1, since 2 Nov
SETTINGS = dict(relaxation_parameter=1.0, max_iterations=200, abs_tolerance=1e-11, rel_tolerance=1e-13,
                delta_h=0.0, eps_solid=1e-8, eps_smooth=1e-3, jacobian_reg_eps=-1e-7,
                pattern_flux_contact=False, level_constraint="whole_cell")
AH = 4e-7
N = 24


def make(params=None, settings=None, ah=AH, phi_x=0.0, phi_y=0.0, n=N, cls=None):
    from fenics import Constant, UnitSquareMesh
    from microcell.film import make_ideal_ht
    from microcell.solver_mixed import MicroPhaseNeutralMixedSolver
    Cls = cls or MicroPhaseNeutralMixedSolver
    p = {**PARAMS, **(params or {})}
    st = {**SETTINGS, **(settings or {})}
    zeta = parent_zeta(**{k: p[k] for k in ("Ux", "Uy", "p0", "dpdx", "dpdy", "H0", "HT", "PT", "UxT", "UyT",
                                            "dpdx_dot", "dpdy_dot")})
    S = parent_settings(**{k: p[k] for k in ("eta0", "penalty_gamma", "xmax", "ymax", "h_min")}, **st)
    ht = make_ideal_ht(Ah=ah, kx=1.0, ky=1.0, phi_x_const=Constant(phi_x), phi_y_const=Constant(phi_y))
    mesh = UnitSquareMesh(n, n)
    return Cls(mesh_m=mesh, zeta=zeta, t_end=p["Tend"], settings=S, ht=ht,
               velocity_ratio=p["roughness_velocity_ratio"], pair=False, auto_solve=False)


FM = {"level_constraint": "fluid_mean_v1"}


def solve(s):
    s.run_steady()
    s.spatial_homogenisation()
    return s


def ledger(s):
    r = dict(s.pressure_ledger)
    lam = float(s.dP)                                   # lambda_dim
    P_re = float(s.zeta.P)
    A = float(s._cell_area)
    from fenics import assemble, dx
    A_f = float(assemble(s.chi_fluid * dx(domain=s.mesh_m)))
    r.update(lam=lam, P_re=P_re, A_p=1.0 - A_f / A, dP=r["P_f"] - P_re, A_f=A_f, A=A)
    return r


def H_final(s):
    return s._constraint_residual(s._constraint_vector(), float(s._cell_area))


def interior_nodes(s):
    return np.flatnonzero(s._lc_mf.get_local() == 0.0) if s._lc_mf is not None else np.zeros(0, int)


# ------------------------------------------------------------------------------------------------ (i), (ii)
def test_i_ii_flooded_fluid_mean_and_dP():
    s = solve(make(settings=FM))
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and r["A_p"] > 0.0
    assert r["lam"] == 0.0 or abs(r["lam"]) < 1e-9 * r["P_re"]                 # flooded: no cavitation lift
    scale = abs(r["P_re"] + r["lam"])
    assert abs(r["Pst_fluid"] - (r["P_re"] + r["lam"])) <= 1e-12 * scale        # (i)
    assert abs(r["dP"] + r["A_p"] * r["P_re"] - (1 - r["A_p"]) * r["lam"]) <= 1e-12 * r["P_re"]   # (ii)
    assert abs(H_final(s)) <= 1e-11


def test_iii_cavitating_dP_identity():
    s = solve(make(params={"p0": 1.137147e1}, settings=FM))           # the contact-load tests' low level
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and r["lam"] > 0.0 and r["A_p"] > 0.0
    pred = (1 - r["A_p"]) * r["lam"] - r["A_p"] * r["P_re"]
    assert abs(r["dP"] - pred) <= 1e-9 * max(abs(r["lam"]), 1.0)
    # mu = -beta lambda A / A_f on the masked branch (plan section 3.5)
    beta = float(base.BETA)                             # fixed in this package (the parent read settings.beta)
    lam_nd = float(s.lambda_nd)
    assert abs(s.mu_nd + beta * lam_nd * r["A"] / r["A_f"]) <= 1e-6 * max(abs(beta * lam_nd), 1e-12)


# ------------------------------------------------------------------------------------------------ (iv)
def test_iv_no_fluid_bypass():
    s = solve(make(params={"H0": 2.5e-8}, ah=2e-8, settings=FM))      # h in [5, 45] nm < h_min = 50 nm
    r = ledger(s)
    assert r["A_f"] == 0.0 and s._lc_path == "no_fluid"
    assert s._n_lc_no_fluid >= 1 and s._n_lc_fluid_mean == 0
    assert float(s.lambda_nd) == 0.0 and r["P_f"] == 0.0
    assert r["dP"] == -r["P_re"] and math.isnan(r["Pst_fluid"])


def test_iv_small_fluid_region_converges(caplog):
    # h_max = H0 + Ah at the corner; h_min just below it: a fluid region of a few elements
    H0, ah = 2.0e-7, 1e-7
    with caplog.at_level(logging.WARNING):
        s = solve(make(params={"H0": H0, "h_min": H0 + ah - 2e-9}, ah=ah, settings=FM))
    assert not [rec for rec in caplog.records if "Schur complement near-singular" in rec.getMessage()]
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and 0.0 < r["A_f"] < 0.02 * r["A"]
    assert abs(H_final(s)) <= 1e-12
    assert abs(r["Pst_fluid"] - (r["P_re"] + r["lam"])) <= 1e-9 * abs(r["P_re"] + r["lam"])


# ------------------------------------------------------------------------------------------------ (v)
@pytest.mark.parametrize("shift", ["one_node", "quarter"])
def test_v_rolls_leave_dP_invariant(shift):
    ramp = {"dpdx": 3e11, "dpdy": -1e11}                     # Pa/m: a pressure ramp across the cell
    base = ledger(solve(make(params=ramp, settings=FM)))
    dphi = 2 * math.pi / N if shift == "one_node" else math.pi / 2
    rolled = ledger(solve(make(params=ramp, settings=FM, phi_x=dphi)))
    assert abs(rolled["A_p"] - base["A_p"]) <= 1e-12
    assert abs(rolled["dP"] - base["dP"]) <= 1e-9 * max(abs(base["dP"]), 1e-9 * base["P_re"])


# ------------------------------------------------------------------------------------------------ (vi), (vii)
def test_vi_vii_interior_perturbation_and_pin():
    s = solve(make(settings=FM))
    r0 = ledger(s)
    idx = interior_nodes(s)
    assert idx.size > 0
    w = s.w.vector().get_local()
    assert np.max(np.abs(w[idx])) <= 1e-10                                      # (vii) steady
    w[idx] += 1e6 / float(s.p_scale)                                           # +1 MPa on the interior
    s.w.vector().set_local(w)
    s.w.vector().apply("insert")
    s._reset_timers()
    s.solve()
    s.spatial_homogenisation()
    r1 = ledger(s)
    assert abs(r1["dP"] - r0["dP"]) <= 1e-12 * max(abs(r0["dP"]), r0["P_re"])  # (vi)
    # the pin undoes it, to the field tolerance: the pinned row's residual kappa m_i w_i is below abs_tolerance
    kappa_m = float(base.PATCH_PIN_KAPPA) * s._mean_vec.get_local()[idx]     # fixed here (parent: settings.patch_pin_kappa)
    assert np.all(np.abs(s.w.vector().get_local()[idx]) <= float(s.settings.abs_tolerance) / kappa_m)



# ------------------------------------------------------------------------------------------------ off / full film
def _outputs(s):
    return [float(v) for v in s.spatial_homogenisation()] + [float(s.lambda_nd), float(s.mu_nd)] + \
        list(s.w.vector().get_local())


def test_off_is_bitwise_and_takes_no_fluid_mean_path():
    a = make()
    a.run_steady()
    oa = _outputs(a)
    assert a._n_lc_whole_cell >= 1 and a._n_lc_fluid_mean == 0 and a._lc_mf is None
    b = make(settings={"level_constraint": "whole_cell"})
    b.run_steady()
    assert _outputs(b) == oa


def test_full_film_fluid_mean_is_bitwise_whole_cell():
    over = {"H0": 5.5e-7}                                  # min h 150 nm > h_min: no patch
    a = make(params=over)
    a.run_steady()
    b = make(params=over, settings=FM)
    b.run_steady()
    assert b._n_lc_fluid_mean == 0 and b._n_lc_whole_cell >= 1
    assert _outputs(b) == _outputs(a)


# ------------------------------------------------------------------------------------------------ refresh / refusals
def test_mf_reassembled_when_chi_changes():
    s = solve(make(settings=FM))
    mf0, Af0 = s._lc_mf.get_local().copy(), s._lc_Af
    s.H0_const.assign(float(s.zeta.H) * 1.05)              # a thicker film: a smaller patch
    s._reset_timers()
    s.solve()
    assert s._lc_Af > Af0
    assert not np.array_equal(s._lc_mf.get_local(), mf0)
    assert np.array_equal(s._mean_vec.get_local(), make()._mean_vec.get_local())   # m6: never overwritten


def test_unknown_setting_refused():
    """An unknown tag is refused.  (The parent's test went on to the LC class, which is not in this
    package.)"""
    s = make(settings={"level_constraint": "fluid_mean_v2"})
    with pytest.raises(ValueError):
        s.run_steady()


def test_base_class_full_film_is_bitwise():
    """V1's base-class item on a manufactured cell: the full-film solver (h_min = 0) takes the whole-cell path under
    either setting, bit for bit."""
    from microcell.solver_base import MicroPhaseNeutralSolver

    outs = []
    for lc in ("whole_cell", "fluid_mean_v1"):
        s = make(params={"h_min": 0.0}, settings={"level_constraint": lc}, cls=MicroPhaseNeutralSolver)
        s.run_steady()
        assert s._n_lc_whole_cell >= 1 and s._n_lc_fluid_mean == 0
        outs.append([float(v) for v in s.spatial_homogenisation()] + [float(s.lambda_nd)]
                    + list(s.w.vector().get_local()))
    assert outs[0] == outs[1]


# ------------------------------------------------------------------------------------------------ the sampler (m2)
def test_sampler_leaves_out_every_mf_zero_vertex():
    """Under the fluid mean the extrema samples never include a vertex with m_f,i = 0 (pinned), whatever the
    sampler's own fluid test says; on the no-fluid path none is fluid (code review, 30 Oct)."""
    from microcell.samplers import _vertex_maps, fluid_pressure_samples
    s = solve(make(settings=FM))
    assert s._lc_path == "fluid_mean"
    _, v_per, _ = _vertex_maps(s)
    zero = (s._lc_mf.get_local() == 0.0)[v_per]
    smp = fluid_pressure_samples(s)
    assert zero.any() and not np.any(smp.fluid & zero)
    off = solve(make())                                   # off: the sampler's own test, as before
    assert getattr(off, "_lc_mf_zero", None) is None
    nf = solve(make(params={"H0": 2.5e-8}, ah=2e-8, settings=FM))
    assert nf._lc_path == "no_fluid" and not fluid_pressure_samples(nf).fluid.any()
