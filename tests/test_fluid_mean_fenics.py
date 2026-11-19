"""The fluid-region level constraint ``fluid_mean_v1`` in the mixed solver class (seconds per cell).

The cells are steady solves of a cosine film with the fluid indicator ``chi = [h - h_min > 0]``
and ``pattern_flux_contact=False``, built by ``make`` through ``tests/parent_fixtures.py``.
``A_f`` is the fluid area, ``A_p = 1 - A_f / A`` the area fraction outside it, ``P_re`` the macro
pressure and ``H`` the residual of the constraint.  Checked:

* (i)   the fluid mean holds: ``(1/A_f) int chi P = P_re + lambda`` to a relative 1e-12;
* (ii)  a flooded cell: ``dP = -A_p P_re`` to 1e-12 (``lambda = 0``);
* (iii) a cavitating cell: ``dP = (1 - A_p) lambda - A_p P_re``;
* (iv)  ``A_f = 0``: ``dP = -P_re``, ``lambda = 0``, ``Pst_fluid`` NaN and the no-fluid path
        counted.  A fluid region of a few elements converges without the Schur warning, with
        ``|H| <= 1e-12`` and the fluid mean to 1e-9;
* (v)   a roll of the film by one node and by L/4, with a pressure ramp: ``dP`` to a relative 1e-9;
* (vi)  +1 MPa on the interior of the patch, solved again: ``dP`` to a relative 1e-12;
* (vii) the pin: ``|w| <= 1e-10`` on the interior of the patch in steady cells;
* ``whole_cell`` stated and by default, and a full-film cell under either setting: bitwise equal;
* ``m_f`` is assembled again when ``chi`` changes; an unknown setting is refused;
* the extrema sampler leaves out every vertex with ``m_f = 0``.
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
#: ``level_constraint="whole_cell"`` is stated, as the default of ``parent_settings`` is
#: ``fluid_mean_v1``; the tests compare the whole cell with the fluid mean (``FM``)
SETTINGS = dict(relaxation_parameter=1.0, max_iterations=200, abs_tolerance=1e-11, rel_tolerance=1e-13,
                delta_h=0.0, eps_solid=1e-8, eps_smooth=1e-3, jacobian_reg_eps=-1e-7,
                pattern_flux_contact=False, level_constraint="whole_cell")
AH = 4e-7
N = 24


def make(params=None, settings=None, ah=AH, phi_x=0.0, phi_y=0.0, n=N, cls=None):
    """Mixed solver class (or ``cls``) on an ``n`` by ``n`` mesh, not yet solved.

    params, settings: overrides of ``PARAMS`` and ``SETTINGS``.  ah: cosine amplitude [m].
    phi_x, phi_y: cosine phases [rad].
    """
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
    """Runs the steady solve and the homogenisation.  Returns the solver."""
    s.run_steady()
    s.spatial_homogenisation()
    return s


def ledger(s):
    """Pressure ledger of the solver, with ``lam`` (lambda [Pa]), ``P_re`` [Pa], the areas ``A``
    and ``A_f``, ``A_p = 1 - A_f / A`` and ``dP = P_f - P_re`` [Pa] added."""
    r = dict(s.pressure_ledger)
    lam = float(s.dP)                                   # lambda [Pa]
    P_re = float(s.zeta.P)
    A = float(s._cell_area)
    from fenics import assemble, dx
    A_f = float(assemble(s.chi_fluid * dx(domain=s.mesh_m)))
    r.update(lam=lam, P_re=P_re, A_p=1.0 - A_f / A, dP=r["P_f"] - P_re, A_f=A_f, A=A)
    return r


def H_final(s):
    """Residual of the level constraint at the solved state."""
    return s._constraint_residual(s._constraint_vector(), float(s._cell_area))


def interior_nodes(s):
    """Indices of the nodes with ``m_f = 0``; empty when the solver has no ``m_f``."""
    return np.flatnonzero(s._lc_mf.get_local() == 0.0) if s._lc_mf is not None else np.zeros(0, int)


# ---- (i), (ii) ----
def test_i_ii_flooded_fluid_mean_and_dP():
    s = solve(make(settings=FM))
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and r["A_p"] > 0.0
    assert r["lam"] == 0.0 or abs(r["lam"]) < 1e-9 * r["P_re"]                 # flooded: no level shift from cavitation
    scale = abs(r["P_re"] + r["lam"])
    assert abs(r["Pst_fluid"] - (r["P_re"] + r["lam"])) <= 1e-12 * scale        # (i)
    assert abs(r["dP"] + r["A_p"] * r["P_re"] - (1 - r["A_p"]) * r["lam"]) <= 1e-12 * r["P_re"]   # (ii)
    assert abs(H_final(s)) <= 1e-11


def test_iii_cavitating_dP_identity():
    s = solve(make(params={"p0": 1.137147e1}, settings=FM))           # a low macro pressure: the cell cavitates
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and r["lam"] > 0.0 and r["A_p"] > 0.0
    pred = (1 - r["A_p"]) * r["lam"] - r["A_p"] * r["P_re"]
    assert abs(r["dP"] - pred) <= 1e-9 * max(abs(r["lam"]), 1.0)
    # mu = -beta lambda A / A_f with pattern_flux_contact off
    beta = float(base.BETA)                             # module constant
    lam_nd = float(s.lambda_nd)
    assert abs(s.mu_nd + beta * lam_nd * r["A"] / r["A_f"]) <= 1e-6 * max(abs(beta * lam_nd), 1e-12)


# ---- (iv) ----
def test_iv_no_fluid_bypass():
    s = solve(make(params={"H0": 2.5e-8}, ah=2e-8, settings=FM))      # h in [5, 45] nm < h_min = 50 nm
    r = ledger(s)
    assert r["A_f"] == 0.0 and s._lc_path == "no_fluid"
    assert s._n_lc_no_fluid >= 1 and s._n_lc_fluid_mean == 0
    assert float(s.lambda_nd) == 0.0 and r["P_f"] == 0.0
    assert r["dP"] == -r["P_re"] and math.isnan(r["Pst_fluid"])


def test_iv_small_fluid_region_converges(caplog):
    # the largest film is H0 + Ah, at the corner; h_min is 2 nm below it: a fluid region of a few elements
    H0, ah = 2.0e-7, 1e-7
    with caplog.at_level(logging.WARNING):
        s = solve(make(params={"H0": H0, "h_min": H0 + ah - 2e-9}, ah=ah, settings=FM))
    assert not [rec for rec in caplog.records if "Schur complement near-singular" in rec.getMessage()]
    r = ledger(s)
    assert s._lc_path == "fluid_mean" and 0.0 < r["A_f"] < 0.02 * r["A"]
    assert abs(H_final(s)) <= 1e-12
    assert abs(r["Pst_fluid"] - (r["P_re"] + r["lam"])) <= 1e-9 * abs(r["P_re"] + r["lam"])


# ---- (v) ----
@pytest.mark.parametrize("shift", ["one_node", "quarter"])
def test_v_rolls_leave_dP_invariant(shift):
    ramp = {"dpdx": 3e11, "dpdy": -1e11}                     # pressure gradient across the cell [Pa/m]
    base = ledger(solve(make(params=ramp, settings=FM)))
    dphi = 2 * math.pi / N if shift == "one_node" else math.pi / 2
    rolled = ledger(solve(make(params=ramp, settings=FM, phi_x=dphi)))
    assert abs(rolled["A_p"] - base["A_p"]) <= 1e-12
    assert abs(rolled["dP"] - base["dP"]) <= 1e-9 * max(abs(base["dP"]), 1e-9 * base["P_re"])


# ---- (vi), (vii) ----
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
    # the pin returns w to 0 there: the residual kappa m_i w_i of a pinned row is below abs_tolerance
    kappa_m = float(base.PATCH_PIN_KAPPA) * s._mean_vec.get_local()[idx]     # module constant
    assert np.all(np.abs(s.w.vector().get_local()[idx]) <= float(s.settings.abs_tolerance) / kappa_m)



# ---- whole cell and full film ----
def _outputs(s):
    """Homogenised outputs, ``lambda_nd``, ``mu_nd`` and the nodal ``w`` as one list."""
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
    over = {"H0": 5.5e-7}                                  # smallest film 150 nm > h_min: no patch
    a = make(params=over)
    a.run_steady()
    b = make(params=over, settings=FM)
    b.run_steady()
    assert b._n_lc_fluid_mean == 0 and b._n_lc_whole_cell >= 1
    assert _outputs(b) == _outputs(a)


# ---- m_f update and refusals ----
def test_mf_reassembled_when_chi_changes():
    s = solve(make(settings=FM))
    mf0, Af0 = s._lc_mf.get_local().copy(), s._lc_Af
    s.H0_const.assign(float(s.zeta.H) * 1.05)              # larger film: smaller patch
    s._reset_timers()
    s.solve()
    assert s._lc_Af > Af0
    assert not np.array_equal(s._lc_mf.get_local(), mf0)
    assert np.array_equal(s._mean_vec.get_local(), make()._mean_vec.get_local())   # the whole-cell mean vector is unchanged


def test_unknown_setting_refused():
    """An unknown ``level_constraint`` raises ``ValueError`` at the solve."""
    s = make(settings={"level_constraint": "fluid_mean_v2"})
    with pytest.raises(ValueError):
        s.run_steady()


def test_base_class_full_film_is_bitwise():
    """The full-film solver class (``h_min = 0``) takes the whole-cell path under either setting,
    with bitwise equal results."""
    from microcell.solver_base import MicroPhaseNeutralSolver

    outs = []
    for lc in ("whole_cell", "fluid_mean_v1"):
        s = make(params={"h_min": 0.0}, settings={"level_constraint": lc}, cls=MicroPhaseNeutralSolver)
        s.run_steady()
        assert s._n_lc_whole_cell >= 1 and s._n_lc_fluid_mean == 0
        outs.append([float(v) for v in s.spatial_homogenisation()] + [float(s.lambda_nd)]
                    + list(s.w.vector().get_local()))
    assert outs[0] == outs[1]


# ---- extrema sampler ----
def test_sampler_leaves_out_every_mf_zero_vertex():
    """Under the fluid mean, the extrema samples include no vertex with ``m_f = 0`` (pinned).  On
    the no-fluid path no sample is fluid."""
    from microcell.samplers import _vertex_maps, fluid_pressure_samples
    s = solve(make(settings=FM))
    assert s._lc_path == "fluid_mean"
    _, v_per, _ = _vertex_maps(s)
    zero = (s._lc_mf.get_local() == 0.0)[v_per]
    smp = fluid_pressure_samples(s)
    assert zero.any() and not np.any(smp.fluid & zero)
    off = solve(make())                                   # whole cell: the sampler uses its own fluid test
    assert getattr(off, "_lc_mf_zero", None) is None
    nf = solve(make(params={"H0": 2.5e-8}, ah=2e-8, settings=FM))
    assert nf._lc_path == "no_fluid" and not fluid_pressure_samples(nf).fluid.any()
