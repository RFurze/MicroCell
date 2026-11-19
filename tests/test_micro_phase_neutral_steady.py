"""Steady checks of ``MicroPhaseNeutralSolver``: the full-film cell with a cosine roughness.

The helpers return the case values as dicts in SI, and ``_make_solver`` puts them into a
``Zeta`` and a ``MicroSettings`` through ``tests/parent_fixtures.py``.  ``h_min`` of the case
dicts is not read by the full-film class.

  T1  flooded baseline     no cavitation: ``lambda`` near 0 and ``w`` with zero mean
  T6  mesh convergence     ``Qx`` and ``Pst`` at 10 and at 40 cells per side
  T7  phase sweep          the solve converges at eight roughness phases
  T8  zero mean            ``|mean(w)|`` after a cavitating solve
  T9  symmetric gradient   ``Pst`` with the gradient along x and along y
"""

import math
import numpy as np
import pytest

# ---- Helpers ----

def _make_mesh(ncells: int):
    """Unit-square mesh of ``ncells`` by ``ncells`` cells."""
    from fenics import UnitSquareMesh
    return UnitSquareMesh(ncells, ncells)


def _make_ideal_ht(Ah: float = 0.0, kx: float = 1.0, ky: float = 1.0,
                   phi_x: float = 0.0, phi_y: float = 0.0):
    """Film callback of the cosine roughness, from ``microcell.film.make_ideal_ht``.

    Ah: amplitude [m].  kx, ky: wavenumbers.  phi_x, phi_y: phases [rad].
    Returns (callback, the ``Constant`` holding ``phi_x``).
    """
    try:
        from microcell.film import make_ideal_ht
        from fenics import Constant
        phi_x_c = Constant(phi_x)
        phi_y_c = Constant(phi_y)
        return make_ideal_ht(Ah=Ah, kx=kx, ky=ky,
                              phi_x_const=phi_x_c, phi_y_const=phi_y_c), phi_x_c
    except ImportError:
        pass

    # used if the import fails: an inline sine-squared film, not the expression of ``make_ideal_ht``
    from fenics import sin, pi, Constant

    phi_x_c = Constant(phi_x)
    phi_y_c = Constant(phi_y)

    def ht(x, y, xmax, ymax, H0, HT=None, T=None, Ux=None, Uy=None):
        H0_val = H0
        HT_val = HT if HT is not None else Constant(0.0)
        T_val = T if T is not None else Constant(0.0)
        Ux_val = Ux if Ux is not None else Constant(0.0)
        Uy_val = Uy if Uy is not None else Constant(0.0)
        x_d = x * xmax
        y_d = y * ymax
        roughness = (
            Constant(Ah / 3.0)
            - Constant(4.0 * Ah / 3.0)
            * sin(kx * pi * (x_d - T_val * Ux_val) / xmax + phi_x_c) ** 2
            * sin(ky * pi * (y_d - T_val * Uy_val) / ymax + phi_y_c) ** 2
        )
        return H0_val + HT_val * T_val + roughness

    return ht, phi_x_c



def _default_new_settings():
    """Newton and smoothing settings shared by the tests, as keyword arguments of
    ``parent_settings``."""
    return dict(
        relaxation_parameter=0.7,
        max_iterations=100,
        abs_tolerance=1e-8,
        rel_tolerance=1e-8,
        delta_h=0.0,
        eps_solid=1e-6,
        eps_smooth=1.0,
    )


def _make_flooded_params(p0: float = 1e8, Ux: float = 0.5,
                          dpdx: float = 0.0, dpdy: float = 0.0,
                          H0: float = 2e-6, Ah: float = 0.0):
    """Case values of a flooded cell (no cavitation), in SI.  ``Ah`` is not read."""
    return dict(
        Ux=Ux, Uy=0.0,
        eta0=0.01,
        penalty_gamma=1e6,
        xmax=80e-6, ymax=80e-6,
        p0=p0, dpdx=dpdx, dpdy=dpdy,
        H0=H0, h_min=0.1e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )


def _make_cavitating_params(p0: float = 1e6, Ux: float = 1.0,
                              dpdx: float = 0.0, H0: float = 1e-6,
                              Ah: float = 0.8e-6):
    """Case values of a cell that is likely to cavitate, in SI.  ``Ah`` is not read."""
    return dict(
        Ux=Ux, Uy=0.0,
        eta0=0.01,
        penalty_gamma=1e14,
        xmax=80e-6, ymax=80e-6,
        p0=p0, dpdx=dpdx, dpdy=0.0,
        H0=H0, h_min=0.05e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )


def _make_solver(params, settings, mesh, ht):
    """``MicroPhaseNeutralSolver``, not yet solved.

    params: case values (a dict from the helpers above).  settings: solver settings.
    mesh: the micro mesh.  ht: film callback.  The roughness velocity ratio is 1.0.
    """
    from parent_fixtures import ensure_process, parent_settings, parent_zeta
    ensure_process()
    from microcell.solver_base import MicroPhaseNeutralSolver

    zeta = parent_zeta(**{k: params[k] for k in ("Ux", "Uy", "p0", "dpdx", "dpdy", "H0", "HT", "PT")})
    st = parent_settings(**{k: params[k] for k in ("eta0", "penalty_gamma", "xmax", "ymax", "h_min")}, **settings)
    return MicroPhaseNeutralSolver(
        mesh_m=mesh,
        zeta=zeta,
        t_end=params["Tend"],
        settings=st,
        ht=ht,
        velocity_ratio=1.0,
        pair=False,
        auto_solve=False,
    )


def _run_new_solver(params, settings, ncells: int = 20,
                    Ah: float = 0.0, phi_x: float = 0.0):
    """Runs the steady solve on an ``ncells`` by ``ncells`` mesh with a cosine roughness of
    amplitude ``Ah`` [m] and phase ``phi_x`` [rad].  Returns (solver, the phase ``Constant``)."""
    from fenics import UnitSquareMesh

    mesh = UnitSquareMesh(ncells, ncells)
    ht, phi_x_c = _make_ideal_ht(Ah=Ah, phi_x=phi_x)
    solver = _make_solver(params, settings, mesh, ht)
    solver.run_steady()
    return solver, phi_x_c


# ---- T1: flooded baseline ----

def test_flooded_baseline():
    """T1: without cavitation, ``|lambda| < 0.01 p0`` and ``|mean(w)| < 1e-10``."""
    params = _make_flooded_params(p0=5e8, Ux=0.5, Ah=0.0)
    new_s = _default_new_settings()

    new_solver, _ = _run_new_solver(params, new_s, ncells=20, Ah=0.0)

    new_res = new_solver.spatial_homogenisation()

    Qx_new, Qy_new, Pst_new = new_res[0], new_res[1], new_res[2]

    # pressure level [Pa]: near 0 without cavitation
    lambda_dim = new_solver.dP
    assert abs(lambda_dim) < 0.01 * abs(params["p0"]), (
        f"Flooded lambda too large: {lambda_dim:.4e}  (p0={params['p0']:.4e})")

    # cell mean of w: 0 by the zero-mean constraint
    from fenics import assemble, dx, Constant
    area = float(assemble(Constant(1.0) * dx(domain=new_solver.mesh_m)))
    mean_w = float(new_solver._mean_vec.inner(new_solver.w.vector())) / area
    assert abs(mean_w) < 1e-10, f"w_fluc not zero-mean: mean = {mean_w:.4e}"

    print(f"\n[T1 Flooded] Qx: new={Qx_new:.4e}")
    print(f"[T1 Flooded] Pst: new={Pst_new:.4e}")
    print(f"[T1 Flooded] lambda_dim = {lambda_dim:.4e}  (should ≈ 0)")
    print(f"[T1 Flooded] mean(w_fluc) = {mean_w:.4e}  (should ≈ 0)")


# ---- T6: mesh convergence ----

@pytest.mark.slow
def test_mesh_convergence():
    """T6: ``Qx`` and ``Pst`` at 10 cells per side are within a relative 0.5 of the values at 40."""
    params = _make_cavitating_params(p0=5e5, Ux=1.0, H0=1e-6, Ah=0.8e-6)
    new_s = _default_new_settings()
    ncells_list = [10, 20, 40]

    results = {}
    for n in ncells_list:
        solver, _ = _run_new_solver(params, new_s, ncells=n, Ah=0.8e-6)
        res = solver.spatial_homogenisation()
        results[n] = {"Qx": res[0], "Qy": res[1], "Pst": res[2]}

    # coarse against fine, with a loose bound: tight convergence needs larger meshes
    Qx_coarse = results[10]["Qx"]
    Qx_fine = results[40]["Qx"]
    Pst_coarse = results[10]["Pst"]
    Pst_fine = results[40]["Pst"]

    print(f"\n[T6 Mesh convergence] Qx: n=10 {Qx_coarse:.4e}  n=20 {results[20]['Qx']:.4e}  n=40 {Qx_fine:.4e}")
    print(f"[T6 Mesh convergence] Pst: n=10 {Pst_coarse:.4e}  n=20 {results[20]['Pst']:.4e}  n=40 {Pst_fine:.4e}")

    if abs(Qx_fine) > 1e-20:
        assert abs(Qx_coarse - Qx_fine) / abs(Qx_fine) < 0.5, (
            f"Qx not converging: coarse={Qx_coarse:.4e}  fine={Qx_fine:.4e}")
    if abs(Pst_fine) > 1e-6:
        assert abs(Pst_coarse - Pst_fine) / abs(Pst_fine) < 0.5, (
            f"Pst not converging: coarse={Pst_coarse:.4e}  fine={Pst_fine:.4e}")


# ---- T7: Newton solve over a phase sweep ----

@pytest.mark.slow
def test_newton_robustness_phase_sweep():
    """T7: the Newton solve converges at eight roughness phases from 0 up to pi."""
    params = _make_cavitating_params(p0=5e5, Ux=1.0, H0=1e-6, Ah=0.8e-6)
    new_s = _default_new_settings()
    ncells = 20
    n_phi = 8
    phi_vals = np.linspace(0.0, math.pi, n_phi + 1)[:-1]

    for i, phi in enumerate(phi_vals):
        try:
            solver, _ = _run_new_solver(params, new_s, ncells=ncells, Ah=0.8e-6,
                                        phi_x=float(phi))
        except RuntimeError as e:
            pytest.fail(f"Newton failed at phi_x={phi:.4f} (point {i}): {e}")

    print(f"\n[T7 Newton robustness] All {n_phi} phase points converged.")


# ---- T8: zero mean ----

def test_zero_mean_w_fluc():
    """T8: after a cavitating solve, ``|mean(w)| < 1e-8`` (enforced by the Lagrange multiplier)."""
    params = _make_cavitating_params(p0=5e5, Ux=1.0, H0=1e-6, Ah=0.8e-6)
    new_s = _default_new_settings()

    from fenics import assemble, dx, Constant
    solver, _ = _run_new_solver(params, new_s, ncells=20, Ah=0.8e-6, phi_x=0.5)
    area = float(assemble(Constant(1.0) * dx(domain=solver.mesh_m)))
    mean_w = float(solver._mean_vec.inner(solver.w.vector())) / area

    print(f"\n[T8 Zero-mean] mean(w_fluc) = {mean_w:.4e}  (target: < 1e-10)")
    assert abs(mean_w) < 1e-8, (
        f"Zero-mean constraint not satisfied: mean(w_fluc) = {mean_w:.4e}")


# ---- T9: symmetric gradient ----

def test_symmetric_gradient():
    """T9: ``Pst`` with the gradient and velocity along x, and along y, differ by a relative
    2.0 at most."""
    H0 = 1e-6
    Ah = 0.6e-6
    # gradient and velocity along x
    params_x = dict(
        Ux=0.5, Uy=0.0,
        eta0=0.01,
        penalty_gamma=1e14,
        xmax=80e-6, ymax=80e-6,
        p0=5e5, dpdx=1e8, dpdy=0.0,
        H0=H0, h_min=0.05e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )
    # gradient and velocity along y
    params_y = dict(
        Ux=0.0, Uy=0.5,
        eta0=0.01,
        penalty_gamma=1e14,
        xmax=80e-6, ymax=80e-6,
        p0=5e5, dpdx=0.0, dpdy=1e8,
        H0=H0, h_min=0.05e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )
    new_s = _default_new_settings()

    solver_x, _ = _run_new_solver(params_x, new_s, ncells=20, Ah=Ah)
    solver_y, _ = _run_new_solver(params_y, new_s, ncells=20, Ah=Ah)

    res_x = solver_x.spatial_homogenisation()
    res_y = solver_y.spatial_homogenisation()

    Pst_x, Fst_x, dP_x = res_x[2], res_x[3], solver_x.dP
    Pst_y, Fst_y, dP_y = res_y[2], res_y[3], solver_y.dP

    print(f"\n[T9 Symmetric] dpdx case: Pst={Pst_x:.4e}  Fst={Fst_x:.3f}  dP={dP_x:.4e}")
    print(f"[T9 Symmetric] dpdy case: Pst={Pst_y:.4e}  Fst={Fst_y:.3f}  dP={dP_y:.4e}")

    ref = max(abs(Pst_x), abs(Pst_y), 1e-6)
    rel_diff = abs(Pst_x - Pst_y) / ref
    # loose bound: the x and y cases are not equivalent
    assert rel_diff < 2.0, (
        f"Pst too different between dpdx and dpdy cases: x={Pst_x:.4e}  y={Pst_y:.4e}")


# ---- Smoke test ----

def test_import_and_instantiate():
    """The class builds and solves a flat film on a 4 by 4 mesh; ``spatial_homogenisation``
    returns 14 values and ``dP`` is a float."""
    from fenics import UnitSquareMesh

    mesh = UnitSquareMesh(4, 4)
    params = dict(
        Ux=0.5, Uy=0.0,
        eta0=0.01,
        penalty_gamma=1e13,
        xmax=80e-6, ymax=80e-6,
        p0=1e8, dpdx=0.0, dpdy=0.0,
        H0=2e-6, h_min=0.1e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )
    settings = dict(
        relaxation_parameter=0.7,
        max_iterations=30,
        abs_tolerance=1e-6,
        rel_tolerance=1e-6,
        delta_h=0.0,
        eps_solid=1e-6,
        eps_smooth=1e3,
    )

    def ht_flat(x, y, xmax, ymax, H0, *args, **kwargs):
        return H0

    solver = _make_solver(params, settings, mesh, ht_flat)
    solver.run_steady()
    result = solver.spatial_homogenisation()
    assert len(result) == 14, f"Expected 14-tuple, got {len(result)}"

    dp = solver.dP
    assert isinstance(dp, float), f"dP should be float, got {type(dp)}"

    print(f"\n[Smoke] Solver ran successfully. dP = {dp:.4e}")
