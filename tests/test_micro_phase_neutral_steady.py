"""
Stage 1 steady-state test suite for MicroPhaseNeutralSolver
============================================================

Copied in WP5 from the parent ``MultiscaleMixedBallinSocket`` at 2aaccad,
``tests/test_micro_phase_neutral_steady.py``.  Expected numbers and tolerances are the parent's.
Changed: the imports; the parameter and settings helpers return the parent's values by the parent's
names and ``_make_solver`` puts them into a ``Zeta`` and a ``MicroSettings``
(``tests/parent_fixtures.py``); ``rho0`` (never read) and ``k_spring`` (0; the spring is removed)
are not carried, ``beta`` (1.0) and ``print_newton`` (False) are fixed in this package.  The
full-film class takes no contact onset film here; the parent's ``h_min`` (0.05 or 0.1 um) was below
the minimum film of every cell in this file, so it masked nothing there either.

Test matrix (the parent's plan section 1.10), and what is here:
  T1  Flooded baseline     — non-cavitating case: lambda ≈ 0 and zero mean (the comparison with the
                             old pinned-BC solver ``MicroMixedSolver`` is left out: not in this package)
  T2  Cavitating baseline  — left out (compares with ``MicroMixedSolver``)
  T3  Phase sweep          — left out (``MicroMixedSolver``)
  T4  Phase-averaged match — left out (``MicroMixedSolver``)
  T5  Beta sensitivity     — left out (``beta`` is fixed at 1.0 in this package)
  T6  Mesh convergence     — Qx, Qy, Pst converge under refinement
  T7  Newton robustness    — no convergence failures across phase sweep
  T8  Zero-mean verify     — |mean(w_fluc)| < machine precision after solve
  T9  Symmetric gradient   — consistent in x and y directions
"""

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(ncells: int):
    """Return a UnitSquareMesh with ncells × ncells cells."""
    from fenics import UnitSquareMesh
    return UnitSquareMesh(ncells, ncells)


def _make_ideal_ht(Ah: float = 0.0, kx: float = 1.0, ky: float = 1.0,
                   phi_x: float = 0.0, phi_y: float = 0.0):
    """Canonical sinusoidal roughness film-thickness callback.

    Uses the same expression as ``microcell/ideal_ht.py``
    so results are directly comparable to existing tests.
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

    # Fallback: inline implementation
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
    """The parent's default ``PhaseNeutralSettings`` for the new solver, by the parent's names
    (``beta = 1.0`` and ``print_newton = False`` are fixed in this package).

    Closure: G = beta * lambda_nd - gamma_nd * mean(spos(-P_nd)) = 0.
    At equilibrium lambda = gamma * mean(spos) / beta ≥ 0.
    """
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
    """Physical parameters for a flooded (non-cavitating) case (the parent's names)."""
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
    """Physical parameters likely to produce cavitation (the parent's names)."""
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
    """``MicroPhaseNeutralSolver`` from the parent's parameter and settings values.  The parent's
    ``MicroPhysicalParameters.roughness_velocity_ratio`` default is 1.0."""
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
    """Instantiate and run MicroPhaseNeutralSolver; return the solver."""
    from fenics import UnitSquareMesh

    mesh = UnitSquareMesh(ncells, ncells)
    ht, phi_x_c = _make_ideal_ht(Ah=Ah, phi_x=phi_x)
    solver = _make_solver(params, settings, mesh, ht)
    solver.run_steady()
    return solver, phi_x_c


# ---------------------------------------------------------------------------
# T1: Flooded baseline
# ---------------------------------------------------------------------------

def test_flooded_baseline():
    """T1: Non-cavitating case — lambda ≈ 0 and zero mean.  (The parent also compared Qx, Qy with
    the old pinned-BC solver to <0.1%; that solver is not in this package.)"""
    params = _make_flooded_params(p0=5e8, Ux=0.5, Ah=0.0)
    new_s = _default_new_settings()

    new_solver, _ = _run_new_solver(params, new_s, ncells=20, Ah=0.0)

    new_res = new_solver.spatial_homogenisation()

    Qx_new, Qy_new, Pst_new = new_res[0], new_res[1], new_res[2]

    # lambda should be ≈ 0 for non-cavitating case
    lambda_dim = new_solver.dP
    assert abs(lambda_dim) < 0.01 * abs(params["p0"]), (
        f"Flooded lambda too large: {lambda_dim:.4e}  (p0={params['p0']:.4e})")

    # Mean of w_fluc should be ≈ 0 (zero-mean constraint)
    from fenics import assemble, dx, Constant
    area = float(assemble(Constant(1.0) * dx(domain=new_solver.mesh_m)))
    mean_w = float(new_solver._mean_vec.inner(new_solver.w.vector())) / area
    assert abs(mean_w) < 1e-10, f"w_fluc not zero-mean: mean = {mean_w:.4e}"

    print(f"\n[T1 Flooded] Qx: new={Qx_new:.4e}")
    print(f"[T1 Flooded] Pst: new={Pst_new:.4e}")
    print(f"[T1 Flooded] lambda_dim = {lambda_dim:.4e}  (should ≈ 0)")
    print(f"[T1 Flooded] mean(w_fluc) = {mean_w:.4e}  (should ≈ 0)")


# ---------------------------------------------------------------------------
# T6: Mesh convergence
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mesh_convergence():
    """T6: Qx, Qy, Pst converge under refinement."""
    params = _make_cavitating_params(p0=5e5, Ux=1.0, H0=1e-6, Ah=0.8e-6)
    new_s = _default_new_settings()
    ncells_list = [10, 20, 40]

    results = {}
    for n in ncells_list:
        solver, _ = _run_new_solver(params, new_s, ncells=n, Ah=0.8e-6)
        res = solver.spatial_homogenisation()
        results[n] = {"Qx": res[0], "Qy": res[1], "Pst": res[2]}

    # Compare coarse vs fine: each quantity should be within 20% (loose bound)
    # (tight convergence needs larger meshes which are slow)
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


# ---------------------------------------------------------------------------
# T7: Newton robustness across phase sweep
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_newton_robustness_phase_sweep():
    """T7: No convergence failures across phase sweep; iteration count stable."""
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


# ---------------------------------------------------------------------------
# T8: Zero-mean verification
# ---------------------------------------------------------------------------

def test_zero_mean_w_fluc():
    """T8: After solve, |mean(w_fluc)| < 1e-10 (Lagrange multiplier enforcement)."""
    params = _make_cavitating_params(p0=5e5, Ux=1.0, H0=1e-6, Ah=0.8e-6)
    new_s = _default_new_settings()

    from fenics import assemble, dx, Constant
    solver, _ = _run_new_solver(params, new_s, ncells=20, Ah=0.8e-6, phi_x=0.5)
    area = float(assemble(Constant(1.0) * dx(domain=solver.mesh_m)))
    mean_w = float(solver._mean_vec.inner(solver.w.vector())) / area

    print(f"\n[T8 Zero-mean] mean(w_fluc) = {mean_w:.4e}  (target: < 1e-10)")
    assert abs(mean_w) < 1e-8, (
        f"Zero-mean constraint not satisfied: mean(w_fluc) = {mean_w:.4e}")


# ---------------------------------------------------------------------------
# T9: Symmetric gradient
# ---------------------------------------------------------------------------

def test_symmetric_gradient():
    """T9: Results are consistent with dpdx > 0 vs dpdy > 0 (by symmetry, Pst ≈ same)."""
    H0 = 1e-6
    Ah = 0.6e-6
    # params with gradient in x direction
    params_x = dict(
        Ux=0.5, Uy=0.0,
        eta0=0.01,
        penalty_gamma=1e14,
        xmax=80e-6, ymax=80e-6,
        p0=5e5, dpdx=1e8, dpdy=0.0,
        H0=H0, h_min=0.05e-6,
        HT=0.0, PT=0.0, Tend=0.0,
    )
    # params with gradient in y direction (swap dpdx/dpdy and Ux/Uy)
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

    # Both should solve without errors and produce physically similar Pst
    ref = max(abs(Pst_x), abs(Pst_y), 1e-6)
    rel_diff = abs(Pst_x - Pst_y) / ref
    # No strict tolerance here (x and y are not exactly equivalent when Ux/Uy differ),
    # but both should be in the same ballpark
    assert rel_diff < 2.0, (
        f"Pst too different between dpdx and dpdy cases: x={Pst_x:.4e}  y={Pst_y:.4e}")


# ---------------------------------------------------------------------------
# Quick smoke test (always runs, no @pytest.mark.slow)
# ---------------------------------------------------------------------------

def test_import_and_instantiate():
    """Smoke test: class imports and instantiates without error on a tiny mesh."""
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

    # dP property
    dp = solver.dP
    assert isinstance(dp, float), f"dP should be float, got {type(dp)}"

    print(f"\n[Smoke] Solver ran successfully. dP = {dp:.4e}")
