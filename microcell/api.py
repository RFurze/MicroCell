"""``microsolver()`` -- one micro cell, one call.

Written (WP1) from the body of the parent's worker ``run_micro_task_phase_neutral``
(``scripts/run/run_microscale_phase_neutral.py`` at 2aaccad): the film callback, the
physical parameters, the settings, the run, the homogenisation, the asperity load, the
Reynolds reference flux and the corrections dQ and dP.  Left out, by the plan's decisions:
the per-worker solver cache, the two mesh tiers and their predictor (D9), the contact
table and its composition, xwarm, dphi sampling (D4) and every environment override (D10).

Conventions
-----------
* SI units throughout.
* ``Zeta.Ux, Zeta.Uy`` are the mean (entrainment) velocities of the macro task; the cell's
  surfaces slide at ``2 Ux`` (D8).
* Transient (``steady=False``): the cell is solved over the window that ends at ``t_end``;
  the last ``n_steps`` CFL steps are solved and the outputs are the end state (D3).
  ``zeta`` is the state at t = 0 of that window; the end state is ``zeta + rate * t_end``
  (e.g. ``Pst`` is returned at ``P + Pdot * t_end``), as in the parent.
* Steady (``steady=True``): one steady solve at ``(H, P, Ux, Uy, dpdx, dpdy)``; the rates of
  ``zeta`` and ``t_end`` are not used.
* A stationary + translating pair takes one ``offset`` [m] per call, the translating
  surface's starting displacement; the caller loops over offsets to average (D4).
* A failed solve returns a result with a non-OK ``status`` and NaN outputs (D11).

``MicroSettings`` is the one definition and the one default of every setting (WP4): the solver
classes read it, and the ``Zeta`` of the call, directly.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from microcell.contact import FFT as CONTACT_FFT     # the one contact law (no FEniCS import)

OK = "OK"
#: ``contact_model`` of a result: the one contact law (``CONTACT_FFT``), or "off" for a full-film cell
#: (the parent's result carried its default tag ``legacy_patch_v1`` there; WP2 dropped that law)
CONTACT_OFF = "off"
FAILED_NEWTON = "FAILED_NEWTON"
FAILED_RIPPLE = "FAILED_RIPPLE"
FAILED_CONTACT = "FAILED_CONTACT"
STATUSES = (OK, FAILED_NEWTON, FAILED_RIPPLE, FAILED_CONTACT)

_NAN = float("nan")


# ----------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Zeta:
    """The macro state the cell is solved at (SI).  ``Ux, Uy``: mean (entrainment) velocities."""
    H: float                    # film thickness [m]
    P: float                    # pressure [Pa]
    Ux: float = 0.0             # [m/s]
    Uy: float = 0.0
    dpdx: float = 0.0           # pressure gradient [Pa/m]
    dpdy: float = 0.0
    Hdot: float = 0.0           # [m/s]
    Pdot: float = 0.0           # [Pa/s]
    Uxdot: float = 0.0          # [m/s^2]
    Uydot: float = 0.0
    dpdx_dot: float = 0.0       # [Pa/m/s]
    dpdy_dot: float = 0.0

    @classmethod
    def from_task(cls, task) -> "Zeta":
        """From the parent's task tuple (``micro_task_helpers._parse_task``): 13 elements
        (steady) or 15 / 16 / 18 / 20+ (transient).  ``task[15]`` is the task's time window
        (``t_end``), which the caller passes to ``microsolver`` itself."""
        t = tuple(task)
        if len(t) < 13:
            raise ValueError(f"a task has at least 13 elements, got {len(t)}")
        H, P, Ux, Uy = (float(v) for v in t[2:6])
        gx, gy = float(t[7]), float(t[8])
        Hdot = float(t[13]) if len(t) >= 15 else 0.0
        Pdot = float(t[14]) if len(t) >= 15 else 0.0
        UxT = float(t[16]) if len(t) >= 18 else 0.0
        UyT = float(t[17]) if len(t) >= 18 else 0.0
        gxd = float(t[18]) if len(t) >= 20 else 0.0
        gyd = float(t[19]) if len(t) >= 20 else 0.0
        return cls(H=H, P=P, Ux=Ux, Uy=Uy, dpdx=gx, dpdy=gy, Hdot=Hdot, Pdot=Pdot,
                   Uxdot=UxT, Uydot=UyT, dpdx_dot=gxd, dpdy_dot=gyd)


@dataclass(frozen=True)
class ContactSettings:
    """Elastic asperity contact (``fft_elastic_v1``).  ``MicroSettings.contact = None`` is the
    full-film mode.  The modulus is ``MicroSettings.E_star``."""
    tol: float = 1e-11                      # contact solve tolerance (1e-11 for h_min near 1e-11 m)
    maxit: int = 500
    sigma_yield: float = 250e6              # [Pa]
    mu: float = 0.8                         # boundary friction coefficient of the asperity traction
    k_asp_mode: str = "closure"             # "closure" | "off"


@dataclass(frozen=True)
class RippleSettings:
    """Fluid-pressure ripple deformation (``ripple_halfspace``).  ``MicroSettings.ripple = None``
    is off.  The modulus is ``MicroSettings.E_star``."""
    tol: float = 1e-6
    maxpass: int = 12
    omega: float = 1.0


@dataclass
class MicroSettings:
    """Every setting of a cell solve, each defined here once.  The defaults are the parent's
    ``GenSa40-ripple`` case (``cases/mixed_transfull_cup90_genrough_ripple.py`` at 2aaccad)."""
    mesh_n: int = 100                       # the cell mesh is UnitSquareMesh(mesh_n, mesh_n)
    xmax: float = 5e-6                      # cell size [m]
    ymax: float = 5e-6
    eta0: float = 1e-3                      # viscosity [Pa s] (isoviscous)
    penalty_gamma: float = 1e10             # cavitation penalty
    h_min: float = 1.0331e-11               # contact onset film [m] (used when contact is on)
    E_star: float = 203775203775.2038       # reduced two-body modulus [Pa]: the contact law and the ripple
    delta_h: float = 0.0                    # width of the contact mask transition [m] (0: sharp)
    eps_solid: float = 1e-8                 # mobility factor inside contact
    eps_smooth: float = 1e-3                # smoothing of the cavitation switch [Pa]
    spos_form: str = "sqrt"                 # "sqrt" | "spline"
    level_constraint: str = "whole_cell"    # "whole_cell" | "fluid_mean_v1"
    pattern_flux_contact: bool = True
    jacobian_reg_eps: float = -1e-7
    # Newton tolerances and damping
    relaxation_parameter: float = 1.0
    max_iterations: int = 300
    abs_tolerance: float = 1e-6
    rel_tolerance: float = 1e-6
    contact: Optional[ContactSettings] = field(default_factory=ContactSettings)
    ripple: Optional[RippleSettings] = field(default_factory=RippleSettings)
    cfl_target: float = 1.0
    roughness_velocity_ratio: Optional[float] = None    # None: taken from the roughness

    def check(self) -> None:
        if self.spos_form not in ("sqrt", "spline"):
            raise ValueError(f"spos_form must be 'sqrt' or 'spline', got {self.spos_form!r}")
        if self.level_constraint not in ("whole_cell", "fluid_mean_v1"):
            raise ValueError(f"level_constraint must be 'whole_cell' or 'fluid_mean_v1', got {self.level_constraint!r}")
        if int(self.mesh_n) < 2:
            raise ValueError("mesh_n must be at least 2")
        # every value the solver classes refuse (they keep their own refusals; these come first)
        if not np.isfinite(float(self.jacobian_reg_eps)):
            raise ValueError("jacobian_reg_eps must be finite; zero disables the shift")
        for name in ("delta_h", "eps_solid", "eps_smooth"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and not negative, got {getattr(self, name)!r}")
        r, c = self.ripple, self.contact
        if r is not None or c is not None:
            if self.E_star is None or not np.isfinite(float(self.E_star)) or float(self.E_star) <= 0.0:
                raise ValueError("contact and the ripple need a positive, finite E_star (the two-body modulus)")
        if r is not None:
            if not (float(r.tol) > 0.0 and int(r.maxpass) >= 1 and 0.0 < float(r.omega) <= 1.0):
                raise ValueError("ripple settings: need tol > 0, maxpass >= 1 and 0 < omega <= 1; got "
                                 f"{r.tol}, {r.maxpass}, {r.omega}")
        if c is not None:
            if c.k_asp_mode not in ("off", "closure"):
                raise ValueError(f"contact.k_asp_mode must be 'off' or 'closure', got {c.k_asp_mode!r}")
            if not (np.isfinite(float(self.h_min)) and float(self.h_min) > 0.0):
                raise ValueError(f"contact needs a positive contact onset film h_min, got {self.h_min!r}")


# ----------------------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------------------
@dataclass
class MicroResult:
    """One cell's outputs (SI).  ``status`` is ``"OK"`` or a ``FAILED_*`` code with ``reason``;
    every homogenised output of a failed cell is NaN and it has no fields.  The asperity and total
    shear means of a full-film cell are NaN with ``info["shear_asperity_status"] == "unavailable"``.

    Each output is in one place: the homogenised outputs below, the arrays in ``fields``, and
    everything else in ``info`` under the parent's ``MicroTaskResult`` name -- the statuses and
    formulation tags, the counters, the pressure ledger, the ripple and contact numbers, the
    sampled shear statistics, the call (``mesh_n``, ``steady``, ``t_end``, ``n_steps``, ``offset``,
    ``roughness_velocity_ratio``, ``t_start``, ``dt``), ``wall_time`` and the field diagnostics
    (``info["fields"]``).  An OK and a failed result carry the same ``info`` entries."""
    status: str
    reason: str
    dQ: np.ndarray              # (2,) Q - Q_reynolds [m^2/s]
    dP: float                   # P_f - P_reynolds [Pa]
    Q: np.ndarray               # (2,) homogenised flux [m^2/s]
    Pst: float                  # whole-cell pressure diagnostic [Pa]
    P_f: float                  # signed fluid pressure over the cell area [Pa]
    Fst: float
    tau_fluid: np.ndarray       # (2,) homogenised fluid shear stress [Pa]
    tau_asperity: np.ndarray    # (2,) sampled asperity shear mean, local frame [Pa]
    tau_total: np.ndarray       # (2,) sampled fluid + asperity shear mean, local frame [Pa]
    P_asp: float                # asperity load per cell area [Pa]
    contact_frac: float
    A_mech_frac: float
    k_asp: float                # closure tangent -dP_asp/dH [Pa/m]
    p_max: float                # fluid pressure extrema [Pa]
    p_min: float
    h_max: float                # film extrema [m]
    h_min: float
    yield_flag: int
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK


# ----------------------------------------------------------------------------------------
# Process set-up: before the first form is built
# ----------------------------------------------------------------------------------------
_PROCESS_READY = False


def _ensure_process() -> None:
    """What the parent's pool worker applies before its first task (``_init_worker``): the
    macro process settings (PETSc options, form-compiler flags), then the micro quadrature
    degree pinned to 2.  Results differ without them."""
    global _PROCESS_READY
    if _PROCESS_READY:
        return
    from microcell import form_compiler, process_settings
    process_settings.apply_macro_process_settings()
    form_compiler.pin_micro_worker("microcell.api")
    _PROCESS_READY = True


# ----------------------------------------------------------------------------------------
# Helpers ported from the worker
# ----------------------------------------------------------------------------------------
def reynolds_flux(steady: bool, zeta: Zeta, t_end: float, eta0: float) -> np.ndarray:
    """The smooth-film Reynolds flux at the end state, shape (1, 2)
    (``micro_task_helpers.q_re`` with the constant viscosity ``eta0``)."""
    H, P, Ux, Uy, gx, gy = zeta.H, zeta.P, zeta.Ux, zeta.Uy, zeta.dpdx, zeta.dpdy
    if not steady:
        dt = t_end
        Hdt = H + zeta.Hdot * dt
        Uxdt = Ux + zeta.Uxdot * dt
        Uydt = Uy + zeta.Uydot * dt
        gx_dt = gx + zeta.dpdx_dot * dt
        gy_dt = gy + zeta.dpdy_dot * dt
    else:
        Hdt, Uxdt, Uydt, gx_dt, gy_dt = H, Ux, Uy, gx, gy
    eta = float(eta0)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta0 must be finite and positive")
    qx = Uxdt * Hdt - (Hdt**3) * gx_dt / (12 * eta)
    qy = Uydt * Hdt - (Hdt**3) * gy_dt / (12 * eta)
    return np.column_stack((qx, qy))


#: ``info`` numbers of the solve: NaN in a failed result
_SOLVE_NUMBERS = ("Pst_fluid", "P_patch", "P_fluid_pos", "P_fluid_neg", "fmax", "fmin", "lambda_nd", "spos_mean",
                  "A_patch_frac", "n_patches", "pasp_patch_max", "contact_residual", "contact_resolution")
#: ``info`` numbers of the ripple: 0.0 without the ripple, NaN in a failed result with it
_RIPPLE_NUMBERS = ("ripple_max", "ripple_resid", "ripple_mean_resid", "ripple_seam_mismatch", "ripple_proj_err")


def _vec2(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1)[:2].copy()


def _full_film_contact_shear() -> dict:
    """The asperity and total shear entries of a full-film cell: no contact law ran, so every
    statistic is NaN with status ``unavailable`` (the parent's ``MicroTaskResult`` defaults)."""
    from microcell import contact_shear as _cshear

    out = {k: _NAN for k in _cshear.FIELDS}
    out.update(shear_asperity_mean_local_Pa=(_NAN, _NAN), shear_total_mean_local_Pa=(_NAN, _NAN),
               shear_asperity_status=_cshear.UNAVAILABLE, shear_total_status=_cshear.UNAVAILABLE,
               shear_contact_definition="unversioned", shear_contact_n_samples=0)
    return out


def _failed_result(status: str, reason: str, info: dict, mixed: bool, ripple_on: bool) -> MicroResult:
    """The one place a failed result is built (the worker's ``_nan_return``).

    Every named output is NaN (``yield_flag`` 0) and there are no fields.  ``info`` is the
    caller's outcome-independent entries plus the entries an OK result adds, with NaN numbers
    and ``failed`` statuses; the asperity and total shear of a full-film cell stay ``unavailable``."""
    from microcell import contact_shear as _cshear
    from microcell.fluid_shear import failed_shear_fields
    from microcell.pressure import EXTREMA_FAILED

    info = dict(info)
    info.update({k: _NAN for k in _SOLVE_NUMBERS})
    info.update({k: (_NAN if ripple_on else 0.0) for k in _RIPPLE_NUMBERS})
    info.update(contact_geometry_valid=0.0, pressure_extrema_status=EXTREMA_FAILED, p_fluid_n_samples=0,
                fields={})
    info.update(failed_shear_fields())
    contact_shear = _cshear.failed_contact_shear_fields() if mixed else _full_film_contact_shear()
    tau_asperity = _vec2(contact_shear.pop("shear_asperity_mean_local_Pa"))
    tau_total = _vec2(contact_shear.pop("shear_total_mean_local_Pa"))
    info.update(contact_shear)
    nan2 = np.array([_NAN, _NAN])
    return MicroResult(
        status=status, reason=reason, dQ=nan2.copy(), dP=_NAN, Q=nan2.copy(), Pst=_NAN, P_f=_NAN, Fst=_NAN,
        tau_fluid=nan2.copy(), tau_asperity=tau_asperity, tau_total=tau_total,
        P_asp=_NAN, contact_frac=_NAN, A_mech_frac=_NAN, k_asp=_NAN, p_max=_NAN, p_min=_NAN,
        h_max=_NAN, h_min=_NAN, yield_flag=0, fields={}, info=info)


# ----------------------------------------------------------------------------------------
# The call
# ----------------------------------------------------------------------------------------
def microsolver(zeta: Zeta, t_end: Optional[float] = None, roughness=None,
                settings: Optional[MicroSettings] = None, n_steps: int = 3,
                offset: Tuple[float, float] = (0.0, 0.0), steady: bool = False,
                save_paraview: bool = False, output_dir=None) -> MicroResult:
    """Solve one micro cell and return its homogenised outputs and fields.

    zeta           the macro state (:class:`Zeta`)
    t_end          the time window [s] of a transient solve (> 0); not used when ``steady``
    roughness      a :class:`microcell.Roughness`
    settings       :class:`MicroSettings` (default: ``MicroSettings()``)
    n_steps        transient: the number of CFL steps solved, ending at ``t_end``
    offset         ``(o_x, o_y)`` [m], the translating surface's starting displacement; pairs only
    steady         True: one steady solve; False: the transient window
    save_paraview  write ``<output_dir>/cell.vts`` and ``cell.xdmf`` (+ ``.h5``) of the end state
    output_dir     where to write them (default ``./microcell_output``)
    """
    from microcell.roughness.frontend import Roughness

    start = time.time()
    st = settings if settings is not None else MicroSettings()
    st.check()
    if not isinstance(roughness, Roughness):
        raise TypeError("roughness must be a microcell.Roughness")
    steady = bool(steady)
    if steady:
        tend = 0.0
    else:
        if t_end is None or not float(t_end) > 0.0:
            raise ValueError("a transient solve needs t_end > 0 (pass steady=True for a steady solve)")
        tend = float(t_end)
        if n_steps is None:
            raise NotImplementedError("n_steps=None (march from t = 0, plan D3) is not implemented at WP1")
        if int(n_steps) < 1:
            raise ValueError("n_steps must be at least 1")
        n_steps = int(n_steps)
    offset = (float(offset[0]), float(offset[1]))
    pair = roughness.is_pair
    if offset != (0.0, 0.0) and not pair:
        raise ValueError("an offset is the translating surface's starting displacement of a stationary + "
                         "translating pair; this roughness is not a pair")
    mixed = st.contact is not None
    ripple_on = st.ripple is not None

    _ensure_process()
    from dolfin import Constant, UnitSquareMesh, assemble, dx

    from microcell import contact_shear as _cshear
    from microcell import fields as _fields
    from microcell import ripple as _ripple
    from microcell.contact import ContactResolutionError
    from microcell.fluid_shear import shear_fields, union_fluid_traction
    from microcell.ideal_ht import make_ideal_ht
    from microcell.micro_phase_neutral_class import MicroPhaseNeutralSolver
    from microcell.micro_phase_neutral_class_contact import MicroPhaseNeutralContactSolver
    from microcell.pressure import (DELTA_P_DEFINITION, PRESSURE_EXTREMA_DEFINITION, PRESSURE_FIELDS,
                                    average_pressure_ledgers, union_pressure_extrema)
    from microcell.surface_ht import make_surface_ht

    # the pin was applied before the solver modules were imported (their import-time
    # set_at_import(2) is then a no-op); check nothing wrote the global since
    from microcell import form_compiler
    form_compiler.check("microcell.api.microsolver")

    mesh = UnitSquareMesh(int(st.mesh_n), int(st.mesh_n))

    # --- film callback -----------------------------------------------------------------
    if roughness.kind == "surface":
        fld = roughness.field()
        if mixed and int(st.mesh_n) < fld.source_n:
            raise ValueError(f"mesh_n = {st.mesh_n} is below the surface's source grid {fld.source_n} "
                             "(fft_elastic_v1 needs n >= N)")
        ht_cb = make_surface_ht(fld)
        # the translating member's starting displacement of THIS call (None = none)
        ht_cb.surface_motion.set_offset(offset if pair else None)
    else:
        c = roughness.cosine_parameters()
        # single surface: phases (phi_x, phi_y, dphi_x, dphi_y) = (0, phi_y, 0, 0)
        ht_cb = make_ideal_ht(
            Ah=c["Ah"], kx=c["kx"], ky=c["ky"],
            phi_x_const=Constant(0.0), phi_y_const=Constant(c["phi_y"] + 0.0),
            dphi_x_const=Constant(0.0), dphi_y_const=Constant(0.0),
            two_surface=False, static_moving=False,
        )

    # --- the task state the solver reads ---------------------------------------------------
    ratio = roughness.velocity_ratio if st.roughness_velocity_ratio is None else float(st.roughness_velocity_ratio)
    H, P, Ux, Uy, gx, gy = (float(zeta.H), float(zeta.P), float(zeta.Ux), float(zeta.Uy),
                            float(zeta.dpdx), float(zeta.dpdy))
    if steady:
        Hdot = Pdot = UxT = UyT = gxd = gyd = 0.0       # a steady task carries no rates
    else:
        Hdot, Pdot, UxT, UyT, gxd, gyd = (float(zeta.Hdot), float(zeta.Pdot), float(zeta.Uxdot),
                                          float(zeta.Uydot), float(zeta.dpdx_dot), float(zeta.dpdy_dot))
    task = Zeta(H=H, P=P, Ux=Ux, Uy=Uy, dpdx=gx, dpdy=gy, Hdot=Hdot, Pdot=Pdot, Uxdot=UxT, Uydot=UyT,
                dpdx_dot=gxd, dpdy_dot=gyd)
    SolverCls = MicroPhaseNeutralContactSolver if mixed else MicroPhaseNeutralSolver
    lc_tag = str(st.level_constraint)
    counters = dict(newton=0, contact=0, ripple=0, contact_s=0.0)
    lc_counts = [0, 0, 0]

    out_dir = Path(output_dir) if output_dir is not None else Path(os.getcwd()) / "microcell_output"
    solver = SolverCls(
        mesh_m=mesh, zeta=task, t_end=tend, settings=st, ht=ht_cb, velocity_ratio=ratio, pair=pair,
        auto_solve=False,
    )
    ci_seen = int(getattr(solver, "_contact_iterations", 0))

    def info_now() -> dict:
        """The ``info`` entries that do not depend on the solve's outcome, as they stand now."""
        # the parent reports no level-constraint counts with "whole_cell"
        counts = (0, 0, 0) if lc_tag == "whole_cell" else lc_counts
        return dict(
            mesh_n=int(st.mesh_n), steady=steady, t_end=tend, n_steps=(0 if steady else n_steps), offset=offset,
            roughness_velocity_ratio=float(ratio),
            t_start=(None if steady else float(getattr(solver, "T_start", _NAN))),
            dt=(None if steady else float(getattr(solver, "dt_dim_run", _NAN))),
            contact_model=CONTACT_FFT if mixed else CONTACT_OFF,
            deformation_model=_ripple.RIPPLE_MODEL if ripple_on else "off",
            level_constraint=lc_tag, n_whole_cell=int(counts[0]), n_fluid_mean=int(counts[1]),
            n_no_fluid=int(counts[2]),
            delta_P_definition=DELTA_P_DEFINITION, pressure_extrema_definition=PRESSURE_EXTREMA_DEFINITION,
            newton_iters=counters["newton"], contact_iters=counters["contact"], ripple_passes=counters["ripple"],
            contact_eval_s=counters["contact_s"], wall_time=time.time() - start)

    def failed(status: str, reason: str) -> MicroResult:
        return _failed_result(status, reason, info_now(), mixed, ripple_on)

    # --- solve -----------------------------------------------------------------------------
    try:
        if steady:
            solver.run_steady()
        else:
            solver.run(Tf=tend, n_steps=n_steps, cfl_target=float(st.cfl_target))
    except RuntimeError as exc:
        status = (FAILED_CONTACT if isinstance(exc, ContactResolutionError)
                  else FAILED_RIPPLE if isinstance(exc, _ripple.RippleNotConvergedError)
                  else FAILED_NEWTON)
        return_failed = (status, repr(exc))
    else:
        return_failed = None
    finally:
        counters["newton"] += getattr(solver, "_n_newton_iters_total", 0)
        for j, name in enumerate(("_n_lc_whole_cell", "_n_lc_fluid_mean", "_n_lc_no_fluid")):
            lc_counts[j] += int(getattr(solver, name, 0))
        if ripple_on:
            counters["ripple"] = int(getattr(solver, "_ripple_pass", 0))
        if mixed:
            ci_now = int(getattr(solver, "_contact_iterations", 0))
            counters["contact"] += ci_now - ci_seen
    if return_failed is not None:
        return failed(*return_failed)

    # --- homogenise --------------------------------------------------------------------------
    solver.pressure_extrema = None
    solver.fluid_traction = None
    solver.fluid_traction_samples = None
    solver.contact_traction = None
    (Qx, Qy, Pst, Fst, taust_x, taust_y,
     _p_max, _p_min, h_max, h_min, _qymax, _qymin, fmax, fmin) = solver.spatial_homogenisation()

    ledger = solver.pressure_ledger.copy()
    ripple_info = dict(solver.ripple_info) if ripple_on else None
    if solver.pressure_extrema is None:
        raise RuntimeError("spatial_homogenisation() did not run the shared pressure extrema sampler")
    if solver.fluid_traction is None:
        raise RuntimeError("spatial_homogenisation() did not run the shared fluid traction sampler")
    extrema_phase = solver.pressure_extrema
    traction_phase = solver.fluid_traction
    lambda_nd = float(solver.lambda_nd)
    spos_mean = float(assemble(solver.negP_pos * dx(domain=solver.mesh_m)))

    Pasp = contact_frac = A_patch = n_patches = k_asp = A_mech = 0.0
    yield_flag = 0
    pasp_patch_max = 0.0
    geometry_valid = 1.0
    contact_residual = contact_resolution = 0.0
    contact_phase = None
    if mixed:
        t0 = time.perf_counter()
        try:
            asp = solver.compute_asperity_load()
        except ContactResolutionError as exc:
            counters["contact_s"] += time.perf_counter() - t0
            return failed(FAILED_CONTACT, repr(exc))
        counters["contact_s"] += time.perf_counter() - t0
        if solver.contact_traction is None:
            raise RuntimeError("the contact class did not sample the asperity / total traction")
        contact_phase = solver.contact_traction
        Pasp += asp["Pasp_hom"]
        contact_frac += asp["contact_frac"]
        yield_flag = max(yield_flag, int(asp["yield_flag"]))
        A_patch += asp["A_patch_frac"]
        n_patches += float(asp["n_patches_loaded"])
        pasp_patch_max = max(pasp_patch_max, float(asp["pasp_max"]))
        k_asp += float(asp.get("k_asp_hom", 0.0))
        A_mech += asp["A_mech_frac"]
        geometry_valid = min(geometry_valid, float(asp["contact_geometry_valid"]))
        contact_residual = max(contact_residual, float(asp["contact_residual_m"]))
        if asp["contact_min_patch_nodes"] > 0:
            contact_resolution = float(asp["contact_min_patch_nodes"])
        counters["contact_s"] += float(getattr(solver, "_contact_solve_s", 0.0))

    # one phase per call: the worker's phase reductions with one member
    ledger = average_pressure_ledgers([ledger])
    extrema = union_pressure_extrema([extrema_phase])
    shear = union_fluid_traction([traction_phase])
    contact_shear = _cshear.union_contact_traction([contact_phase]) if contact_phase is not None else None
    Q = np.column_stack((Qx, Qy))
    Q_re = reynolds_flux(steady, zeta, tend, st.eta0)
    P_re = P if steady else (P + Pdot * tend)

    # --- fields ------------------------------------------------------------------------------
    shown, cell_fields, field_info = _fields.extract_fields(solver, mixed)
    fields = _fields.to_si(shown)
    del fields["p_fluid_masked"]        # p_fluid where chi_fluid is 1, NaN elsewhere: kept in the ParaView files only
    n = int(st.mesh_n)
    xs, ys = np.linspace(0.0, float(st.xmax), n + 1), np.linspace(0.0, float(st.ymax), n + 1)
    fields["x"], fields["y"] = np.meshgrid(xs, ys, indexing="ij")
    fields["chi_fluid_dg0"] = cell_fields["chi_fluid_dg0"]      # per mesh cell (DG0 dof order), not a grid

    paraview = None
    if save_paraview:
        out_dir.mkdir(parents=True, exist_ok=True)
        Lx_um, Ly_um = float(st.xmax) * 1e6, float(st.ymax) * 1e6
        _fields.write_vts(out_dir / "cell.vts", shown, Lx_um, Ly_um)
        _fields.write_xdmf(out_dir / "cell.xdmf", solver, shown, cell_fields, Lx_um, Ly_um)
        paraview = [str(out_dir / "cell.vts"), str(out_dir / "cell.xdmf")]

    # --- the result: each output once ------------------------------------------------------------
    info = info_now()
    info.update(
        **{k: ledger[k] for k in PRESSURE_FIELDS if k not in ("Pst", "P_f")},
        fmax=fmax, fmin=fmin, lambda_nd=lambda_nd, spos_mean=spos_mean,
        A_patch_frac=A_patch, n_patches=n_patches, pasp_patch_max=pasp_patch_max,
        contact_geometry_valid=geometry_valid, contact_residual=contact_residual,
        contact_resolution=contact_resolution,
        pressure_extrema_status=extrema["status"], p_fluid_n_samples=int(extrema["n_fluid_samples"]),
        fields=field_info)
    if ripple_on:
        info.update(ripple_passes=int(ripple_info["passes"]),
                    **{k: float(ripple_info[k[len("ripple_"):]]) for k in _RIPPLE_NUMBERS})
    else:
        info.update({k: 0.0 for k in _RIPPLE_NUMBERS})
    info.update(shear_fields(shear))
    contact_shear = (_cshear.contact_shear_fields(contact_shear) if contact_shear is not None
                     else _full_film_contact_shear())
    tau_asperity = _vec2(contact_shear.pop("shear_asperity_mean_local_Pa"))
    tau_total = _vec2(contact_shear.pop("shear_total_mean_local_Pa"))
    info.update(contact_shear)
    if paraview is not None:
        info["paraview"] = paraview
    return MicroResult(
        status=OK, reason="", dQ=_vec2(Q - Q_re), dP=float(ledger["P_f"] - P_re), Q=_vec2(Q), Pst=float(Pst),
        P_f=float(ledger["P_f"]), Fst=float(Fst), tau_fluid=_vec2((taust_x, taust_y)),
        tau_asperity=tau_asperity, tau_total=tau_total,
        P_asp=float(Pasp), contact_frac=float(contact_frac), A_mech_frac=float(A_mech), k_asp=float(k_asp),
        p_max=float(extrema["p_max"]), p_min=float(extrema["p_min"]), h_max=float(h_max), h_min=float(h_min),
        yield_flag=int(yield_flag), fields=fields, info=info)
