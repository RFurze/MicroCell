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

At WP1 ``MicroSettings`` wraps the copied settings dataclasses: the fields the plan lists
are named; the copied fields it does not list travel in ``MicroSettings.legacy`` until WP4.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

OK = "OK"
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
    full-film mode."""
    E_star: float = 203775203775.2038       # reduced modulus [Pa]
    nu: float = 0.26
    tol: float = 1e-11                      # contact solve tolerance (1e-11 for h_min near 1e-11 m)
    maxit: int = 500
    sigma_yield: float = 250e6              # [Pa]
    mu: float = 0.8                         # boundary friction coefficient of the asperity traction
    k_asp_mode: str = "closure"
    k_asp_exponent: float = 1.5


@dataclass(frozen=True)
class RippleSettings:
    """Fluid-pressure ripple deformation (``ripple_halfspace``).  ``MicroSettings.ripple = None``
    is off."""
    E_star: float = 203775203775.2038       # [Pa]
    tol: float = 1e-6
    maxpass: int = 12
    omega: float = 1.0


#: WP1 only: fields of the copied settings dataclasses / MicroPhysicalParameters that the plan's
#: MicroSettings does not list, with the GenSa40-ripple case's values.  Removed at WP4.
LEGACY_DEFAULTS: Dict[str, Any] = dict(
    beta=1.0, delta_h=0.0, eps_solid=1e-8, eps_smooth=1e-3, k_spring=0.0,
    contact_E=380e9, contact_model="fft_elastic_v1", use_reduced_modulus=False,
    overclosure_measure="patch_max", patch_pin=None, patch_pin_kappa=1.0,
)


@dataclass
class MicroSettings:
    """Every setting of a cell solve.  Defaults at WP1: the parent's ``GenSa40-ripple`` case
    (``cases/mixed_transfull_cup90_genrough_ripple.py`` at 2aaccad); revisited at WP4."""
    mesh_n: int = 100                       # the cell mesh is UnitSquareMesh(mesh_n, mesh_n)
    xmax: float = 5e-6                      # cell size [m]
    ymax: float = 5e-6
    eta0: float = 1e-3                      # viscosity [Pa s] (isoviscous)
    rho0: float = 1000.0
    penalty_gamma: float = 1e10             # cavitation penalty
    h_min: float = 1.0331e-11               # contact onset film [m] (used when contact is on)
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
    legacy: Dict[str, Any] = field(default_factory=dict)    # WP1 only, see LEGACY_DEFAULTS

    def legacy_value(self, key: str):
        if key not in LEGACY_DEFAULTS:
            raise KeyError(key)
        return self.legacy.get(key, LEGACY_DEFAULTS[key])

    def check(self) -> None:
        unknown = sorted(set(self.legacy) - set(LEGACY_DEFAULTS))
        if unknown:
            raise ValueError(f"MicroSettings.legacy: unknown keys {unknown}; allowed: {sorted(LEGACY_DEFAULTS)}")
        if self.spos_form not in ("sqrt", "spline"):
            raise ValueError(f"spos_form must be 'sqrt' or 'spline', got {self.spos_form!r}")
        if self.level_constraint not in ("whole_cell", "fluid_mean_v1"):
            raise ValueError(f"level_constraint must be 'whole_cell' or 'fluid_mean_v1', got {self.level_constraint!r}")
        if int(self.mesh_n) < 2:
            raise ValueError("mesh_n must be at least 2")


# ----------------------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------------------
@dataclass
class MicroResult:
    """One cell's outputs (SI).  ``status`` is ``"OK"`` or a ``FAILED_*`` code with ``reason``;
    every homogenised output of a failed cell is NaN.  The asperity and total shear means of a
    full-film cell are NaN with ``info["shear_asperity_status"] == "unavailable"``."""
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
    #: the outputs under the parent's ``MicroTaskResult`` field names (parity work; WP4 removes it)
    raw: Dict[str, Any] = field(default_factory=dict)

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


def _solver_settings(st: MicroSettings, mixed: bool, pair: bool):
    """The copied settings dataclass of this call (the worker's ``_build_phase_neutral_settings``
    without the environment overrides)."""
    from microcell.contact import PHASE_AVERAGE_TABLE
    from microcell.micro_phase_neutral_class import PhaseNeutralSettings
    from microcell.micro_phase_neutral_class_contact import PhaseNeutralContactSettings

    common = dict(
        relaxation_parameter=st.relaxation_parameter,
        max_iterations=st.max_iterations,
        abs_tolerance=st.abs_tolerance,
        rel_tolerance=st.rel_tolerance,
        delta_h=st.legacy_value("delta_h"),
        eps_solid=st.legacy_value("eps_solid"),
        eps_smooth=st.legacy_value("eps_smooth"),
        spos_form=str(st.spos_form).strip().lower(),
        beta=float(st.legacy_value("beta")),
        jacobian_reg_eps=float(st.jacobian_reg_eps),
        level_constraint=str(st.level_constraint),
    )
    if st.level_constraint != "whole_cell":
        common.update(patch_pin=st.legacy_value("patch_pin"),
                      patch_pin_kappa=float(st.legacy_value("patch_pin_kappa")))
    if st.ripple is not None:
        common.update(fluid_deformation="ripple_halfspace", fluid_E_star=float(st.ripple.E_star),
                      ripple_tol=float(st.ripple.tol), ripple_maxpass=int(st.ripple.maxpass),
                      ripple_omega=float(st.ripple.omega))
    if not mixed:
        return PhaseNeutralSettings(**common)
    c = st.contact
    return PhaseNeutralContactSettings(
        **common,
        contact_E=float(st.legacy_value("contact_E")),
        contact_nu=float(c.nu),
        contact_model=str(st.legacy_value("contact_model")),
        contact_E_star=c.E_star,
        use_reduced_modulus=bool(st.legacy_value("use_reduced_modulus")),
        overclosure_measure=str(st.legacy_value("overclosure_measure")),
        sigma_yield=float(c.sigma_yield),
        k_asp_mode=str(c.k_asp_mode),
        k_asp_exponent=float(c.k_asp_exponent),
        pattern_flux_contact=bool(st.pattern_flux_contact),
        contact_tol=float(c.tol),
        contact_maxit=int(c.maxit),
        mu_material=float(c.mu),
        # A stationary + translating pair runs the solver's pair branch (the converged per-step
        # ripple, the warm-start contact rescue), which the copied class keys on this tag.  No
        # table is read or composed here: P_asp is the coupled solve's.
        phase_average=PHASE_AVERAGE_TABLE if pair else "off",
    )


def _failed_raw(st: MicroSettings, mixed: bool, ripple_on: bool, lc_fields: dict, reason: str,
                counters: dict, wall: float) -> dict:
    """The worker's ``_nan_return`` by field name."""
    from microcell import contact_shear as _cshear
    from microcell import ripple as _ripple
    from microcell.fluid_shear import failed_shear_fields
    from microcell.pressure import DELTA_P_DEFINITION, EXTREMA_FAILED, PRESSURE_EXTREMA_DEFINITION

    nan2 = np.array([np.nan, np.nan])
    raw = dict(
        delta_Q=nan2.copy(), delta_P=_NAN, delta_P_definition=DELTA_P_DEFINITION, Pst=_NAN, Q=nan2.copy(),
        Fst=_NAN, tau_res=nan2.copy(), p_max=_NAN, p_min=_NAN, h_max=_NAN, h_min=_NAN, fmax=_NAN, fmin=_NAN,
        wall_time=wall, lambda_nd=_NAN, spos_mean=_NAN, Pasp_hom=_NAN, contact_frac=_NAN, yield_flag=0,
        mesh_used=int(st.mesh_n), A_patch_frac=_NAN, n_patches=_NAN, pasp_patch_max=_NAN, k_asp_hom=_NAN,
        A_mech_frac=_NAN, contact_geometry_valid=0.0,
        contact_model=str(st.legacy_value("contact_model")) if mixed else "legacy_patch_v1",
        failure_reason=reason, contact_eval_s=counters["contact_s"], newton_iters=counters["newton"],
        contact_iters=counters["contact"], contact_residual=_NAN, contact_resolution=_NAN,
        ripple_passes=counters["ripple"],
        ripple_max=_NAN if ripple_on else 0.0, ripple_resid=_NAN if ripple_on else 0.0,
        ripple_mean_resid=_NAN if ripple_on else 0.0, ripple_seam_mismatch=_NAN if ripple_on else 0.0,
        ripple_proj_err=_NAN if ripple_on else 0.0,
        deformation_model=_ripple.RIPPLE_MODEL if ripple_on else "off",
        P_f=_NAN, Pst_fluid=_NAN, P_patch=_NAN, P_fluid_pos=_NAN, P_fluid_neg=_NAN,
        pressure_extrema_status=EXTREMA_FAILED, pressure_extrema_definition=PRESSURE_EXTREMA_DEFINITION,
        **failed_shear_fields(),
        **(_cshear.failed_contact_shear_fields() if mixed else {}),
        **lc_fields,
    )
    return raw


#: the parent's ``MicroTaskResult`` defaults of the appended fields a call may not fill
_RAW_DEFAULTS: Dict[str, Any] = dict(
    A_mech_frac=0.0, contact_geometry_valid=1.0, contact_model="legacy_patch_v1", failure_reason="",
    contact_eval_s=0.0, newton_iters=0, contact_iters=0, contact_residual=0.0, contact_resolution=0.0,
    ripple_passes=0, ripple_max=0.0, ripple_resid=0.0, ripple_mean_resid=0.0, ripple_seam_mismatch=0.0,
    ripple_proj_err=0.0, deformation_model="off",
    shear_asperity_mean_local_Pa=(_NAN, _NAN), shear_asperity_gx_min_Pa=_NAN, shear_asperity_gx_max_Pa=_NAN,
    shear_asperity_gx_sample_mean_Pa=_NAN, shear_asperity_support_fraction=_NAN,
    shear_asperity_status="unavailable",
    shear_total_mean_local_Pa=(_NAN, _NAN), shear_total_gx_min_Pa=_NAN, shear_total_gx_max_Pa=_NAN,
    shear_total_gx_sample_mean_Pa=_NAN, shear_total_status="unavailable",
    shear_contact_overlap_fraction=_NAN, shear_contact_force_on_fluid_fraction=_NAN,
    shear_contact_obstruction_without_pressure_fraction=_NAN, shear_contact_mean_check=_NAN,
    shear_contact_definition="unversioned", shear_contact_n_samples=0,
    level_constraint="whole_cell", n_whole_cell=0, n_fluid_mean=0, n_no_fluid=0,
)


def _result(status: str, reason: str, raw: dict, fields: dict, info: dict) -> MicroResult:
    def vec(key):
        return np.asarray(raw.get(key, (_NAN, _NAN)), dtype=float).reshape(-1)[:2].copy()

    info = dict(info)
    for key in ("pressure_extrema_status", "shear_extrema_status", "shear_asperity_status",
                "shear_total_status", "deformation_model", "contact_model", "level_constraint"):
        info[key] = raw.get(key)
    return MicroResult(
        status=status, reason=reason,
        dQ=vec("delta_Q"), dP=float(raw["delta_P"]), Q=vec("Q"), Pst=float(raw["Pst"]),
        P_f=float(raw["P_f"]), Fst=float(raw["Fst"]),
        tau_fluid=vec("tau_res"), tau_asperity=vec("shear_asperity_mean_local_Pa"),
        tau_total=vec("shear_total_mean_local_Pa"),
        P_asp=float(raw["Pasp_hom"]), contact_frac=float(raw["contact_frac"]),
        A_mech_frac=float(raw["A_mech_frac"]), k_asp=float(raw["k_asp_hom"]),
        p_max=float(raw["p_max"]), p_min=float(raw["p_min"]),
        h_max=float(raw["h_max"]), h_min=float(raw["h_min"]),
        yield_flag=int(raw["yield_flag"]), fields=fields, info=info, raw=raw)


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
    if mixed and str(st.legacy_value("contact_model")) != "fft_elastic_v1" and (ripple_on or roughness.kind == "surface"):
        raise ValueError("ripple deformation and generated / array surfaces need contact_model 'fft_elastic_v1'")

    _ensure_process()
    from dolfin import Constant, UnitSquareMesh, assemble, dx

    from microcell import contact_shear as _cshear
    from microcell import fields as _fields
    from microcell import ripple as _ripple
    from microcell.contact import ContactResolutionError
    from microcell.fluid_shear import shear_fields, union_fluid_traction
    from microcell.ideal_ht import make_ideal_ht
    from microcell.micro_phase_neutral_class import MicroPhaseNeutralSolver, MicroPhysicalParameters
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

    # --- physical parameters ---------------------------------------------------------------
    ratio = roughness.velocity_ratio if st.roughness_velocity_ratio is None else float(st.roughness_velocity_ratio)
    H, P, Ux, Uy, gx, gy = (float(zeta.H), float(zeta.P), float(zeta.Ux), float(zeta.Uy),
                            float(zeta.dpdx), float(zeta.dpdy))
    if steady:
        Hdot = Pdot = UxT = UyT = gxd = gyd = 0.0       # a steady task carries no rates
    else:
        Hdot, Pdot, UxT, UyT, gxd, gyd = (float(zeta.Hdot), float(zeta.Pdot), float(zeta.Uxdot),
                                          float(zeta.Uydot), float(zeta.dpdx_dot), float(zeta.dpdy_dot))
    phys = MicroPhysicalParameters(
        Ux=Ux, Uy=Uy, eta0=st.eta0, rho0=st.rho0, penalty_gamma=float(st.penalty_gamma),
        xmax=st.xmax, ymax=st.ymax, p0=P, dpdx=gx, dpdy=gy, H0=H,
        h_min=(float(st.h_min) if mixed else 0.0),      # 0.0 keeps the mixed masking off (full film)
        HT=Hdot, PT=Pdot, Tend=tend, k_spring=st.legacy_value("k_spring"),
        UxT=UxT, UyT=UyT, dpdx_dot=gxd, dpdy_dot=gyd, roughness_velocity_ratio=ratio,
    )
    solver_settings = _solver_settings(st, mixed, pair)
    SolverCls = MicroPhaseNeutralContactSolver if mixed else MicroPhaseNeutralSolver
    fft = mixed and str(st.legacy_value("contact_model")) == "fft_elastic_v1"
    lc_tag = str(solver_settings.level_constraint)
    counters = dict(newton=0, contact=0, ripple=0, contact_s=0.0)
    lc_counts = [0, 0, 0]

    def lc_fields() -> dict:
        if lc_tag == "whole_cell":
            return {}
        return dict(level_constraint=lc_tag, n_whole_cell=int(lc_counts[0]),
                    n_fluid_mean=int(lc_counts[1]), n_no_fluid=int(lc_counts[2]))

    out_dir = Path(output_dir) if output_dir is not None else Path(os.getcwd()) / "microcell_output"
    solver = SolverCls(
        mesh_m=mesh, mesh_h=mesh, physical_params=phys, solver_settings=solver_settings, k=1,
        ht=ht_cb, export_vtk=False, output_dir=os.path.join(os.getcwd(), "output"), auto_solve=False,
    )
    ci_seen = int(getattr(solver, "_contact_iterations", 0))

    base_info = dict(mesh_n=int(st.mesh_n), steady=steady, t_end=tend, n_steps=(0 if steady else n_steps),
                     offset=offset, roughness_velocity_ratio=float(ratio))

    def failed(status: str, reason: str) -> MicroResult:
        raw = _failed_raw(st, mixed, ripple_on, lc_fields(), reason, counters, time.time() - start)
        info = dict(base_info, newton_iters=counters["newton"], contact_iters=counters["contact"],
                    ripple_passes=counters["ripple"], wall_time=raw["wall_time"])
        return _result(status, reason, raw, {}, info)

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
        if fft:
            ci_now = int(getattr(solver, "_contact_iterations", 0))
            counters["contact"] += ci_now - ci_seen
    if return_failed is not None:
        return failed(*return_failed)

    # --- homogenise --------------------------------------------------------------------------
    solver.pressure_extrema = None
    solver.fluid_traction = None
    solver.fluid_traction_samples = None
    solver.contact_traction = None
    solver.shear_frame = None           # no macro frame: the global-x shear statistics are "no_frame"
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
        if st.legacy_value("contact_model") in ("hertz_summit_v1", "fft_elastic_v1"):
            A_mech += asp["A_mech_frac"]
            geometry_valid = min(geometry_valid, float(asp["contact_geometry_valid"]))
        if fft:
            contact_residual = max(contact_residual, float(asp["contact_residual_m"]))
            if asp["contact_min_patch_nodes"] > 0:
                contact_resolution = float(asp["contact_min_patch_nodes"])
    if fft:
        counters["contact_s"] += float(getattr(solver, "_contact_solve_s", 0.0))

    # one phase per call: the worker's phase reductions with one member
    ledger = average_pressure_ledgers([ledger])
    extrema = union_pressure_extrema([extrema_phase])
    shear = union_fluid_traction([traction_phase])
    contact_shear = _cshear.union_contact_traction([contact_phase]) if contact_phase is not None else None
    inv_n = 1.0
    Qx, Qy, Pst, Fst = Qx * inv_n, Qy * inv_n, Pst * inv_n, Fst * inv_n
    taust_x, taust_y = taust_x * inv_n, taust_y * inv_n

    Q = np.column_stack((Qx, Qy))
    tau_res = np.array([taust_x, taust_y])
    Q_re = reynolds_flux(steady, zeta, tend, st.eta0)
    P_re = P if steady else (P + Pdot * tend)
    delta_Q = Q - Q_re
    delta_P = ledger["P_f"] - P_re

    raw = dict(_RAW_DEFAULTS)
    raw.update(
        delta_Q=delta_Q, delta_P=delta_P, Pst=Pst, Q=Q, Fst=Fst, tau_res=tau_res,
        p_max=extrema["p_max"], p_min=extrema["p_min"], h_max=h_max * inv_n, h_min=h_min * inv_n,
        fmax=fmax * inv_n, fmin=fmin * inv_n, wall_time=time.time() - start,
        lambda_nd=lambda_nd * inv_n, spos_mean=spos_mean * inv_n,
        Pasp_hom=Pasp * inv_n, contact_frac=contact_frac * inv_n, yield_flag=int(yield_flag),
        mesh_used=int(st.mesh_n), A_patch_frac=A_patch * inv_n, n_patches=n_patches * inv_n,
        pasp_patch_max=pasp_patch_max, k_asp_hom=k_asp * inv_n, A_mech_frac=A_mech * inv_n,
        contact_geometry_valid=geometry_valid,
        contact_model=str(st.legacy_value("contact_model")) if mixed else "legacy_patch_v1",
        contact_eval_s=counters["contact_s"], newton_iters=counters["newton"],
        contact_iters=counters["contact"], contact_residual=contact_residual,
        contact_resolution=contact_resolution,
        **{k: ledger[k] for k in PRESSURE_FIELDS if k != "Pst"},
        delta_P_definition=DELTA_P_DEFINITION,
    )
    if ripple_on:
        raw.update(ripple_passes=int(ripple_info["passes"]), ripple_max=float(ripple_info["max"]),
                   ripple_resid=float(ripple_info["resid"]), ripple_mean_resid=float(ripple_info["mean_resid"]),
                   ripple_seam_mismatch=float(ripple_info["seam_mismatch"]),
                   ripple_proj_err=float(ripple_info["proj_err"]), deformation_model=str(ripple_info["model"]))
    raw.update(p_fluid_sample_max_Pa=extrema["p_max"], p_fluid_sample_min_Pa=extrema["p_min"],
               pressure_extrema_status=extrema["status"], pressure_extrema_definition=PRESSURE_EXTREMA_DEFINITION,
               p_max_phase_mean=extrema["p_max_phase_mean"], p_min_phase_mean=extrema["p_min_phase_mean"],
               p_fluid_n_samples=int(extrema["n_fluid_samples"]))
    raw.update(shear_fields(shear))
    if contact_shear is not None:
        raw.update(_cshear.contact_shear_fields(contact_shear))
    raw.update(lc_fields())

    # --- fields ------------------------------------------------------------------------------
    shown, cell_fields, field_info = _fields.extract_fields(solver, mixed)
    fields = _fields.to_si(shown)
    n = int(st.mesh_n)
    xs, ys = np.linspace(0.0, float(st.xmax), n + 1), np.linspace(0.0, float(st.ymax), n + 1)
    fields["x"], fields["y"] = np.meshgrid(xs, ys, indexing="ij")
    fields["chi_fluid_dg0"] = cell_fields["chi_fluid_dg0"]      # per mesh cell (DG0 dof order), not a grid

    info = dict(base_info)
    info.update(newton_iters=counters["newton"], contact_iters=counters["contact"],
                ripple_passes=int(raw["ripple_passes"]), contact_residual=contact_residual,
                contact_resolution=contact_resolution, lambda_nd=lambda_nd, spos_mean=spos_mean,
                t_start=(None if steady else float(getattr(solver, "T_start", _NAN))),
                dt=(None if steady else float(getattr(solver, "dt_dim_run", _NAN))),
                fields=field_info)
    if save_paraview:
        out_dir.mkdir(parents=True, exist_ok=True)
        Lx_um, Ly_um = float(st.xmax) * 1e6, float(st.ymax) * 1e6
        _fields.write_vts(out_dir / "cell.vts", shown, Lx_um, Ly_um)
        _fields.write_xdmf(out_dir / "cell.xdmf", solver, shown, cell_fields, Lx_um, Ly_um)
        info["paraview"] = [str(out_dir / "cell.vts"), str(out_dir / "cell.xdmf")]
    raw["wall_time"] = time.time() - start
    info["wall_time"] = raw["wall_time"]
    return _result(OK, "", raw, fields, info)
