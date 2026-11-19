"""Inputs of the solver-class tests, under the argument names and defaults of the parent
repository (``MultiscaleMixedBallinSocket``, commit ``2aaccad``), where their expected values
were computed.

The helpers build a ``Zeta`` and a ``MicroSettings``.  Their keyword defaults are the values
those tests rely on, which differ from the defaults of ``MicroSettings``:

    spos_form                        "spline"
    jacobian_reg_eps                 1e-8
    level_constraint                 "fluid_mean_v1"
    pattern_flux_contact             True
    ripple tol / maxpass / omega     1e-6 / 12 / 1.0
    contact tol / maxit              1e-8 / 500
    sigma_yield / mu / k_asp_mode    250e6 Pa / NaN / "off"
    roughness_velocity_ratio         1.0 (the solver's ``velocity_ratio``)

``BETA``, ``PATCH_PIN`` and ``PATCH_PIN_KAPPA`` (1.0, None, 1.0) and the damping values are
constants of the solver module, not settings.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import ContactSettings, MicroSettings, RippleSettings, Zeta  # noqa: E402

NAN = float("nan")


def effective_modulus(E1, nu1, E2=math.inf, nu2=0.0):
    """Two-body contact modulus ``1 / sum((1 - nu^2) / E)``, in the unit of ``E``.

    E1, E2: Young's moduli (``E2`` infinite: a rigid second body).  nu1, nu2: Poisson's ratios.
    """
    compliance = 0.0
    for E, nu in ((E1, nu1), (E2, nu2)):
        if not E > 0 or math.isnan(E) or not math.isfinite(nu) or not -1 < nu < 0.5:
            raise ValueError("contact materials require E > 0 and -1 < nu < 0.5")
        compliance += (1.0 - nu * nu) / E
    if compliance <= 0:
        raise ValueError("at least one contact body must be deformable")
    return 1.0 / compliance


def parent_zeta(*, Ux, Uy, p0, dpdx, dpdy, H0, HT=0.0, PT=0.0, UxT=0.0, UyT=0.0, dpdx_dot=0.0,
                dpdy_dot=0.0) -> Zeta:
    """Macro state as a ``Zeta``: ``H0``, ``p0`` and the rates ``HT``, ``PT``, ``UxT``, ``UyT`` in SI."""
    return Zeta(H=H0, P=p0, Ux=Ux, Uy=Uy, dpdx=dpdx, dpdy=dpdy, Hdot=HT, Pdot=PT, Uxdot=UxT, Uydot=UyT,
                dpdx_dot=dpdx_dot, dpdy_dot=dpdy_dot)


def parent_contact(*, contact_tol=1e-8, contact_maxit=500, sigma_yield=250e6, mu_material=NAN,
                   k_asp_mode="off") -> ContactSettings:
    """``ContactSettings`` with the defaults listed in the module docstring."""
    return ContactSettings(tol=contact_tol, maxit=contact_maxit, sigma_yield=sigma_yield, mu=mu_material,
                           k_asp_mode=k_asp_mode)


def parent_ripple(*, ripple_tol=1e-6, ripple_maxpass=12, ripple_omega=1.0) -> RippleSettings:
    """``RippleSettings`` with the defaults listed in the module docstring."""
    return RippleSettings(tol=ripple_tol, maxpass=ripple_maxpass, omega=ripple_omega)


def parent_settings(*, eta0, penalty_gamma, xmax, ymax, h_min, relaxation_parameter, max_iterations,
                    abs_tolerance, rel_tolerance, delta_h, eps_solid, eps_smooth, spos_form="spline",
                    level_constraint="fluid_mean_v1", jacobian_reg_eps=1.0e-8, pattern_flux_contact=True,
                    contact=None, ripple=None, E_star=None, mesh_n=0) -> MicroSettings:
    """Cell values and solver settings as one ``MicroSettings``.

    ``mesh_n`` is not read by the solver classes, which are given the mesh.
    """
    return MicroSettings(
        mesh_n=mesh_n, xmax=xmax, ymax=ymax, eta0=eta0, penalty_gamma=penalty_gamma, h_min=h_min,
        E_star=E_star, delta_h=delta_h, eps_solid=eps_solid, eps_smooth=eps_smooth, spos_form=spos_form,
        level_constraint=level_constraint, pattern_flux_contact=pattern_flux_contact,
        jacobian_reg_eps=jacobian_reg_eps, relaxation_parameter=relaxation_parameter,
        max_iterations=max_iterations, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance,
        contact=contact, ripple=ripple)


def ensure_process() -> None:
    """Sets the process state that ``microsolver`` sets before its first form: PETSc options,
    form-compiler flags and quadrature degree 2.  A test that builds a solver class directly
    then runs in the same state whichever test file runs first."""
    from microcell.api import _ensure_process
    _ensure_process()
