"""What the parent's invariant tests build, in this package's names (WP5).

The tests copied from ``MultiscaleMixedBallinSocket`` at 2aaccad built a ``MicroPhysicalParameters``
tuple and a settings dataclass and handed both to a solver class.  Here the same values go into a
``Zeta`` and a ``MicroSettings``.  The parent's dataclass defaults differ from ``MicroSettings``'
defaults (which are one production case's values), so every default a copied test relied on is a
keyword default below, with the parent's value:

    parent field (file at 2aaccad)                                          parent default   here
    MicroSolverSettings.spos_form (micro_phase_neutral_class.py)            "spline"         spos_form
    PhaseNeutralSettings.jacobian_reg_eps                                   1e-8             jacobian_reg_eps
    PhaseNeutralSettings.level_constraint (ehl/pressure.py RUN_DEFAULT)     "fluid_mean_v1"  level_constraint
    PhaseNeutralSettings.pattern_flux_contact (..._class_mixed.py)          True             pattern_flux_contact
    PhaseNeutralSettings.ripple_tol / ripple_maxpass / ripple_omega         1e-6 / 12 / 1.0  parent_ripple()
    PhaseNeutralContactSettings.contact_tol / contact_maxit                 1e-8 / 500       parent_contact()
    PhaseNeutralContactSettings.sigma_yield / mu_material / k_asp_mode      250e6 / nan / "off"
    MicroPhysicalParameters.roughness_velocity_ratio                        1.0              the solver's velocity_ratio

Fixed in this package (no setting): ``beta = 1.0``, ``patch_pin = None``, ``patch_pin_kappa = 1.0``
(``microcell`` solver module constants ``BETA``, ``PATCH_PIN``, ``PATCH_PIN_KAPPA``), the damping
constants, and ``k_spring = 0`` (the spring is removed).  ``rho0`` was never read.
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
    """The parent's ``ehl/contact.py:effective_modulus`` at 2aaccad (removed from this package in
    WP3), copied: the two-body contact modulus ``1 / sum((1 - nu^2) / E)``."""
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
    """``MicroPhysicalParameters``' macro state as a ``Zeta`` (the parent's names)."""
    return Zeta(H=H0, P=p0, Ux=Ux, Uy=Uy, dpdx=dpdx, dpdy=dpdy, Hdot=HT, Pdot=PT, Uxdot=UxT, Uydot=UyT,
                dpdx_dot=dpdx_dot, dpdy_dot=dpdy_dot)


def parent_contact(*, contact_tol=1e-8, contact_maxit=500, sigma_yield=250e6, mu_material=NAN,
                   k_asp_mode="off") -> ContactSettings:
    """``PhaseNeutralContactSettings``' contact fields with the parent's defaults."""
    return ContactSettings(tol=contact_tol, maxit=contact_maxit, sigma_yield=sigma_yield, mu=mu_material,
                           k_asp_mode=k_asp_mode)


def parent_ripple(*, ripple_tol=1e-6, ripple_maxpass=12, ripple_omega=1.0) -> RippleSettings:
    """``PhaseNeutralSettings``' ripple fields with the parent's defaults."""
    return RippleSettings(tol=ripple_tol, maxpass=ripple_maxpass, omega=ripple_omega)


def parent_settings(*, eta0, penalty_gamma, xmax, ymax, h_min, relaxation_parameter, max_iterations,
                    abs_tolerance, rel_tolerance, delta_h, eps_solid, eps_smooth, spos_form="spline",
                    level_constraint="fluid_mean_v1", jacobian_reg_eps=1.0e-8, pattern_flux_contact=True,
                    contact=None, ripple=None, E_star=None, mesh_n=0) -> MicroSettings:
    """The parent's ``MicroPhysicalParameters`` cell values and settings dataclass as one
    ``MicroSettings``.  ``mesh_n`` is not read by the solver classes (they are given the mesh)."""
    return MicroSettings(
        mesh_n=mesh_n, xmax=xmax, ymax=ymax, eta0=eta0, penalty_gamma=penalty_gamma, h_min=h_min,
        E_star=E_star, delta_h=delta_h, eps_solid=eps_solid, eps_smooth=eps_smooth, spos_form=spos_form,
        level_constraint=level_constraint, pattern_flux_contact=pattern_flux_contact,
        jacobian_reg_eps=jacobian_reg_eps, relaxation_parameter=relaxation_parameter,
        max_iterations=max_iterations, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance,
        contact=contact, ripple=ripple)


def ensure_process() -> None:
    """The process state ``microsolver`` sets before its first form (PETSc options, form-compiler
    flags, quadrature degree pinned to 2), so a copied test that builds a solver class directly
    runs in the same state whichever test file pytest runs first."""
    from microcell.api import _ensure_process
    _ensure_process()
