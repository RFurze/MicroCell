"""MicroCell: the micro-cell solver of a transient multiscale mixed-lubrication model.

    from microcell import microsolver, Zeta, Roughness, MicroSettings

FEniCS (dolfin) is imported at the first ``microsolver`` call, not on package import.
"""
from microcell.api import (FAILED_CONTACT, FAILED_NEWTON, FAILED_NONFINITE, FAILED_RIPPLE, NAN_STATUS, OK,
                           ContactSettings, MicroResult, MicroSettings, RippleSettings, Zeta, microsolver)
from microcell.roughness.frontend import Roughness

__all__ = ["microsolver", "Zeta", "Roughness", "MicroSettings", "ContactSettings", "RippleSettings",
           "MicroResult", "OK", "FAILED_NEWTON", "FAILED_RIPPLE", "FAILED_CONTACT", "FAILED_NONFINITE",
           "NAN_STATUS"]
