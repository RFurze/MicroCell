"""Contact identities and the shared contact error (SI), independent of FEniCS.

The macro owns mean deformation.  The parent's ``legacy_patch_v1`` and
``hertz_summit_v1`` laws are not in this repo (MicroCell WP2, 11 Nov 2026).
"""
import math

#: periodic elastic (FFT) contact solve per micro cell, ehl.elastic_contact (16 Oct 2026)
FFT = "fft_elastic_v1"
CONTACT_MODELS = (FFT,)


class ContactResolutionError(RuntimeError):
    """Valid geometry but an unresolved fluid obstruction mesh."""


def effective_modulus(E1, nu1, E2=math.inf, nu2=0.0):
    """Two-body contact modulus; +inf denotes a rigid counterface."""
    compliance = 0.0
    for E, nu in ((E1, nu1), (E2, nu2)):
        if not E > 0 or math.isnan(E) or not math.isfinite(nu) or not -1 < nu < 0.5:
            raise ValueError("contact materials require E > 0 and -1 < nu < 0.5")
        compliance += (1.0 - nu * nu) / E
    if compliance <= 0:
        raise ValueError("at least one contact body must be deformable")
    return 1.0 / compliance
