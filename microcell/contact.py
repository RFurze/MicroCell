"""Local assumed-overclosure contact laws (SI), independent of FEniCS.

The macro owns mean deformation. Hertz represents summit self-indentation;
neighbouring displacement and asperity interactions are not resolved here.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

LEGACY = "legacy_patch_v1"
HERTZ = "hertz_summit_v1"
#: periodic elastic (FFT) contact solve per micro cell, ehl.elastic_contact (16 Oct 2026)
FFT = "fft_elastic_v1"
CONTACT_MODELS = (LEGACY, HERTZ, FFT)
#: [phase-table WP2, 4 Nov 2026] ``Asperity.phase_average``: the phase average of a surface PAIR's contact
#: (route A, docs/plans/phase_average_route_a_production_plan_2026-11-04.md D1).  "off" = every run as today;
#: "contact_table_v1" = P_asp and k_asp from the contact-only phase-mean table of the pair (ehl.contact_table)
#: plus one coupled solve per contact cell (session B).  fft_elastic_v1 and a stationary + translating
#: SurfaceRoughness only.
PHASE_AVERAGE_OFF = "off"
PHASE_AVERAGE_TABLE = "contact_table_v1"
PHASE_AVERAGES = (PHASE_AVERAGE_OFF, PHASE_AVERAGE_TABLE)


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


def hertz_response(radius, indentation, E_star):
    """(force N, mechanical area m², mean pressure Pa, dF/dδ N/m).

No spatial or resolved peak pressure is supplied. The derivative with
respect to separation is the negative of the returned stiffness.
"""
    if not all(math.isfinite(v) for v in (radius, indentation, E_star)) or radius <= 0 or E_star <= 0:
        raise ValueError("Hertz requires finite R > 0, E_star > 0 and indentation")
    if indentation <= 0:
        return 0.0, 0.0, 0.0, 0.0
    a = math.sqrt(radius * indentation)
    force = (4.0 / 3.0) * E_star * a * indentation
    area = math.pi * a * a
    return force, area, force / area, 2.0 * E_star * a


@dataclass(frozen=True)
class CosineSummits:
    amplitude: float
    radius: float
    count: int
    spacing: float

    def indicators(self, indentation):
        """Conservative qualified geometry screen, not an interaction guarantee.

        The isolated cosine dry reference supports δ/Ah <=0.1. The provisional
        0.3 limit missed the 5% mechanical-area target and was narrowed.
        a/R <=0.1 screens the small-slope assumption.
        """
        d = max(indentation, 0.0)
        a = math.sqrt(self.radius * d)
        ratio = d / self.amplitude
        return dict(indentation_ratio=ratio, a_over_R=a / self.radius,
                    a_over_spacing=a / self.spacing,
                    contact_geometry_valid=ratio <= 0.1 + 1e-12 and a / self.radius <= 0.1)


@lru_cache(maxsize=64)
def cosine_summits(amplitude, kx, ky, Lx, Ly):
    """Exact summit data for make_ideal_ht's single sum cosine.

    Circular summits only. Integer wave counts retain periodicity; rectangular
    cells are allowed when their physical wavelengths agree. Cached per geometry.
    """
    if not all(math.isfinite(v) and v > 0 for v in (amplitude, kx, ky, Lx, Ly)):
        raise ValueError("hertz_summit_v1 requires positive cosine amplitude, periods and lengths")
    if kx != round(kx) or ky != round(ky):
        raise ValueError("hertz_summit_v1 requires integer periodic wave counts")
    qx, qy = 2 * math.pi * kx / Lx, 2 * math.pi * ky / Ly
    if not math.isclose(qx, qy, rel_tol=1e-12):
        raise ValueError("hertz_summit_v1 supports circular summits only (equal physical wavelengths)")
    return CosineSummits(amplitude, 2 / (amplitude * qx * qx), int(kx * ky), Lx / kx)
