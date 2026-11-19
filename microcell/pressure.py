"""Pressure ledger fields and the fluid pressure extrema of a cell (numpy only).

Pressures are signed [Pa] and are distinct from the shear traction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELTA_P_DEFINITION = "fluid_per_cell"
PRESSURE_FIELDS = ("Pst", "P_f", "Pst_fluid", "P_patch", "P_fluid_pos", "P_fluid_neg")

# Fluid pressure extrema, definition "fluid_nodal_sharp_v1/phase_union".
#   Per phase (the sampler in samplers.py): min and max of the signed fluid pressure [Pa] over the
#     mesh vertices whose nodal obstruction level set is > 0.  No projection, threshold or clipping,
#     and no contact pressure.
#   Per cell: the min of the phase minima and the max of the phase maxima.
# The values are extrema of a finite sample set, not of the continuum.
PRESSURE_EXTREMA_SAMPLER = "fluid_nodal_sharp_v1"
PRESSURE_EXTREMA_DEFINITION = PRESSURE_EXTREMA_SAMPLER + "/phase_union"

#: status of the cell's extrema
EXTREMA_OK = "ok"                         # finite min and max over at least one fluid sample
EXTREMA_NO_FLUID = "no_fluid"             # no fluid sample in any phase: NaN
EXTREMA_PHASE_INVALID = "phase_invalid"   # a phase had non-finite fluid pressure: NaN, a failure
EXTREMA_FAILED = "failed"                 # the micro solve failed

#: status of one phase's sample
PHASE_OK, PHASE_NO_FLUID, PHASE_NONFINITE = "ok", "no_fluid", "nonfinite"


@dataclass(frozen=True)
class FluidPressureExtrema:
    """Fluid pressure extrema of one phase, as returned by the sampler."""
    p_min_Pa: float
    p_max_Pa: float
    n_fluid_samples: int
    n_samples: int
    fluid_frac: float          # n_fluid_samples / n_samples: a vertex-count fraction, not an area
    status: str                # PHASE_OK | PHASE_NO_FLUID | PHASE_NONFINITE
    sampler: str = PRESSURE_EXTREMA_SAMPLER
    mask_source: str = ""      # "phic_nd_func" (contact class) | "projected_h" (the other classes)


def union_pressure_extrema(phases) -> dict:
    """Extrema of the cell from the samples of its phases.

    phases: ``FluidPressureExtrema`` objects, one per phase.
    Returns a dict:
      ``status``: ``phase_invalid`` if any phase had non-finite fluid pressure, else
        ``no_fluid`` if no phase had a fluid sample, else ``ok``.
      ``p_min``, ``p_max`` [Pa]: min of the minima and max of the maxima over the phases with
        fluid samples; NaN unless ``ok``.
      ``p_min_phase_mean``, ``p_max_phase_mean`` [Pa]: mean over those phases of their extrema;
        NaN unless ``ok``.
      ``n_fluid_samples``: summed over the phases.
    """
    phases = list(phases)
    if not phases:
        raise ValueError("no phase sample")
    good = [ph for ph in phases if ph.status == PHASE_OK]
    if any(ph.status not in (PHASE_OK, PHASE_NO_FLUID) for ph in phases):
        status = EXTREMA_PHASE_INVALID
    elif not good:
        status = EXTREMA_NO_FLUID
    else:
        status = EXTREMA_OK
    nan = float("nan")
    out = dict(status=status, p_min=nan, p_max=nan, p_min_phase_mean=nan, p_max_phase_mean=nan,
               n_fluid_samples=int(sum(int(ph.n_fluid_samples) for ph in phases)))
    if status == EXTREMA_OK:
        lo = [float(ph.p_min_Pa) for ph in good]
        hi = [float(ph.p_max_Pa) for ph in good]
        out.update(p_min=min(lo), p_max=max(hi),
                   p_min_phase_mean=float(np.mean(lo)), p_max_phase_mean=float(np.mean(hi)))
    return out


def average_pressure_ledgers(ledgers):
    """Mean over the phases of each field in ``PRESSURE_FIELDS``.

    ledgers: one dict per phase, keyed by ``PRESSURE_FIELDS`` [Pa].
    Returns a dict of the means with ``delta_P_definition``.  ``P_f`` stays per cell area and
    is not renormalised by the fluid area.
    """
    if not ledgers:
        raise ValueError("empty pressure phase sample")
    return {"delta_P_definition": DELTA_P_DEFINITION,
            **{key: float(np.mean([r[key] for r in ledgers])) for key in PRESSURE_FIELDS}}
