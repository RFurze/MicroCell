"""Pressure exchange contract (signed pressure is distinct from traction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELTA_P_DEFINITION = "fluid_per_cell"
PRESSURE_FIELDS = ("Pst", "P_f", "Pst_fluid", "P_patch", "P_fluid_pos", "P_fluid_neg")

# ---------------------------------------------------------------------------------------
# [extrema M1, 23 Oct 2026] The fluid pressure extrema of a micro task (pmin / pmax).
# Plan docs/plans/masked_pressure_extrema_and_shear_exports_2026-10-23.md sections 2.1 and
# 3.1, decisions D1-D4 of docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md
# section 2.3.  The sampler is microcell/samplers.py; this is the
# numpy-only part that the worker, run_micro, the pressure guard, the macro driver and the
# checkpoint share.
#
# Definition "fluid_nodal_sharp_v1/phase_union":
#   per phase (the sampler, fluid_nodal_sharp_v1): the signed physical pressure P_dim [Pa]
#     at every vertex of the micro mesh whose nodal obstruction level set is > 0 -- the
#     sharp chi_fluid test itself: no projection of chi, no 0.5 threshold, no clipping, no
#     asperity or fictitious contact pressure; min and max over those samples;
#   per task: the min of the phase minima and the max of the phase maxima over the phases
#     with fluid samples -- an envelope over every fluid sample of every phase.  The old
#     statistic (the mean over phases of the phase extrema) is kept separately, labelled.
# They are extrema of a finite sample set, not continuum extrema.
# ---------------------------------------------------------------------------------------
PRESSURE_EXTREMA_SAMPLER = "fluid_nodal_sharp_v1"
PRESSURE_EXTREMA_DEFINITION = PRESSURE_EXTREMA_SAMPLER + "/phase_union"

#: per-task status of the extrema
EXTREMA_OK = "ok"                         # finite min / max over >= 1 fluid sample
EXTREMA_NO_FLUID = "no_fluid"             # no fluid sample in any phase: NaN, legitimately
EXTREMA_PHASE_INVALID = "phase_invalid"   # a phase had non-finite fluid pressure: NaN, a failure
EXTREMA_FAILED = "failed"                 # the micro solve failed

#: per-phase status of the sampler
PHASE_OK, PHASE_NO_FLUID, PHASE_NONFINITE = "ok", "no_fluid", "nonfinite"


@dataclass(frozen=True)
class FluidPressureExtrema:
    """One phase's fluid pressure extrema, as the sampler returns them (D1)."""
    p_min_Pa: float
    p_max_Pa: float
    n_fluid_samples: int
    n_samples: int
    fluid_frac: float          # n_fluid_samples / n_samples: a vertex-count fraction, not an area
    status: str                # PHASE_OK | PHASE_NO_FLUID | PHASE_NONFINITE
    sampler: str = PRESSURE_EXTREMA_SAMPLER
    mask_source: str = ""      # "phic_nd_func" (fft_elastic_v1) | "projected_h" (every other class)


def union_pressure_extrema(phases) -> dict:
    """The task's extrema from its phases' samples (D3).

    ``p_min`` / ``p_max``: min of the minima / max of the maxima over the phases with
    fluid samples.  Status: ``phase_invalid`` if any phase had non-finite fluid pressure,
    else ``no_fluid`` if no phase had a fluid sample, else ``ok``; NaN unless ``ok``.
    ``p_min_phase_mean`` / ``p_max_phase_mean``: the old statistic, the mean over the
    phases with fluid samples of their extrema (NaN unless ``ok``), labelled and never
    used as the envelope.  ``n_fluid_samples``: summed over the phases.
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


# ---------------------------------------------------------------------------------------
# [fluid-mean WP3, 30 Oct 2026] The micro pressure-level constraint (the formulation tag).
# Plan docs/notes/micro_fluid_mean_constraint_plan_2026-10-30.md rev 3, sections 2.3, 3.6 and
# WP3 (S3 (c), (d)); decision D8.  "whole_cell": (1/A) int P = P_re + lambda;
# "fluid_mean_v1": (1/A_f) int chi P = P_re + lambda.
# The setting and its one default are ``microcell.api.MicroSettings.level_constraint`` (MicroCell
# WP4; the parent's second default LEVEL_CONSTRAINT_RUN_DEFAULT = "fluid_mean_v1" is not here).
# ---------------------------------------------------------------------------------------


def average_pressure_ledgers(ledgers):
    """Average each diagnostic over phases, never renormalise P_f by fluid area."""
    if not ledgers:
        raise ValueError("empty pressure phase sample")
    return {"delta_P_definition": DELTA_P_DEFINITION,
            **{key: float(np.mean([r[key] for r in ledgers])) for key in PRESSURE_FIELDS}}
