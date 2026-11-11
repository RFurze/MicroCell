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
# section 2.3.  The sampler is microscale/src/functions/micro_diagnostics.py; this is the
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

#: per-task status of the extrema (a fixed-width string array per node in run_micro)
EXTREMA_OK = "ok"                         # finite min / max over >= 1 fluid sample
EXTREMA_NO_FLUID = "no_fluid"             # no fluid sample in any phase: NaN, legitimately
EXTREMA_PHASE_INVALID = "phase_invalid"   # a phase had non-finite fluid pressure: NaN, a failure
EXTREMA_FAILED = "failed"                 # the micro solve failed (or, under dedup, its representative's
                                          # did); after run_micro's kNN fill the values are neighbour
                                          # estimates, never measured bounds
EXTREMA_DEDUP_MEMBER = "dedup_member"     # (run_micro, MICRO_DEDUP_TOL > 0) a dedup member of a solved
                                          # representative: NaN; the extrema of its representative's
                                          # cell are not its own (D6)
EXTREMA_LEGACY = "legacy"                 # restored from a checkpoint written before this definition
EXTREMA_UNAVAILABLE = "unavailable"       # a result built without the extrema fields
EXTREMA_STATUS_DTYPE = "<U16"
#: statuses whose NaN extrema are legitimate, so not a failed task (D4; with D6's dedup members,
#: Rob's decision of 23 Oct 2026)
EXTREMA_NAN_IS_VALID = (EXTREMA_NO_FLUID, EXTREMA_DEDUP_MEMBER)

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


def require_definition(value, context="pressure results", expected=DELTA_P_DEFINITION):
    """Fail closed for untagged, legacy, or mixed-definition data."""
    tag = value.get("delta_P_definition") if isinstance(value, dict) else getattr(value, "delta_P_definition", None)
    if tag is None or np.asarray(tag).size == 0 or not np.all(np.asarray(tag) == expected):
        raise ValueError(f"{context}: incompatible delta_P_definition {tag!r}; expected {expected!r}")


# ---------------------------------------------------------------------------------------
# [fluid-mean WP3, 30 Oct 2026] The micro pressure-level constraint (the formulation tag).
# Plan docs/notes/micro_fluid_mean_constraint_plan_2026-10-30.md rev 3, sections 2.3, 3.6 and
# WP3 (S3 (c), (d)); decision D8.  "whole_cell": (1/A) int P = P_re + lambda (the default until 2 Nov 2026);
# "fluid_mean_v1": (1/A_f) int chi P = P_re + lambda.  The tag is written only when it is not
# "whole_cell": absent means "whole_cell" everywhere (results, batch frames, used_corrections,
# checkpoints), so an off run's records are byte-identical to the ones before the tag existed.
#
# [fluid-mean WP8, 2 Nov 2026] Two defaults, kept apart (review docs/notes/micro_fluid_mean_default_
# review_2026-11-02.md section 3):
#   LEVEL_CONSTRAINT_DEFAULT      what an ABSENT tag or attribute means -- a record, a batch frame, a
#                                 checkpoint, a legacy CONFIG-*.py module or a MicroCase rebuilt from a
#                                 JSON record written before the field existed.  Stays "whole_cell":
#                                 every such object was made under the whole cell.
#   LEVEL_CONSTRAINT_RUN_DEFAULT  what a NEW run gets when its Case does not say (ehl.spec.MicroNumerics)
#                                 and what a solver-settings object built without the field gets
#                                 (PhaseNeutralSettings).  "fluid_mean_v1" since 2 Nov 2026 (WP8, option A).
# A mixed multiscale case that must stay on the whole cell (an in-flight run, a resume of an untagged
# checkpoint, the averaging campaign) pins numerics.micro.level_constraint = "whole_cell".  A full-film or
# single-scale case has no contact patch: its translation resolves the tag to "whole_cell"
# (effective_level_constraint), so its records and checkpoints are the ones they always were.
# ---------------------------------------------------------------------------------------
LEVEL_CONSTRAINT_DEFAULT = "whole_cell"
LEVEL_CONSTRAINT_RUN_DEFAULT = "fluid_mean_v1"
LEVEL_CONSTRAINTS = ("whole_cell", "fluid_mean_v1")


def effective_level_constraint(setting: str, mixed_micro: bool) -> str:
    """[fluid-mean WP8] The formulation a run with ``setting`` actually solves: ``setting`` for a mixed
    multiscale micro, ``"whole_cell"`` otherwise (no contact patch: the base class holds the whole cell by
    construction, bit for bit, V1).  The translator and ``MicroCase.from_case`` write this value."""
    if setting not in LEVEL_CONSTRAINTS:
        raise ValueError(f"level_constraint {setting!r}: one of {LEVEL_CONSTRAINTS}")
    return setting if mixed_micro else LEVEL_CONSTRAINT_DEFAULT


def level_constraint_of(value, default=LEVEL_CONSTRAINT_DEFAULT):
    """The tag a result / batch / carried set holds (``default`` when it has none)."""
    tag = value.get("level_constraint", None) if isinstance(value, dict) else getattr(value, "level_constraint", None)
    return default if tag is None else tag


def require_level_constraint(value, context="micro results", expected=LEVEL_CONSTRAINT_DEFAULT):
    """Fail closed unless ``value`` (a dict or an object) carries the level constraint ``expected``.

    An absent tag reads as ``"whole_cell"`` (S3 (c)), so untagged data passes only where the whole cell is
    expected; an unknown tag, a mixed tag array, or another formulation is refused."""
    if expected not in LEVEL_CONSTRAINTS:
        raise ValueError(f"{context}: unknown expected level_constraint {expected!r}; one of {LEVEL_CONSTRAINTS}")
    tag = level_constraint_of(value)
    arr = np.asarray(tag)
    if arr.size == 0:
        raise ValueError(f"{context}: empty level_constraint tag; expected {expected!r}")
    vals = {str(v) for v in arr.ravel()}
    if len(vals) != 1:
        raise ValueError(f"{context}: mixed level_constraint tags {sorted(vals)}; expected {expected!r}")
    (v,) = vals
    if v not in LEVEL_CONSTRAINTS:
        raise ValueError(f"{context}: unknown level_constraint {v!r}; expected {expected!r}")
    if v != expected:
        raise ValueError(f"{context}: level_constraint {v!r}, expected {expected!r} (results, checkpoints and "
                         f"carried corrections of the two formulations never mix; plan D8)")


def average_pressure_ledgers(ledgers):
    """Average each diagnostic over phases, never renormalise P_f by fluid area."""
    if not ledgers:
        raise ValueError("empty pressure phase sample")
    return {"delta_P_definition": DELTA_P_DEFINITION,
            **{key: float(np.mean([r[key] for r in ledgers])) for key in PRESSURE_FIELDS}}
