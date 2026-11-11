"""Fluid shear traction statistics of a micro task: the exchange contract (extrema M2b, 23 Oct 2026).

Pure numpy, no FEniCS (the worker, ``run_micro``, the pressure guard, the macro driver, the
checkpoint and FEniCS-free tests share it).  Plan
``docs/plans/masked_pressure_extrema_and_shear_exports_2026-10-23.md`` sections 2.2 (fluid), 2.4,
3.2 (fluid) and 5 C-E; decisions D5 (global x only) and D6 (dedup members invalid) of
``docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md`` section 2.3; the convention
(surface, sign, basis, rotation, torque) as M2a found it, ``docs/notes/extrema_m2a_implementation_log_
2026-10-23.md`` section 1, unchanged.  The reduction is M2a's ``ehl/shear_reduce.py``; the sampler is
``microscale/src/functions/micro_diagnostics.py`` (``sample_fluid_traction``).

Definition ``fluid_quadrature_sharp_v1/global_x/phase_union`` (Rob's choice of 23 Oct 2026: the
ledger's quadrature points, not the M1 vertices -- on the vertices the sampled mean differs from
``taust`` by the vertex rule's sampling of eta U / H near a contact edge as well as by the mask):

per phase (the sampler ``fluid_quadrature_sharp_v1``)
    samples   the points of the form compiler's ``default`` rule of the degree in force (the rule
              ``assemble`` uses for ``taust``) on every cell of the micro mesh; weight = rule weight
              x cell Jacobian, in unit-cell area (sum 1);
    traction  the existing micro shear expression at each point, local frame, Pa:
              ``tau = eta U_s / (H + 1e-30) + (H / 2) grad P`` (the cell's gradient of the CG1
              pressure; the film as the form evaluates it);
    support   fluid where the extrema M1 nodal level set (``phic_nd_func`` | the projected
              ``h_nd - h_min``), interpolated linearly to the point, is > 0 -- the sharp
              ``chi_fluid`` test at the point; the traction is zero outside it (t_f = chi_f tau_f);
    reduced   by ``reduce_traction``: the local-frame mean over the whole set (per total cell
              area).  The parent's global-x envelope (min / max / sample mean, with the macro
              frame ``R_inv``) is not produced: a standalone cell has no macro frame (WP4);
per task      means by phase weight (equal); any invalid phase invalidates.

The two cell means, kept apart (plan 3.2): ``micro_raw/mean_existing_integration`` is the existing
``taust`` (``assemble(chi_dg0 * tau)`` / area: the DG0-projected mask; ``tau_res`` -> ``taust_rot``),
``micro_raw/mean_sampled`` is the sampled mean here (``shear_fluid_mean_local_Pa`` ->
``shear_fluid_mean_sampled``).  On the same representation (the samples weighted by the DG0 mask) the
sampler reproduces ``taust`` to roundoff; the sampled mean differs from it by the mask only.  The DG0
mask is the rule's cell mean of the class's own ``chi_fluid`` at the same points, so the difference is
sum_T sum_q W (chi_q - chibar_T)(t_q - tbar_T) / area -- the in-cell covariance of mask and traction,
zero on uncut cells and for a traction constant in a cell -- when ``chi_fluid`` is the conditional on the
sampler's level set (``phic_nd_func``; a CG1 film), plus sum W (chi_M1 - chi_class) t / area for an
analytic film under the projected-h mask (the class tests the analytic film, M1 its CG1 projection).
Nothing rescales either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from microcell.shear_reduce import (STATUS_INVALID_SAMPLES, STATUS_NO_SUPPORT, STATUS_OK,
                                    STATUS_PHASE_INVALID, STATUS_STATE_INVALID)

SHEAR_SAMPLER = "fluid_quadrature_sharp_v1"
SHEAR_EXTREMA_DEFINITION = SHEAR_SAMPLER + "/global_x/phase_union"

# ------------------------------------------------------------------------------- statuses
#: per-task status of the shear statistics: ``ok`` (a phase has fluid samples), ``no_support``
#: (a valid state with no fluid sample: the mean is the physical zero) or one of
SHEAR_PHASE_INVALID = STATUS_PHASE_INVALID  # a phase had a fluid point without a finite traction: NaN
SHEAR_FAILED = "failed"                    # the micro solve failed


# ------------------------------------------------------------------------- one phase, the union
@dataclass(frozen=True)
class FluidTractionPhase:
    """One phase's sampled fluid traction, as the sampler returns it.

    ``mean_local``       (t_x, t_y) [Pa]: the mean over the whole sample set, zero outside the fluid,
                         per total cell area, in the micro's local frame (micro_raw/mean_sampled)
    ``mean_existing_weights``  (t_x, t_y) [Pa]: the same samples weighted by the class's own mask of
                         ``taust`` (the DG0 ``chi`` of the mixed classes; none in the base class): a
                         check that the sampler evaluates the existing expression -- ``taust`` to roundoff
    ``status``           ``ok`` | ``no_support`` | ``invalid_samples`` | ``state_invalid``
    """
    mean_local: tuple
    mean_existing_weights: tuple
    status: str
    n_samples: int
    n_support: int
    support_fraction: float
    quadrature_degree: int
    mask_source: str
    film_source: str
    sampler: str = SHEAR_SAMPLER


_NAN2 = (float("nan"), float("nan"))


def union_fluid_traction(phases: Sequence[FluidTractionPhase], weights=None) -> dict:
    """The task's statistics from its phases (equal weights unless given): means by weight; a phase
    whose samples are invalid invalidates the task (``phase_invalid``, everything NaN); ``ok`` when a
    phase has fluid samples, else ``no_support``."""
    phases = list(phases)
    if not phases:
        raise ValueError("no phase sample")
    w = np.ones(len(phases)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (len(phases),) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("one finite positive weight per phase is required")
    out = dict(status=SHEAR_PHASE_INVALID, mean_local=_NAN2, support_fraction=float("nan"),
               n_samples=int(sum(int(p.n_samples) for p in phases)))
    if any(p.status in (STATUS_INVALID_SAMPLES, STATUS_STATE_INVALID) for p in phases):
        return out
    ws = float(np.sum(w))
    out["mean_local"] = tuple(float(sum(wi * p.mean_local[c] for wi, p in zip(w, phases)) / ws)
                              for c in range(2))
    out["support_fraction"] = float(sum(wi * p.support_fraction for wi, p in zip(w, phases)) / ws)
    out["status"] = STATUS_OK if any(p.status == STATUS_OK for p in phases) else STATUS_NO_SUPPORT
    return out


def shear_fields(union: dict) -> dict:
    """The appended ``MicroTaskResult`` fields of a task's union (the worker's success return)."""
    return {"shear_fluid_mean_local_Pa": tuple(float(v) for v in union["mean_local"]),
            "shear_fluid_support_fraction": float(union["support_fraction"]),
            "shear_extrema_status": str(union["status"]),
            "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": int(union["n_samples"])}


def failed_shear_fields() -> dict:
    """The worker's NaN return: every statistic NaN, status ``failed`` (run_micro's kNN fill then
    writes neighbour estimates, never measured statistics)."""
    nan = float("nan")
    return {"shear_fluid_mean_local_Pa": _NAN2, "shear_fluid_support_fraction": nan,
            "shear_extrema_status": SHEAR_FAILED, "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": 0}
