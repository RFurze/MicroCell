"""Asperity and total shear traction of a micro task (extrema M3b, 23 Oct 2026).

Pure numpy, no FEniCS (the worker, ``run_micro``, the pressure guard, the macro driver, the checkpoint and
FEniCS-free tests share it, as ``ehl/fluid_shear.py``).  Plan
``docs/plans/masked_pressure_extrema_and_shear_exports_2026-10-23.md`` sections 2.2 (contact / total,
paragraphs 4-5), 2.3, 2.4, 3.2, 4, 5 C-E and 6.3; decisions D5 (global x only), D6 (dedup members invalid),
D7 (the contact pressure from the stored FFT solution, never ``p_asp_field``) of
``docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md`` section 2.3.  Reused, not re-invented:
the fluid sample set, its status and transport pattern (extrema M2b, ``ehl/fluid_shear.py``), the reduction
(M2a, ``ehl/shear_reduce.py``) and the contact sampler and traction algebra (M3a,
``microscale/src/functions/contact_sample.py``).

Definition ``fft_nodal_p1_periodic_v1@fluid_quadrature_sharp_v1/global_x/phase_union``
------------------------------------------------------------------------------------
per phase (``microscale/src/functions/micro_diagnostics.py`` ``sample_contact_traction``)
    samples   the fluid sampler's points and weights (M2b: the ``default`` rule of the form compiler's
              quadrature degree on every cell of the micro mesh, unit-cell area, sum 1) -- one common set;
    contact   ``p_a`` = the periodic P1 interpolant of ``sol.p`` of the stored ``fft_elastic_v1`` solution
              of this phase's state (M3a's ``sample_contact``); the solution is used only if its state key
              is the current one (``_contact_key == _contact_state_key()``), it is never refreshed here (a
              refresh would change the mask the fluid was solved with), and a refresh by the extraction
              (``compute_asperity_load``) after the samples were taken makes the phase ``state_mismatch``;
    traction  ``t_a = mu_material p_a U_hat`` with ``U_hat = U / (|U| + 1e-30)`` of the micro's own sliding
              velocity at the end of its run (``Ux_nd_const``, ``Uy_nd_const``: the velocity of the Couette
              term of the fluid traction; the macro's law ``calc_asp_friction``, unchanged); zero outside
              its mechanical support ``p_a > 0``; ``t_t = t_f + t_a`` at each sample, each contribution its
              physical zero outside its own support, both kept where they overlap;
    reduced   by ``ehl.shear_reduce.reduce_traction``: the local-frame mean over the whole set (per total
              cell area), and with the task's frame ``R_inv`` the global-x min / max / sample mean (every
              sample rotated and projected first; the total formed per sample before it is reduced);
    checks    the common-support construction is published only when (a) the contact grid is the micro
              mesh's n x n grid and (b) the sampled mean of ``p_a`` reproduces the solution's own
              ``Pasp_hom`` to ``MEAN_CHECK_BUDGET`` (M3a's gate-1 budget, stated before its comparison);
              otherwise ``check_failed``.  The supports' mismatch (fluid samples carrying contact pressure,
              obstruction without pressure, the contact force on fluid samples) is recorded, never
              suppressed;
per task      means by phase weight (equal), extrema by union over the phases (``union_phases``); a phase
              whose state is invalid invalidates the task (``phase_invalid``).

Contact model: ``fft_elastic_v1``, the only one in this repo (``micro_diagnostics.sample_contact_traction``
raises for any other).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from microcell.shear_reduce import (STATUS_NO_SUPPORT, STATUS_OK,
                              VALID_STATUSES, ShearReduction, union_phases)

#: M3a's sampler (``contact_sample.CONTACT_SAMPLER``; equal by test) on M2b's sample set
CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"
FLUID_SAMPLER = "fluid_quadrature_sharp_v1"            # ehl.fluid_shear.SHEAR_SAMPLER (equal by test)
CONTACT_SHEAR_DEFINITION = f"{CONTACT_SAMPLER}@{FLUID_SAMPLER}/global_x/phase_union"
#: M3a's gate-1 budget, stated before its comparison (M3a log 1.1): the sampled mean of p_a against the
#: solution's own Pasp_hom, relative; exactly 0 when Pasp_hom is 0
MEAN_CHECK_BUDGET = 1e-12

from microcell.contact import FFT as FFT_MODEL      # the one contact law (one definition)

# ----------------------------------------------------------------------------------------- statuses
#: per component (asperity, total), per phase and per task; a fixed-width string array per node
OK = STATUS_OK                       # evaluated, some support
NO_SUPPORT = STATUS_NO_SUPPORT       # a valid state without the contribution (asperity: p = 0 everywhere):
                                     # whole-cell mean / min / max are the physical zero
STATE_INVALID = "state_invalid"      # no current contact solution at the samples (missing, or its state key
                                     # stale), or a fluid sample without a finite traction (total): NaN
STATE_MISMATCH = "state_mismatch"    # the extraction refreshed the contact solution after the samples were
                                     # taken: fluid and contact describe different states, NaN
CHECK_FAILED = "check_failed"        # the common-support checks failed (grid, sampled mean vs Pasp_hom): NaN
PHASE_INVALID = "phase_invalid"      # task: a phase was invalid (any of the three above)
FAILED = "failed"                    # the micro solve failed (or, under dedup, its representative's)
NO_FRAME = "no_frame"                # the batch came without the macro frames: global-x NaN, local means kept
UNAVAILABLE = "unavailable"          # a result or a correction set without the fields (full film, older code)
#: the phase statuses that invalidate a task
_PHASE_BAD = (STATE_INVALID, STATE_MISMATCH, CHECK_FAILED)


def _nan2():
    return (float("nan"), float("nan"))


# ------------------------------------------------------------------------------------ one phase
@dataclass(frozen=True)
class ContactTractionPhase:
    """One phase's asperity and total traction on the fluid sample set.  Pa; local means in the micro's
    frame, envelopes along global x (None without a frame)."""
    asperity_status: str
    total_status: str
    asperity_mean_local: tuple            # (t_x, t_y) per total cell area
    total_mean_local: tuple
    asperity: Optional[ShearReduction]    # the global-x reduction (None: no frame / invalid)
    total: Optional[ShearReduction]
    asperity_support_fraction: float
    overlap_fraction: float
    force_on_fluid_fraction: float
    obstruction_without_pressure_fraction: float
    mean_check: float                     # relative |sampled mean - Pasp_hom| (NaN when not checked)
    n_samples: int
    contact_model: str
    u_hat_local: tuple = (float("nan"), float("nan"), float("nan"))
    fluid_mean_local: tuple = (float("nan"), float("nan"))
    sampler: str = CONTACT_SAMPLER
    # the stored contact solution these samples came from (identity only; never pickled through IPC)
    solution: object = field(default=None, compare=False, repr=False)


def _contact_support_numbers(report: dict) -> tuple:
    return (float(report["fluid_and_mech"]["fraction"]), float(report["contact_force_share_on_fluid_samples"]),
            float(report["obstruction_without_pressure"]["fraction"]))


def invalid_phase(status: str, n_samples: int, contact_model: str, u_hat=None, fluid_mean=None,
                  mean_check: float = float("nan"), solution=None) -> ContactTractionPhase:
    """A phase whose asperity and total statistics cannot be published (NaN, the given status)."""
    nan = float("nan")
    return ContactTractionPhase(
        asperity_status=status, total_status=status, asperity_mean_local=_nan2(), total_mean_local=_nan2(),
        asperity=None, total=None, asperity_support_fraction=nan, overlap_fraction=nan,
        force_on_fluid_fraction=nan, obstruction_without_pressure_fraction=nan, mean_check=float(mean_check),
        n_samples=int(n_samples), contact_model=str(contact_model),
        u_hat_local=tuple(float(v) for v in (u_hat if u_hat is not None else (nan, nan, nan))),
        fluid_mean_local=tuple(float(v) for v in (fluid_mean if fluid_mean is not None else (nan, nan))),
        solution=solution)


def mean_check(p, weights, pasp_hom: float) -> float:
    """Relative difference of the weighted sample mean of ``p`` and the solution's ``Pasp_hom`` (0 when both
    are exactly 0; inf when only the sample mean is nonzero)."""
    w = np.asarray(weights, dtype=float)
    m = float(np.sum(w * np.asarray(p, dtype=float)) / np.sum(w))
    P = float(pasp_hom)
    if P > 0.0:
        return abs(m - P) / P
    return 0.0 if m == 0.0 else float("inf")


def phase_from_reductions(local, glob, *, contact_status: str, contact_model: str, fluid_state_valid: bool,
                          mean_check_rel: float, u_hat, fluid_mean, solution=None,
                          grid_ok: bool = True) -> ContactTractionPhase:
    """A phase from M3a's ``reduce_contact_and_total`` results on the common samples: ``local`` (no
    rotation: the local-frame means) and ``glob`` (with the task's ``R_inv``: the global-x envelope; None
    without a frame).  ``contact_status`` is the contact sampler's (``ok`` / ``no_support`` /
    ``state_invalid``); the checks of the module docstring are applied here."""
    n = int(local.asperity.n_samples)
    if contact_status not in (OK, NO_SUPPORT):
        return invalid_phase(STATE_INVALID, n, contact_model, u_hat, fluid_mean, solution=solution)
    if not grid_ok or not (mean_check_rel <= MEAN_CHECK_BUDGET):
        return invalid_phase(CHECK_FAILED, n, contact_model, u_hat, fluid_mean, mean_check_rel, solution)
    a_st = local.asperity.status
    if a_st not in VALID_STATUSES:
        return invalid_phase(STATE_INVALID, n, contact_model, u_hat, fluid_mean, mean_check_rel, solution)
    t_st = local.total.status if fluid_state_valid else STATE_INVALID
    if t_st not in VALID_STATUSES:
        t_st = STATE_INVALID
    if glob is None:
        a_st = NO_FRAME
        t_st = NO_FRAME if t_st in VALID_STATUSES else t_st
    else:
        a_st = glob.asperity.status if glob.asperity.status in VALID_STATUSES else STATE_INVALID
        if t_st in VALID_STATUSES:
            t_st = glob.total.status if glob.total.status in VALID_STATUSES else STATE_INVALID
    ov, ff, onp = _contact_support_numbers(local.support)
    t_ok = t_st in VALID_STATUSES + (NO_FRAME,)
    return ContactTractionPhase(
        asperity_status=a_st, total_status=t_st,
        asperity_mean_local=tuple(float(v) for v in local.asperity.mean[:2]),
        total_mean_local=(tuple(float(v) for v in local.total.mean[:2]) if t_ok else _nan2()),
        asperity=(glob.asperity if glob is not None else None),
        total=(glob.total if (glob is not None and t_ok) else None),
        asperity_support_fraction=float(local.asperity.support_fraction), overlap_fraction=ov,
        force_on_fluid_fraction=ff, obstruction_without_pressure_fraction=onp, mean_check=float(mean_check_rel),
        n_samples=n, contact_model=contact_model, u_hat_local=tuple(float(v) for v in u_hat),
        fluid_mean_local=tuple(float(v) for v in fluid_mean), solution=solution)


def mismatch(phase: ContactTractionPhase) -> ContactTractionPhase:
    """The extraction refreshed the solution after the samples were taken: nothing is published."""
    return invalid_phase(STATE_MISMATCH, phase.n_samples, phase.contact_model, phase.u_hat_local,
                         phase.fluid_mean_local)


# ----------------------------------------------------------------------------------- the union
def _wmean(vals, w):
    return float(sum(wi * float(v) for wi, v in zip(w, vals)) / float(np.sum(w)))


def union_contact_traction(phases: Sequence[ContactTractionPhase], weights=None) -> dict:
    """The task's asperity and total statistics from its phases (equal weights unless given): means by
    weight, extrema by union (``ehl.shear_reduce.union_phases``), an invalid phase invalidates the task
    (``phase_invalid``); without a frame the envelopes are NaN (``no_frame``) and the means kept."""
    phases = list(phases)
    if not phases:
        raise ValueError("no phase sample")
    w = np.ones(len(phases)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (len(phases),) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("one finite positive weight per phase is required")
    nan = float("nan")
    out = {"n_samples": int(sum(int(p.n_samples) for p in phases)),
           "contact_model": phases[0].contact_model}

    def component(name):
        st = [getattr(p, f"{name}_status") for p in phases]
        res = dict(status=PHASE_INVALID, mean_local=_nan2(), gx_min=nan, gx_max=nan, gx_mean=nan)
        if any(s in _PHASE_BAD for s in st):
            return res
        if any(s == UNAVAILABLE for s in st):            # not computed (no friction coefficient given)
            res["status"] = UNAVAILABLE
            return res
        means = [getattr(p, f"{name}_mean_local") for p in phases]
        res["mean_local"] = tuple(_wmean([m[c] for m in means], w) for c in range(2))
        if any(s == NO_FRAME for s in st):
            res["status"] = NO_FRAME
            return res
        u = union_phases([getattr(p, name) for p in phases], w)
        res.update(status=u.status, gx_min=float(u.min), gx_max=float(u.max), gx_mean=float(u.sample_mean))
        return res

    out["asperity"] = component("asperity")
    out["total"] = component("total")
    asp_ok = out["asperity"]["status"] in (OK, NO_SUPPORT, NO_FRAME)
    for key, attr in (("support_fraction", "asperity_support_fraction"), ("overlap", "overlap_fraction"),
                      ("force_on_fluid", "force_on_fluid_fraction"),
                      ("obstruction_without_pressure", "obstruction_without_pressure_fraction")):
        out[key] = _wmean([getattr(p, attr) for p in phases], w) if asp_ok else nan
    checks = [float(p.mean_check) for p in phases if np.isfinite(float(p.mean_check))]
    out["mean_check"] = max(checks) if checks else nan      # the worst phase (kept for a failed check too)
    return out


# --------------------------------------------------------------------- the worker's result fields
#: the appended ``MicroTaskResult`` fields, in their order (the worker's success and NaN returns)
FIELDS = ("shear_asperity_mean_local_Pa", "shear_asperity_gx_min_Pa", "shear_asperity_gx_max_Pa",
          "shear_asperity_gx_sample_mean_Pa", "shear_asperity_support_fraction", "shear_asperity_status",
          "shear_total_mean_local_Pa", "shear_total_gx_min_Pa", "shear_total_gx_max_Pa",
          "shear_total_gx_sample_mean_Pa", "shear_total_status", "shear_contact_overlap_fraction",
          "shear_contact_force_on_fluid_fraction", "shear_contact_obstruction_without_pressure_fraction",
          "shear_contact_mean_check", "shear_contact_definition", "shear_contact_n_samples")


def contact_shear_fields(union: dict) -> dict:
    """The appended ``MicroTaskResult`` fields of a task's union."""
    a, t = union["asperity"], union["total"]
    return {"shear_asperity_mean_local_Pa": tuple(float(v) for v in a["mean_local"]),
            "shear_asperity_gx_min_Pa": float(a["gx_min"]), "shear_asperity_gx_max_Pa": float(a["gx_max"]),
            "shear_asperity_gx_sample_mean_Pa": float(a["gx_mean"]),
            "shear_asperity_support_fraction": float(union["support_fraction"]),
            "shear_asperity_status": str(a["status"]),
            "shear_total_mean_local_Pa": tuple(float(v) for v in t["mean_local"]),
            "shear_total_gx_min_Pa": float(t["gx_min"]), "shear_total_gx_max_Pa": float(t["gx_max"]),
            "shear_total_gx_sample_mean_Pa": float(t["gx_mean"]), "shear_total_status": str(t["status"]),
            "shear_contact_overlap_fraction": float(union["overlap"]),
            "shear_contact_force_on_fluid_fraction": float(union["force_on_fluid"]),
            "shear_contact_obstruction_without_pressure_fraction": float(union["obstruction_without_pressure"]),
            "shear_contact_mean_check": float(union["mean_check"]),
            "shear_contact_definition": CONTACT_SHEAR_DEFINITION,
            "shear_contact_n_samples": int(union["n_samples"])}


def failed_contact_shear_fields() -> dict:
    """The worker's NaN return: every statistic NaN, both statuses ``failed``."""
    nan = float("nan")
    out = {k: nan for k in FIELDS}
    out.update(shear_asperity_mean_local_Pa=_nan2(), shear_total_mean_local_Pa=_nan2(),
               shear_asperity_status=FAILED, shear_total_status=FAILED,
               shear_contact_definition=CONTACT_SHEAR_DEFINITION, shear_contact_n_samples=0)
    return out
