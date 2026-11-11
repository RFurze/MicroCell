"""Asperity and total shear traction of a micro task, raw micro products vs applied macro products: the
exchange contract (extrema M3b, 23 Oct 2026).

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

Contact models: ``fft_elastic_v1`` is supported.  ``hertz_summit_v1`` and ``legacy_patch_v1`` have no
resolved pressure distribution (their ``p_asp_field`` is a display): their envelopes are
``unsupported_model`` (NaN, never approximated); their asperity mean is the patch law's cell mean,
``mu_material Pasp_hom U_hat`` -- the quantity the macro consumes -- and the total mean is the sampled fluid
mean plus it.

Raw micro products vs applied macro products (plan section 4), kept apart, never blended
-----------------------------------------------------------------------------------
``micro_raw/*``      the statistics of one micro batch (its ``micro_batch_id``, the batch stamp): the fluid
                     (M2b), asperity and total means and envelopes of the correction set the accepted solve
                     applied (or a carried set at a head solve) -- committed-step series
                     ``shear_fluid_*`` (M2b), ``shear_asperity_*``, ``shear_total_*``;
``macro_applied/*``  what the macro integrated into the reported torque: the fluid mean after correction
                     handling, ``taust_rot`` (M2a), and the asperity pressure the friction consumed,
                     ``_pasp_consumer()`` (``Pasp`` after the guard and the relaxation, or the Newton-like
                     predictor ``Pasp_eff``), exported as ``applied_Pasp`` (nondimensional, like ``Pasp``) with
                     its traction ``applied_asperity_mean = mu_material Pc p0 U / (|U| + 1e-30)`` at the nodes
                     (a nodal display: the torque integrates the law at the macro's quadrature points from
                     ``applied_Pasp``).  ``Ty_Nm`` integrates ``taust_rot``, ``Tasp_y_Nm`` the law of
                     ``applied_Pasp``; their sum is the reported total.
The raw / applied difference is visible node by node (``shear_asperity_mean`` against
``applied_asperity_mean``; ``shear_fluid_mean_sampled`` against ``taust_rot``) and integrated per committed
step in ``shear_extrema_steps.jsonl`` (``contact.Fx_N``).  A raw envelope describes its batch's state; the
applied means are not presented as lying inside it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from microcell.shear_reduce import (GLOBAL_X, SHEAR_REDUCE_VERSION, STATUS_NO_SUPPORT, STATUS_OK,
                              VALID_STATUSES, ShearReduction, union_phases)

#: M3a's sampler (``contact_sample.CONTACT_SAMPLER``; equal by test) on M2b's sample set
CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"
FLUID_SAMPLER = "fluid_quadrature_sharp_v1"            # ehl.fluid_shear.SHEAR_SAMPLER (equal by test)
CONTACT_SHEAR_DEFINITION = f"{CONTACT_SAMPLER}@{FLUID_SAMPLER}/global_x/phase_union"
#: M3a's gate-1 budget, stated before its comparison (M3a log 1.1): the sampled mean of p_a against the
#: solution's own Pasp_hom, relative; exactly 0 when Pasp_hom is 0
MEAN_CHECK_BUDGET = 1e-12
#: the regularisation of the macro's asperity direction (friction.py 188), in the units of the velocity
ZERO_SPEED_EPS = 1e-30

FFT_MODEL = "fft_elastic_v1"
UNSUPPORTED_MODELS = ("legacy_patch_v1", "hertz_summit_v1")

# ----------------------------------------------------------------------------------------- statuses
#: per component (asperity, total), per phase and per task; a fixed-width string array per node
OK = STATUS_OK                       # evaluated, some support
NO_SUPPORT = STATUS_NO_SUPPORT       # a valid state without the contribution (asperity: p = 0 everywhere):
                                     # whole-cell mean / min / max are the physical zero
UNSUPPORTED = "unsupported_model"    # hertz_summit_v1 / legacy_patch_v1: no resolved distribution; the
                                     # envelope NaN, the means the patch law's cell mean (asperity) and the
                                     # sampled fluid mean plus it (total)
STATE_INVALID = "state_invalid"      # no current contact solution at the samples (missing, or its state key
                                     # stale), or a fluid sample without a finite traction (total): NaN
STATE_MISMATCH = "state_mismatch"    # the extraction refreshed the contact solution after the samples were
                                     # taken: fluid and contact describe different states, NaN
CHECK_FAILED = "check_failed"        # the common-support checks failed (grid, sampled mean vs Pasp_hom): NaN
PHASE_INVALID = "phase_invalid"      # task: a phase was invalid (any of the three above)
FAILED = "failed"                    # the micro solve failed (or, under dedup, its representative's)
DEDUP_MEMBER = "dedup_member"        # (MICRO_DEDUP_TOL > 0) global-x statistics NaN; means the representative's
NO_FRAME = "no_frame"                # the batch came without the macro frames: global-x NaN, local means kept
LEGACY = "legacy"                    # restored from a checkpoint written before this definition
UNAVAILABLE = "unavailable"          # a result or a correction set without the fields (full film, older code)
STATUS_DTYPE = "<U24"
STATUSES = (OK, NO_SUPPORT, UNSUPPORTED, STATE_INVALID, STATE_MISMATCH, CHECK_FAILED, PHASE_INVALID, FAILED,
            DEDUP_MEMBER, NO_FRAME, LEGACY, UNAVAILABLE)
#: statuses under which the means are physical numbers (the local / macro mean vectors)
MEAN_VALID = (OK, NO_SUPPORT, UNSUPPORTED, NO_FRAME, DEDUP_MEMBER)
#: statuses under which the global-x envelope is physical
ENVELOPE_VALID = (OK, NO_SUPPORT)
#: the phase statuses that invalidate a task
_PHASE_BAD = (STATE_INVALID, STATE_MISMATCH, CHECK_FAILED)

#: the committed-step XDMF status code
XDMF_STATUS_CODE = {s: float(k) for k, s in enumerate(STATUSES)}

# ------------------------------------------------------------------------------- run_micro arrays
ASP_MEAN = "shear_asp_mean_local_results"                     # (N, 2) local frame [Pa]
ASP_GX_MIN = "shear_asp_gx_min_results"
ASP_GX_MAX = "shear_asp_gx_max_results"
ASP_GX_MEAN = "shear_asp_gx_mean_results"
ASP_SUPPORT = "shear_asp_support_fraction_results"            # mechanical support p_a > 0, by weight
TOT_MEAN = "shear_tot_mean_local_results"                     # (N, 2)
TOT_GX_MIN = "shear_tot_gx_min_results"
TOT_GX_MAX = "shear_tot_gx_max_results"
TOT_GX_MEAN = "shear_tot_gx_mean_results"
OVERLAP = "shear_contact_overlap_results"                     # fluid samples with p_a > 0, by weight
FORCE_ON_FLUID = "shear_contact_force_on_fluid_results"       # share of the contact force on fluid samples
OBSTRUCTION_NO_P = "shear_contact_obstruction_without_pressure_results"   # obstructed samples with p_a = 0
MEAN_CHECK = "shear_contact_mean_check_results"               # |sampled mean p_a - Pasp_hom| / Pasp_hom
ASP_STATUS = "shear_asperity_status"
TOT_STATUS = "shear_total_status"
DEFINITION_KEY = "shear_contact_definition"
LABELS = (ASP_STATUS, TOT_STATUS, DEFINITION_KEY)
GX_ARRAYS = (ASP_GX_MIN, ASP_GX_MAX, ASP_GX_MEAN, TOT_GX_MIN, TOT_GX_MAX, TOT_GX_MEAN)

_MEAN_NAN = (STATE_INVALID, STATE_MISMATCH, CHECK_FAILED, PHASE_INVALID, FAILED, UNAVAILABLE, LEGACY)
_GX_NAN = _MEAN_NAN + (UNSUPPORTED, DEDUP_MEMBER, NO_FRAME)
_SUPPORT_NAN = _MEAN_NAN + (UNSUPPORTED,)
#: run_micro array -> (the status array that governs it, the statuses under which its NaN is legitimate:
#: the M1 rule -- not a failure of the task, accepted by the batch stamp; a NaN under any other status is a
#: failure as before).  ``failed`` is among them: a failed task is failed by its solver channels anyway,
#: and its diagnostics are kNN estimates where neighbours have a value, else NaN (never a refused batch).
NAN_IS_VALID = {
    ASP_MEAN: (ASP_STATUS, _MEAN_NAN), TOT_MEAN: (TOT_STATUS, _MEAN_NAN),
    ASP_GX_MIN: (ASP_STATUS, _GX_NAN), ASP_GX_MAX: (ASP_STATUS, _GX_NAN), ASP_GX_MEAN: (ASP_STATUS, _GX_NAN),
    TOT_GX_MIN: (TOT_STATUS, _GX_NAN), TOT_GX_MAX: (TOT_STATUS, _GX_NAN), TOT_GX_MEAN: (TOT_STATUS, _GX_NAN),
    ASP_SUPPORT: (ASP_STATUS, _SUPPORT_NAN), OVERLAP: (ASP_STATUS, _SUPPORT_NAN),
    FORCE_ON_FLUID: (ASP_STATUS, _SUPPORT_NAN), OBSTRUCTION_NO_P: (ASP_STATUS, _SUPPORT_NAN),
    MEAN_CHECK: (ASP_STATUS, _SUPPORT_NAN),
}
#: what a dedup member gets instead of its representative's value (D6): its global-x statistics are NaN;
#: means and fractions are scattered as every other channel
DEDUP_MEMBER_NAN = GX_ARRAYS
#: a dedup member of a representative with one of these statuses keeps the representative's status
DEDUP_KEEP_REP_STATUS = tuple(s for s in STATUSES if s not in (OK, NO_SUPPORT, NO_FRAME))


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
    asperity: Optional[ShearReduction]    # the global-x reduction (None: no frame / invalid / unsupported)
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
                          grid_ok: bool = True, n_samples: Optional[int] = None) -> ContactTractionPhase:
    """A phase from M3a's ``reduce_contact_and_total`` results on the common samples: ``local`` (no
    rotation: the local-frame means) and ``glob`` (with the task's ``R_inv``: the global-x envelope; None
    without a frame).  ``contact_status`` is the contact sampler's (``ok`` / ``no_support`` /
    ``unsupported_model`` / ``state_invalid``); the checks of the module docstring are applied here.
    ``local`` may be None for an unsupported model (``n_samples`` then gives the count)."""
    n = int(n_samples) if local is None else int(local.asperity.n_samples)
    if contact_status == UNSUPPORTED:
        # no resolved distribution: the means are completed from Pasp_hom after the extraction
        # (finish_unsupported); the envelopes are never approximated
        nan = float("nan")
        return ContactTractionPhase(
            asperity_status=UNSUPPORTED, total_status=(UNSUPPORTED if fluid_state_valid else STATE_INVALID),
            asperity_mean_local=_nan2(), total_mean_local=_nan2(), asperity=None, total=None,
            asperity_support_fraction=nan, overlap_fraction=nan, force_on_fluid_fraction=nan,
            obstruction_without_pressure_fraction=nan, mean_check=nan, n_samples=n,
            contact_model=contact_model, u_hat_local=tuple(float(v) for v in u_hat),
            fluid_mean_local=tuple(float(v) for v in fluid_mean), solution=solution)
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


def finish_unsupported(phase: ContactTractionPhase, pasp_hom: float, mu_material: float) -> ContactTractionPhase:
    """``hertz_summit_v1`` / ``legacy_patch_v1`` after the extraction: the asperity mean is the patch law's
    cell mean ``mu Pasp_hom U_hat`` (local frame), the total mean the sampled fluid mean plus it; the
    envelopes stay NaN."""
    if phase.asperity_status != UNSUPPORTED:
        return phase
    a = tuple(float(mu_material) * float(pasp_hom) * float(u) for u in phase.u_hat_local[:2])
    if phase.total_status == UNSUPPORTED and all(np.isfinite(phase.fluid_mean_local)):
        t = tuple(float(f) + float(x) for f, x in zip(phase.fluid_mean_local, a))
    else:
        t = _nan2()
    if not all(np.isfinite(a)):
        return replace(phase, asperity_status=STATE_INVALID, total_status=STATE_INVALID)
    return replace(phase, asperity_mean_local=a, total_mean_local=t)


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
        if all(s == UNSUPPORTED for s in st):
            res["status"] = UNSUPPORTED
            return res
        if any(s == UNSUPPORTED for s in st):           # one model per task: cannot happen
            res["mean_local"] = _nan2()
            return res
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


#: MicroTaskResult field -> run_micro array (the per-node table); the definition and the sample count are
#: not kept per node
RESULT_ARRAYS = (("shear_asperity_mean_local_Pa", ASP_MEAN), ("shear_asperity_gx_min_Pa", ASP_GX_MIN),
                 ("shear_asperity_gx_max_Pa", ASP_GX_MAX), ("shear_asperity_gx_sample_mean_Pa", ASP_GX_MEAN),
                 ("shear_asperity_support_fraction", ASP_SUPPORT), ("shear_asperity_status", ASP_STATUS),
                 ("shear_total_mean_local_Pa", TOT_MEAN), ("shear_total_gx_min_Pa", TOT_GX_MIN),
                 ("shear_total_gx_max_Pa", TOT_GX_MAX), ("shear_total_gx_sample_mean_Pa", TOT_GX_MEAN),
                 ("shear_total_status", TOT_STATUS), ("shear_contact_overlap_fraction", OVERLAP),
                 ("shear_contact_force_on_fluid_fraction", FORCE_ON_FLUID),
                 ("shear_contact_obstruction_without_pressure_fraction", OBSTRUCTION_NO_P),
                 ("shear_contact_mean_check", MEAN_CHECK))


def allocate_arrays(n: int) -> dict:
    """The run_micro arrays of a batch of n tasks: NaN, statuses ``unavailable``."""
    out = {}
    for _f, key in RESULT_ARRAYS:
        if key in (ASP_STATUS, TOT_STATUS):
            out[key] = np.full(int(n), UNAVAILABLE, dtype=STATUS_DTYPE)
        elif key in (ASP_MEAN, TOT_MEAN):
            out[key] = np.full((int(n), 2), np.nan)
        else:
            out[key] = np.full(int(n), np.nan)
    return out


def legitimate_nan_rows(results: dict, n: int) -> dict:
    """Per run_micro array, the rows whose NaN is legitimate by the status beside it (``NAN_IS_VALID``).  A
    result without the status gets no entry (every NaN is then a failure, as before)."""
    out = {}
    for name, (status_key, valid) in NAN_IS_VALID.items():
        st = results.get(status_key)
        if name in results and st is not None and np.shape(st) == (n,):
            out[name] = np.isin(np.asarray(st), valid)
    return out


# --------------------------------------------------------------- run_micro -> predictions -> records
_PRED_KEYS = ((ASP_GX_MIN, "shear_asp_gx_min"), (ASP_GX_MAX, "shear_asp_gx_max"), (ASP_GX_MEAN, "shear_asp_gx_mean"),
              (ASP_SUPPORT, "shear_asp_support_frac"), (TOT_GX_MIN, "shear_tot_gx_min"),
              (TOT_GX_MAX, "shear_tot_gx_max"), (TOT_GX_MEAN, "shear_tot_gx_mean"),
              (OVERLAP, "shear_contact_overlap_frac"), (FORCE_ON_FLUID, "shear_contact_force_on_fluid_frac"),
              (OBSTRUCTION_NO_P, "shear_contact_obstruction_frac"), (MEAN_CHECK, "shear_contact_mean_check"),
              (ASP_STATUS, "shear_asp_status"), (TOT_STATUS, "shear_tot_status"),
              (DEFINITION_KEY, "shear_contact_definition"))
#: the keys a correction set carries for the asperity / total statistics (predictions, used_corrections,
#: checkpoint); ``shear_contact_batch_id`` is the batch stamp's id (the raw products' state id)
RECORD_KEYS = (("tau_asp_sampled_x", "tau_asp_sampled_y", "tau_tot_sampled_x", "tau_tot_sampled_y")
               + tuple(k for _, k in _PRED_KEYS) + ("shear_contact_batch_id",))


def contact_shear_predictions(micro_results: dict) -> dict:
    """The predictions entries of a run_micro result's asperity / total statistics (empty for a result
    without them: an older micro driver)."""
    if micro_results.get(ASP_MEAN) is None:
        return {}
    a = np.asarray(micro_results[ASP_MEAN], dtype=float)
    t = np.asarray(micro_results[TOT_MEAN], dtype=float)
    out = {"tau_asp_sampled_x": a[:, 0], "tau_asp_sampled_y": a[:, 1],
           "tau_tot_sampled_x": t[:, 0], "tau_tot_sampled_y": t[:, 1]}
    for src, dst in _PRED_KEYS:
        if src in micro_results:
            out[dst] = micro_results[src]
    if micro_results.get("micro_batch_id") is not None:
        out["shear_contact_batch_id"] = micro_results["micro_batch_id"]
    return out


def contact_shear_record(predictions) -> dict:
    """The asperity / total entries of a correction set about to be applied: the raw statistics of the
    batch, never relaxed -- they describe one micro state (``micro_raw``)."""
    if not predictions:
        return {}
    return {k: predictions[k] for k in RECORD_KEYS if k in predictions}


# ------------------------------------------------------------------------ state ids (checkpoint)
#: the batch ids a correction set can carry: the batch stamp's, and the copies each diagnostic family keeps
BATCH_ID_KEYS = ("micro_batch_id", "shear_batch_id", "shear_contact_batch_id")


def _id(v) -> str:
    return str(np.asarray(v).reshape(-1)[0])


def diagnostic_state_mismatch(record) -> Optional[str]:
    """None when the batch ids a correction set carries agree; otherwise what disagrees (a set whose
    diagnostics come from another batch than its solver channels: a mixed-state checkpoint)."""
    if not record:
        return None
    ids = {k: _id(record[k]) for k in BATCH_ID_KEYS if record.get(k) is not None and np.size(record[k])}
    if len(set(ids.values())) > 1:
        return ", ".join(f"{k}={v}" for k, v in sorted(ids.items()))
    return None


# ------------------------------------------------------------------------------ the macro series
#: the committed-step XDMF series of the raw asperity / total statistics (solver attributes;
#: ``fe_problem.load_contact_shear_diagnostics``): the means in the global frame (vector, Pa), the global-x
#: whole-cell min / max / sample mean (Pa), the status codes; and the support-mismatch map
RAW_SERIES = ("shear_asperity_mean", "shear_asperity_gx_min", "shear_asperity_gx_max",
              "shear_asperity_gx_sample_mean", "shear_asperity_status", "shear_total_mean", "shear_total_gx_min",
              "shear_total_gx_max", "shear_total_gx_sample_mean", "shear_total_status",
              "shear_contact_force_on_fluid")
#: the applied products (``fe_problem.applied_asperity_diagnostics``): the consumer field and its traction
APPLIED_SERIES = ("applied_Pasp", "applied_asperity_mean")
SERIES = RAW_SERIES + APPLIED_SERIES
_SCALAR_FROM = (("shear_asperity_gx_min", "shear_asp_gx_min"), ("shear_asperity_gx_max", "shear_asp_gx_max"),
                ("shear_asperity_gx_sample_mean", "shear_asp_gx_mean"), ("shear_total_gx_min", "shear_tot_gx_min"),
                ("shear_total_gx_max", "shear_tot_gx_max"), ("shear_total_gx_sample_mean", "shear_tot_gx_mean"),
                ("shear_contact_force_on_fluid", "shear_contact_force_on_fluid_frac"))


def _status(source, key, n):
    st = None if source is None else source.get(key)
    if st is not None and np.shape(st) == (n,):
        return np.asarray(st).astype(STATUS_DTYPE)
    return np.full(int(n), UNAVAILABLE, dtype=STATUS_DTYPE)


def export_arrays(source, n: int) -> dict:
    """The per-node values of the raw committed-step series from a correction set (predictions or
    used_corrections): ``asperity_local`` / ``total_local`` (n, 3), the six global-x statistics and the
    force share, both statuses and their codes.  A set without them gives NaN with status ``unavailable``
    (``legacy`` where its status array says so): explicit, never zero-filled."""
    n = int(n)
    nan = np.full(n, np.nan)
    have = source is not None and "tau_asp_sampled_x" in source and "tau_tot_sampled_x" in source
    out = {"asperity_status": _status(source, "shear_asp_status", n),
           "total_status": _status(source, "shear_tot_status", n)}
    for name, keys in (("asperity_local", ("tau_asp_sampled_x", "tau_asp_sampled_y")),
                       ("total_local", ("tau_tot_sampled_x", "tau_tot_sampled_y"))):
        out[name] = (np.column_stack((np.asarray(source[keys[0]], dtype=float),
                                      np.asarray(source[keys[1]], dtype=float), np.zeros(n)))
                     if have else np.full((n, 3), np.nan))
    for series, key in _SCALAR_FROM:
        out[series] = (np.asarray(source[key], dtype=float).copy() if have and key in source else nan.copy())
    for k in ("asperity", "total"):
        out[f"{k}_status_code"] = np.array([XDMF_STATUS_CODE.get(str(s), XDMF_STATUS_CODE[UNAVAILABLE])
                                            for s in out[f"{k}_status"]], dtype=float)
    batch = None if source is None else source.get("shear_contact_batch_id")
    out["batch_id"] = None if batch is None else _id(batch)
    defin = None if source is None else source.get("shear_contact_definition")
    out["definition"] = None if defin is None else _id(defin)
    return out


def applied_asperity_traction(pc_nd, p0: float, mu_material: float, u_nodal) -> np.ndarray:
    """``mu_material Pc p0 U / (|U| + 1e-30)`` per node (n, 3) [Pa]: the integrand of ``calc_asp_friction``
    (friction.py 186-191) at the nodes, from the consumer field ``Pc`` (nondimensional, per node) and the
    macro's nodal velocity ``U`` (n, 3).  Zero at zero speed (no static friction)."""
    u = np.asarray(u_nodal, dtype=float).reshape(-1, 3)
    pc = np.asarray(pc_nd, dtype=float).reshape(-1)
    mag = np.sqrt(np.sum(u * u, axis=1)) + ZERO_SPEED_EPS
    return float(mu_material) * (pc * float(p0))[:, None] * u / mag[:, None]


# --------------------------------------------------------------------------------- the metadata
METADATA = {
    "definition": CONTACT_SHEAR_DEFINITION,
    "contact_sampler": CONTACT_SAMPLER,
    "fluid_sampler": FLUID_SAMPLER,
    "reduction": SHEAR_REDUCE_VERSION + " (ehl/shear_reduce.py)",
    "units": "Pa (traction); applied_Pasp nondimensional (x p0 = Pa), like Pasp; status series are codes",
    "quantities": {
        "asperity": ("t_a = mu_material p_a U_hat at the fluid sampler's points; p_a the periodic P1 interpolant "
                     "of sol.p of the stored fft_elastic_v1 solution (D7; never p_asp_field); U_hat = U / (|U| + "
                     "1e-30) of the micro's sliding velocity at the end of its run, the macro's law "
                     "(calc_asp_friction) unchanged; zero outside p_a > 0"),
        "total": "t_t = t_f + t_a per sample, then reduced (extrema of components are never added)",
    },
    "contact_models": {"fft_elastic_v1": "supported",
                       "hertz_summit_v1": ("unsupported_model: envelope NaN; asperity mean = mu Pasp_hom U_hat "
                                           "(the patch law's cell mean), total mean = sampled fluid mean + it"),
                       "legacy_patch_v1": "unsupported_model, as hertz_summit_v1"},
    "checks": {"grid": "the contact grid is the micro mesh's n x n grid",
               "mean": f"sampled mean of p_a vs the solution's Pasp_hom <= {MEAN_CHECK_BUDGET:g} relative "
                       "(M3a gate-1 budget); else check_failed and nothing is published",
               "state": ("the solution's state key is current when the samples are taken (state_invalid "
                         "otherwise, no refresh); a refresh by the extraction afterwards: state_mismatch")},
    "support_mismatch": {"shear_contact_overlap": "fluid samples with p_a > 0 (by weight)",
                         "shear_contact_force_on_fluid": "share of the sampled contact force on fluid samples",
                         "shear_contact_obstruction_without_pressure": "obstructed samples with p_a = 0 (by weight)",
                         "policy": "reported, never suppressed"},
    "direction": {"name": "global_x", "vector": list(GLOBAL_X), "note": "decision D5: fixed global x only"},
    "mean_denominator": "total micro-cell area",
    "phases": "equal weights: means by weight, extrema by union; an invalid phase invalidates the task",
    "products": {
        "micro_raw": {"fluid": ["shear_fluid_mean_sampled", "shear_fluid_gx_min", "shear_fluid_gx_max",
                                "shear_fluid_gx_sample_mean", "shear_fluid_status"],
                      "asperity": ["shear_asperity_mean", "shear_asperity_gx_min", "shear_asperity_gx_max",
                                   "shear_asperity_gx_sample_mean", "shear_asperity_status"],
                      "total": ["shear_total_mean", "shear_total_gx_min", "shear_total_gx_max",
                                "shear_total_gx_sample_mean", "shear_total_status"],
                      "state_id": "the batch's micro_batch_id (shear_extrema_steps.jsonl: contact.micro_batch_id)"},
        "macro_applied": {"fluid": "taust_rot (the fluid mean after correction handling; Ty_Nm integrates it)",
                          "asperity": ("applied_Pasp = _pasp_consumer() (Pasp after the guard and the relaxation, "
                                       "or the Newton-like predictor Pasp_eff); Tasp_y_Nm integrates "
                                       "mu_material applied_Pasp p0 U / (|U| + 1e-30) at the macro's quadrature "
                                       "points; applied_asperity_mean is that traction at the nodes (a display)"),
                          "total": "Ty_Nm + Tasp_y_Nm"},
        "difference": ("raw minus applied, never blended: node by node (shear_asperity_mean vs "
                       "applied_asperity_mean, shear_fluid_mean_sampled vs taust_rot) and as forces per committed "
                       "step (shear_extrema_steps.jsonl contact.Fx_N); a raw envelope describes its batch, the "
                       "applied means are not presented as lying inside it"),
    },
    "statuses": list(STATUSES),
    "xdmf_status_codes": dict(XDMF_STATUS_CODE),
    "series": list(SERIES),
}


def metadata_json() -> str:
    return json.dumps(METADATA, indent=1, sort_keys=True)
