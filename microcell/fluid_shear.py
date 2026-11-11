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
              area), and -- with the task's frame ``R_inv`` (local -> global) -- the global-x
              min / max / sample mean over the whole set (zero extension) and the fluid-only min /
              max (``support_*``, conditional statistics);
per task      means by phase weight (equal), extrema by union over the phases, any invalid phase
              invalidates (``union_phases``).

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

import functools
import inspect
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from microcell.shear_reduce import (GLOBAL_X, SHEAR_REDUCE_VERSION, STATUS_INVALID_SAMPLES,
                              STATUS_NO_SUPPORT, STATUS_OK, STATUS_PHASE_INVALID, STATUS_STATE_INVALID,
                              VALID_STATUSES, ShearReduction, union_phases)

SHEAR_SAMPLER = "fluid_quadrature_sharp_v1"
SHEAR_EXTREMA_DEFINITION = SHEAR_SAMPLER + "/global_x/phase_union"

# ------------------------------------------------------------------------------- statuses
#: per-task status of the shear statistics (a fixed-width string array per node in run_micro)
SHEAR_OK = STATUS_OK                       # evaluated, some fluid support
SHEAR_NO_SUPPORT = STATUS_NO_SUPPORT       # no fluid point in any phase: whole-cell mean / min / max are
                                           # the physical zero, the support-only extrema NaN
SHEAR_PHASE_INVALID = STATUS_PHASE_INVALID  # a phase had a fluid point without a finite traction: NaN
SHEAR_FAILED = "failed"                    # the micro solve failed (or, under dedup, its representative's);
                                           # after run_micro's kNN fill the values are neighbour estimates
SHEAR_DEDUP_MEMBER = "dedup_member"        # (MICRO_DEDUP_TOL > 0) a member of a solved representative: the
                                           # global-x statistics are NaN (another cell, another frame, D6);
                                           # its means are the representative's, as every other channel's
SHEAR_NO_FRAME = "no_frame"                # the batch came without the macro frames (a caller that passes no
                                           # shear_frames): global-x statistics NaN, the local mean computed
SHEAR_LEGACY = "legacy"                    # restored from a checkpoint written before this definition
SHEAR_UNAVAILABLE = "unavailable"          # a result or a correction set without the shear fields
SHEAR_STATUS_DTYPE = "<U16"
SHEAR_STATUSES = (SHEAR_OK, SHEAR_NO_SUPPORT, SHEAR_PHASE_INVALID, SHEAR_FAILED, SHEAR_DEDUP_MEMBER,
                  SHEAR_NO_FRAME, SHEAR_LEGACY, SHEAR_UNAVAILABLE)

#: the run_micro arrays of the shear statistics and, per array, the statuses under which a NaN in it is
#: legitimate (the M1 rule, D4/D6: not a failed task, not kNN-filled, accepted by the batch stamp); a NaN
#: under any other status is a failure as before.  ``unavailable`` (a worker that predates the fields)
#: makes every shear array's NaN legitimate: the statistics are absent, the task is not failed.
SHEAR_NAN_IS_VALID = {
    "shear_mean_local_results": (SHEAR_UNAVAILABLE,),
    "shear_support_fraction_results": (SHEAR_UNAVAILABLE,),
    "shear_gx_min_results": (SHEAR_DEDUP_MEMBER, SHEAR_NO_FRAME, SHEAR_UNAVAILABLE),
    "shear_gx_max_results": (SHEAR_DEDUP_MEMBER, SHEAR_NO_FRAME, SHEAR_UNAVAILABLE),
    "shear_gx_mean_results": (SHEAR_DEDUP_MEMBER, SHEAR_NO_FRAME, SHEAR_UNAVAILABLE),
    "shear_gx_support_min_results": (SHEAR_NO_SUPPORT, SHEAR_DEDUP_MEMBER, SHEAR_NO_FRAME, SHEAR_UNAVAILABLE),
    "shear_gx_support_max_results": (SHEAR_NO_SUPPORT, SHEAR_DEDUP_MEMBER, SHEAR_NO_FRAME, SHEAR_UNAVAILABLE),
}
#: the frame-dependent (global-x) arrays among them
SHEAR_GX_ARRAYS = ("shear_gx_min_results", "shear_gx_max_results", "shear_gx_mean_results",
                   "shear_gx_support_min_results", "shear_gx_support_max_results")
#: the labels of the shear statistics in a run_micro result (not numerical outputs)
SHEAR_LABELS = ("shear_extrema_status", "shear_extrema_definition")

#: the status as a number per node in the committed-step XDMF series ``shear_fluid_status``
XDMF_STATUS_CODE = {SHEAR_OK: 0, SHEAR_NO_SUPPORT: 1, SHEAR_FAILED: 2, SHEAR_DEDUP_MEMBER: 3,
                    SHEAR_NO_FRAME: 4, SHEAR_PHASE_INVALID: 5, SHEAR_LEGACY: 6, SHEAR_UNAVAILABLE: 7}


# ------------------------------------------------------------------------- one phase, the union
@dataclass(frozen=True)
class FluidTractionPhase:
    """One phase's sampled fluid traction, as the sampler returns it.

    ``mean_local``       (t_x, t_y) [Pa]: the mean over the whole sample set, zero outside the fluid,
                         per total cell area, in the micro's local frame (micro_raw/mean_sampled)
    ``mean_existing_weights``  (t_x, t_y) [Pa]: the same samples weighted by the class's own mask of
                         ``taust`` (the DG0 ``chi`` of the mixed classes; none in the base class): a
                         check that the sampler evaluates the existing expression -- ``taust`` to roundoff
    ``reduction``        the global-x reduction (``ShearReduction``), None without a frame
    ``status``           ``ok`` | ``no_support`` | ``invalid_samples`` | ``state_invalid`` | ``no_frame``
    """
    mean_local: tuple
    mean_existing_weights: tuple
    reduction: Optional[ShearReduction]
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
    """The task's statistics from its phases (equal weights unless given): means by weight, extrema by
    union (``ehl.shear_reduce.union_phases``); a phase whose samples are invalid invalidates the task
    (``phase_invalid``, everything NaN); without a frame the global-x statistics are NaN
    (``no_frame``) and the local mean is kept."""
    phases = list(phases)
    if not phases:
        raise ValueError("no phase sample")
    w = np.ones(len(phases)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (len(phases),) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("one finite positive weight per phase is required")
    nan = float("nan")
    out = dict(status=SHEAR_PHASE_INVALID, mean_local=_NAN2, gx_min=nan, gx_max=nan, gx_mean=nan,
               gx_support_min=nan, gx_support_max=nan, support_fraction=nan,
               n_samples=int(sum(int(p.n_samples) for p in phases)))
    if any(p.status in (STATUS_INVALID_SAMPLES, STATUS_STATE_INVALID) for p in phases):
        return out
    ws = float(np.sum(w))
    out["mean_local"] = tuple(float(sum(wi * p.mean_local[c] for wi, p in zip(w, phases)) / ws)
                              for c in range(2))
    out["support_fraction"] = float(sum(wi * p.support_fraction for wi, p in zip(w, phases)) / ws)
    if any(p.reduction is None for p in phases):
        out["status"] = SHEAR_NO_FRAME
        return out
    u = union_phases([p.reduction for p in phases], w)
    out.update(status=u.status, gx_min=float(u.min), gx_max=float(u.max), gx_mean=float(u.sample_mean),
               gx_support_min=float(u.support_min), gx_support_max=float(u.support_max))
    return out


def shear_fields(union: dict) -> dict:
    """The appended ``MicroTaskResult`` fields of a task's union (the worker's success return)."""
    return {"shear_fluid_mean_local_Pa": tuple(float(v) for v in union["mean_local"]),
            "shear_fluid_gx_min_Pa": float(union["gx_min"]),
            "shear_fluid_gx_max_Pa": float(union["gx_max"]),
            "shear_fluid_gx_sample_mean_Pa": float(union["gx_mean"]),
            "shear_fluid_support_gx_min_Pa": float(union["gx_support_min"]),
            "shear_fluid_support_gx_max_Pa": float(union["gx_support_max"]),
            "shear_fluid_support_fraction": float(union["support_fraction"]),
            "shear_extrema_status": str(union["status"]),
            "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": int(union["n_samples"])}


def failed_shear_fields() -> dict:
    """The worker's NaN return: every statistic NaN, status ``failed`` (run_micro's kNN fill then
    writes neighbour estimates, never measured statistics)."""
    nan = float("nan")
    return {"shear_fluid_mean_local_Pa": _NAN2, "shear_fluid_gx_min_Pa": nan, "shear_fluid_gx_max_Pa": nan,
            "shear_fluid_gx_sample_mean_Pa": nan, "shear_fluid_support_gx_min_Pa": nan,
            "shear_fluid_support_gx_max_Pa": nan, "shear_fluid_support_fraction": nan,
            "shear_extrema_status": SHEAR_FAILED, "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": 0}


# --------------------------------------------------------------- run_micro -> predictions -> records
#: run_micro result key -> predictions / used_corrections key (the local mean's two components apart,
#: as ``taustx`` / ``tausty``)
_PRED_KEYS = (("shear_gx_min_results", "shear_gx_min"), ("shear_gx_max_results", "shear_gx_max"),
              ("shear_gx_mean_results", "shear_gx_mean"),
              ("shear_gx_support_min_results", "shear_gx_support_min"),
              ("shear_gx_support_max_results", "shear_gx_support_max"),
              ("shear_support_fraction_results", "shear_support_frac"),
              ("shear_extrema_status", "shear_extrema_status"),
              ("shear_extrema_definition", "shear_extrema_definition"))
#: the keys a correction set carries for the shear statistics (predictions, used_corrections, checkpoint)
SHEAR_RECORD_KEYS = ("tau_sampled_x", "tau_sampled_y") + tuple(k for _, k in _PRED_KEYS) + ("shear_batch_id",)


def shear_predictions(micro_results: dict) -> dict:
    """The predictions entries of a run_micro result's shear statistics (empty for a result without
    them: an older micro driver).  ``shear_batch_id`` is the batch stamp's id when there is one."""
    m = micro_results.get("shear_mean_local_results")
    if m is None:
        return {}
    m = np.asarray(m, dtype=float)
    out = {"tau_sampled_x": m[:, 0], "tau_sampled_y": m[:, 1]}
    for src, dst in _PRED_KEYS:
        if src in micro_results:
            out[dst] = micro_results[src]
    if micro_results.get("micro_batch_id") is not None:
        out["shear_batch_id"] = micro_results["micro_batch_id"]
    return out


def shear_record(predictions) -> dict:
    """The shear entries of a correction set about to be applied (``new_used`` / the steady handoff):
    the raw statistics of the batch, never relaxed -- they describe one micro state."""
    if not predictions:
        return {}
    return {k: predictions[k] for k in SHEAR_RECORD_KEYS if k in predictions}


def legitimate_nan_rows(results: dict, n: int) -> dict:
    """[extrema M1 + M2b + M3b] Per run_micro array, the rows whose NaN is legitimate by the status beside
    it: ``pmax_results`` / ``pmin_results`` by ``pressure_extrema_status`` (``ehl.pressure.
    EXTREMA_NAN_IS_VALID``), the shear arrays by ``shear_extrema_status`` (``SHEAR_NAN_IS_VALID``), the
    asperity / total arrays by ``shear_asperity_status`` / ``shear_total_status``
    (``ehl.contact_shear.NAN_IS_VALID``).
    A result without a status gets no entry for it (every NaN is then a failure, as before)."""
    out = {}
    st = results.get("pressure_extrema_status")
    if st is not None and np.shape(st) == (n,):
        from microcell.pressure import EXTREMA_NAN_IS_VALID
        m = np.isin(np.asarray(st), EXTREMA_NAN_IS_VALID)
        out["pmax_results"] = out["pmin_results"] = m
    st = results.get("shear_extrema_status")
    if st is not None and np.shape(st) == (n,):
        s = np.asarray(st)
        for name, valid in SHEAR_NAN_IS_VALID.items():
            out[name] = np.isin(s, valid)
    # [extrema M3b] the asperity / total statistics by the same rule, per array, by their own statuses
    # (ehl.contact_shear.NAN_IS_VALID)
    from microcell.contact_shear import legitimate_nan_rows as _contact_rows
    out.update(_contact_rows(results, n))
    return out


# --------------------------------------------------------------------- the frames (macro -> micro)
def macro_frames(solver, n: int):
    """(n, 3, 3) ``R_inv`` per macro node (local -> global; ``R v_global = v_local``), the macro solver's
    nodal CG1 9-vector ``R_inv`` (``ehl/macro/fe_problem.py`` RotationMatrixInverse) in node order --
    the matrix ``unrotate_taust`` applies to the batch's ``taust``.  None when the solver has none or its
    size is not 9 n (a caller whose tasks are not one per node)."""
    R_inv = getattr(solver, "R_inv", None)
    if R_inv is None:
        return None
    try:
        a = np.asarray(R_inv.vector().get_local(), dtype=float)
    except Exception:
        return None
    if a.size != 9 * int(n):
        return None
    return a.reshape(int(n), 3, 3).copy()


def takes_shear_frames(fn) -> bool:
    try:
        return "shear_frames" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def with_shear_frames(run_micro, state):
    """``run_micro`` with every batch given the macro's frames, ``shear_frames=macro_frames(state.solver,
    len(tasks))``, when the runner takes them (the phase-neutral pool; the legacy pool does not, and is
    returned unchanged).  An explicit ``shear_frames`` of the caller wins."""
    if run_micro is None or not takes_shear_frames(run_micro):
        return run_micro

    @functools.wraps(run_micro)
    def run_micro_with_frames(tasks, *args, **kwargs):
        if "shear_frames" not in kwargs:
            kwargs["shear_frames"] = macro_frames(getattr(state, "solver", None), len(tasks))
        return run_micro(tasks, *args, **kwargs)
    return run_micro_with_frames


# --------------------------------------------------------------------------- the macro series
#: the committed-step XDMF series of the shear statistics (solver attributes; ``fe_problem.
#: load_shear_diagnostics``): the sampled mean in the global frame (vector, Pa), the global-x whole-cell
#: min / max / sample mean and the fluid-only min / max (Pa), and the status code
SHEAR_SERIES = ("shear_fluid_mean_sampled", "shear_fluid_gx_min", "shear_fluid_gx_max",
                "shear_fluid_gx_sample_mean", "shear_fluid_support_gx_min", "shear_fluid_support_gx_max",
                "shear_fluid_status")
_SERIES_FROM = (("shear_fluid_gx_min", "shear_gx_min"), ("shear_fluid_gx_max", "shear_gx_max"),
                ("shear_fluid_gx_sample_mean", "shear_gx_mean"),
                ("shear_fluid_support_gx_min", "shear_gx_support_min"),
                ("shear_fluid_support_gx_max", "shear_gx_support_max"))


def export_arrays(source, n: int) -> dict:
    """The per-node values of the committed-step series from a correction set (predictions or
    used_corrections) as float64 arrays of length n: ``mean_local`` (n, 3) and the five global-x
    statistics, ``status`` (strings) and ``status_code``.  A set without the statistics (None, an older
    batch, a legacy checkpoint) gives NaN everywhere with status ``unavailable`` (``legacy`` where its
    status array says so): explicit, never zero-filled."""
    nan = np.full(int(n), np.nan)
    st = None if source is None else source.get("shear_extrema_status")
    status = (np.asarray(st).astype(SHEAR_STATUS_DTYPE) if st is not None and np.shape(st) == (n,)
              else np.full(int(n), SHEAR_UNAVAILABLE, dtype=SHEAR_STATUS_DTYPE))
    have = source is not None and "tau_sampled_x" in source and "tau_sampled_y" in source
    if have:
        mean = np.column_stack((np.asarray(source["tau_sampled_x"], dtype=float),
                                np.asarray(source["tau_sampled_y"], dtype=float), np.zeros(int(n))))
    else:
        mean = np.full((int(n), 3), np.nan)
        if st is None:
            status = np.full(int(n), SHEAR_UNAVAILABLE, dtype=SHEAR_STATUS_DTYPE)
    out = {"mean_local": mean, "status": status}
    for series, key in _SERIES_FROM:
        out[series] = (np.asarray(source[key], dtype=float).copy() if have and key in source else nan.copy())
    out["status_code"] = np.array([XDMF_STATUS_CODE.get(str(s), XDMF_STATUS_CODE[SHEAR_UNAVAILABLE])
                                   for s in status], dtype=float)
    batch = None if source is None else source.get("shear_batch_id")
    out["batch_id"] = None if batch is None else str(np.asarray(batch).reshape(-1)[0])
    defin = None if source is None else source.get("shear_extrema_definition")
    out["definition"] = None if defin is None else str(np.asarray(defin).reshape(-1)[0])
    return out


# ------------------------------------------------------------------------------- the metadata
#: the diagnostic contract (plan 2.4), versioned apart from the solver physics; saved beside the
#: committed-step series (``shear_extrema_metadata.json``), in the checkpoint
#: (``shear_extrema_definition.txt`` holds the definition) and the manifest's multiscale segments
SHEAR_METADATA = {
    "definition": SHEAR_EXTREMA_DEFINITION,
    "sampler": SHEAR_SAMPLER,
    "reduction": SHEAR_REDUCE_VERSION + " (ehl/shear_reduce.py)",
    "units": "Pa (traction); the status series is a code (xdmf_status_codes)",
    "quantity": ("fluid traction t_f = chi_f tau_f of one micro cell; tau = eta U_s / (H + 1e-30) + "
                 "(H / 2) grad P, the existing micro shear expression (micro_phase_neutral_class_mixed.py "
                 "spatial_homogenisation; the base class the same), U_s = 2 U the micro's surface speed"),
    "surface_and_sign": ("as found at 8ef021d, unchanged: tau is minus the fluid traction on the moving "
                         "surface (the ball), i.e. the traction of the ball on the film (inferred, M2a log "
                         "1.2); the macro force F = 2 Rc^2 int taust_rot and T_y = -Rc F_x (friction.py)"),
    "local_basis": ("e3 = radial unit normal, e1 = U_tan / |U_tan|, e2 = e3 x e1; at |U_tan| < 1e-18 the "
                    "fallback e1 = global x, not projected on the tangent plane (fe_problem.py "
                    "RotationMatrix)"),
    "rotation": ("R v_global = v_local; v_global = R_inv v_local, R_inv = R^T, nodal (CG1).  The micro "
                 "reduces with its node's R_inv applied as given (shear_frames); the macro rotates the "
                 "local sampled mean with the same nodal R_inv, as taust -> taust_rot"),
    "direction": {"name": "global_x", "vector": list(GLOBAL_X), "note": "decision D5: fixed global x only"},
    "samples": ("the form compiler's 'default' rule of the quadrature degree in force (the rule assemble "
                "uses for taust; 2 in the multiscale processes, utils/form_compiler.py) on every cell of the "
                "micro mesh, weight = rule weight x cell Jacobian (unit-cell area, sum 1)"),
    "film_at_samples": ("an analytic film (make_ideal_ht, CosineContactDescriptor.height) by its own "
                        "formula plus the CG1 deformation parts; every other film from the CG1 projection of "
                        "H0 h_nd the class computes (exact for a CG1 film: generated surfaces, contact / "
                        "ripple deformation)"),
    "pressure_gradient": "the cell's gradient of the CG1 pressure (macro ramp + w_fluc), x and y scaled by p_scale / Lx",
    "support": ("fluid where the extrema M1 nodal level set (phic_nd_func | projected h_nd - h_min) "
                "interpolated linearly to the sample is > 0; zero outside it (whole-cell set); the "
                "support_* extrema are over the fluid samples only, conditional statistics that need not "
                "contain the whole-cell mean"),
    "mean_denominator": "total micro-cell area",
    "statistics": {"per_phase": "sample_extremum", "per_task": "phase_union_extremum",
                   "sample_mean": "weighted mean of d . t on the same whole-cell sample set",
                   "mean": "three-component mean over total area (local frame per task; global frame on "
                           "the macro side)"},
    "phases": ("n_dphi phases with equal weights: means by weight, extrema by union; a phase with an "
               "invalid sample invalidates the task (phase_invalid)"),
    "means_kept_apart": {
        "micro_raw/mean_existing_integration": ("taust_x, taust_y = assemble(chi_dg0 tau dx) / area (DG0 "
                                                "mask) -> tau_res / taustx, tausty -> taust_rot"),
        "micro_raw/mean_sampled": ("shear_fluid_mean_local_Pa -> tau_sampled_x, tau_sampled_y -> "
                                   "shear_fluid_mean_sampled; the same samples with the sharp mask instead of "
                                   "the DG0 one (the rule's cell mean of the class's chi): differs from the "
                                   "existing mean by the in-cell covariance of mask and traction (and, for an "
                                   "analytic film under the projected-h mask, by the points where the "
                                   "analytic and projected level sets disagree); nothing is rescaled"),
    },
    "raw_vs_applied": ("the committed-step series hold the raw statistics of the correction set applied in "
                       "the accepted solve -- the latest batch's, or the carried set at a head solve: the "
                       "source of pmin / pmax and of the unrelaxed taust_rot; never relaxed or blended; "
                       "failed rows are kNN estimates (status failed), dedup members have no envelope"),
    "statuses": list(SHEAR_STATUSES),
    "xdmf_status_codes": dict(XDMF_STATUS_CODE),
    "series": list(SHEAR_SERIES),
}


def metadata_json() -> str:
    return json.dumps(SHEAR_METADATA, indent=1, sort_keys=True)
