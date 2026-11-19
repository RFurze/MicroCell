"""Shear traction of a micro cell (numpy only): the reduction of sampled traction to means and
extrema, and the fluid, asperity and total statistics of a cell.

The samples come from ``microcell/samplers.py``: the fluid traction at the quadrature points of
every cell (``sample_fluid_traction``) and the contact pressure at the same points
(``sample_contact_traction``).  All tractions are in Pa, in the cell frame.

Reduction of one state (``reduce_traction``).  The inputs are the traction at N samples, the
sample weights, the samples that were evaluated (``valid``), the samples where the contribution
is present (``support``) and a unit direction ``d``.  The traction is zero outside the support.
The result is the weighted mean vector over all samples (the mean per cell area for area
weights), the min, max and weighted mean of ``d . t`` over all samples, and the min and max of
``d . t`` over the support alone.

Fluid traction: ``tau = eta U_s / (H + 1e-30) + (H / 2) grad P`` in the fluid, zero outside it.
Asperity traction: ``t_a = mu p_a U / (|U| + 1e-30)`` where ``p_a > 0``.  Total traction:
``t_f + t_a``, formed at each sample before it is reduced.

The asperity and total statistics are valid only where the contact grid is the n by n grid of
the mesh and the sampled mean of ``p_a`` reproduces ``Pasp_hom`` of the contact solution to
``MEAN_CHECK_BUDGET``.  Otherwise the status is ``check_failed``.

The statistics of a cell are the means over its phases, with equal weights unless given.  One
invalid phase makes the cell ``phase_invalid``.

Statuses of a reduction:
  ``ok``               evaluated, some support
  ``no_support``       valid state without the contribution: mean, min and max are zero and the
                       support extrema are NaN
  ``invalid_samples``  a support sample was not evaluated or is not finite: NaN
  ``state_invalid``    the state is invalid or has no samples: NaN
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STATUS_OK = "ok"
STATUS_NO_SUPPORT = "no_support"
STATUS_INVALID_SAMPLES = "invalid_samples"
STATUS_STATE_INVALID = "state_invalid"
STATUS_PHASE_INVALID = "phase_invalid"
#: statuses with valid numbers; the others carry NaN
VALID_STATUSES = (STATUS_OK, STATUS_NO_SUPPORT)

STATISTIC_SAMPLE = "sample_extremum"

#: default projection direction: the x axis of the frame of the samples
GLOBAL_X = (1.0, 0.0, 0.0)

#: tolerance on |d| = 1 for a direction
UNIT_TOL = 1e-12

_NAN3 = (float("nan"),) * 3


@dataclass(frozen=True)
class ShearReduction:
    """One reduced traction distribution [Pa], with vectors in the frame of the samples.

    ``mean``          weighted mean vector over all samples (zero outside the support)
    ``min``, ``max``  extrema of ``d . t`` over all samples, the zeros outside the support included
    ``sample_mean``   weighted mean of ``d . t`` over all samples (``d . mean``)
    ``support_min``, ``support_max``  extrema of ``d . t`` over the support samples; NaN without
                      support
    ``support_fraction``  support weight divided by total weight
    """
    mean: tuple
    min: float
    max: float
    sample_mean: float
    support_min: float
    support_max: float
    support_fraction: float
    n_samples: int
    n_support: int
    status: str
    direction: tuple
    statistic: str = STATISTIC_SAMPLE

    @property
    def valid(self) -> bool:
        """True for the statuses ``ok`` and ``no_support``."""
        return self.status in VALID_STATUSES


def _direction(direction) -> np.ndarray:
    """``direction`` as a (3,) array.  Raises unless it is a finite unit vector."""
    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        raise ValueError(f"direction must be three finite numbers, got {direction!r}")
    if abs(float(np.linalg.norm(d)) - 1.0) > UNIT_TOL:
        raise ValueError(f"direction must be a unit vector (|d| = {np.linalg.norm(d):.15g})")
    return d


def _invalid(status: str, d: np.ndarray, n: int, n_support: int, frac: float,
             statistic: str = STATISTIC_SAMPLE) -> ShearReduction:
    """``ShearReduction`` with NaN statistics and the given status."""
    nan = float("nan")
    return ShearReduction(mean=_NAN3, min=nan, max=nan, sample_mean=nan, support_min=nan,
                          support_max=nan, support_fraction=frac, n_samples=int(n),
                          n_support=int(n_support), status=status,
                          direction=tuple(float(v) for v in d), statistic=statistic)


def reduce_traction(vectors, weights, valid, direction=GLOBAL_X, *, support=None,
                    state_valid: bool = True) -> ShearReduction:
    """Reduces the traction sampled in one state of the cell.

    vectors: sampled traction, (N, 2) or (N, 3) [Pa], cell frame.  A zero third component is
        added to two columns.
    weights: (N,) finite positive sample weights.  The means are normalised by their sum over
        all N samples, not by the weight of the support.
    valid: (N,) bool, the sample was evaluated.  Samples outside the support are not read.
    direction: unit vector in the frame of the samples, on which each sample is projected.
    support: (N,) bool, where the contribution is present.  Default: every sample.
    state_valid: False for an invalid state, which gives NaN and the status ``state_invalid``.
    Returns a ``ShearReduction``.
    """
    d = _direction(direction)
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError(f"vectors must have shape (N, 2) or (N, 3), got {v.shape}")
    n = v.shape[0]
    if v.shape[1] == 2:
        v = np.column_stack((v, np.zeros(n)))
    w = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    sup = np.ones(n, dtype=bool) if support is None else np.asarray(support, dtype=bool).reshape(-1)
    if w.shape != (n,) or ok.shape != (n,) or sup.shape != (n,):
        raise ValueError(f"weights, valid and support must have shape ({n},)")
    if n and (not np.all(np.isfinite(w)) or np.any(w <= 0.0)):
        raise ValueError("weights must be finite and positive")

    n_support = int(np.count_nonzero(sup))
    total = float(np.sum(w)) if n else 0.0
    frac = float(np.sum(w[sup]) / total) if n else float("nan")
    if not state_valid or n == 0:
        return _invalid(STATUS_STATE_INVALID, d, n, n_support, frac)
    evaluated = ok & np.all(np.isfinite(v), axis=1)
    if np.any(sup & ~evaluated):
        return _invalid(STATUS_INVALID_SAMPLES, d, n, n_support, frac)

    t = np.where(sup[:, None], v, 0.0)          # zero outside the support
    s = t @ d                                    # projection on d
    mean = tuple(float(c) for c in (w @ t) / total)
    sample_mean = float(w @ s / total)
    if n_support:
        smin, smax = float(np.min(s[sup])), float(np.max(s[sup]))
        status = STATUS_OK
    else:
        smin = smax = float("nan")
        status = STATUS_NO_SUPPORT
    return ShearReduction(mean=mean, min=float(np.min(s)), max=float(np.max(s)),
                          sample_mean=sample_mean, support_min=smin, support_max=smax,
                          support_fraction=frac, n_samples=n, n_support=n_support, status=status,
                          direction=tuple(float(c) for c in d))




from dataclasses import dataclass
from typing import Sequence

import numpy as np


SHEAR_SAMPLER = "fluid_quadrature_sharp_v1"
SHEAR_EXTREMA_DEFINITION = SHEAR_SAMPLER + "/global_x/phase_union"

# ---- fluid traction of a cell ----
#: status of the fluid shear statistics of a cell: ``ok`` (a phase has fluid samples),
#: ``no_support`` (valid, no fluid sample: the mean is zero) or one of
SHEAR_PHASE_INVALID = STATUS_PHASE_INVALID  # a phase had a fluid point without a finite traction: NaN
SHEAR_FAILED = "failed"                    # the micro solve failed


@dataclass(frozen=True)
class FluidTractionPhase:
    """Sampled fluid traction of one phase, as returned by ``samplers.sample_fluid_traction``.

    ``mean_local``       (t_x, t_y) [Pa]: mean over all samples, zero outside the fluid, per cell
                         area, in the cell frame
    ``mean_existing_weights``  (t_x, t_y) [Pa]: the same samples weighted by the mask of the
                         solver's ``taust`` integral (DG0 ``chi`` in the mixed classes, none in the
                         base class); it reproduces ``taust`` to rounding
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
    """Fluid traction statistics of the cell from its phases.

    phases: ``FluidTractionPhase`` objects.  weights: one positive weight per phase; equal if None.
    Returns a dict: ``mean_local`` (t_x, t_y) [Pa] and ``support_fraction``, the weighted means
    over the phases; ``n_samples``; ``status``, which is ``phase_invalid`` (with NaN) if any
    phase is invalid, ``ok`` if a phase has fluid samples, else ``no_support``.
    """
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
    """Result entries of the fluid shear statistics, from the dict of ``union_fluid_traction``."""
    return {"shear_fluid_mean_local_Pa": tuple(float(v) for v in union["mean_local"]),
            "shear_fluid_support_fraction": float(union["support_fraction"]),
            "shear_extrema_status": str(union["status"]),
            "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": int(union["n_samples"])}


def failed_shear_fields() -> dict:
    """Result entries of the fluid shear statistics for a failed solve: NaN, status ``failed``."""
    nan = float("nan")
    return {"shear_fluid_mean_local_Pa": _NAN2, "shear_fluid_support_fraction": nan,
            "shear_extrema_status": SHEAR_FAILED, "shear_extrema_definition": SHEAR_EXTREMA_DEFINITION,
            "shear_fluid_n_samples": 0}



from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---- asperity and total traction of a cell ----
#: contact pressure sampler (equal to ``samplers.CONTACT_SAMPLER``) on the fluid sample set
CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"
FLUID_SAMPLER = "fluid_quadrature_sharp_v1"            # equal to SHEAR_SAMPLER
CONTACT_SHEAR_DEFINITION = f"{CONTACT_SAMPLER}@{FLUID_SAMPLER}/global_x/phase_union"
#: relative tolerance of the sampled mean of p_a against ``Pasp_hom`` of the contact solution
MEAN_CHECK_BUDGET = 1e-12

from microcell.contact import FFT as FFT_MODEL      # the only contact law

#: statuses of the asperity and total statistics, per phase and per cell
OK = STATUS_OK                       # evaluated, some support
NO_SUPPORT = STATUS_NO_SUPPORT       # valid state without the contribution (asperity: p = 0 everywhere):
                                     # mean, min and max are zero
STATE_INVALID = "state_invalid"      # no current contact solution at the samples, or (total) a fluid
                                     # sample without a finite traction: NaN
STATE_MISMATCH = "state_mismatch"    # the contact solution was replaced after the samples were taken, so
                                     # fluid and contact describe different states: NaN
CHECK_FAILED = "check_failed"        # the grid check or the mean check against Pasp_hom failed: NaN
PHASE_INVALID = "phase_invalid"      # cell: a phase had one of the three statuses above
FAILED = "failed"                    # the micro solve failed
UNAVAILABLE = "unavailable"          # not computed: full film, or no friction coefficient
#: phase statuses that make the cell ``phase_invalid``
_PHASE_BAD = (STATE_INVALID, STATE_MISMATCH, CHECK_FAILED)


def _nan2():
    return (float("nan"), float("nan"))


@dataclass(frozen=True)
class ContactTractionPhase:
    """Asperity and total traction of one phase on the fluid sample set [Pa], with means in the
    cell frame."""
    asperity_status: str
    total_status: str
    asperity_mean_local: tuple            # (t_x, t_y) per cell area
    total_mean_local: tuple               # (t_x, t_y) per cell area
    asperity_support_fraction: float
    overlap_fraction: float
    force_on_fluid_fraction: float
    obstruction_without_pressure_fraction: float
    mean_check: float                     # relative |sampled mean - Pasp_hom|; NaN when not checked
    n_samples: int
    contact_model: str
    u_hat_local: tuple = (float("nan"), float("nan"), float("nan"))
    fluid_mean_local: tuple = (float("nan"), float("nan"))
    sampler: str = CONTACT_SAMPLER
    # the contact solution the samples came from, held for an identity check only
    solution: object = field(default=None, compare=False, repr=False)


def _contact_support_numbers(report: dict) -> tuple:
    """(overlap fraction, contact force share on fluid samples, fraction obstructed without
    pressure) from a ``samplers.support_report`` dict."""
    return (float(report["fluid_and_mech"]["fraction"]), float(report["contact_force_share_on_fluid_samples"]),
            float(report["obstruction_without_pressure"]["fraction"]))


def invalid_phase(status: str, n_samples: int, contact_model: str, u_hat=None, fluid_mean=None,
                  mean_check: float = float("nan"), solution=None) -> ContactTractionPhase:
    """``ContactTractionPhase`` with NaN asperity and total statistics and the given status."""
    nan = float("nan")
    return ContactTractionPhase(
        asperity_status=status, total_status=status, asperity_mean_local=_nan2(), total_mean_local=_nan2(),
        asperity_support_fraction=nan, overlap_fraction=nan,
        force_on_fluid_fraction=nan, obstruction_without_pressure_fraction=nan, mean_check=float(mean_check),
        n_samples=int(n_samples), contact_model=str(contact_model),
        u_hat_local=tuple(float(v) for v in (u_hat if u_hat is not None else (nan, nan, nan))),
        fluid_mean_local=tuple(float(v) for v in (fluid_mean if fluid_mean is not None else (nan, nan))),
        solution=solution)


def mean_check(p, weights, pasp_hom: float) -> float:
    """Relative difference between the weighted sample mean of ``p`` [Pa] and ``pasp_hom`` [Pa].

    Returns 0 when both are zero and inf when only the sample mean is non-zero.
    """
    w = np.asarray(weights, dtype=float)
    m = float(np.sum(w * np.asarray(p, dtype=float)) / np.sum(w))
    P = float(pasp_hom)
    if P > 0.0:
        return abs(m - P) / P
    return 0.0 if m == 0.0 else float("inf")


def phase_from_reductions(local, *, contact_status: str, contact_model: str, fluid_state_valid: bool,
                          mean_check_rel: float, u_hat, fluid_mean, solution=None,
                          grid_ok: bool = True) -> ContactTractionPhase:
    """``ContactTractionPhase`` from the reductions on the common samples.

    local: the ``ContactTotalReduction`` of ``samplers.reduce_contact_and_total``.
    contact_status: status of the contact samples (``ok``, ``no_support`` or ``state_invalid``).
    fluid_state_valid: the fluid traction of the phase is valid.
    mean_check_rel, grid_ok: results of the mean check and the grid check; a failure of either
        gives ``check_failed``.
    u_hat, fluid_mean, solution: copied to the phase.
    """
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
    ov, ff, onp = _contact_support_numbers(local.support)
    t_ok = t_st in VALID_STATUSES
    return ContactTractionPhase(
        asperity_status=a_st, total_status=t_st,
        asperity_mean_local=tuple(float(v) for v in local.asperity.mean[:2]),
        total_mean_local=(tuple(float(v) for v in local.total.mean[:2]) if t_ok else _nan2()),
        asperity_support_fraction=float(local.asperity.support_fraction), overlap_fraction=ov,
        force_on_fluid_fraction=ff, obstruction_without_pressure_fraction=onp, mean_check=float(mean_check_rel),
        n_samples=n, contact_model=contact_model, u_hat_local=tuple(float(v) for v in u_hat),
        fluid_mean_local=tuple(float(v) for v in fluid_mean), solution=solution)


def mismatch(phase: ContactTractionPhase) -> ContactTractionPhase:
    """The phase with NaN statistics and the status ``state_mismatch``."""
    return invalid_phase(STATE_MISMATCH, phase.n_samples, phase.contact_model, phase.u_hat_local,
                         phase.fluid_mean_local)


def _wmean(vals, w):
    """Weighted mean of ``vals`` with the weights ``w``."""
    return float(sum(wi * float(v) for wi, v in zip(w, vals)) / float(np.sum(w)))


def union_contact_traction(phases: Sequence[ContactTractionPhase], weights=None) -> dict:
    """Asperity and total traction statistics of the cell from its phases.

    phases: ``ContactTractionPhase`` objects.  weights: one positive weight per phase; equal if None.
    Returns a dict: ``asperity`` and ``total``, each with ``mean_local`` (t_x, t_y) [Pa] and a
    ``status`` (``phase_invalid`` if any phase is invalid, ``unavailable`` if any was not
    computed, ``ok`` if a phase has support, else ``no_support``); the weighted means of the
    support fractions; ``mean_check``, the largest over the phases; ``n_samples``;
    ``contact_model``.
    """
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
        res = dict(status=PHASE_INVALID, mean_local=_nan2())
        if any(s in _PHASE_BAD for s in st):
            return res
        if any(s == UNAVAILABLE for s in st):            # not computed: no friction coefficient
            res["status"] = UNAVAILABLE
            return res
        means = [getattr(p, f"{name}_mean_local") for p in phases]
        res["mean_local"] = tuple(_wmean([m[c] for m in means], w) for c in range(2))
        res["status"] = OK if any(s == OK for s in st) else NO_SUPPORT
        return res

    out["asperity"] = component("asperity")
    out["total"] = component("total")
    asp_ok = out["asperity"]["status"] in (OK, NO_SUPPORT)
    for key, attr in (("support_fraction", "asperity_support_fraction"), ("overlap", "overlap_fraction"),
                      ("force_on_fluid", "force_on_fluid_fraction"),
                      ("obstruction_without_pressure", "obstruction_without_pressure_fraction")):
        out[key] = _wmean([getattr(p, attr) for p in phases], w) if asp_ok else nan
    checks = [float(p.mean_check) for p in phases if np.isfinite(float(p.mean_check))]
    out["mean_check"] = max(checks) if checks else nan      # largest over the phases
    return out


#: result entries of the asperity and total shear statistics, in order
FIELDS = ("shear_asperity_mean_local_Pa", "shear_asperity_support_fraction", "shear_asperity_status",
          "shear_total_mean_local_Pa", "shear_total_status", "shear_contact_overlap_fraction",
          "shear_contact_force_on_fluid_fraction", "shear_contact_obstruction_without_pressure_fraction",
          "shear_contact_mean_check", "shear_contact_definition", "shear_contact_n_samples")


def contact_shear_fields(union: dict) -> dict:
    """Result entries (``FIELDS``) from the dict of ``union_contact_traction``."""
    a, t = union["asperity"], union["total"]
    return {"shear_asperity_mean_local_Pa": tuple(float(v) for v in a["mean_local"]),
            "shear_asperity_support_fraction": float(union["support_fraction"]),
            "shear_asperity_status": str(a["status"]),
            "shear_total_mean_local_Pa": tuple(float(v) for v in t["mean_local"]),
            "shear_total_status": str(t["status"]),
            "shear_contact_overlap_fraction": float(union["overlap"]),
            "shear_contact_force_on_fluid_fraction": float(union["force_on_fluid"]),
            "shear_contact_obstruction_without_pressure_fraction": float(union["obstruction_without_pressure"]),
            "shear_contact_mean_check": float(union["mean_check"]),
            "shear_contact_definition": CONTACT_SHEAR_DEFINITION,
            "shear_contact_n_samples": int(union["n_samples"])}


def failed_contact_shear_fields() -> dict:
    """Result entries (``FIELDS``) for a failed solve: NaN, both statuses ``failed``."""
    nan = float("nan")
    out = {k: nan for k in FIELDS}
    out.update(shear_asperity_mean_local_Pa=_nan2(), shear_total_mean_local_Pa=_nan2(),
               shear_asperity_status=FAILED, shear_total_status=FAILED,
               shear_contact_definition=CONTACT_SHEAR_DEFINITION, shear_contact_n_samples=0)
    return out
