"""Shear traction of a micro cell: the reduction of sampled traction, the fluid statistics, and the
asperity and total statistics.

MicroCell WP5 (11 Nov 2026): ``shear_reduce.py``, ``fluid_shear.py`` and ``contact_shear.py`` in one
file, in that order.  The code is moved verbatim; the imports between the three are gone.  Their
module docstrings follow.  Where one names one of the three files, read this file.

``shear_reduce.py`` (the parent's ``ehl/shear_reduce.py``)
------------------------------------------------------------------------------------------
Reduction of sampled shear traction to a mean and a directional envelope (extrema M2a, 23 Oct 2026).

Pure numpy, no FEniCS: the micro classes (M2b), the pool driver's phase union, the result
transport and FEniCS-free tests can all import it.  (``microscale/src/functions/__init__.py``
imports the FEniCS micro classes, so a module there would pull dolfin into every consumer.)
Plan ``docs/plans/masked_pressure_extrema_and_shear_exports_2026-10-23.md`` sections 2.2-2.4,
6.2.1-6.2.3 and 6.2.5; decision D5 (fixed global x only) of
``docs/notes/extrema_shear_exports_review_and_launch_2026-10-23.md`` section 2.3.  The fluid
shear convention the samples follow is written down, with file:line references, in
``docs/notes/extrema_m2a_implementation_log_2026-10-23.md`` section 1.

One micro state (one phase), :func:`reduce_traction`:

* the input is the traction sampled at N points of the cell (Pa), the samples' weights (area
  or quadrature weights: their sum is the measure of the whole sample set), which samples were
  evaluated (``valid``), where the contribution is physically present (``support``: the fluid
  samples for fluid traction) and a fixed unit direction ``d`` in the macro (global) frame;
* every sample is projected on ``d`` in the frame the samples are given in (the micro's local
  frame: a standalone cell has no macro frame, so the parent's rotation with ``R_inv`` is not
  here, WP4); only then is anything reduced;
* outside the support the contribution is the physical zero (zero extension), whatever was
  sampled there, so the whole-cell set contains 0 wherever the support does not cover it;
* the result is the three-component weighted mean over the whole sample set (the mean per
  total cell area when the weights are area weights), the min / max and the weighted mean of
  ``d . t`` over the whole sample set, and -- conditional, labelled ``support_*`` -- the min /
  max of ``d . t`` over the support samples alone.  A support-only band need not contain the
  whole-cell mean: that is a property of the two statistics, not an error.

Statuses (strings; never inferred from NaN):

``ok``               evaluated, some support
``no_support``       a valid state without the contribution: whole-cell mean, min and max are
                     the physical zero; the support-only extrema are undefined (NaN)
``invalid_samples``  a support sample was not evaluated (``valid`` False or non-finite): NaN
``state_invalid``    the state failed or is unsupported (``state_valid=False``) or has no
                     samples: no physical zero can be inferred, NaN
``phase_invalid``    a task's status (``fluid_shear``, ``contact_shear``): one of its phases was
                     ``invalid_samples`` / ``state_invalid``: NaN

``fluid_shear.py`` (the parent's ``ehl/fluid_shear.py``)
------------------------------------------------------------------------------------------
Fluid shear traction statistics of a micro task: the exchange contract (extrema M2b, 23 Oct 2026).

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

``contact_shear.py`` (the parent's ``ehl/contact_shear.py``)
------------------------------------------------------------------------------------------
Asperity and total shear traction of a micro task (extrema M3b, 23 Oct 2026).

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
    reduced   by ``shear_reduce.reduce_traction``: the local-frame mean over the whole set (per total
              cell area; the total formed per sample before it is reduced).  The parent's global-x
              envelope (with the macro frame ``R_inv``) is not produced: a standalone cell has no
              macro frame (WP4);
    checks    the common-support construction is published only when (a) the contact grid is the micro
              mesh's n x n grid and (b) the sampled mean of ``p_a`` reproduces the solution's own
              ``Pasp_hom`` to ``MEAN_CHECK_BUDGET`` (M3a's gate-1 budget, stated before its comparison);
              otherwise ``check_failed``.  The supports' mismatch (fluid samples carrying contact pressure,
              obstruction without pressure, the contact force on fluid samples) is recorded, never
              suppressed;
per task      means by phase weight (equal); a phase whose state is invalid invalidates the task
              (``phase_invalid``).

Contact model: ``fft_elastic_v1``, the only one in this repo (``micro_diagnostics.sample_contact_traction``
raises for any other).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SHEAR_REDUCE_VERSION = "shear_reduce_v1"

STATUS_OK = "ok"
STATUS_NO_SUPPORT = "no_support"
STATUS_INVALID_SAMPLES = "invalid_samples"
STATUS_STATE_INVALID = "state_invalid"
STATUS_PHASE_INVALID = "phase_invalid"
#: the statuses whose numbers are physical (the others carry NaN)
VALID_STATUSES = (STATUS_OK, STATUS_NO_SUPPORT)

STATISTIC_SAMPLE = "sample_extremum"

#: the default (and, by decision D5, the only production) direction: the macro frame's x axis,
#: the axis of the existing friction force F_x and torque T_y = -Rc * F_x
GLOBAL_X = (1.0, 0.0, 0.0)

#: tolerance of the unit-direction check (programming errors raise ValueError)
UNIT_TOL = 1e-12

_NAN3 = (float("nan"),) * 3


@dataclass(frozen=True)
class ShearReduction:
    """One reduced traction distribution.  Pa; vectors in the frame of the samples.

    ``mean``          weighted mean vector over the whole sample set (zero outside the support)
    ``min``, ``max``  extrema of ``d . t`` over the whole sample set (the zero extension included)
    ``sample_mean``   weighted mean of ``d . t`` over the whole sample set (= ``d . mean``)
    ``support_min``, ``support_max``  extrema of ``d . t`` over the support samples only
                      (conditional statistics; NaN without support)
    ``support_fraction``  support weight / total weight
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
        """The numbers are physical (status ``ok`` or ``no_support``)."""
        return self.status in VALID_STATUSES

    def as_dict(self) -> dict:
        """Plain floats / ints / strings (picklable, JSON-ready)."""
        return {
            "mean": [float(v) for v in self.mean], "min": float(self.min), "max": float(self.max),
            "sample_mean": float(self.sample_mean), "support_min": float(self.support_min),
            "support_max": float(self.support_max), "support_fraction": float(self.support_fraction),
            "n_samples": int(self.n_samples), "n_support": int(self.n_support), "status": self.status,
            "direction": [float(v) for v in self.direction], "statistic": self.statistic,
            "version": SHEAR_REDUCE_VERSION,
        }


def _direction(direction) -> np.ndarray:
    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        raise ValueError(f"direction must be three finite numbers, got {direction!r}")
    if abs(float(np.linalg.norm(d)) - 1.0) > UNIT_TOL:
        raise ValueError(f"direction must be a unit vector (|d| = {np.linalg.norm(d):.15g})")
    return d


def _invalid(status: str, d: np.ndarray, n: int, n_support: int, frac: float,
             statistic: str = STATISTIC_SAMPLE) -> ShearReduction:
    nan = float("nan")
    return ShearReduction(mean=_NAN3, min=nan, max=nan, sample_mean=nan, support_min=nan,
                          support_max=nan, support_fraction=frac, n_samples=int(n),
                          n_support=int(n_support), status=status,
                          direction=tuple(float(v) for v in d), statistic=statistic)


def reduce_traction(vectors, weights, valid, direction=GLOBAL_X, *, support=None,
                    state_valid: bool = True) -> ShearReduction:
    """Reduce the traction sampled in one micro state.

    Parameters
    ----------
    vectors : (N, 2) or (N, 3) array, Pa
        The sampled traction, in the micro's local basis.  Two columns are padded with a zero third component (the macro
        sets the normal component of the micro traction to zero).
    weights : (N,) array
        Finite, positive sample weights.  The means are normalised by their sum over all N
        samples (the whole sample set), not by the support's weight.
    valid : (N,) bool array
        The sample was evaluated.  Samples outside the support need not be valid: their
        value is replaced by the physical zero and never read.
    direction : three floats, default ``GLOBAL_X``
        Unit vector, in the frame of the samples, on which every sample is projected.
    support : (N,) bool array, optional
        Where the contribution is physically present (default: every sample).
    state_valid : bool
        False for a failed or unsupported state: everything NaN, status ``state_invalid``.
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

    t = np.where(sup[:, None], v, 0.0)          # zero extension: nothing outside the support is read
    s = t @ d                                    # project every sample on d
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



from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


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
UNAVAILABLE = "unavailable"          # a result or a correction set without the fields (full film, older code)
#: the phase statuses that invalidate a task
_PHASE_BAD = (STATE_INVALID, STATE_MISMATCH, CHECK_FAILED)


def _nan2():
    return (float("nan"), float("nan"))


# ------------------------------------------------------------------------------------ one phase
@dataclass(frozen=True)
class ContactTractionPhase:
    """One phase's asperity and total traction on the fluid sample set.  Pa; means in the micro's
    local frame."""
    asperity_status: str
    total_status: str
    asperity_mean_local: tuple            # (t_x, t_y) per total cell area
    total_mean_local: tuple
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
        asperity_support_fraction=nan, overlap_fraction=nan,
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


def phase_from_reductions(local, *, contact_status: str, contact_model: str, fluid_state_valid: bool,
                          mean_check_rel: float, u_hat, fluid_mean, solution=None,
                          grid_ok: bool = True) -> ContactTractionPhase:
    """A phase from M3a's ``reduce_contact_and_total`` result on the common samples (``local``: the
    local-frame means).  ``contact_status`` is the contact sampler's (``ok`` / ``no_support`` /
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
    """The extraction refreshed the solution after the samples were taken: nothing is published."""
    return invalid_phase(STATE_MISMATCH, phase.n_samples, phase.contact_model, phase.u_hat_local,
                         phase.fluid_mean_local)


# ----------------------------------------------------------------------------------- the union
def _wmean(vals, w):
    return float(sum(wi * float(v) for wi, v in zip(w, vals)) / float(np.sum(w)))


def union_contact_traction(phases: Sequence[ContactTractionPhase], weights=None) -> dict:
    """The task's asperity and total statistics from its phases (equal weights unless given): means by
    weight; an invalid phase invalidates the task (``phase_invalid``); ``ok`` when a phase has support,
    else ``no_support``."""
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
        if any(s == UNAVAILABLE for s in st):            # not computed (no friction coefficient given)
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
    out["mean_check"] = max(checks) if checks else nan      # the worst phase (kept for a failed check too)
    return out


# --------------------------------------------------------------------- the worker's result fields
#: the appended ``MicroTaskResult`` fields, in their order (the worker's success and NaN returns)
FIELDS = ("shear_asperity_mean_local_Pa", "shear_asperity_support_fraction", "shear_asperity_status",
          "shear_total_mean_local_Pa", "shear_total_status", "shear_contact_overlap_fraction",
          "shear_contact_force_on_fluid_fraction", "shear_contact_obstruction_without_pressure_fraction",
          "shear_contact_mean_check", "shear_contact_definition", "shear_contact_n_samples")


def contact_shear_fields(union: dict) -> dict:
    """The appended ``MicroTaskResult`` fields of a task's union."""
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
    """The worker's NaN return: every statistic NaN, both statuses ``failed``."""
    nan = float("nan")
    out = {k: nan for k in FIELDS}
    out.update(shear_asperity_mean_local_Pa=_nan2(), shear_total_mean_local_Pa=_nan2(),
               shear_asperity_status=FAILED, shear_total_status=FAILED,
               shear_contact_definition=CONTACT_SHEAR_DEFINITION, shear_contact_n_samples=0)
    return out
