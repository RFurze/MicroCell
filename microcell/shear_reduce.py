"""Reduction of sampled shear traction to a mean and a directional envelope (extrema M2a, 23 Oct 2026).

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

