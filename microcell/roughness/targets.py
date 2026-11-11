"""Engineering amplitude targets: Sa and Sq policies, feasibility, reports (R2, M1).

Targets refer to the **finite master array** (plan §5.1): after centring once,

    Sa = mean|z|,   Sq = sqrt(mean z^2),   Sz = max z - min z.

Policies (:class:`ehl.roughness.spec.HeightFit`):

``fixed``  change nothing; check every requested target.
``scale``  one positive multiplier makes the anchor (Sa or Sq) exact; every
           other requested target is then checked.  Ratios cannot change, so
           a fixed height distribution with the wrong Sa/Sq is *unreachable*
           and is reported as such -- never "fixed" by scaling twice.
``ratio``  generated surfaces with a generalised-normal HPD: choose beta so
           the exact final-grid quantile vector has the requested Sa/Sq
           (:func:`fit_gennorm_ratio`), then scale to Sq.  With ``take='Zh'``
           the field is a permutation of that vector, so Sa/Sq carries over
           exactly; with ``'Zs'`` the result is re-checked like any other.

Sz is never fitted here (joint Sa/Sq/Sz fitting is milestone M3).  A
requested Sz is checked and can fail acceptance.

Outcome classes (``TargetReport.status``):

``accepted``                  every requested target within tolerance
``invalid_request``           impossible for any data: a target <= 0,
                              Sa > Sq, or Sz < 2 Sq
``unreachable_fixed_shape``   the array's (or HPD's) fixed ratios cannot meet
                              the targets under this policy
``outside_family_range``      ratio fit: Sa/Sq outside what gennorm can give
                              on this grid (envelope reported)
``not_met``                   fixed mode, values off target

A rejected report is data, not an exception, so a best-effort field can be
inspected; :mod:`ehl.roughness.prepare` refuses to publish it.

Only NumPy is imported.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .generation import HPD_SHAPE_BRACKET, hpd_sorted_sample
from .spec import HeightFit, SurfaceTargets

__all__ = ["TargetCheck", "TargetReport", "realised_amplitudes", "check_targets",
           "screen_request", "apply_policy", "gennorm_ratio_continuous",
           "gennorm_ratio_at", "gennorm_ratio_envelope", "RatioFit", "fit_gennorm_ratio",
           "TargetInfeasibleError"]


class TargetInfeasibleError(ValueError):
    """Raised with a :class:`TargetReport` when a request cannot be met."""

    def __init__(self, report: "TargetReport"):
        super().__init__(f"{report.status}: {report.reason}")
        self.report = report


@dataclass(frozen=True)
class TargetCheck:
    name: str
    requested: float
    realised: float
    tolerance: float
    abs_error: float
    rel_error: float
    passed: bool


@dataclass
class TargetReport:
    mode: str
    anchor: Optional[str]
    status: str
    reason: str = ""
    scale_factor: float = 1.0
    requested: Dict[str, float] = field(default_factory=dict)
    realised: Dict[str, float] = field(default_factory=dict)
    checks: List[TargetCheck] = field(default_factory=list)
    fit: Dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["accepted"] = self.accepted
        return d


def realised_amplitudes(z: np.ndarray) -> Dict[str, float]:
    zc = np.asarray(z, dtype=float) - float(np.mean(z))
    Sq = float(np.sqrt(np.mean(zc ** 2)))
    Sa = float(np.mean(np.abs(zc)))
    return {"Sa": Sa, "Sq": Sq, "Sz": float(zc.max() - zc.min()),
            "Sa_over_Sq": Sa / Sq if Sq > 0 else float("nan")}


def _tol(t: SurfaceTargets, want: float) -> float:
    return max(t.atol, t.rtol * want)


def check_targets(z: np.ndarray, targets: SurfaceTargets) -> Tuple[Dict[str, float], List[TargetCheck]]:
    got = realised_amplitudes(z)
    checks = []
    for k, want in targets.requested().items():
        tol = _tol(targets, want)
        err = abs(got[k] - want)
        checks.append(TargetCheck(k, float(want), got[k], tol, err, err / want, err <= tol))
    return got, checks


def screen_request(targets: SurfaceTargets) -> Optional[str]:
    """Reason a request is impossible for *any* zero-mean data, or None.
    (``0 < Sa <= Sq`` and ``Sz >= 2*Sq`` are necessary, not sufficient.)"""
    Sa, Sq, Sz = targets.Sa, targets.Sq, targets.Sz
    if Sa is not None and Sq is not None and Sa > Sq * (1.0 + targets.rtol):
        return f"Sa = {Sa:.6g} m exceeds Sq = {Sq:.6g} m (Sa <= Sq for any data)"
    if Sz is not None and Sq is not None and Sz < 2.0 * Sq * (1.0 - targets.rtol):
        return f"Sz = {Sz:.6g} m is below 2*Sq = {2 * Sq:.6g} m (impossible for zero-mean data)"
    return None


def apply_policy(z: np.ndarray, targets: SurfaceTargets,
                 fit: HeightFit) -> Tuple[np.ndarray, TargetReport]:
    """Apply a ``fixed`` or ``scale`` policy (``ratio`` reduces to ``scale``
    on Sq once the HPD shape has been fitted).  Returns the centred, scaled
    field and the report; never raises for an unmet target."""
    zc = np.asarray(z, dtype=float) - float(np.mean(z))
    req = targets.requested()
    mode = fit.mode
    anchor = "Sq" if mode == "ratio" else fit.anchor
    rep = TargetReport(mode=mode, anchor=anchor, status="accepted", requested=dict(req))
    bad = screen_request(targets)
    if bad:
        rep.status, rep.reason = "invalid_request", bad
        rep.realised, rep.checks = check_targets(zc, targets)
        return zc, rep
    if mode in ("scale", "ratio"):
        if anchor not in req:
            raise ValueError(f"{mode} policy needs a requested {anchor} target")
        now = realised_amplitudes(zc)[anchor]
        if not now > 0.0:
            raise ValueError("cannot scale a zero field")
        s = req[anchor] / now
        zc = zc * s
        rep.scale_factor = float(s)
    rep.realised, rep.checks = check_targets(zc, targets)
    failed = [c for c in rep.checks if not c.passed]
    if failed:
        names = ", ".join(f"{c.name} (want {c.requested:.6g}, got {c.realised:.6g})" for c in failed)
        if mode == "fixed":
            rep.status, rep.reason = "not_met", f"fixed field misses {names}"
        else:
            r = rep.realised["Sa_over_Sq"]
            rep.status = "unreachable_fixed_shape"
            rep.reason = (f"one scale factor on the {anchor} anchor leaves {names}; the field's "
                          f"Sa/Sq = {r:.6f} and Sz/Sq = {rep.realised['Sz'] / rep.realised['Sq']:.4f} "
                          "are fixed by its height distribution")
    return zc, rep


# ---------------------------------------------------------------------------
# Generalised-normal Sa/Sq fit on the exact final-grid quantile vector
# ---------------------------------------------------------------------------
def gennorm_ratio_continuous(beta: float) -> float:
    """E|Z| / sqrt(E Z^2) of the continuous gennorm(beta):
    Gamma(2/b) / sqrt(Gamma(1/b) Gamma(3/b))."""
    b = float(beta)
    return math.exp(math.lgamma(2.0 / b) - 0.5 * (math.lgamma(1.0 / b) + math.lgamma(3.0 / b)))


def gennorm_ratio_at(M: int, beta: float) -> float:
    """Sa/Sq of the standardised M-point gennorm quantile vector."""
    return float(np.mean(np.abs(hpd_sorted_sample(int(M), "gennorm", float(beta)))))


def gennorm_ratio_envelope(M: int, bracket: Tuple[float, float] = HPD_SHAPE_BRACKET["gennorm"]
                           ) -> Tuple[float, float]:
    """(min, max) Sa/Sq reachable with beta in ``bracket`` at M points."""
    return gennorm_ratio_at(M, bracket[0]), gennorm_ratio_at(M, bracket[1])


@dataclass
class RatioFit:
    target: float
    beta: float
    achieved: float
    M: int
    iterations: int
    envelope: Tuple[float, float]
    beta_continuous: float

    def to_dict(self):
        return asdict(self)


def fit_gennorm_ratio(target: float, M: int,
                      bracket: Tuple[float, float] = HPD_SHAPE_BRACKET["gennorm"],
                      tol: float = 1e-12, max_iter: int = 60) -> RatioFit:
    """beta such that the M-point gennorm quantile vector has Sa/Sq = target.

    Starts from the continuous-distribution solution (cheap), then a
    bracketed Illinois secant on the exact M-point ratio.  Raises
    :class:`TargetInfeasibleError` (status ``outside_family_range``) when the
    target lies outside the bracket's envelope at this M.
    """
    lo_b, hi_b = float(bracket[0]), float(bracket[1])
    env = gennorm_ratio_envelope(M, (lo_b, hi_b))
    if not (env[0] <= target <= env[1]):
        rep = TargetReport(mode="ratio", anchor="Sq", status="outside_family_range",
                           reason=(f"Sa/Sq = {target:.6f} is outside the gennorm envelope "
                                   f"[{env[0]:.6f}, {env[1]:.6f}] for beta in "
                                   f"[{lo_b}, {hi_b}] at M = {M}"),
                           fit={"envelope": list(env), "bracket": [lo_b, hi_b], "M": int(M)})
        raise TargetInfeasibleError(rep)
    # continuous start (bisection on log beta)
    a, b = math.log(lo_b), math.log(hi_b)
    for _ in range(200):
        m = 0.5 * (a + b)
        if gennorm_ratio_continuous(math.exp(m)) < target:
            a = m
        else:
            b = m
    beta_c = math.exp(0.5 * (a + b))
    # bracket on the M-point ratio around the continuous solution
    x0, x1 = max(lo_b, beta_c * 0.95), min(hi_b, beta_c * 1.05)
    f0, f1 = gennorm_ratio_at(M, x0) - target, gennorm_ratio_at(M, x1) - target
    its = 2
    while f0 > 0 and x0 > lo_b:
        x0 = max(lo_b, x0 * 0.8)
        f0 = gennorm_ratio_at(M, x0) - target
        its += 1
    while f1 < 0 and x1 < hi_b:
        x1 = min(hi_b, x1 * 1.25)
        f1 = gennorm_ratio_at(M, x1) - target
        its += 1
    # Illinois regula falsi on [x0, x1] with f0 <= 0 <= f1
    x, fx = (x0, f0) if abs(f0) <= abs(f1) else (x1, f1)
    side = 0
    for _ in range(max_iter):
        if abs(fx) <= tol or f1 == f0:
            break
        x = (x0 * f1 - x1 * f0) / (f1 - f0)
        fx = gennorm_ratio_at(M, x) - target
        its += 1
        if fx > 0:
            x1, f1 = x, fx
            if side == 1:
                f0 *= 0.5
            side = 1
        else:
            x0, f0 = x, fx
            if side == -1:
                f1 *= 0.5
            side = -1
    return RatioFit(target=float(target), beta=float(x), achieved=float(fx + target), M=int(M),
                    iterations=its, envelope=env, beta_continuous=beta_c)
