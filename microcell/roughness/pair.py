"""Surface arrangements: stationary + translating composition and kinematics.

Gap in the stationary-body frame (Case API plan §13), both fields in the
canonical gap convention:

    stationary only:  h = H(t) + z_s(x - s_s)
    translating only: h = H(t) + z_m(x - s_m(t))
    both:             h = H(t) + z_s(x - s_s) + z_m(x - s_m(t))

``s_s`` and ``s_m`` are independent signed displacement vectors [m] (a
stationary surface simply keeps its offset).  :class:`SurfacePair` evaluates
the composite on any nodal grid through the canonical Fourier evaluator and
reports per-surface and combined statistics.  The combined statistics follow

    Sq_gap(s)^2 = Sq_s^2 + Sq_m^2 + 2*mean(z_s * z_m(. - s))

and are *reported over displacement, never renormalised*: per-surface
targets belong to each physical surface (plan §13).

Kinematics.  The production micro solver's legacy single-pattern convention
is ``s(t) = ratio * (2*U*t + UT*t**2)`` with ``U`` the solver's
half-velocity (``micro_phase_neutral_class``: ``_Ux_dim_const =
ratio*(2*Ux + UxT*T)``, displacement ``T*_Ux_dim_const``).
:func:`pattern_displacement` evaluates it *at the requested time*: the
history seeding at ``T - dt`` must use ``s(T - dt)``, not
``(T - dt) * _Ux_dim_const(T)``, which differs whenever ``UT != 0``
(plan §7.2, §15.2).  New code passes displacements in metres; the factor 2
is applied here once and nowhere else (radius plan §9.7).

Only NumPy is imported.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .evaluator import FourierSurface
from .stats import height_moments

__all__ = ["pattern_displacement", "legacy_effective_displacement",
           "LEGACY_ROUGHNESS_MODES", "arrangement_for_legacy_mode",
           "SurfacePair", "cross_term_spectral"]

Vec = Tuple[float, float]


def pattern_displacement(t: float, U: float, UT: float = 0.0,
                         ratio: float = 1.0) -> float:
    """Pattern displacement [m] at time ``t`` in the solver's convention:
    ``ratio * (2*U*t + UT*t**2)`` (integral of the full surface velocity
    ``2*(U + UT*t)`` from 0 to t).  Valid for negative (history) times."""
    t = float(t)
    return float(ratio) * (2.0 * float(U) * t + float(UT) * t * t)


def legacy_effective_displacement(t_eval: float, t_now: float, U: float,
                                  UT: float = 0.0, ratio: float = 1.0) -> float:
    """What the current solver produces when ``T_dim_const`` is set to
    ``t_eval`` while ``_Ux_dim_const`` still holds its value for ``t_now``:
    ``t_eval * ratio*(2U + UT*t_now)``.  Equals
    :func:`pattern_displacement` only when ``t_eval == t_now`` or ``UT == 0``.
    Recorded so tests can show the BDF-history trap explicitly."""
    return float(t_eval) * float(ratio) * (2.0 * float(U) + float(UT) * float(t_now))


#: How each existing ``Roughness.roughness_mode`` maps onto the one pair
#: representation (``SurfacePairSpec``).  ``motion`` gives each member's
#: displacement in terms of the solver convention above.
LEGACY_ROUGHNESS_MODES: Dict[str, Dict[str, object]] = {
    "cup": {
        "arrangement": "stationary_only",
        "members": {"stationary": "s = offset"},
        "roughness_velocity_ratio": 0.0,
        "generated_support": "M1",
        "note": "static roughness on the cup",
    },
    "ball": {
        "arrangement": "translating_only",
        "members": {"translating": "s = offset + pattern_displacement(t, U, UT, 1.0)"},
        "roughness_velocity_ratio": 1.0,
        "generated_support": "M1",
        "note": "roughness carried by the ball at the full surface speed",
    },
    "two_surface_split": {
        "arrangement": "stationary_and_translating",
        "members": {"stationary": "s = offset",
                    "translating": "s = offset + carried + pattern_displacement(t, U, UT, 1.0)"},
        "roughness_velocity_ratio": 1.0,
        "generated_support": "M1 full film; mixed is M2 (R5b)",
        "note": "phase_sync's carried kx-phase becomes a carried displacement in metres "
                "(no sinusoid-specific phase for generated surfaces)",
    },
    "two_surface_split_avg": {
        "arrangement": "stationary_and_translating",
        "members": {"stationary": "s = offset",
                    "translating": "s = offset + carried + pattern_displacement(t, U, UT, 1.0)"},
        "roughness_velocity_ratio": 1.0,
        "averaging": "explicit list of relative offsets [m]; the sinusoid's "
                     "{0, 2pi/3, 4pi/3} has no generated-surface meaning",
        "generated_support": "M1 full film once an offset set is declared; mixed is M2",
        "note": "split model plus an offset average",
    },
    "two_surface": {
        "arrangement": "two_translating",
        "members": {"surface_1": "s = +pattern_displacement(t, U, UT, ratio)",
                    "surface_2": "s = -pattern_displacement(t, U, UT, ratio)"},
        "roughness_velocity_ratio": 0.5,
        "generated_support": "not supported (legacy counter-propagating frame)",
        "note": "identical surfaces counter-propagating at +-U in a frame moving with "
                "the mean pattern speed; relative displacement equals the split model's",
    },
}


def arrangement_for_legacy_mode(mode: str,
                                roughness_velocity_ratio: Optional[float] = None) -> Dict[str, object]:
    """The pair-representation entry for a legacy ``roughness_mode``; checks
    the velocity ratio the legacy docs require."""
    if mode not in LEGACY_ROUGHNESS_MODES:
        raise ValueError(f"unknown roughness_mode {mode!r}")
    entry = dict(LEGACY_ROUGHNESS_MODES[mode])
    want = float(entry["roughness_velocity_ratio"])
    if roughness_velocity_ratio is not None and abs(float(roughness_velocity_ratio) - want) > 0.0:
        raise ValueError(f"roughness_mode {mode!r} expects roughness_velocity_ratio={want}, "
                         f"got {roughness_velocity_ratio}")
    return entry


def cross_term_spectral(a: FourierSurface, b: FourierSurface,
                        s_a: Sequence[float] = (0.0, 0.0),
                        s_b: Sequence[float] = (0.0, 0.0)) -> float:
    """Continuous ``mean over the cell of z_a(x - s_a) * z_b(x - s_b)`` from
    the retained coefficients (exact for the trigonometric polynomials)."""
    _same_cell(a, b)
    ca = {(int(ky), int(kx)): c for ky, row in zip(a.ky, a._shifted(s_a, (0, 0)))
          for kx, c in zip(a.kx, row) if c != 0}
    total = 0.0 + 0.0j
    for ky, row in zip(b.ky, b._shifted(s_b, (0, 0))):
        for kx, c in zip(b.kx, row):
            if c == 0:
                continue
            ca_conj = ca.get((-int(ky), -int(kx)))
            if ca_conj is not None:
                total += ca_conj * c
    return float(total.real)


def _same_cell(a: FourierSurface, b: FourierSurface) -> None:
    if abs(a.Lx - b.Lx) > 1e-12 * a.Lx or abs(a.Ly - b.Ly) > 1e-12 * a.Ly:
        raise ValueError(f"surfaces must share the periodic cell: "
                         f"({a.Lx}, {a.Ly}) vs ({b.Lx}, {b.Ly}); resample or tile explicitly")


class SurfacePair:
    """Composite gap perturbation of a stationary and/or a translating surface.

    Either member may be None (single-surface arrangements); a pair with one
    member reduces exactly to that member.  Members must share Lx, Ly.
    """

    def __init__(self, stationary: Optional[FourierSurface] = None,
                 translating: Optional[FourierSurface] = None):
        if stationary is None and translating is None:
            raise ValueError("SurfacePair needs at least one surface")
        if stationary is not None and translating is not None:
            _same_cell(stationary, translating)
        self.stationary = stationary
        self.translating = translating
        ref = stationary if stationary is not None else translating
        self.Lx, self.Ly = ref.Lx, ref.Ly

    @property
    def arrangement(self) -> str:
        if self.translating is None:
            return "stationary_only"
        if self.stationary is None:
            return "translating_only"
        return "stationary_and_translating"

    def gap(self, mx: int, my: Optional[int] = None,
            s_stationary: Sequence[float] = (0.0, 0.0),
            s_translating: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
        """``z_s(x - s_s) + z_m(x - s_m)`` on the ``my x mx`` nodal grid."""
        my = mx if my is None else my
        out = np.zeros((my, mx))
        if self.stationary is not None:
            out += self.stationary.nodal(mx, my, shift=s_stationary)
        if self.translating is not None:
            out += self.translating.nodal(mx, my, shift=s_translating)
        return out

    def cross_term(self, mx: int, my: Optional[int] = None,
                   s_stationary: Sequence[float] = (0.0, 0.0),
                   s_translating: Sequence[float] = (0.0, 0.0)) -> float:
        """Nodal ``mean(z_s * z_m)`` at the given displacements (0 for a
        single-surface arrangement)."""
        if self.stationary is None or self.translating is None:
            return 0.0
        my = mx if my is None else my
        a = self.stationary.nodal(mx, my, shift=s_stationary)
        b = self.translating.nodal(mx, my, shift=s_translating)
        return float(np.mean(a * b))

    def statistics(self, mx: int, my: Optional[int] = None,
                   displacements: Iterable[Tuple[Vec, Vec]] = (((0.0, 0.0), (0.0, 0.0)),)
                   ) -> Dict[str, object]:
        """Per-surface and combined statistics on the ``my x mx`` grid.

        ``displacements``: iterable of ``(s_stationary, s_translating)``.
        Per-surface values are evaluated at each member's first displacement
        (its own statistics are displacement-invariant on its source grid).
        The combined values are listed per displacement and summarised; they
        are never used to rescale anything.
        """
        my = mx if my is None else my
        disp = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                for a, b in displacements]
        if not disp:
            raise ValueError("need at least one displacement")
        out: Dict[str, object] = {"arrangement": self.arrangement, "grid": [my, mx]}
        for name, surf, idx in (("stationary", self.stationary, 0),
                                ("translating", self.translating, 1)):
            if surf is None:
                out[name] = None
                continue
            v = surf.nodal(mx, my, shift=disp[0][idx])
            m = height_moments(v)
            m.update(z_min=float(v.min()), z_max=float(v.max()),
                     Sz=float(v.max() - v.min()))
            out[name] = m
        rows: List[Dict[str, float]] = []
        for s_s, s_m in disp:
            g = self.gap(mx, my, s_s, s_m)
            m = height_moments(g)
            cross = self.cross_term(mx, my, s_s, s_m)
            parts = 0.0
            for surf, s in ((self.stationary, s_s), (self.translating, s_m)):
                if surf is not None:
                    parts += float(np.mean(surf.nodal(mx, my, shift=s) ** 2))
            rows.append({"s_stationary": list(s_s), "s_translating": list(s_m),
                         "Sa": m["Sa"], "Sq": m["Sq"], "Ssk": m["Ssk"], "Sku": m["Sku"],
                         "z_min": float(g.min()), "z_max": float(g.max()),
                         "Sz": float(g.max() - g.min()),
                         "film_reducing_extreme": max(0.0, -float(g.min())),
                         "cross_term": cross,
                         "identity_residual": float(m["Sq"] ** 2 - (parts + 2.0 * cross))})
        out["combined"] = rows
        sq = np.array([r["Sq"] for r in rows])
        zmin = np.array([r["z_min"] for r in rows])
        out["combined_summary"] = {
            "n_displacements": len(rows),
            "Sq_min": float(sq.min()), "Sq_max": float(sq.max()), "Sq_mean": float(sq.mean()),
            "z_min_min": float(zmin.min()), "z_min_max": float(zmin.max()),
            "renormalised": False,
        }
        return out
