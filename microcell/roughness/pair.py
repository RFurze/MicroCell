"""Kinematics of the translating surface.

The production micro solver's legacy single-pattern convention is
``s(t) = ratio * (2*U*t + UT*t**2)`` with ``U`` the solver's half-velocity
(``micro_phase_neutral_class``: ``_Ux_dim_const = ratio*(2*Ux + UxT*T)``,
displacement ``T*_Ux_dim_const``).  :func:`pattern_displacement` evaluates it
*at the requested time*: the history seeding at ``T - dt`` must use
``s(T - dt)``, not ``(T - dt) * _Ux_dim_const(T)``, which differs whenever
``UT != 0`` (plan §7.2, §15.2).  Displacements are in metres; the factor 2 is
applied here once and nowhere else (radius plan §9.7).

[MicroCell WP3] The parent's ``SurfacePair`` (composite gap and its statistics
over displacement), the legacy roughness-mode table and the cross-term helpers
were removed: the solver composes the members in
``roughness/runtime.SurfaceField``.  They are in the parent
(``ehl/roughness/pair.py``).
"""
from __future__ import annotations

__all__ = ["pattern_displacement"]


def pattern_displacement(t: float, U: float, UT: float = 0.0,
                         ratio: float = 1.0) -> float:
    """Pattern displacement [m] at time ``t`` in the solver's convention:
    ``ratio * (2*U*t + UT*t**2)`` (integral of the full surface velocity
    ``2*(U + UT*t)`` from 0 to t).  Valid for negative (history) times."""
    t = float(t)
    return float(ratio) * (2.0 * float(U) * t + float(UT) * t * t)
