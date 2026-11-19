"""Kinematics of the translating surface.

The surface is displaced by ``s(t) = ratio * (2*U*t + UT*t**2)`` [m], with U the mean
(entrainment) velocity [m/s] and UT its rate [m/s^2].  The factor 2 is applied here only.
"""
from __future__ import annotations

__all__ = ["pattern_displacement"]


def pattern_displacement(t: float, U: float, UT: float = 0.0,
                         ratio: float = 1.0) -> float:
    """Displacement of the roughness pattern [m] at time t [s]: ``ratio * (2*U*t + UT*t**2)``,
    the integral of the surface velocity ``2*(U + UT*t)`` from 0 to t, times the velocity ratio.
    Valid for negative times."""
    t = float(t)
    return float(ratio) * (2.0 * float(U) * t + float(UT) * t * t)
