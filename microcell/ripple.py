"""Ripple deformation of the micro cell: the elastic response to its own fluid pressure (NumPy only).

The film gains a zero-mean, periodic deformation ``u_r`` [m] of the fluid traction ``s`` [Pa],
through the periodic two-body half-space operator H of ``microcell.contact.PeriodicHalfSpace``:

    u_r^(q) = 2 s^(q) / (E* |q|)  (q != 0),   u_r^(0) = 0,   1/E* = sum_i (1 - nu_i^2) / E_i

The mean deformation and the macroscale pressure ramp belong to the macroscale, so the cell-mean
film is unchanged.  The traction is formed at the grid nodes:

    s = chi * [P > 0] * (P - G.(x - x_c)),    u_r = H[s]

where P is the full pressure [Pa], chi the fluid indicator and G the macroscale pressure gradient
[Pa/m].  A positive traction opens the film (``u_r > 0``).

The solver holds ``u_r`` fixed during a pass and updates it between passes.  A pass is accepted by
``residual_acceptance``; ``transition_checks`` is a reported diagnostic.

Arrays are ``[i, j]``, with i along x and node ``(i, j)`` at ``(i/nx, j/ny)`` of the unit cell.
"""
from __future__ import annotations

import numpy as np

from microcell.contact import PeriodicHalfSpace, band_limited_sample

__all__ = [
    "RIPPLE_MODEL", "ZERO_MEAN_TOL",
    "RippleNotConvergedError", "ripple_operator", "fluid_traction", "ripple_traction",
    "ripple", "zero_mean_residual", "check_zero_mean", "seam_traction_jump",
    "pattern_shift", "transition_checks", "CONTRACTION_MAX", "residual_acceptance",
]

#: name of the deformation model: the field is held fixed during a pass, in the frame of the
#: rough surface
RIPPLE_MODEL = "ripple_halfspace_v1"

#: zero-mean check: |mean u| <= tol * max|u|.  The nodes of the uniform periodic grid have equal
#: areas, so the nodal mean is the area mean.
ZERO_MEAN_TOL = 1e-10


class RippleNotConvergedError(RuntimeError):
    """The ripple passes did not meet the acceptance, or a field failed a check."""


def ripple_operator(nx: int, ny: int, Lx: float, Ly: float, E_star: float) -> PeriodicHalfSpace:
    """Half-space operator on the ``nx x ny`` grid of an ``Lx x Ly`` cell [m], modulus ``E_star`` [Pa]."""
    return PeriodicHalfSpace(nx, ny, Lx, Ly, E_star)


def fluid_traction(P, chi, ramp) -> np.ndarray:
    """Fluid traction at the grid nodes, ``s = chi * [P > 0] * (P - ramp)`` [Pa].

    P: full pressure [Pa], ramp included.
    chi: fluid indicator (1 fluid, 0 contact; values above 0.5 count as fluid).
    ramp: ``G.(x - x_c)`` [Pa] at the same nodes, or a scalar 0 without a gradient.
    Returns s, zero in contact and in cavitation.  The value is signed and not clipped.
    """
    P = np.asarray(P, dtype=float)
    chi = np.asarray(chi)
    ramp = np.broadcast_to(np.asarray(ramp, dtype=float), P.shape)
    carried = (chi > 0.5) & (P > 0.0)
    return np.where(carried, P - ramp, 0.0)


def ripple_traction(s) -> np.ndarray:
    """Traction with its nodal mean over the cell removed, ``s - <s>`` [Pa]."""
    s = np.asarray(s, dtype=float)
    return s - s.mean()


def ripple(s, op: PeriodicHalfSpace) -> np.ndarray:
    """Ripple deformation ``u_r = H[s]`` [m] of the traction s [Pa].  The operator removes the mean."""
    return op.apply(np.asarray(s, dtype=float))


def zero_mean_residual(u) -> float:
    """``|mean u| / max|u|``: 0 for a zero field, inf for a non-finite one."""
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)):
        return float("inf")
    m = float(np.max(np.abs(u))) if u.size else 0.0
    if m == 0.0:
        return 0.0
    return abs(float(u.mean())) / m


def check_zero_mean(u, tol: float = ZERO_MEAN_TOL, what: str = "ripple field") -> float:
    """Returns ``zero_mean_residual(u)``.  Raises ``RippleNotConvergedError`` when it exceeds
    ``tol`` or a value is not finite; ``what`` names the field in the message."""
    res = zero_mean_residual(u)
    if not res <= tol:
        raise RippleNotConvergedError(
            f"ripple_zero_mean: {what} has |mean|/max = {res:.3e} > {tol:.1e}")
    return res


def seam_traction_jump(Q, chi, Gx: float, Gy: float, Lx: float, Ly: float) -> dict:
    """Jump of the traction s across the two periodic boundaries (seams) of the cell [Pa].

    Q: periodic part of the pressure, ``P - G.(x - x_c)``, at the grid nodes [Pa].
    chi: fluid indicator at the nodes.
    Gx, Gy: macroscale pressure gradient [Pa/m]; Lx, Ly: cell size [m].
    The full pressure at a seam node is Q plus the ramp at ``-L/2`` on one side and at ``+L/2``
    on the other, so the two sides differ only in the cavitation indicator ``[P > 0]``.
    Returns ``{"max_x", "max_y", "max", "rms"}``: the largest jump on each seam, the largest of
    both, and the root mean square over all seam nodes.
    """
    Q = np.asarray(Q, dtype=float)
    fluid = np.asarray(chi) > 0.5
    nx, ny = Q.shape
    x_rel = (np.arange(nx) / nx - 0.5) * Lx
    y_rel = (np.arange(ny) / ny - 0.5) * Ly

    def _jump(q, fl, along_other, G_cross, L_cross):
        lo = q + G_cross * (-0.5 * L_cross) + along_other
        hi = q + G_cross * (+0.5 * L_cross) + along_other
        ind = (hi > 0.0).astype(float) - (lo > 0.0).astype(float)
        return np.where(fl, q * ind, 0.0)

    jx = _jump(Q[0, :], fluid[0, :], Gy * y_rel, Gx, Lx)      # x seam: nodes (0, j)
    jy = _jump(Q[:, 0], fluid[:, 0], Gx * x_rel, Gy, Ly)      # y seam: nodes (i, 0)
    allj = np.concatenate([jx, jy])
    return {"max_x": float(np.max(np.abs(jx))), "max_y": float(np.max(np.abs(jy))),
            "max": float(np.max(np.abs(allj))), "rms": float(np.sqrt(np.mean(allj ** 2)))}


#: limits of ``transition_checks`` (A_res: the area of one node, 1/n^2 of the cell)
FIELD_TOL = 1e-6        # change of the fluid traction and contact pressure fields (normalised L2)
FORCE_TOL = 1e-6        # change of the contact force
AREA_ABS_TOL = 1e-6     # area change / cell area
AREA_REL_TOL = 1e-3     # area change / max(area, previous area, A_res)
SET_ABS_TOL = 1e-6      # symmetric-difference area / cell area
SET_REL_TOL = 1e-3      # symmetric-difference area / max(union area, A_res)

#: largest ratio max|R_k| / max|R_{k-1}| of the residuals of consecutive passes at which pass k is
#: accepted
CONTRACTION_MAX = 0.5


def transition_checks(prev: dict, cur: dict, *, resid_max: float, change_max: float, H0: float,
                      p_scale: float, n: int, deform_tol: float, mnorm) -> tuple:
    """Changes between two consecutive ripple passes, compared with the limits above.  A diagnostic.

    prev, cur: ``{level: snapshot}`` for every time level of each pass.  A snapshot holds ``t``
        (fluid traction ``chi max(P, 0)`` as dofs [Pa]) and ``A_mask`` (obstructed area, cell
        fraction), and with contact ``p`` (nodal contact pressure [Pa]), ``P_asp`` [Pa],
        ``A_mech`` (cell fraction) and ``set`` (bool nodal contact set).
    resid_max: max|H[s] - u| [m]; change_max: max|u - u_prev| [m].
    H0: film scale [m]; p_scale: pressure scale [Pa]; n: grid nodes per side.
    deform_tol: limit on the two deformation values, relative to H0.
    mnorm: L2 norm of a dof vector on the unit cell.
    Returns ``(ok, worst)``: ``worst`` holds the largest normalised value of each check over the
    levels, and ``levels`` = "mismatch" when the two passes do not hold the same levels.
    """
    A_res = 1.0 / (n * n)
    worst = {"deform_resid": resid_max / H0, "deform_change": change_max / H0,
             "traction": 0.0, "area_mask_abs": 0.0, "area_mask_rel": 0.0, "contact_p": 0.0,
             "force": 0.0, "area_mech_abs": 0.0, "area_mech_rel": 0.0, "set_abs": 0.0, "set_rel": 0.0}
    if set(prev) != set(cur):
        worst["levels"] = "mismatch"
        return False, worst

    def _area(a, b, name):
        d = abs(b - a)
        worst[f"{name}_abs"] = max(worst[f"{name}_abs"], d)
        worst[f"{name}_rel"] = max(worst[f"{name}_rel"], d / max(a, b, A_res))

    for lev in cur:
        a, b = prev[lev], cur[lev]
        dn = mnorm(np.asarray(b["t"]) - np.asarray(a["t"]))
        worst["traction"] = max(worst["traction"], dn / max(mnorm(a["t"]), mnorm(b["t"]), p_scale))
        _area(a["A_mask"], b["A_mask"], "area_mask")
        if ("p" in a) != ("p" in b) or ("set" in a) != ("set" in b):
            worst["levels"] = "mismatch"
            return False, worst
        if "p" in a:
            pa, pb = np.asarray(a["p"], float), np.asarray(b["p"], float)
            U = (pa > 0.0) | (pb > 0.0)
            if U.any():
                num = float(np.sqrt(np.sum((pb - pa)[U] ** 2)))
                den = max(float(np.sqrt(np.sum(pa[U] ** 2))), float(np.sqrt(np.sum(pb[U] ** 2))),
                          p_scale * float(np.sqrt(U.sum())))
                worst["contact_p"] = max(worst["contact_p"], num / den)
            worst["force"] = max(worst["force"], abs(b["P_asp"] - a["P_asp"])
                                 / max(abs(a["P_asp"]), abs(b["P_asp"]), 1e-6 * p_scale))
            _area(a["A_mech"], b["A_mech"], "area_mech")
        if "set" in a:
            sa, sb = np.asarray(a["set"], bool), np.asarray(b["set"], bool)
            sym = float(np.count_nonzero(sa ^ sb)) * A_res
            union = float(np.count_nonzero(sa | sb)) * A_res
            worst["set_abs"] = max(worst["set_abs"], sym)
            worst["set_rel"] = max(worst["set_rel"], sym / max(union, A_res))
    ok = (worst["deform_resid"] <= deform_tol and worst["deform_change"] <= deform_tol
          and worst["traction"] <= FIELD_TOL and worst["contact_p"] <= FIELD_TOL
          and worst["force"] <= FORCE_TOL
          and worst["area_mask_abs"] <= AREA_ABS_TOL and worst["area_mask_rel"] <= AREA_REL_TOL
          and worst["area_mech_abs"] <= AREA_ABS_TOL and worst["area_mech_rel"] <= AREA_REL_TOL
          and worst["set_abs"] <= SET_ABS_TOL and worst["set_rel"] <= SET_REL_TOL)
    return bool(ok), worst


def residual_acceptance(resid_max, prev_resid_max, prev_set, cur_set, *, H0, deform_tol, n,
                        contraction_max=CONTRACTION_MAX) -> tuple:
    """Acceptance of ripple pass k.  The pass is accepted when:

    1. ``max|H[s_k] - u_k| <= deform_tol * H0``;
    2. for k >= 2, ``max|R_k| <= contraction_max * max|R_{k-1}|``;
    3. for k >= 2, the nodal contact set at the end level is unchanged from pass k - 1, within
       ``SET_ABS_TOL`` and ``SET_REL_TOL`` (equal sets for n < 1000).  With no set in either
       pass (full film) there is no check; a set in one pass only is not accepted.

    resid_max: max|R_k| = max|H[s_k] - u_k| [m]; prev_resid_max: that of pass k - 1, None at k = 1.
    prev_set, cur_set: bool nodal contact sets at the end level, or None.
    H0: film scale [m]; n: grid nodes per side.
    Returns ``(ok, info)``.  ``info`` holds ``contraction`` (resid / prev: 0 when both are 0, inf
    when only prev is 0, nan at k = 1) and ``end_set_abs``, ``end_set_rel`` (0 with no set to
    compare).  A non-finite residual is not accepted.
    """
    resid_max = float(resid_max)
    info = {"contraction": float("nan"), "end_set_abs": 0.0, "end_set_rel": 0.0}
    ok = bool(np.isfinite(resid_max) and resid_max <= float(deform_tol) * float(H0))
    if prev_resid_max is None:
        return ok, info
    prev = float(prev_resid_max)
    if not np.isfinite(prev):
        ratio = float("nan")
    elif prev > 0.0:
        ratio = resid_max / prev
    else:
        ratio = 0.0 if resid_max == 0.0 else float("inf")
    info["contraction"] = float(ratio)
    ok = ok and bool(np.isfinite(ratio) and ratio <= float(contraction_max))
    have = (prev_set is not None, cur_set is not None)
    if have == (True, True):
        A_res = 1.0 / (int(n) * int(n))
        sa, sb = np.asarray(prev_set, bool), np.asarray(cur_set, bool)
        if sa.shape != sb.shape:
            info["end_set_abs"] = info["end_set_rel"] = float("inf")
            return False, info
        sym = float(np.count_nonzero(sa ^ sb)) * A_res
        union = float(np.count_nonzero(sa | sb)) * A_res
        info["end_set_abs"] = sym
        info["end_set_rel"] = sym / max(union, A_res)
        ok = ok and sym <= SET_ABS_TOL and info["end_set_rel"] <= SET_REL_TOL
    elif have != (False, False):
        info["end_set_abs"] = info["end_set_rel"] = float("inf")
        ok = False
    return bool(ok), info


def pattern_shift(u, shift_x: float, shift_y: float) -> np.ndarray:
    """Field u translated by ``(shift_x, shift_y)`` cell fractions.

    The trigonometric interpolant of the grid is evaluated at ``x - shift``, which is exact for a
    band-limited field.  It carries the ripple field with a translating rough surface.
    """
    u = np.asarray(u, dtype=float)
    nx, ny = u.shape
    return band_limited_sample(u, nx, ny, shift_x=float(shift_x), shift_y=float(shift_y))
