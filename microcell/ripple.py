"""Option H micro fluid deformation, ``ripple_halfspace`` (numpy only, FEniCS-free).

Combined plan ``docs/notes/micro_mixed_roughness_location_phase_test_plan_2026-10-13.md``
section 18.15; plan ``docs/notes/ripple_halfspace_plan_2026-10-17.md`` (H1).

The micro film gains a zero-mean, cell-periodic elastic deformation ``u_r`` of the
cell's own fluid traction, through the periodic two-body half-space operator::

    u_r^(q) = 2 s^(q) / (E* |q|)  (q != 0),   u_r^(0) = 0,   1/E* = sum_i (1 - nu_i^2) / E_i

The macro keeps the cell-mean (q = 0) displacement and the macro ramp, so the
micro's cell-mean film is the macro nodal film by construction.

The traction is built in the order of 18.13.3, at the grid nodes, signed, with no
clips:

1. ``t = chi * max(P, 0)`` from the FULL pressure ``P = P0 + G.(x - x_c) + p~``
   (the full pressure decides where fluid is and where it cavitates);
2. ``s = t - chi * [P > 0] * G.(x - x_c)``: the ramp removed only where it is
   carried, i.e. ``s = chi * [P > 0] * (P - ramp)``;
3. ``r = s - <s>`` (the operator drops q = 0, which is the same operation);
4. evaluate on the periodic nodes (the seam mismatch of ``s`` is reported first:
   it is non-zero only where the cavitation boundary crosses the seam differently
   on the two sides, because ``P(x + L) = P(x) + G L``);
5. ``u_r = H[r]``.

The operator is :class:`ehl.elastic_contact.PeriodicHalfSpace`, the one
implementation shared with ``fft_elastic_v1``'s contact solve (FFT contact plan
D6 / WP8).  Grid convention as there: arrays ``[i, j]``, ``i`` along x, node
``(i, j)`` at ``(i/nx, j/ny)`` of the unit cell, periodic images dropped.

Sign: a positive traction opens the gap (``u_r > 0``); the unloaded part of the
cell (contact patch, cavitation) sits below the cell-mean film.

The pass loop's acceptance (plan ``docs/notes/ripple_fast_loop_plan_2026-10-19.md``,
section 4.6; Rob's decision of 19 October 2026, replacing O5's 13.4 rule): pass k is
accepted by :func:`residual_acceptance` when its deformation residual max|H[s_k] - u_k|
is within the budget ``ripple_tol`` x H0, from pass 2 the residual contracted by at
least :data:`CONTRACTION_MAX` since the previous pass, and the end level's nodal
contact set is unchanged.  :func:`transition_checks` (the 13.4 checks of one
transition) stays as a reported diagnostic.
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

#: deformation-model tag carried by results and checkpoints (plan O7). v1: the field is
#: held fixed during each micro run in the frame of the rough surface (plan O6).
RIPPLE_MODEL = "ripple_halfspace_v1"

#: zero-mean check of 18.13.2: |integral u dA| <= tol * A * max|u|.  On the uniform
#: periodic grid every node has the same tributary area, so the nodal mean is the
#: area mean of the P1 interpolant.
ZERO_MEAN_TOL = 1e-10


class RippleNotConvergedError(RuntimeError):
    """The ripple fixed point did not meet its acceptance, or a field failed a check.

    A ``RuntimeError``, so the micro worker's existing failure path records the task
    (with this message as its reason) and the batch kNN-fills it (plan O5)."""


def ripple_operator(nx: int, ny: int, Lx: float, Ly: float, E_star: float) -> PeriodicHalfSpace:
    """The periodic two-body half-space operator on the ``nx x ny`` grid of an
    ``Lx x Ly`` cell [m] with combined modulus ``E_star`` [Pa].  Built once per mesh
    and modulus; depends on nothing else (not the roughness, not the pressure)."""
    return PeriodicHalfSpace(nx, ny, Lx, Ly, E_star)


def fluid_traction(P, chi, ramp) -> np.ndarray:
    """Steps 1-2 of 18.13.3 at the grid nodes: ``s = chi * [P > 0] * (P - ramp)`` [Pa].

    ``P``: full pressure [Pa] (ramp included); ``chi``: fluid indicator (1 fluid, 0
    contact; values > 0.5 count as fluid); ``ramp``: ``G.(x - x_c)`` [Pa] at the same
    nodes (a scalar 0 for no gradient).  Signed: ``P - ramp`` may be negative where
    ``P > 0``; nothing is clipped.  Zero in contact and in cavitation.
    """
    P = np.asarray(P, dtype=float)
    chi = np.asarray(chi)
    ramp = np.broadcast_to(np.asarray(ramp, dtype=float), P.shape)
    carried = (chi > 0.5) & (P > 0.0)
    return np.where(carried, P - ramp, 0.0)


def ripple_traction(s) -> np.ndarray:
    """Step 3: ``r = s - <s>`` (whole-cell nodal mean)."""
    s = np.asarray(s, dtype=float)
    return s - s.mean()


def ripple(s, op: PeriodicHalfSpace) -> np.ndarray:
    """Step 5: ``u_r = H[s]`` [m].  The operator drops q = 0, so ``<s>`` need not be
    removed first (``ripple(s) == ripple(ripple_traction(s))`` to rounding)."""
    return op.apply(np.asarray(s, dtype=float))


def zero_mean_residual(u) -> float:
    """``|mean u| / max|u|`` (0 for an identically zero field; inf if non-finite)."""
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)):
        return float("inf")
    m = float(np.max(np.abs(u))) if u.size else 0.0
    if m == 0.0:
        return 0.0
    return abs(float(u.mean())) / m


def check_zero_mean(u, tol: float = ZERO_MEAN_TOL, what: str = "ripple field") -> float:
    """Raise :class:`RippleNotConvergedError` unless ``|mean u| <= tol * max|u|`` and
    every value is finite (18.13.2: after every update, history write and restore).
    Returns the residual."""
    res = zero_mean_residual(u)
    if not res <= tol:
        raise RippleNotConvergedError(
            f"ripple_zero_mean: {what} has |mean|/max = {res:.3e} > {tol:.1e}")
    return res


def seam_traction_jump(Q, chi, Gx: float, Gy: float, Lx: float, Ly: float) -> dict:
    """Jump of ``s`` across the two cell seams before projection (18.15 item 2) [Pa].

    ``Q``: the periodic part of the pressure, ``P - G.(x - x_c)``, at the grid nodes
    (``P0 + lambda + p~`` in the phase-neutral solver); ``chi``: fluid indicator at
    the nodes; ``Gx, Gy``: the imposed gradient [Pa/m].  The seam nodes ``i = 0``
    (``j = 0``) are seen from ``x = 0`` and from ``x = Lx``: there the full pressure
    is ``Q + ramp`` with the ramp at ``-Lx/2`` and at ``+Lx/2``, so the two sides
    differ only in the cavitation indicator ``[P > 0]``.

    Returns ``{"max_x", "max_y", "max", "rms"}``: the largest jump on each seam, the
    overall largest, and the root-mean-square over all seam nodes.
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

    jx = _jump(Q[0, :], fluid[0, :], Gy * y_rel, Gx, Lx)      # the x seam: nodes (0, j)
    jy = _jump(Q[:, 0], fluid[:, 0], Gx * x_rel, Gy, Ly)      # the y seam: nodes (i, 0)
    allj = np.concatenate([jx, jy])
    return {"max_x": float(np.max(np.abs(jx))), "max_y": float(np.max(np.abs(jy))),
            "max": float(np.max(np.abs(allj))), "rms": float(np.sqrt(np.mean(allj ** 2)))}


#: 13.4 budgets (radius plan 13.4; plan O5, frozen 17 October)
FIELD_TOL = 1e-6        # fluid traction and contact pressure field changes (normalised L2)
FORCE_TOL = 1e-6        # contact force change
AREA_ABS_TOL = 1e-6     # area change / cell area
AREA_REL_TOL = 1e-3     # area change / max(area, area_prev, A_res)
SET_ABS_TOL = 1e-6      # symmetric-difference area / cell area
SET_REL_TOL = 1e-3      # symmetric-difference area / max(union area, A_res)

#: the residual acceptance of the pass loop (plan ripple_fast_loop_plan_2026-10-19 section 4.6): the
#: observed contraction max|R_k| / max|R_{k-1}| a pass must show to be accepted on its residual
CONTRACTION_MAX = 0.5


def transition_checks(prev: dict, cur: dict, *, resid_max: float, change_max: float, H0: float,
                      p_scale: float, n: int, deform_tol: float, mnorm) -> tuple:
    """The 13.4 checks of one transition between consecutive passes, reported as diagnostics
    of the pass loop (plan O5 defined them as the acceptance; since the ripple fast loop,
    plan ripple_fast_loop_plan_2026-10-19, the acceptance is :func:`residual_acceptance` and
    these are computed over the levels both passes hold and kept in ``ripple_info["worst"]``).

    ``prev``, ``cur``: ``{level: snapshot}`` for every time level of the pass (the seed and
    each step of a transient, the one solve of a steady task).  A snapshot holds ``t`` (the
    fluid traction ``chi max(P, 0)`` as Vper dofs [Pa]) and ``A_mask`` (obstruction area,
    cell fraction), and optionally the contact quantities: ``p`` (nodal contact pressure
    [Pa]), ``P_asp`` [Pa], ``A_mech`` (cell fraction) and ``set`` (bool nodal contact set).
    ``resid_max`` = max|H[s] - u| and ``change_max`` = max|u - u_prev| [m] of the field;
    ``mnorm`` is the consistent-mass L2 norm of a dof vector on the unit cell; ``n`` the grid.

    Returns ``(ok, worst)`` where ``worst`` holds the largest normalised value of each
    check over the levels (and ``levels`` = "mismatch" if the level sets differ).
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
    """Accept pass k of the option H pass loop (plan ripple_fast_loop_plan_2026-10-19 section 4.6)
    when

    1. ``max|H[s_k] - u_k| <= deform_tol * H0`` (the 13.4 deformation budget);
    2. k >= 2: ``max|R_k| <= contraction_max * max|R_{k-1}|``;
    3. k >= 2: the end level's nodal contact set unchanged from pass k - 1, by 13.4's set budgets
       (``SET_ABS_TOL`` and ``SET_REL_TOL``; one node is 1/n^2 of the cell, so for n < 1000 this
       means equal sets; the micro grid is 100 x 100).  No set in either pass (full film): no
       check; a set in one only: not accepted.

    At k = 1 (``prev_resid_max`` None) only 1 applies: the ripple is below the budget and u = 0
    stands.  A non-finite residual is never accepted.

    Returns ``(ok, info)``: ``info`` holds ``contraction`` (resid / prev; 0 when both are 0, inf
    when only prev is 0; nan at k = 1), ``end_set_abs`` and ``end_set_rel`` (named apart from
    ``transition_checks``' set_abs / set_rel, which ``_ripple_run`` keeps in the same ``worst``
    dict; 0 when there is no set to compare).
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
    """``u`` translated by ``(shift_x, shift_y)`` cell fractions: the grid's
    trigonometric interpolant evaluated at ``x - shift`` (``band_limited_sample``,
    the canonical Fourier evaluator of D4).  Exact for band-limited fields; commutes
    with :func:`ripple`.  Used to carry the field with a translating rough surface
    (plan O6)."""
    u = np.asarray(u, dtype=float)
    nx, ny = u.shape
    return band_limited_sample(u, nx, ny, shift_x=float(shift_x), shift_y=float(shift_y))
