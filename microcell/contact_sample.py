"""Contact pressure of the resolved ``fft_elastic_v1`` solution at cell-local sample points, and the
asperity / total traction on a common sample set (extrema M3a, 23 October 2026).

Pure numpy; nothing from FEniCS is imported or accepted.  ``microscale/src/functions/__init__.py``
imports the FEniCS micro classes, so importing this module through the package imports dolfin; a
FEniCS-free consumer loads the file by path (as ``tests/elastic_contact/test_elastic_contact.py``
loads ``generated_ht_cell.py``).  Plan ``docs/plans/masked_pressure_extrema_and_shear_exports_
2026-10-23.md`` sections 2.2 (contact / total), 3.2 and 6.2.4; review-and-launch note section 2.3
(D5 global x only, D7 the contact pressure source); log ``docs/notes/extrema_m3a_implementation_log_
2026-10-23.md``.

Source (D7)
-----------
The nodal solution the contact class stores, ``MicroPhaseNeutralContactSolver._contact_solution``:
an ``ehl.elastic_contact.ContactSolution`` written by ``_refresh_contact`` and re-checked against the
state key by ``_fft_asperity_load`` (``micro_phase_neutral_class_contact.py`` 210-300).  Its arrays are
indexed ``[i, j]``, ``i`` along x and ``j`` along y; node ``(i, j)`` sits at ``(i/nx, j/ny)`` of the
unit cell; the periodic images at x = 1 and y = 1 are not stored (``ehl/elastic_contact.py`` 21-25).
The class writes the grid onto the Vper dofs as ``field[i, j]``, through the map of ``_periodic_grid``
(``micro_phase_neutral_class.py`` 1731-1753; ``_contact.py`` 237-239 and 273-275).  This module never
goes through dof or CG indices and never reads ``p_asp_field``.  For ``hertz_summit_v1`` that name is
a DG0 force-conserving display over the obstruction mask (``_contact.py`` 351-388), not a pressure
distribution.

Coordinates
-----------
Cell-local, in fractions of the cell: the micro mesh is ``UnitSquareMesh(n, n)``
(``scripts/run/run_microscale_phase_neutral.py`` 372), so its own vertex and quadrature coordinates
can be passed unchanged.  With ``cell_size=(Lx, Ly)`` the coordinates are read in metres instead.
Points outside [0, 1)^2 are wrapped: the cell is periodic.

Interpolant
-----------
The periodic P1 interpolant on the 'right'-diagonal triangulation of the grid.  Each square
``(i, j)``-``(i+1, j+1)`` is split along that diagonal, indices taken mod n:

    lower triangle (i,j),(i+1,j),(i+1,j+1)  for s >= t:  p = (1-s) p00 + (s-t) p10 + t p11
    upper triangle (i,j),(i,j+1),(i+1,j+1)  for t >  s:  p = (1-t) p00 + (t-s) p01 + s p11

with ``(s, t)`` the point's offset inside its square, in grid units.  These are the cells of dolfin's
``UnitSquareMesh(n, n)`` (default diagonal), so the interpolant is what the mesh's CG1 / Vper space
makes of the same nodal values.  ``phic_nd_func``, ``uc_nd_func`` and the fft ``p_asp_field``
(``_contact.py`` 237-239, 273-275) all use that space, and ``ehl.elastic_contact.p1_area`` integrates
the same triangulation.  Properties used by the gates:

* at a grid node it returns the nodal value exactly.  A coordinate within ``SNAP_TOL`` grid units of a
  node is snapped to it, so mesh-vertex coordinates such as 0.34 hit the node;
* ``p_a >= 0`` everywhere, and its maximum over any sample set is at most ``max(sol.p)``: no overshoot;
* it integrates to ``mean(sol.p) = Pasp_hom`` exactly over the cell, because every hat function on a
  uniform periodic grid integrates to 1/(nx ny).  Any positive rule that is exact on each triangle
  therefore reproduces ``Pasp_hom`` to rounding (gate 1; budget in the log, section 1.1);
* the seams are periodic: x and x + 1 give the same value, and the square between node n-1 and the
  image at x = 1 uses node 0.

The native nodal peak ``max(sol.p)``, the peak of this interpolant at points that are not nodes, and
any projected display peak (a DG0 cell mean, say) are three different statistics.  ``statistic`` says
which one a sample set gives: ``native_nodal`` when every sample is a node, ``p1_interpolated``
otherwise.

Supports at the samples
-----------------------
``mech_support``  ``p_a > 0``: where the contact traction is not zero.  Under P1 this is the open star
                  of the active nodes.
``obstruction``   ``phi_c <= 0``, with ``phi_c`` the P1 interpolant of ``sol.phi_c = g - p/k_N``: the
                  complement of the fluid solver's ``chi_fluid = [phi_c > 0]`` (``_contact.py``
                  130-133).  At a vertex it is the class's nodal level set, so M1's fluid samples
                  (``fluid_nodal_sharp_v1``, ``phic_nd_func > 0``) are its complement there.

The two need not agree, and plan section 2.2 says not to assume they do.  Under P1 the pressure is
positive on a fringe on the fluid side of the obstruction edge.  A node can also be obstructed with
p = 0, or carry p > 0 on the fluid side, within the complementarity tolerance.  ``support_report``
measures all of this; nothing here suppresses either contribution.

Traction (plan section 2.2)
---------------------------
    t_a(x) = mu_material p_a(x) U_hat,     t_t(x) = t_f(x) + t_a(x)    at the common samples

Zero-speed law, exactly as the existing code has it (``ehl/macro/friction.py`` 186-191,
``calc_asp_friction``; documented, not changed)::

    tau_asp_i = mu_material * Pc * p0 * U_i / (sqrt(U.U) + 1e-30)

``U`` is the macro's CG1 field ``Rball U0 omega (x_z, 0, -x_x)`` (``ehl/macro/fe_problem.py`` 156-166),
so at a reversal (omega = 0) and on the rotation axis ``U = 0`` and the asperity traction is zero: there
is no static friction.  For ``|U| > 0`` the factor is ``U/|U|`` to within ``1e-30/|U|``, so it is
regularised, not normalised.  The 1e-30 is in the units of that field.  ``sliding_direction``
reproduces the law, bit for bit when it is given the macro's own ``U``.  In the micro's local basis
the micro receives ``R U = (|U|, 0, 0)`` up to rounding (``fe_problem.py`` 836-858), so the local
direction is ``(|U|/(|U| + 1e-30), 0, 0)``.  ``p_a`` is in Pa here.  The macro's ``Pc`` is
nondimensional and is multiplied by ``p0``.

Statuses (strings, never inferred from NaN)
-------------------------------------------
``ok``                 ``fft_elastic_v1``, some contact pressure
``no_support``         ``fft_elastic_v1`` with ``p = 0`` everywhere: a valid state without the
                       contribution; p_a = 0 at every sample
``state_invalid``      no stored solution, or arrays that are not a finite, non-negative grid: NaN
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from microcell.shear_reduce import GLOBAL_X, reduce_traction   # pure numpy; ehl/__init__ imports nothing

CONTACT_SAMPLER = "fft_nodal_p1_periodic_v1"
SAMPLER_DEFINITION = ("sol.p of the stored fft_elastic_v1 solution, periodic P1 interpolant on the "
                      "'right'-diagonal grid (UnitSquareMesh cells), Pa")

from microcell.contact import FFT as FFT_MODEL      # the one contact law (one definition)

STATUS_OK = "ok"
STATUS_NO_SUPPORT = "no_support"
STATUS_STATE_INVALID = "state_invalid"
CONTACT_VALID_STATUSES = (STATUS_OK, STATUS_NO_SUPPORT)

STATISTIC_NATIVE = "native_nodal"
STATISTIC_INTERPOLATED = "p1_interpolated"

#: a coordinate this close to a node, in grid units, is the node
SNAP_TOL = 1e-9
#: the regularisation of the macro's asperity direction, friction.py 188 (units of the velocity field)
ZERO_SPEED_EPS = 1e-30


@dataclass(frozen=True)
class ContactSamples:
    """The resolved contact pressure at N samples.  Pa; boolean masks per sample."""
    p: np.ndarray                 # (N,) contact pressure [Pa]; NaN unless the status is ok / no_support
    mech_support: np.ndarray      # (N,) p > 0
    obstruction: np.ndarray       # (N,) phi_c <= 0: the fluid solver's contact mask
    at_node: np.ndarray           # (N,) the sample is a grid node (after snapping)
    status: str
    contact_model: str
    grid: tuple                   # (nx, ny) of the solution; (0, 0) when there is none
    native_peak: float            # max(sol.p) [Pa] (the class's pasp_max, _contact.py 263)
    Pasp_hom: float               # mean(sol.p) [Pa] (the class's Pasp_hom, _contact.py 260)
    statistic: str                # native_nodal | p1_interpolated
    sampler: str = CONTACT_SAMPLER

    @property
    def valid(self) -> bool:
        return self.status in CONTACT_VALID_STATUSES

    def summary(self) -> dict:
        """Plain numbers and strings (no arrays): picklable, JSON-ready."""
        finite = np.isfinite(self.p)
        return {"status": self.status, "contact_model": self.contact_model, "grid": list(self.grid),
                "n_samples": int(self.p.size), "n_mech_support": int(np.count_nonzero(self.mech_support)),
                "n_obstruction": int(np.count_nonzero(self.obstruction)),
                "n_at_node": int(np.count_nonzero(self.at_node)),
                "sample_peak_Pa": float(np.max(self.p)) if finite.any() else float("nan"),
                "native_peak_Pa": float(self.native_peak), "Pasp_hom_Pa": float(self.Pasp_hom),
                "statistic": self.statistic, "sampler": self.sampler}


# ---------------------------------------------------------------------------
# the interpolant
# ---------------------------------------------------------------------------
def _grid_coords(xy, shape, cell_size):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"sample coordinates must have shape (N, 2), got {xy.shape}")
    if not np.all(np.isfinite(xy)):
        raise ValueError("sample coordinates must be finite")
    u = xy.copy()
    if cell_size is not None:
        lx, ly = (float(v) for v in cell_size)
        if not (np.isfinite(lx) and np.isfinite(ly) and lx > 0 and ly > 0):
            raise ValueError(f"cell_size must be two positive lengths, got {cell_size!r}")
        u[:, 0] /= lx
        u[:, 1] /= ly
    nx, ny = shape
    gx, gy = u[:, 0] * nx, u[:, 1] * ny
    rx, ry = np.rint(gx), np.rint(gy)
    sx, sy = np.abs(gx - rx) <= SNAP_TOL, np.abs(gy - ry) <= SNAP_TOL
    gx, gy = np.where(sx, rx, gx), np.where(sy, ry, gy)
    return gx, gy, sx & sy


def periodic_p1(values, xy, *, cell_size=None):
    """The periodic P1 interpolant of nodal ``values`` (``[i, j]``, node at ``(i/nx, j/ny)``) at ``xy``
    (N x 2, cell fractions, or metres with ``cell_size``).  Returns ``(v, at_node)``."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 2:
        raise ValueError(f"nodal values must be an nx x ny grid with nx, ny >= 2, got shape {v.shape}")
    nx, ny = v.shape
    gx, gy, at_node = _grid_coords(xy, v.shape, cell_size)
    fx, fy = np.floor(gx), np.floor(gy)
    s, t = gx - fx, gy - fy
    i0 = fx.astype(np.int64) % nx
    j0 = fy.astype(np.int64) % ny
    i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
    p00, p11 = v[i0, j0], v[i1, j1]
    lower = s >= t
    out = np.where(lower,
                   (1.0 - s) * p00 + (s - t) * v[i1, j0] + t * p11,
                   (1.0 - t) * p00 + (t - s) * v[i0, j1] + s * p11)
    out = np.where(at_node, p00, out)            # a node is the node's value, bit for bit
    return out, at_node


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
def _empty(n, status, model, grid=(0, 0)):
    nan = np.full(n, np.nan)
    no = np.zeros(n, dtype=bool)
    return ContactSamples(p=nan, mech_support=no, obstruction=no.copy(), at_node=no.copy(), status=status,
                          contact_model=str(model), grid=tuple(int(g) for g in grid),
                          native_peak=float("nan"), Pasp_hom=float("nan"), statistic=STATISTIC_INTERPOLATED)


def sample_contact(solution: Optional[object], xy, *, contact_model: str = FFT_MODEL, cell_size=None) -> ContactSamples:
    """``p_a`` [Pa] and the supports at the samples ``xy``, from the stored contact solution.

    ``solution`` is the class's ``_contact_solution`` (anything with ``p`` and ``phi_c`` grids, as
    ``ehl.elastic_contact.ContactSolution``), or None.  ``contact_model`` is the task's identity:
    ``fft_elastic_v1``, the only one in this repo (any other raises)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"sample coordinates must have shape (N, 2), got {xy.shape}")
    n = xy.shape[0]
    model = str(contact_model)
    if model != FFT_MODEL:
        raise ValueError(f"unknown contact model {model!r}")
    if solution is None:
        return _empty(n, STATUS_STATE_INVALID, model)
    p_grid = np.asarray(getattr(solution, "p", None), dtype=float)
    phi_grid = np.asarray(getattr(solution, "phi_c", None), dtype=float)
    if (p_grid.ndim != 2 or phi_grid.shape != p_grid.shape or min(p_grid.shape) < 2
            or not np.all(np.isfinite(p_grid)) or not np.all(np.isfinite(phi_grid)) or np.any(p_grid < 0.0)):
        return _empty(n, STATUS_STATE_INVALID, model, p_grid.shape if p_grid.ndim == 2 else (0, 0))
    p, at_node = periodic_p1(p_grid, xy, cell_size=cell_size)
    phi, _ = periodic_p1(phi_grid, xy, cell_size=cell_size)
    peak = float(np.max(p_grid))
    status = STATUS_OK if peak > 0.0 else STATUS_NO_SUPPORT
    return ContactSamples(p=p, mech_support=p > 0.0, obstruction=phi <= 0.0, at_node=at_node, status=status,
                          contact_model=model, grid=tuple(int(g) for g in p_grid.shape), native_peak=peak,
                          Pasp_hom=float(p_grid.mean()),
                          statistic=STATISTIC_NATIVE if n and bool(np.all(at_node)) else STATISTIC_INTERPOLATED)


# ---------------------------------------------------------------------------
# weights on a sample set (P1-exact)
# ---------------------------------------------------------------------------
def p1_lumped_weights(coords, cells) -> np.ndarray:
    """Lumped P1 weights of the vertices of a triangle mesh: each vertex gets a third of the area of
    every triangle it belongs to.  Exact for every function linear on each triangle, so on the micro
    mesh (``mesh_m.coordinates()``, ``mesh_m.cells()``) the weighted vertex mean of ``p_a`` is
    ``Pasp_hom`` up to rounding.  They sum to the mesh area (1 on the unit cell)."""
    X = np.asarray(coords, dtype=float)
    C = np.asarray(cells, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 2 or C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("p1_lumped_weights needs (nv, 2) coordinates and (nc, 3) triangles")
    a, b, c = X[C[:, 0]], X[C[:, 1]], X[C[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    w = np.zeros(X.shape[0])
    for k in range(3):
        np.add.at(w, C[:, k], area / 3.0)
    return w


# ---------------------------------------------------------------------------
# asperity and total traction
# ---------------------------------------------------------------------------
def sliding_direction(u, eps: float = ZERO_SPEED_EPS) -> np.ndarray:
    """``U / (|U| + eps)``: the asperity direction of ``calc_asp_friction`` (friction.py 186-191),
    zero at zero speed.  ``u`` is (3,) or (N, 3); the result has the same shape."""
    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 3:
        raise ValueError(f"velocity must have 3 components, got shape {u.shape}")
    mag = np.sqrt(np.sum(u * u, axis=-1, keepdims=True))
    return u / (mag + float(eps))


def asperity_traction(p, mu_material: float, u_hat) -> np.ndarray:
    """``t_a = mu_material p_a U_hat`` (N, 3) [Pa] for p_a [Pa] at N samples and the direction
    ``u_hat`` (3,) for the whole cell or (N, 3) per sample (from ``sliding_direction``)."""
    p = np.asarray(p, dtype=float).reshape(-1)
    d = np.asarray(u_hat, dtype=float)
    mu = float(mu_material)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu_material must be finite and >= 0, got {mu_material!r}")
    if d.shape == (3,):
        d = np.broadcast_to(d, (p.size, 3))
    if d.shape != (p.size, 3):
        raise ValueError(f"u_hat must have shape (3,) or ({p.size}, 3), got {d.shape}")
    return mu * p[:, None] * d


def _pad3(v, n, what):
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] != n or v.shape[1] not in (2, 3):
        raise ValueError(f"{what} must have shape ({n}, 2) or ({n}, 3), got {v.shape}")
    return v if v.shape[1] == 3 else np.column_stack((v, np.zeros(n)))


def total_traction(t_fluid, fluid_support, t_asp, mech_support) -> np.ndarray:
    """``t_t = t_f [fluid] + t_a [mech]`` at each common sample: each contribution is its physical zero
    outside its own support, and both are kept where the supports overlap."""
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    ms = np.asarray(mech_support, dtype=bool).reshape(-1)
    n = fs.size
    if ms.shape != (n,):
        raise ValueError("fluid and mechanical supports must be on the same samples")
    tf, ta = _pad3(t_fluid, n, "t_fluid"), _pad3(t_asp, n, "t_asp")
    return np.where(fs[:, None], tf, 0.0) + np.where(ms[:, None], ta, 0.0)


def support_report(fluid_support, contact: ContactSamples, weights) -> dict:
    """Where the fluid set, the obstruction set and the mechanical support agree and where they do not,
    by sample weight (fractions of the whole sample set) and count.  Reported, never corrected."""
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = fs.size
    if w.shape != (n,) or contact.p.shape != (n,):
        raise ValueError("fluid support, weights and contact samples must have the same length")
    ms, ob = contact.mech_support, contact.obstruction
    W = float(np.sum(w)) if n else float("nan")

    def frac(m):
        return float(np.sum(w[m]) / W) if n else float("nan")

    sets = {
        "fluid": fs, "mech_support": ms, "obstruction": ob,
        "fluid_and_mech": fs & ms,                          # both tractions present: the overlap
        "mech_outside_obstruction": ms & ~ob,               # contact pressure where the fluid sees fluid
        "obstruction_without_pressure": ob & ~ms,           # obstructed, no contact traction
        "neither": ~fs & ~ms,                               # no traction contribution at all
        "fluid_not_complement_of_obstruction": fs == ob,    # the caller's fluid set vs the contact level set
    }
    out = {k: {"count": int(np.count_nonzero(m)), "fraction": frac(m)} for k, m in sets.items()}
    if contact.valid and np.all(np.isfinite(contact.p)):
        f_all = float(np.sum(w * contact.p))
        f_fluid = float(np.sum(w[fs & ms] * contact.p[fs & ms]))
        out["contact_force_share_on_fluid_samples"] = (f_fluid / f_all) if f_all > 0.0 else 0.0
    else:
        out["contact_force_share_on_fluid_samples"] = float("nan")
    out["status"] = contact.status
    return out


@dataclass(frozen=True)
class ContactTotalReduction:
    """Fluid, asperity and total traction of one micro state, reduced on one common sample set."""
    fluid: object                  # ehl.shear_reduce.ShearReduction
    asperity: object
    total: object
    support: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"fluid": self.fluid.as_dict(), "asperity": self.asperity.as_dict(),
                "total": self.total.as_dict(), "support": dict(self.support), "contact": dict(self.contact)}


def reduce_contact_and_total(t_fluid, fluid_support, fluid_valid, contact: ContactSamples, mu_material: float,
                             u_hat, weights, *, direction=GLOBAL_X,
                             fluid_state_valid: bool = True) -> ContactTotalReduction:
    """The three contributions on the same samples, each reduced by ``ehl.shear_reduce.reduce_traction``
    (every sample projected on ``direction``, in the local basis, before anything is reduced).  The total is formed sample by sample first and reduced afterwards; extrema
    of the components are never added.

    ``t_fluid`` (N, 2|3) [Pa] is the fluid traction in the local basis (M2b), ``fluid_support`` its
    samples (``chi_f``), ``fluid_valid`` which were evaluated.  ``u_hat`` is the asperity direction in the
    same basis (``sliding_direction`` of the velocity, (3,) or (N, 3)).  An unsupported or invalid
    contact state leaves the fluid reduction as it is and makes the asperity and total ``state_invalid``.
    """
    fs = np.asarray(fluid_support, dtype=bool).reshape(-1)
    fv = np.asarray(fluid_valid, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = fs.size
    tf = _pad3(t_fluid, n, "t_fluid")
    if fv.shape != (n,) or w.shape != (n,) or contact.p.shape != (n,):
        raise ValueError("fluid arrays, weights and contact samples must have the same length")
    fluid = reduce_traction(tf, w, fv, direction, support=fs, state_valid=fluid_state_valid)

    c_ok = contact.valid
    p = np.where(np.isfinite(contact.p), contact.p, 0.0) if c_ok else np.zeros(n)
    ta = asperity_traction(p, mu_material, u_hat)
    a_valid = np.isfinite(contact.p) if c_ok else np.zeros(n, dtype=bool)
    asperity = reduce_traction(ta, w, a_valid, direction, support=contact.mech_support,
                               state_valid=c_ok)

    ms = contact.mech_support
    tt = total_traction(tf, fs, ta, ms)
    t_valid = (fv | ~fs) & (a_valid | ~ms)
    total = reduce_traction(tt, w, t_valid, direction, support=fs | ms,
                            state_valid=bool(fluid_state_valid and c_ok))
    return ContactTotalReduction(fluid=fluid, asperity=asperity, total=total,
                                 support=support_report(fs, contact, w), contact=contact.summary())
