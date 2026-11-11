"""Periodic elastic micro contact for ``fft_elastic_v1`` (numpy only, FEniCS-free).

One displacement-controlled frictionless normal contact problem per micro cell,
on the cell's periodic n x n nodal grid (plan
``docs/notes/fft_elastic_contact_plan_2026-10-16.md`` sections 3.1-3.4)::

    find p >= 0 with  g = h_rigid - h_min + u[p] >= 0  and  p * g = 0,
    u[p]^(q) = 2 p^(q) / (E* |q|)   (q != 0),   u[p]^(0) = 0.

* ``E*`` is the combined two-body modulus (``ehl.contact.effective_modulus``).
* The q = 0 mode is removed: the macro already applies the cell-mean contact
  deformation through ``K . P_asp`` (decision D3), so the micro carries only the
  zero-mean ripple.  For a square summit lattice this is the beta correction
  ``delta = (a^2/R) (1 - 4 beta a / (3 L))``, which comes out of the operator.
* Displacement control: ``h_rigid`` comes from the macro film and the roughness;
  there is no load target in the micro.
* A mean rigid gap <= h_min has no solution (the ripple cannot lift the mean
  gap).  It is rejected before solving (decision D8), as
  :class:`ContactInfeasibleError`.

Grid convention: arrays are indexed ``[i, j]`` with ``i`` along x and ``j``
along y; node ``(i, j)`` sits at ``(i/nx, j/ny)`` of the unit cell, i.e. at the
vertices of dolfin's ``UnitSquareMesh(n, n)`` (default 'right' diagonal: cells
``(i,j)-(i+1,j)-(i+1,j+1)`` and ``(i,j)-(i,j+1)-(i+1,j+1)``), with the periodic
images at ``x = 1`` / ``y = 1`` dropped.

Every error raised here subclasses :class:`ehl.contact.ContactResolutionError`
(itself a ``RuntimeError``), so the micro worker's failure path records the
task and the kNN fill applies unchanged.

Ported from the 16 October prototype (``_to_delete/fft_contact_prototype_20261016/
contact_fft_proto.py``); its outputs are the expected values of
``tests/elastic_contact/test_elastic_contact.py``.  Departures from the
prototype, each deliberate:

* no rigid overlap at any node returns the exact solution p = 0 at once,
  whatever the warm start (the prototype iterated from a positive warm start);
* the stopping rule is followed by a complementarity check on the returned
  state, and an exhausted iteration budget raises instead of returning;
* the P1 area is vectorised (same formula).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microcell.contact import ContactResolutionError

__all__ = [
    "ContactInfeasibleError", "ContactNotConvergedError",
    "PeriodicHalfSpace", "ContactSolution", "solve_contact", "contact_tangent",
    "p1_area", "periodic_components", "band_limited_sample",
]


class ContactInfeasibleError(ContactResolutionError):
    """No contact solution exists: the mean rigid gap is at or below h_min (D8)."""


class ContactNotConvergedError(ContactResolutionError):
    """The contact solve or its tangent did not meet its tolerances."""


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------
class PeriodicHalfSpace:
    """Combined surface displacement of two periodic elastic half-spaces.

    ``apply(p)`` returns ``u`` [m] for a nodal pressure ``p`` [Pa] on the
    ``(nx, ny)`` grid of a cell ``Lx x Ly`` [m]: ``u^(q) = 2 p^(q) / (E* |q|)``,
    mean removed.  The kernel is computed once (rfft2 layout).  This is the
    one implementation of the operator; option H's fluid ripple uses it too.
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, E_star: float):
        nx, ny = int(nx), int(ny)
        if nx < 2 or ny < 2:
            raise ValueError(f"PeriodicHalfSpace needs at least 2 x 2 nodes, got {nx} x {ny}")
        Lx, Ly, E_star = float(Lx), float(Ly), float(E_star)
        if not all(np.isfinite(v) and v > 0 for v in (Lx, Ly, E_star)):
            raise ValueError("PeriodicHalfSpace needs finite Lx, Ly, E_star > 0")
        qx = 2.0 * np.pi * np.fft.fftfreq(nx, Lx / nx)
        qy = 2.0 * np.pi * np.fft.rfftfreq(ny, Ly / ny)
        Q = np.hypot(qx[:, None], qy[None, :])
        Q[0, 0] = 1.0
        G = 2.0 / (E_star * Q)
        G[0, 0] = 0.0
        self.G = G
        self.shape = (nx, ny)
        self.Lx, self.Ly, self.E_star = Lx, Ly, E_star
        self.dA = Lx * Ly / (nx * ny)
        self.dx, self.dy = Lx / nx, Ly / ny

    @property
    def k_fundamental(self) -> float:
        """Half-space stiffness of the cell's fundamental mode, E* pi / max(Lx, Ly) [Pa/m]."""
        return self.E_star * np.pi / max(self.Lx, self.Ly)

    @property
    def k_nyquist(self) -> float:
        """Stiffness of the grid's Nyquist mode, k_N = E* pi / (2 max(dx, dy)) [Pa/m] (D2)."""
        return self.E_star * np.pi / (2.0 * max(self.dx, self.dy))

    def apply(self, p: np.ndarray) -> np.ndarray:
        """Zero-mean surface displacement u [m] of the nodal pressure p [Pa]."""
        return np.fft.irfft2(np.fft.rfft2(p) * self.G, s=self.shape)

    __call__ = apply


# ---------------------------------------------------------------------------
# Contact solve
# ---------------------------------------------------------------------------
@dataclass
class ContactSolution:
    """Nodal solution of one contact problem (arrays on the solver grid, SI)."""
    p: np.ndarray            # contact pressure [Pa], >= 0
    u: np.ndarray            # zero-mean contact deformation [m]
    g: np.ndarray            # deformed gap above h_min, h_rigid + u - h_min [m]
    phi_c: np.ndarray        # complementarity level set g - p / k_N [m] (< 0 in contact)
    active: np.ndarray       # bool, p > 0
    iterations: int
    residual: float          # complementarity violation of the returned state [m]
    change: float            # last relative pressure change sum|dp| / sum p
    k_N: float               # mask stiffness [Pa/m]

    @property
    def P_asp(self) -> float:
        """Cell-mean contact pressure = force per TOTAL cell area [Pa]."""
        return float(self.p.mean())


def _complementarity(p: np.ndarray, g: np.ndarray) -> float:
    act = p > 0
    on_set = float(np.max(np.abs(g[act]))) if act.any() else 0.0
    off_set = float(max(-np.min(g[~act]), 0.0)) if (~act).any() else 0.0
    return max(on_set, off_set)


def solve_contact(h_rigid: np.ndarray, h_min: float, op: PeriodicHalfSpace,
                  p0: np.ndarray | None = None, tol: float = 1e-8, maxit: int = 500,
                  comp_tol: float | None = None) -> ContactSolution:
    """Polonsky-Keer conjugate gradient with projection, displacement controlled.

    ``h_rigid`` [m] is the rigid (undeformed) film on the grid, ``h_min`` [m]
    the contact datum.  ``p0`` is an optional warm start (negative entries are
    dropped).  Two FFTs per iteration.  Safeguards (plan 3.2):

    * soft start: the rigid overlap times the fundamental-mode stiffness
      E* pi / L (a start at grid stiffness overshoots to a fully loaded cell,
      where the mean-removed problem is singular);
    * whole-cell guard: a step that loads every node is halved;
    * a-priori feasibility: mean(h_rigid) - h_min <= 0 raises
      :class:`ContactInfeasibleError` before any iteration (D8);
    * stopping rule sum|dp| / sum p <= ``tol``, then the complementarity of the
      returned state must be <= ``comp_tol`` (default 1e-6 h_min), else
      :class:`ContactNotConvergedError`.
    """
    h_rigid = np.asarray(h_rigid, dtype=float)
    if h_rigid.shape != op.shape:
        raise ValueError(f"h_rigid shape {h_rigid.shape} != operator grid {op.shape}")
    h_min = float(h_min)
    if not np.isfinite(h_min) or h_min <= 0.0:
        raise ValueError(f"solve_contact needs a positive contact datum h_min, got {h_min!r}")
    if not np.all(np.isfinite(h_rigid)):
        raise ContactNotConvergedError("contact_input_nonfinite: rigid film has NaN/inf nodes")
    if comp_tol is None:
        comp_tol = 1e-6 * h_min
    k_N = op.k_nyquist
    g_r = h_rigid - h_min
    mean_gap = float(g_r.mean())
    if not mean_gap > 0.0:
        raise ContactInfeasibleError(
            f"contact_infeasible: mean rigid gap {mean_gap:.6e} m <= 0 above h_min "
            "(the zero-mean ripple cannot lift the mean gap; D8)")
    if not (g_r < 0.0).any():
        zero = np.zeros_like(g_r)
        return ContactSolution(p=zero, u=zero.copy(), g=g_r.copy(), phi_c=g_r.copy(),
                               active=np.zeros(g_r.shape, dtype=bool), iterations=0,
                               residual=0.0, change=0.0, k_N=k_N)

    p = np.zeros_like(g_r) if p0 is None else np.where(np.asarray(p0, dtype=float) > 0.0, p0, 0.0)
    if p.shape != g_r.shape:
        raise ValueError(f"warm start shape {p.shape} != grid {g_r.shape}")
    if not (p > 0.0).any():
        p = np.where(g_r < 0.0, -g_r * op.k_fundamental, 0.0)

    t = np.zeros_like(g_r)
    G_old = 1.0
    delta = 0
    err = np.inf
    it = 0
    converged = False
    for it in range(1, int(maxit) + 1):
        g = g_r + op.apply(p)
        I = p > 0.0
        G_new = float(np.sum(g[I] ** 2))
        t = np.where(I, g + delta * (G_new / G_old) * t, 0.0)
        G_old = G_new if G_new > 0.0 else 1.0
        q = op.apply(t)
        den = float(np.sum(q[I] * t[I]))
        tau = float(np.sum(g[I] * t[I])) / den if den > 0.0 else 0.0
        p_old = p.copy()
        p = p - tau * t
        p[p < 0.0] = 0.0
        if p.all():
            # whole cell 'in contact': singular for the mean-removed problem
            p = 0.5 * (p_old + p)
            p[p < 0.0] = 0.0
        I_ol = (p == 0.0) & (g < 0.0)
        if I_ol.any():
            p[I_ol] -= tau * g[I_ol]
            delta = 0
        else:
            delta = 1
        err = float(np.sum(np.abs(p - p_old))) / max(float(np.sum(p)), 1e-300)
        if not np.isfinite(err):
            break
        if err <= tol:
            converged = True
            break
    if not converged:
        raise ContactNotConvergedError(
            f"contact_not_converged: relative change {err:.3e} > {tol:.1e} after {it} iterations")
    u = op.apply(p)
    g = g_r + u
    res = _complementarity(p, g)
    if not res <= comp_tol:
        raise ContactNotConvergedError(
            f"contact_complementarity: residual {res:.3e} m > {comp_tol:.1e} m after {it} iterations")
    return ContactSolution(p=p, u=u, g=g, phi_c=g - p / k_N, active=p > 0.0, iterations=it,
                           residual=res, change=err, k_N=k_N)


def contact_tangent(op: PeriodicHalfSpace, active: np.ndarray, tol: float = 1e-8,
                    maxit: int = 500) -> float:
    """dP_asp/dH [Pa/m] (<= 0) on the frozen contact set (plan 3.3, D7).

    Solves ``G_II dp = -1`` on the active set I by conjugate gradients with the
    same operator, and returns ``<dp>`` (cell mean).  ``G_II`` is positive
    definite whenever I is not the whole cell.  The Newton-like macro coupling
    uses ``k_asp = -dP_asp/dH``.
    """
    active = np.asarray(active, dtype=bool)
    if not active.any():
        return 0.0
    if active.all():
        raise ContactInfeasibleError("contact_tangent: the whole cell is in contact (G_II singular)")
    b = np.where(active, -1.0, 0.0)
    x = np.zeros_like(b)
    r = b.copy()
    d = r.copy()
    rr = float(np.sum(r * r))
    b_norm = np.sqrt(rr)
    for _ in range(int(maxit)):
        Ad = np.where(active, op.apply(d), 0.0)
        dAd = float(np.sum(d * Ad))
        if not dAd > 0.0:
            raise ContactNotConvergedError("contact_tangent: G_II not positive on the search direction")
        alpha = rr / dAd
        x += alpha * d
        r -= alpha * Ad
        rn = float(np.sum(r * r))
        if np.sqrt(rn) <= tol * b_norm:
            return float(x.mean())
        d = r + (rn / rr) * d
        rr = rn
    raise ContactNotConvergedError(f"contact_tangent: CG did not reach {tol:.1e} in {maxit} iterations")


# ---------------------------------------------------------------------------
# Mask area and topology (diagnostics on the same nodal solution)
# ---------------------------------------------------------------------------
def p1_area(phi: np.ndarray) -> float:
    """Area fraction of ``{phi < 0}`` for the P1 interpolant of nodal ``phi``.

    Exact for the 'right'-diagonal ``UnitSquareMesh`` triangulation of the
    periodic grid: per square, triangles ``(i,j),(i+1,j),(i+1,j+1)`` and
    ``(i,j),(i,j+1),(i+1,j+1)``.  A triangle with one negative vertex value
    ``f0`` and the others ``f1, f2`` has negative area fraction
    ``f0^2 / ((f0 - f1)(f0 - f2))``; two negatives take the complement of the
    same formula for the positive vertex.  Zero counts as non-negative.
    """
    a = np.asarray(phi, dtype=float)
    b = np.roll(a, -1, 0)
    c = np.roll(b, -1, 1)
    d = np.roll(a, -1, 1)
    total = 0.0
    for f1, f2, f3 in ((a, b, c), (a, d, c)):
        F = np.stack([f1, f2, f3]).reshape(3, -1)
        neg = (F < 0.0).sum(0)
        total += float(np.count_nonzero(neg == 3))
        S = np.sort(F, axis=0)
        one = neg == 1          # odd vertex = the negative one = the minimum
        if one.any():
            fo, fa, fb = S[0, one], S[1, one], S[2, one]
            total += float(np.sum(fo * fo / ((fo - fa) * (fo - fb))))
        two = neg == 2          # odd vertex = the non-negative one = the maximum
        if two.any():
            fo, fa, fb = S[2, two], S[0, two], S[1, two]
            total += float(np.sum(1.0 - fo * fo / ((fo - fa) * (fo - fb))))
    return total / (2.0 * a.size)


def periodic_components(active: np.ndarray) -> tuple[int, int]:
    """(number of connected contact components, fewest nodes in one).

    Nodes connect along the mesh edges of the 'right'-diagonal triangulation
    (x, y and the (+1, +1) diagonal), across the periodic seams.  Diagnostic
    only: the mechanics never use patch topology.  ``(0, 0)`` without contact.
    """
    active = np.asarray(active, dtype=bool)
    nx, ny = active.shape
    idx = np.flatnonzero(active.ravel())
    if idx.size == 0:
        return 0, 0
    parent = {int(k): int(k) for k in idx}

    def find(k):
        root = k
        while parent[root] != root:
            root = parent[root]
        while parent[k] != root:
            parent[k], k = root, parent[k]
        return root

    for k in idx:
        i, j = divmod(int(k), ny)
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            m = ((i + di) % nx) * ny + (j + dj) % ny
            if m in parent:
                ra, rb = find(int(k)), find(m)
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(int(k)) for k in idx])
    _, counts = np.unique(roots, return_counts=True)
    return int(counts.size), int(counts.min())


# ---------------------------------------------------------------------------
# Band-limited (Fourier) evaluator of a sampled periodic surface (D4)
# ---------------------------------------------------------------------------
def band_limited_sample(z: np.ndarray, nx: int, ny: int, shift_x: float = 0.0,
                        shift_y: float = 0.0) -> np.ndarray:
    """Trigonometric interpolant of the periodic source ``z`` (N x M, node
    ``(i, j)`` at ``(i/N, j/M)``) evaluated at the ``nx x ny`` grid nodes
    ``(i/nx - shift_x, j/ny - shift_y)`` (shifts in cell fractions).

    [16 Oct 2026, Rob's decision] A thin wrapper of the canonical evaluator,
    ``ehl.roughness.evaluator.FourierSurface`` (decision D4), so there is one
    reconstruction: this module keeps its x-first ``[i, j]`` layout and
    cell-fraction shifts, and the source mean is kept.  An even axis's
    Nyquist modes are split symmetrically, the (N/2, M/2) corner mode
    included (the earlier real-part formula gave cos(ax + by) there; the two
    differ only in that mode).  Reproduces the source exactly at
    ``nx = N, ny = M`` with no shift.  Bilinear refill makes every source
    minimum a pyramid tip and the contact load grid-dependent (plan 1,
    condition 1).
    """
    from microcell.roughness.evaluator import FourierSurface
    z = np.asarray(z, dtype=float)
    fs = FourierSurface(z.T, 1.0, 1.0)              # canonical z[j_y, i_x] on a unit cell
    return fs.nodal(int(nx), int(ny), shift=(float(shift_x), float(shift_y))).T + fs.mean_removed
