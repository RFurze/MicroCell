"""Report data for a prepared surface or surface pair (R2; the CLI is R3).

:func:`surface_report` returns a JSON-safe dictionary with

``identity``        recipe id, array hash, evaluator identity
``cell``            Lx, Ly, nx, ny, dx, dy
``targets``         requested vs realised Sa/Sq/Sz, tolerances, status
``statistics``      :func:`ehl.roughness.stats.surface_stats` of the master array
``spectrum``        wavenumber band, Nyquist share, nodal vs continuous variance
``reconstruction``  Fourier extrema (oversampled) against the source extrema
``fidelity``        for each requested mesh size n and displacement: Sa, Sq,
                    Sz and the film-reducing extreme of the Fourier nodal field
                    and of the legacy bilinear field, as ratios to the master
                    array
``generation``      generator diagnostics recorded at preparation

:func:`pair_report` adds the combined-gap statistics of a stationary and/or
translating pair over a set of relative displacements (reported, never
renormalised).  :func:`format_report` renders a text table.

Only NumPy is imported.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .assets import PreparedSurface, json_safe
from .evaluator import Bandwidth, FourierSurface, bilinear_nodal
from .pair import SurfacePair
from .stats import height_moments, surface_stats

__all__ = ["fidelity_table", "surface_report", "pair_report", "format_report"]


def _field_numbers(v: np.ndarray) -> Dict[str, float]:
    m = height_moments(v)
    vc = v - v.mean()
    return {"Sa": m["Sa"], "Sq": m["Sq"], "Sz": float(vc.max() - vc.min()),
            "film_reducing_extreme": max(0.0, -float(vc.min()))}


def fidelity_table(z: np.ndarray, Lx: float, Ly: float,
                   mesh_sizes: Sequence[int] = (60, 64, 100, 128),
                   shifts: Sequence[Tuple[float, float]] = ((0.0, 0.0),),
                   bandwidth: Bandwidth = Bandwidth()) -> List[Dict[str, object]]:
    """Fourier and legacy-bilinear nodal statistics relative to the master
    array, per mesh size (square n x n, as the micro UnitSquareMesh) and shift."""
    ref = _field_numbers(np.asarray(z, dtype=float))
    fs = FourierSurface(z, Lx, Ly, bandwidth)
    rows = []
    for n in mesh_sizes:
        for s in shifts:
            row: Dict[str, object] = {"n": int(n), "shift": [float(s[0]), float(s[1])]}
            for name, v in (("fourier", fs.nodal(n, n, shift=s)),
                            ("bilinear_legacy", bilinear_nodal(z, Lx, Ly, n, n, shift=s))):
                got = _field_numbers(v)
                row[name] = {k: (got[k] / ref[k] if ref[k] else float("nan")) for k in got}
            rows.append(row)
    return rows


def surface_report(prepared: PreparedSurface,
                   mesh_sizes: Sequence[int] = (60, 64, 100, 128),
                   shifts: Optional[Sequence[Tuple[float, float]]] = None,
                   bandwidth: Bandwidth = Bandwidth()) -> Dict[str, object]:
    z, Lx, Ly = prepared.z, prepared.Lx, prepared.Ly
    ny, nx = z.shape
    if shifts is None:
        # aligned, half a source step, and an irrational fraction of the cell
        shifts = ((0.0, 0.0), (0.5 * Lx / nx, 0.5 * Ly / ny), (0.3819660 * Lx, 0.6180340 * Ly))
    fs = FourierSurface(z, Lx, Ly, bandwidth)
    st = surface_stats(z, Lx, Ly)
    lo, hi = fs.extrema(oversample=4)
    var = fs.variances()
    rep = {
        "identity": {"recipe_id": prepared.recipe_id, "surface_id": prepared.surface_id,
                     "array_sha256": prepared.array_sha256, "evaluator": fs.identity()},
        "cell": {"Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny, "dx": Lx / nx, "dy": Ly / ny},
        "targets": prepared.reports.get("targets"),
        "accepted": prepared.reports.get("accepted"),
        "statistics": st,
        "spectrum": {
            "q_fundamental_x": 2 * math.pi / Lx, "q_fundamental_y": 2 * math.pi / Ly,
            "q_nyquist_x": math.pi * nx / Lx, "q_nyquist_y": math.pi * ny / Ly,
            "nyquist_fraction": var["nyquist_fraction"],
            "variance_nodal": var["nodal"], "variance_continuous": var["continuous"],
            "retained_fraction": var["retained_fraction"],
        },
        "reconstruction": {"extrema_oversample4": [lo, hi],
                           "source_extrema": [st["z_min"], st["z_max"]],
                           "overshoot_below_min": st["z_min"] - lo,
                           "overshoot_above_max": hi - st["z_max"]},
        "fidelity": fidelity_table(z, Lx, Ly, mesh_sizes, shifts, bandwidth),
        "generation": prepared.reports.get("generation"),
        "versions": prepared.versions,
    }
    return json_safe(rep)


def pair_report(stationary: Optional[PreparedSurface], translating: Optional[PreparedSurface],
                mx: int, my: Optional[int] = None,
                displacements: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                n_offsets: int = 8) -> Dict[str, object]:
    """Per-surface and combined statistics of a pair.  Default displacements:
    the translating member at ``n_offsets`` x ``n_offsets`` relative offsets
    over the cell (stationary at 0)."""
    s = None if stationary is None else FourierSurface(stationary.z, stationary.Lx, stationary.Ly)
    m = None if translating is None else FourierSurface(translating.z, translating.Lx, translating.Ly)
    pr = SurfacePair(s, m)
    if displacements is None:
        displacements = [((0.0, 0.0), (i * pr.Lx / n_offsets, j * pr.Ly / n_offsets))
                         for j in range(n_offsets) for i in range(n_offsets)]
    out = pr.statistics(mx, my, displacements)
    out["members"] = {
        "stationary": None if stationary is None else {
            "recipe_id": stationary.recipe_id, "array_sha256": stationary.array_sha256},
        "translating": None if translating is None else {
            "recipe_id": translating.recipe_id, "array_sha256": translating.array_sha256},
    }
    return json_safe(out)


def format_report(rep: Dict[str, object]) -> str:
    """Short aligned text rendering of :func:`surface_report` output."""
    st = rep["statistics"]
    c = rep["cell"]
    lines = [f"surface {rep['identity']['surface_id']}  array {rep['identity']['array_sha256'][:16]}",
             f"cell {c['Lx'] * 1e6:.4g} x {c['Ly'] * 1e6:.4g} um, {c['nx']} x {c['ny']} "
             f"(dx {c['dx'] * 1e9:.4g} nm)"]
    t = rep.get("targets") or {}
    if t:
        lines.append(f"targets: {t.get('status')}  mode {t.get('mode')} anchor {t.get('anchor')}"
                     + (f"  -- {t.get('reason')}" if t.get("reason") else ""))
        for ch in t.get("checks", []):
            lines.append(f"  {ch['name']:3s} want {ch['requested']:.6e}  got {ch['realised']:.6e}"
                         f"  rel err {ch['rel_error']:.2e}  {'ok' if ch['passed'] else 'FAIL'}")
    lines.append("master array (gap convention):")
    for k in ("Sa", "Sq", "Sz", "Sp", "Sv", "Ssk", "Sku", "Sm", "n_summits"):
        v = st.get(k)
        lines.append(f"  {k:10s} {v:.6e}" if isinstance(v, float) else f"  {k:10s} {v}")
    sp = rep["spectrum"]
    lines.append(f"Nyquist share {sp['nyquist_fraction']:.3e}; continuous/nodal variance "
                 f"{sp['variance_continuous'] / sp['variance_nodal']:.6f}")
    r = rep["reconstruction"]
    lines.append(f"Fourier overshoot below min {r['overshoot_below_min']:.3e} m, "
                 f"above max {r['overshoot_above_max']:.3e} m")
    lines.append(f"{'n':>5s} {'shift (um)':>20s} | {'Fourier Sa/Sq/Sz/ext':>28s} | "
                 f"{'bilinear Sa/Sq/Sz/ext':>28s}")
    for row in rep["fidelity"]:
        f, b = row["fourier"], row["bilinear_legacy"]
        sh = f"({row['shift'][0] * 1e6:.3f},{row['shift'][1] * 1e6:.3f})"
        lines.append(f"{row['n']:5d} {sh:>20s} | "
                     f"{f['Sa']:.3f} {f['Sq']:.3f} {f['Sz']:.3f} {f['film_reducing_extreme']:.3f}"
                     f"{'':5s}| {b['Sa']:.3f} {b['Sq']:.3f} {b['Sz']:.3f} "
                     f"{b['film_reducing_extreme']:.3f}")
    return "\n".join(lines)
