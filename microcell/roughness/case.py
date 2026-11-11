"""A Case's generated roughness, prepared: the bridge between the Case API and
the artefact store (R3).

``ehl.spec.SurfaceRoughness`` is a recipe.  Before a run, each member surface
is resolved **once** into an immutable artefact (:mod:`ehl.roughness.assets`)
in the surface store, and the arrangement is summarised as a *descriptor*: a
canonical JSON string with, per member, the artefact file, its recipe id and
array hash, and its signed offset; the cell; the reconstruction (evaluator
revision and bandwidth); and the roughness envelope.  The descriptor is what
the translated config module and ``MicroCase.surface`` carry to the workers,
which load the bytes it names and check their hash (R4).

Store: ``$EHL_SURFACE_STORE`` if set, else ``<tree>/data/surfaces``.  Paths
inside the tree are recorded relative to it, so a descriptor stays valid in
another clone of the same tree.

The roughness *envelope* is the film-reducing extreme of the reconstruction,
``max(0, -min)`` of each member's Fourier surface on an oversampled grid,
summed over the members (every relative alignment is then covered).  It is an
estimate of the continuous extreme, not a proof of a bound (R0 §5.4).

Only NumPy and the standard library are imported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import assets as A
from .evaluator import EVALUATOR_REVISION, Bandwidth, FourierSurface
from .prepare import prepare, recipe_for

__all__ = ["TREE_ROOT", "SURFACE_STORE_ENV", "DESCRIPTOR_FORMAT", "ENVELOPE_OVERSAMPLE",
           "SurfaceNotPreparedError", "store_dir", "member_specs", "ResolvedArrangement",
           "prepare_arrangement", "load_arrangement", "descriptor_json", "identity_of",
           "identity_mismatch"]

TREE_ROOT = Path(__file__).resolve().parents[2]
SURFACE_STORE_ENV = "EHL_SURFACE_STORE"
DESCRIPTOR_FORMAT = "ehl-surface-arrangement/1"
ENVELOPE_OVERSAMPLE = 8


class SurfaceNotPreparedError(RuntimeError):
    """A member surface has no published artefact in the store."""


def store_dir() -> Path:
    s = os.environ.get(SURFACE_STORE_ENV)
    return Path(s) if s else TREE_ROOT / "data" / "surfaces"


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(TREE_ROOT).as_posix()
    except ValueError:
        return str(Path(p).resolve())


def _abs(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else TREE_ROOT / p


def member_specs(sr) -> Tuple[Tuple[str, Any, Tuple[float, float]], ...]:
    """``((role, recipe, offset), ...)`` for the members present."""
    pair = sr.pair
    out = []
    if pair.stationary is not None:
        out.append(("stationary", pair.stationary, tuple(pair.offset_stationary)))
    if pair.translating is not None:
        out.append(("translating", pair.translating, tuple(pair.offset_translating)))
    return tuple(out)


def _bandwidth(sr) -> Bandwidth:
    return sr.bandwidth if sr.bandwidth is not None else Bandwidth()


@dataclass(frozen=True, eq=False)
class ResolvedArrangement:
    """A SurfaceRoughness with every member prepared."""
    arrangement: str
    members: Dict[str, Dict[str, Any]]      # role -> {prepared, json, offset, envelope_m}
    cell: Tuple[float, float]
    bandwidth: Bandwidth
    envelope_m: float

    def descriptor(self) -> Dict[str, Any]:
        mem = {}
        for role in ("stationary", "translating"):
            m = self.members.get(role)
            mem[role] = None if m is None else {
                "artefact": _rel(m["json"]),
                "recipe_id": m["prepared"].recipe_id,
                "array_sha256": m["prepared"].array_sha256,
                "shape": [int(m["prepared"].z.shape[0]), int(m["prepared"].z.shape[1])],
                "offset_m": list(m["offset"]),
                "envelope_m": m["envelope_m"],
            }
        return {"format": DESCRIPTOR_FORMAT, "arrangement": self.arrangement,
                "cell_m": list(self.cell), "members": mem,
                "reconstruction": {"evaluator": EVALUATOR_REVISION,
                                   "bandwidth": self.bandwidth.to_dict()},
                "envelope_m": self.envelope_m, "envelope_oversample": ENVELOPE_OVERSAMPLE,
                "sign": "gap", "layout": "z[j_y,i_x]"}

    def descriptor_json(self) -> str:
        return A.canonical_json(self.descriptor())


def _envelope(p: A.PreparedSurface, bw: Bandwidth) -> float:
    lo, _hi = FourierSurface(p.z, p.Lx, p.Ly, bw).extrema(oversample=ENVELOPE_OVERSAMPLE)
    return max(0.0, -float(lo))


def _check_cell(sr, p: A.PreparedSurface, role: str) -> None:
    for got, want, ax in ((p.Lx, sr.xmax, "x"), (p.Ly, sr.ymax, "y")):
        if abs(got - want) > 1e-9 * want:
            raise ValueError(f"{sr.name}: the {role} surface's cell L{ax} = {got:.6g} m differs from the "
                             f"case's {ax}max = {want:.6g} m (no implicit stretching; R0 §4.4)")


def _assemble(sr, prepared: Dict[str, Tuple[A.PreparedSurface, Path]]) -> ResolvedArrangement:
    bw = _bandwidth(sr)
    members: Dict[str, Dict[str, Any]] = {}
    for role, _spec, offset in member_specs(sr):
        p, js = prepared[role]
        _check_cell(sr, p, role)
        members[role] = {"prepared": p, "json": js, "offset": offset, "envelope_m": _envelope(p, bw)}
    env = sum(m["envelope_m"] for m in members.values())
    return ResolvedArrangement(arrangement=sr.pair.arrangement, members=members,
                               cell=(float(sr.xmax), float(sr.ymax)), bandwidth=bw, envelope_m=env)


def prepare_arrangement(sr, store: Optional[Path] = None,
                        base_dir: Optional[Path] = None) -> ResolvedArrangement:
    """Prepare and publish every member (idempotent: an existing identical
    artefact is reused, a different one for the same recipe is refused)."""
    store = Path(store) if store is not None else store_dir()
    base = Path(base_dir) if base_dir is not None else TREE_ROOT
    done: Dict[str, Tuple[A.PreparedSurface, Path]] = {}
    for role, spec, _off in member_specs(sr):
        p = prepare(spec, root=store, base_dir=base)
        _npz, js, _lock = A.artefact_paths(store, p.surface_id)
        done[role] = (A.load_artefact(js), js)
    return _assemble(sr, done)


def load_arrangement(sr, store: Optional[Path] = None,
                     base_dir: Optional[Path] = None) -> ResolvedArrangement:
    """The arrangement from already published artefacts; raises
    :class:`SurfaceNotPreparedError` naming the member that is missing."""
    store = Path(store) if store is not None else store_dir()
    base = Path(base_dir) if base_dir is not None else TREE_ROOT
    done: Dict[str, Tuple[A.PreparedSurface, Path]] = {}
    for role, spec, _off in member_specs(sr):
        rid = A.sha256_json({"format": A.ARTEFACT_FORMAT, "recipe": recipe_for(spec, base)})
        _npz, js, _lock = A.artefact_paths(store, rid[:16])
        try:
            p = A.load_artefact(js)
        except A.ArtefactMissingError:
            raise SurfaceNotPreparedError(
                f"{sr.name}: the {role} surface is not prepared (no {js}); run "
                "'python -m ehl surface prepare <case.py>' first") from None
        if p.recipe_id != rid:
            raise A.ArtefactCorruptError(f"{js}: recipe id {p.recipe_id[:16]} != {rid[:16]}")
        done[role] = (p, js)
    return _assemble(sr, done)


def descriptor_json(case) -> Optional[str]:
    """``MicroCase.surface`` / ``ideal_film_thickness.surface`` for a Case:
    the canonical descriptor of its prepared surfaces, or None for idealised
    roughness."""
    sr = case.roughness
    if getattr(sr, "kind", None) != "surface":
        return None
    return load_arrangement(sr).descriptor_json()


def identity_of(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a descriptor a restart must not change: arrangement,
    cell, reconstruction, and per member the recipe id, array hash and offset."""
    mem = {}
    for role, m in descriptor.get("members", {}).items():
        mem[role] = None if m is None else {k: m[k] for k in ("recipe_id", "array_sha256", "offset_m")}
    return {"arrangement": descriptor.get("arrangement"), "cell_m": descriptor.get("cell_m"),
            "reconstruction": descriptor.get("reconstruction"), "members": mem}


def identity_mismatch(stored: Dict[str, Any], current: Dict[str, Any]) -> Optional[str]:
    """None when a restart may continue; otherwise why not."""
    a, b = identity_of(stored), identity_of(current)
    if a == b:
        return None
    diffs = [k for k in ("arrangement", "cell_m", "reconstruction") if a[k] != b[k]]
    for role in ("stationary", "translating"):
        ma, mb = a["members"].get(role), b["members"].get(role)
        if ma != mb:
            if ma is None or mb is None:
                diffs.append(f"{role} member present/absent")
            else:
                diffs += [f"{role}.{k}" for k in ma if ma[k] != mb[k]]
    return ("the case's surfaces differ from the ones this run started with (" + ", ".join(diffs)
            + "); a changed surface is a new case -- give it a new name")
