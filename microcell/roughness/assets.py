"""Strict surface-array inputs and immutable, hashed surface artefacts.

Two file kinds, both NumPy + JSON, nothing pickled.

1. **Surface array file** (an input): ``<name>.npy`` holding ``z[j_y, i_x]``
   and a sidecar ``<name>.json``::

       {"format": "ehl-surface-array/1",
        "units": "m",
        "layout": "z[j_y,i_x]",
        "grid": "half_open_periodic",
        "sign": "gap" | "protrusion",
        "Lx_m": ..., "Ly_m": ..., "nx": ..., "ny": ...,
        "sha256": "<sha256 of the .npy file bytes>",      # optional
        "provenance": {...}}                                # optional

   ``sign="gap"``: positive z thickens the film (the canonical convention,
   ``h = H0 + HT*t + z``).  ``sign="protrusion"``: positive z is material
   protruding into the film; it is converted once by negation.  Every key
   except ``sha256`` and ``provenance`` is required; anything else is refused
   rather than guessed (Case API plan §4, input rules).

   :func:`import_legacy_array` reads the historical arrays (no sidecar) with
   every convention stated explicitly by the caller.

2. **Prepared artefact** (an output of preparation, read by workers):
   ``<root>/<id16>.npz`` (array ``z``, float64, gap convention, zero mean)
   and ``<root>/<id16>.json`` (format ``ehl-surface-artefact/1``) holding
   the recipe, its sha256 (the *recipe id*), the array content hash, the
   ``.npz`` file hash, the cell, reports and library versions.  The json is
   written last and is the completion marker; a missing or inconsistent json
   means "not prepared".  Publication is atomic (temporary file +
   ``os.replace``) under a ``mkdir`` lock, and an existing artefact is never
   overwritten: republishing identical content is a no-op, different content
   for the same recipe raises :class:`ArtefactConflictError`.

Identity: the recipe id says *what was asked for*; the array hash says *which
bytes were produced*.  A seed fixes a recipe, not bytes across library
versions (plan §6.2), so workers and restarts consume the stored bytes and
check their hash.

Only NumPy and the standard library are imported.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

__all__ = [
    "SURFACE_FILE_FORMAT", "ARTEFACT_FORMAT", "SurfaceArray", "PreparedSurface",
    "ArtefactError", "ArtefactMissingError", "ArtefactCorruptError",
    "ArtefactConflictError", "SurfaceFileError",
    "canonical_json", "json_safe", "sha256_json", "array_sha256", "file_sha256",
    "validate_height_array", "write_surface_file", "write_sidecar", "read_surface_file",
    "import_legacy_array", "make_prepared", "publish_artefact",
    "load_artefact", "artefact_paths", "library_versions",
]

SURFACE_FILE_FORMAT = "ehl-surface-array/1"
ARTEFACT_FORMAT = "ehl-surface-artefact/1"
_LAYOUT = "z[j_y,i_x]"
_GRID = "half_open_periodic"
_SIGNS = ("gap", "protrusion")


class SurfaceFileError(ValueError):
    """A surface input file or array violates the strict format."""


class ArtefactError(RuntimeError):
    """Base class for prepared-artefact problems."""


class ArtefactMissingError(ArtefactError):
    """No completed artefact (json completion marker absent)."""


class ArtefactCorruptError(ArtefactError):
    """Artefact present but inconsistent with its recorded hashes/format."""


class ArtefactConflictError(ArtefactError):
    """A different array is already published under the same recipe id."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _canon(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if not math.isfinite(f):
            raise ValueError("canonical_json: non-finite float")
        return f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(f"canonical_json: unsupported type {type(obj).__name__}")


def json_safe(obj: Any) -> Any:
    """Reports may hold non-finite numbers (e.g. Sm = inf with no summits);
    map them to None so the artefact json stays strict JSON."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no spaces, shortest float repr)."""
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def array_sha256(z: np.ndarray) -> str:
    """Content hash of a 2-D float64 array (little-endian bytes + shape)."""
    a = np.ascontiguousarray(np.asarray(z, dtype="<f8"))
    if a.ndim != 2:
        raise ValueError("array_sha256 expects a 2-D array")
    h = hashlib.sha256(f"float64-le:{a.shape[0]}x{a.shape[1]}:".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": sys.platform}


# ---------------------------------------------------------------------------
# Validated source arrays
# ---------------------------------------------------------------------------
def validate_height_array(z: Any, nx: Optional[int] = None,
                          ny: Optional[int] = None) -> np.ndarray:
    """Return ``z`` as float64 after the strict checks (2-D, shape, finite,
    non-constant).  Raises :class:`SurfaceFileError`."""
    a = np.asarray(z)
    if a.dtype.kind not in "fiu":
        raise SurfaceFileError(f"height array must be numeric, got dtype {a.dtype}")
    a = a.astype(float)
    if a.ndim != 2:
        raise SurfaceFileError(f"height array must be 2-D z[j_y,i_x], got shape {a.shape}")
    if nx is not None and ny is not None and a.shape != (int(ny), int(nx)):
        raise SurfaceFileError(f"array shape {a.shape} != declared (ny, nx) = ({ny}, {nx})")
    if min(a.shape) < 2:
        raise SurfaceFileError("height array needs at least 2 samples per axis")
    if not np.all(np.isfinite(a)):
        raise SurfaceFileError("height array contains non-finite values")
    if float(np.ptp(a)) == 0.0:
        raise SurfaceFileError("height array is constant")
    return a


@dataclass(frozen=True, eq=False)
class SurfaceArray:
    """A validated, centred gap-perturbation array with its cell."""
    z: np.ndarray
    Lx: float
    Ly: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        a = validate_height_array(self.z)
        a = np.ascontiguousarray(a)
        a.setflags(write=False)
        object.__setattr__(self, "z", a)
        if not (self.Lx > 0.0 and self.Ly > 0.0):
            raise SurfaceFileError("Lx and Ly must be > 0")

    @property
    def nx(self) -> int:
        return int(self.z.shape[1])

    @property
    def ny(self) -> int:
        return int(self.z.shape[0])

    @property
    def sha256(self) -> str:
        return array_sha256(self.z)


def _sidecar(npy_path: Path) -> Path:
    return npy_path.with_suffix(".json")


def write_surface_file(npy_path: Union[str, Path], z: np.ndarray, Lx: float,
                       Ly: float, sign: str = "gap",
                       provenance: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``z`` and its strict sidecar; returns the sidecar path.  The array
    is written as given (no centring; the reader centres)."""
    if sign not in _SIGNS:
        raise SurfaceFileError(f"sign must be one of {_SIGNS}")
    npy_path = Path(npy_path)
    a = validate_height_array(z)
    np.save(npy_path, a.astype("<f8"))
    meta = {"format": SURFACE_FILE_FORMAT, "units": "m", "layout": _LAYOUT,
            "grid": _GRID, "sign": sign, "Lx_m": float(Lx), "Ly_m": float(Ly),
            "nx": int(a.shape[1]), "ny": int(a.shape[0]),
            "sha256": file_sha256(npy_path), "provenance": provenance or {}}
    side = _sidecar(npy_path)
    side.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return side


def write_sidecar(npy_path: Union[str, Path], Lx: float, Ly: float, sign: str,
                  provenance: Optional[Dict[str, Any]] = None) -> Path:
    """Write the strict sidecar for an EXISTING ``.npy`` (the array file is not
    touched; its sha256 is recorded).  Refuses to overwrite a sidecar."""
    npy_path = Path(npy_path)
    if sign not in _SIGNS:
        raise SurfaceFileError(f"sign must be one of {_SIGNS}")
    a = validate_height_array(np.load(npy_path, allow_pickle=False))
    side = _sidecar(npy_path)
    if side.exists():
        raise SurfaceFileError(f"{side} exists; sidecars are not overwritten")
    meta = {"format": SURFACE_FILE_FORMAT, "units": "m", "layout": _LAYOUT,
            "grid": _GRID, "sign": sign, "Lx_m": float(Lx), "Ly_m": float(Ly),
            "nx": int(a.shape[1]), "ny": int(a.shape[0]),
            "sha256": file_sha256(npy_path), "provenance": provenance or {}}
    side.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return side


def read_surface_file(npy_path: Union[str, Path],
                      meta_path: Optional[Union[str, Path]] = None,
                      expected_sha256: Optional[str] = None) -> SurfaceArray:
    """Read a strict surface-array file (see module docstring) into the
    canonical gap convention, centred once."""
    npy_path = Path(npy_path)
    meta_path = Path(meta_path) if meta_path is not None else _sidecar(npy_path)
    if not meta_path.is_file():
        raise SurfaceFileError(f"{npy_path}: sidecar {meta_path.name} is required "
                               "(units, layout, sign and cell are never guessed)")
    meta = json.loads(meta_path.read_text())
    required = {"format": SURFACE_FILE_FORMAT, "units": "m", "layout": _LAYOUT,
                "grid": _GRID}
    for k, v in required.items():
        if meta.get(k) != v:
            raise SurfaceFileError(f"{meta_path}: '{k}' must be {v!r}, got {meta.get(k)!r}")
    for k in ("sign", "Lx_m", "Ly_m", "nx", "ny"):
        if k not in meta:
            raise SurfaceFileError(f"{meta_path}: missing required key '{k}'")
    if meta["sign"] not in _SIGNS:
        raise SurfaceFileError(f"{meta_path}: sign must be one of {_SIGNS}")
    fh = file_sha256(npy_path)
    for want, label in ((meta.get("sha256"), "sidecar"), (expected_sha256, "caller")):
        if want is not None and want != fh:
            raise SurfaceFileError(f"{npy_path}: sha256 {fh} does not match the {label} "
                                   f"value {want}")
    raw = np.load(npy_path, allow_pickle=False)
    a = validate_height_array(raw, nx=meta["nx"], ny=meta["ny"])
    mean = float(a.mean())
    a = a - mean
    if meta["sign"] == "protrusion":
        a = -a
    prov = {"source": "file", "path": str(npy_path), "file_sha256": fh,
            "sign_in_file": meta["sign"], "mean_removed_m": mean,
            "file_provenance": meta.get("provenance", {})}
    return SurfaceArray(a, float(meta["Lx_m"]), float(meta["Ly_m"]), prov)


def import_legacy_array(npy_path: Union[str, Path], Lx: float, Ly: float, *,
                        layout: str, sign: str, note: str = "",
                        expected_sha256: Optional[str] = None) -> SurfaceArray:
    """Read a historical ``.npy`` (no strict sidecar) with every convention
    stated by the caller: ``layout`` ``"y_x"`` (``z[j_y, i_x]``, the old
    tree's generator and refill) or ``"x_y"`` (``z[i_x, j_y]``, as
    ``generated_ht_cell.sample_periodic`` reads arrays); ``sign`` ``"gap"`` or
    ``"protrusion"``.  The cell lengths are the caller's statement too: the
    study cells were generated at 6.208 um, the WTC26 HDS family at 5 um."""
    npy_path = Path(npy_path)
    if layout not in ("y_x", "x_y"):
        raise SurfaceFileError("layout must be 'y_x' or 'x_y'")
    if sign not in _SIGNS:
        raise SurfaceFileError(f"sign must be one of {_SIGNS}")
    fh = file_sha256(npy_path)
    if expected_sha256 is not None and expected_sha256 != fh:
        raise SurfaceFileError(f"{npy_path}: sha256 {fh} != expected {expected_sha256}")
    a = validate_height_array(np.load(npy_path, allow_pickle=False))
    if layout == "x_y":
        a = np.ascontiguousarray(a.T)
    mean = float(a.mean())
    a = a - mean
    if sign == "protrusion":
        a = -a
    prov = {"source": "legacy_array", "path": str(npy_path), "file_sha256": fh,
            "layout_in_file": layout, "sign_in_file": sign,
            "mean_removed_m": mean, "note": note}
    return SurfaceArray(a, float(Lx), float(Ly), prov)


# ---------------------------------------------------------------------------
# Prepared artefacts
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PreparedSurface:
    """An immutable prepared surface: recipe, identity, bytes and reports."""
    recipe: Dict[str, Any]
    recipe_id: str
    z: np.ndarray
    Lx: float
    Ly: float
    array_sha256: str
    reports: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        a = np.ascontiguousarray(np.asarray(self.z, dtype=float))
        a.setflags(write=False)
        object.__setattr__(self, "z", a)

    @property
    def surface_id(self) -> str:
        """Short id used for file names (first 16 hex of the recipe id)."""
        return self.recipe_id[:16]


def make_prepared(recipe: Dict[str, Any], z: np.ndarray, Lx: float, Ly: float,
                  reports: Optional[Dict[str, Any]] = None) -> PreparedSurface:
    """Wrap a resolved array: centre check, recipe id, content hash."""
    a = validate_height_array(z)
    if abs(float(a.mean())) > 1e-9 * float(np.sqrt(np.mean(a ** 2))):
        raise SurfaceFileError("prepared surfaces must be zero-mean (centre before wrapping)")
    rid = sha256_json({"format": ARTEFACT_FORMAT, "recipe": recipe})
    return PreparedSurface(recipe=json.loads(canonical_json(recipe)), recipe_id=rid,
                           z=a, Lx=float(Lx), Ly=float(Ly),
                           array_sha256=array_sha256(a),
                           reports=json.loads(canonical_json(json_safe(reports or {}))),
                           versions=library_versions())


def artefact_paths(root: Union[str, Path], surface_id: str):
    root = Path(root)
    return root / f"{surface_id}.npz", root / f"{surface_id}.json", root / f"{surface_id}.lock"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _npz_bytes(z: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.savez(buf, z=np.ascontiguousarray(z, dtype="<f8"))
    return buf.getvalue()


def publish_artefact(prepared: PreparedSurface, root: Union[str, Path],
                     lock_timeout: float = 60.0) -> Path:
    """Publish ``prepared`` under ``root``; returns the json path.

    Idempotent for identical content; raises :class:`ArtefactConflictError`
    if a different array is already published for the same recipe id.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    npz, js, lock = artefact_paths(root, prepared.surface_id)
    t0 = time.monotonic()
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if time.monotonic() - t0 > lock_timeout:
                raise ArtefactError(f"timed out waiting for lock {lock}")
            time.sleep(0.05)
    try:
        if js.exists():
            existing = load_artefact(js)
            if existing.recipe_id != prepared.recipe_id:
                raise ArtefactConflictError(f"{js}: short-id collision with a different recipe")
            if existing.array_sha256 != prepared.array_sha256:
                raise ArtefactConflictError(
                    f"{js}: recipe {prepared.recipe_id[:16]} already published with array "
                    f"{existing.array_sha256[:16]}, refusing to replace it with "
                    f"{prepared.array_sha256[:16]} (immutable artefact)")
            return js
        data = _npz_bytes(prepared.z)
        _atomic_write_bytes(npz, data)
        meta = {
            "format": ARTEFACT_FORMAT,
            "recipe_id": prepared.recipe_id,
            "surface_id": prepared.surface_id,
            "recipe": prepared.recipe,
            "array_sha256": prepared.array_sha256,
            "npz_file": npz.name,
            "npz_sha256": hashlib.sha256(data).hexdigest(),
            "shape": [int(prepared.z.shape[0]), int(prepared.z.shape[1])],
            "Lx_m": prepared.Lx, "Ly_m": prepared.Ly,
            "units": "m", "layout": _LAYOUT, "grid": _GRID, "sign": "gap",
            "reports": prepared.reports,
            "versions": prepared.versions,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_write_bytes(js, json.dumps(meta, indent=1, sort_keys=True).encode("utf-8"))
        return js
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def load_artefact(json_path: Union[str, Path], verify: bool = True) -> PreparedSurface:
    """Load a published artefact; with ``verify`` (default) check the format,
    the recipe id, the ``.npz`` file hash and the array content hash."""
    js = Path(json_path)
    if not js.is_file():
        raise ArtefactMissingError(f"{js}: no completed artefact")
    try:
        meta = json.loads(js.read_text())
    except (OSError, ValueError) as exc:
        raise ArtefactCorruptError(f"{js}: unreadable json ({exc})") from exc
    if meta.get("format") != ARTEFACT_FORMAT:
        raise ArtefactCorruptError(f"{js}: format {meta.get('format')!r} != {ARTEFACT_FORMAT!r}")
    npz = js.with_name(meta.get("npz_file", ""))
    if not npz.is_file():
        raise ArtefactCorruptError(f"{js}: array file {npz.name} is missing")
    if verify:
        rid = sha256_json({"format": ARTEFACT_FORMAT, "recipe": meta["recipe"]})
        if rid != meta.get("recipe_id"):
            raise ArtefactCorruptError(f"{js}: recipe id does not match its recipe")
        if file_sha256(npz) != meta.get("npz_sha256"):
            raise ArtefactCorruptError(f"{npz}: file hash does not match {js.name}")
    with np.load(npz, allow_pickle=False) as d:
        z = np.array(d["z"], dtype=float)
    if verify and array_sha256(z) != meta.get("array_sha256"):
        raise ArtefactCorruptError(f"{npz}: array content hash does not match {js.name}")
    return PreparedSurface(recipe=meta["recipe"], recipe_id=meta["recipe_id"], z=z,
                           Lx=float(meta["Lx_m"]), Ly=float(meta["Ly_m"]),
                           array_sha256=meta["array_sha256"],
                           reports=meta.get("reports", {}),
                           versions=meta.get("versions", {}))
