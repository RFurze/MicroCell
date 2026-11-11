"""Frozen per-surface and surface-pair descriptors (recipes, not arrays).

These are the pure-data inputs of preparation.  Stage 2 (R3) wires them into
``ehl.spec.Case.roughness`` as additive variants beside the idealised
``Roughness``; nothing here imports ``ehl.spec``.

Per-surface recipes:

* :class:`GeneratedSurface` -- PSD + HPD + seed (+ targets/fit);
* :class:`FileSurface`      -- an already realised strict surface file;
* :class:`ModeSurface`      -- an explicit cosine mode list (idealised
  profiles, the study's band-limited surfaces).

Arrangement: :class:`SurfacePairSpec` -- a stationary member and/or a
translating member, each with its own recipe, targets and signed initial
offset [m].  "Both surfaces rough" has this one representation;
:data:`ehl.roughness.pair.LEGACY_ROUGHNESS_MODES` maps the old
``roughness_mode`` flags onto it.

Every descriptor has ``to_dict()``: the canonical, JSON-safe recipe whose
sha256 is the recipe id (:func:`ehl.roughness.assets.sha256_json`).  The
``name`` fields are labels and are left out of it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, Union

from .generation import GENERATOR_REVISION, HPD_FAMILIES

__all__ = ["PSDSpec", "HPDSpec", "SurfaceTargets", "HeightFit", "GeneratedSurface",
           "FileSurface", "ModeSurface", "SurfaceSpec", "SurfacePairSpec"]


def _pos(name: str, v: Optional[float]) -> None:
    if v is not None and not (isinstance(v, (int, float)) and v > 0.0):
        raise ValueError(f"{name} must be > 0, got {v!r}")


def _as_dict(obj) -> Dict[str, Any]:
    """Recipe dictionary.  ``name`` is a label, not part of the recipe: two
    specs that differ only in name describe the same surface."""
    out: Dict[str, Any] = {"type": type(obj).__name__}
    for f in fields(obj):
        if f.name == "name":
            continue
        v = getattr(obj, f.name)
        out[f.name] = v.to_dict() if hasattr(v, "to_dict") else v
    return out


@dataclass(frozen=True)
class PSDSpec:
    """Power-spectrum recipe (shape only; amplitude comes from the targets).

    ``kind``: ``gaussian_acf`` (``corr_len``), ``exponential_acf``
    (``corr_len``, optional ``q1``), ``self_affine`` (``H``, optional ``q0``,
    ``q1`` [rad/m]) or ``radial_table`` (``table`` path to an ``.npz`` with
    ``f`` [1/m] and ``P``; ``table_sha256``; ``gamma_tilt``;
    ``out_of_band`` ``"zero"`` | ``"legacy_clip"``).
    """
    kind: str
    corr_len: Optional[float] = None
    H: Optional[float] = None
    q0: Optional[float] = None
    q1: Optional[float] = None
    table: Optional[str] = None
    table_sha256: Optional[str] = None
    gamma_tilt: float = 0.0
    out_of_band: str = "zero"

    def __post_init__(self):
        if self.kind in ("gaussian_acf", "exponential_acf"):
            if self.corr_len is None:
                raise ValueError(f"{self.kind} needs corr_len")
            _pos("corr_len", self.corr_len)
        elif self.kind == "self_affine":
            if self.H is None or not (0.0 < self.H < 1.0):
                raise ValueError("self_affine needs 0 < H < 1")
        elif self.kind == "radial_table":
            if not self.table:
                raise ValueError("radial_table needs a table path")
            if self.out_of_band not in ("zero", "legacy_clip"):
                raise ValueError("out_of_band must be 'zero' or 'legacy_clip'")
        else:
            raise ValueError(f"unknown PSD kind {self.kind!r}")
        _pos("q0", self.q0)
        _pos("q1", self.q1)

    @classmethod
    def gaussian_acf(cls, corr_len: float) -> "PSDSpec":
        return cls("gaussian_acf", corr_len=float(corr_len))

    @classmethod
    def exponential_acf(cls, corr_len: float, q1: Optional[float] = None) -> "PSDSpec":
        return cls("exponential_acf", corr_len=float(corr_len), q1=q1)

    @classmethod
    def self_affine(cls, H: float, q0: Optional[float] = None,
                    q1: Optional[float] = None) -> "PSDSpec":
        return cls("self_affine", H=float(H), q0=q0, q1=q1)

    @classmethod
    def radial_table(cls, table: str, table_sha256: Optional[str] = None,
                     gamma_tilt: float = 0.0, out_of_band: str = "zero") -> "PSDSpec":
        return cls("radial_table", table=str(table), table_sha256=table_sha256,
                   gamma_tilt=float(gamma_tilt), out_of_band=out_of_band)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class HPDSpec:
    """Height-distribution recipe.

    ``family``: one of ``gaussian``, ``gennorm`` (shape beta), ``skewnorm``
    (a), ``weibull`` (k; the old convention: k > 0 has its long tail
    *down*), ``johnsonsu`` ((a, b)), or ``samples`` (``samples`` path to a
    1-D ``.npy`` of measured heights, ``samples_sha256``).

    ``orientation``: ``"gap"`` -- the distribution describes the gap
    perturbation z directly; ``"protrusion"`` -- it describes height
    protruding into the film and is negated into the gap convention.  (The
    WTC26 "worn" family is ``weibull`` k = 2 with ``"protrusion"``: the old
    script negated the old family's sample.)
    """
    family: str = "gaussian"
    shape: Union[None, float, Tuple[float, float]] = None
    samples: Optional[str] = None
    samples_sha256: Optional[str] = None
    orientation: str = "gap"

    def __post_init__(self):
        if self.family not in HPD_FAMILIES + ("samples",):
            raise ValueError(f"unknown HPD family {self.family!r}")
        if self.family == "samples" and not self.samples:
            raise ValueError("HPD 'samples' needs a samples path")
        if self.family in ("gennorm", "skewnorm", "weibull") and self.shape is None:
            raise ValueError(f"HPD {self.family!r} needs a shape parameter")
        if self.family == "johnsonsu" and (self.shape is None or len(self.shape) != 2):
            raise ValueError("HPD 'johnsonsu' needs shape=(a, b)")
        if self.orientation not in ("gap", "protrusion"):
            raise ValueError("orientation must be 'gap' or 'protrusion'")
        if isinstance(self.shape, list):
            object.__setattr__(self, "shape", tuple(self.shape))

    def to_dict(self) -> Dict[str, Any]:
        d = _as_dict(self)
        if isinstance(self.shape, tuple):
            d["shape"] = list(self.shape)
        return d


@dataclass(frozen=True)
class SurfaceTargets:
    """Amplitude targets on the finite master array [m].

    ``Sa``, ``Sq``: fitted/scaled targets (M1).  ``Sz``: checked only -- a
    requested Sz is compared with the result and can fail acceptance, but no
    fitting moves it (joint Sa/Sq/Sz fitting is milestone M3).
    Acceptance per target: ``|got - want| <= max(atol, rtol*want)``.
    """
    Sa: Optional[float] = None
    Sq: Optional[float] = None
    Sz: Optional[float] = None
    rtol: float = 1e-4
    atol: float = 1e-13

    def __post_init__(self):
        for k in ("Sa", "Sq", "Sz"):
            v = getattr(self, k)
            if v is not None and not (v > 0.0):
                raise ValueError(f"target {k} must be > 0, got {v!r}")
        if not (self.rtol >= 0.0 and self.atol >= 0.0):
            raise ValueError("tolerances must be >= 0")

    def requested(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in ("Sa", "Sq", "Sz") if getattr(self, k) is not None}

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class HeightFit:
    """How the height amplitudes are brought to the targets.

    ``mode``:
      * ``"fixed"`` -- change nothing; check the targets;
      * ``"scale"`` -- one positive multiplier chosen so ``anchor``
        (``"Sa"`` or ``"Sq"``) is met exactly; check the rest;
      * ``"ratio"`` -- generated surfaces with ``HPDSpec('gennorm', ...)``:
        fit beta so the final-grid quantile vector has the requested
        Sa/Sq, then scale to Sq (the shape in the HPD spec is the starting
        value only).
    """
    mode: str = "scale"
    anchor: Optional[str] = "Sq"

    def __post_init__(self):
        if self.mode not in ("fixed", "scale", "ratio"):
            raise ValueError(f"unknown fit mode {self.mode!r}")
        if self.mode == "scale" and self.anchor not in ("Sa", "Sq"):
            raise ValueError("scale mode needs anchor 'Sa' or 'Sq'")
        if self.mode == "fixed" and self.anchor is not None:
            object.__setattr__(self, "anchor", None)
        if self.mode == "ratio":
            object.__setattr__(self, "anchor", "Sq")

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class GeneratedSurface:
    """PSD + HPD recipe realised by IAAFT (or the filtered Gaussian generator)."""
    name: str
    Lx: float
    Ly: float
    nx: int
    ny: int
    psd: PSDSpec
    hpd: HPDSpec = HPDSpec()
    seed: int = 1000
    n_iter: int = 150
    take: str = "Zh"
    generator: str = "iaaft"
    targets: SurfaceTargets = SurfaceTargets()
    fit: HeightFit = HeightFit()
    revision: str = GENERATOR_REVISION

    def __post_init__(self):
        _pos("Lx", self.Lx)
        _pos("Ly", self.Ly)
        for k in ("nx", "ny"):
            v = getattr(self, k)
            if int(v) != v or v < 4:
                raise ValueError(f"{k} must be an integer >= 4")
        if int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        if self.take not in ("Zh", "Zs"):
            raise ValueError("take must be 'Zh' or 'Zs'")
        if self.generator not in ("iaaft", "gaussian_filter"):
            raise ValueError("generator must be 'iaaft' or 'gaussian_filter'")
        if self.generator == "gaussian_filter":
            if self.psd.kind != "gaussian_acf" or self.hpd.family != "gaussian":
                raise ValueError("gaussian_filter needs PSD gaussian_acf and HPD gaussian")
            if self.nx != self.ny or self.Lx != self.Ly:
                raise ValueError("gaussian_filter supports square cells only")
        if self.fit.mode == "ratio" and self.hpd.family != "gennorm":
            raise ValueError("fit mode 'ratio' needs HPD family 'gennorm'")
        if self.n_iter < 0:
            raise ValueError("n_iter must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class FileSurface:
    """A realised strict surface file (``.npy`` + sidecar, see assets.py)."""
    name: str
    path: str
    sha256: Optional[str] = None
    targets: SurfaceTargets = SurfaceTargets()
    fit: HeightFit = HeightFit(mode="fixed", anchor=None)

    def __post_init__(self):
        if self.fit.mode == "ratio":
            raise ValueError("fit mode 'ratio' applies to generated surfaces only")

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class ModeSurface:
    """Explicit cosine modes ``(m, n, a [m], phi [rad])``; exact evaluation."""
    name: str
    Lx: float
    Ly: float
    modes: Tuple[Tuple[int, int, float, float], ...]

    def __post_init__(self):
        _pos("Lx", self.Lx)
        _pos("Ly", self.Ly)
        object.__setattr__(self, "modes", tuple(
            (int(m), int(n), float(a), float(p)) for m, n, a, p in self.modes))

    def to_dict(self) -> Dict[str, Any]:
        d = _as_dict(self)
        d["modes"] = [list(m) for m in self.modes]
        return d


SurfaceSpec = Union[GeneratedSurface, FileSurface, ModeSurface]


@dataclass(frozen=True)
class SurfacePairSpec:
    """The roughness arrangement of one Case.

    ``stationary`` / ``translating``: per-surface recipes (either may be
    None, not both).  Offsets are signed initial displacements [m] of each
    surface, ``(sx, sy)``.  Two members with the same recipe are refused
    unless ``reuse_same_field`` says the reuse is intended (plan §13: never
    create a second identical surface implicitly).  Per-surface targets live
    in each recipe; combined-gap statistics are reported, never imposed.
    """
    name: str
    stationary: Optional[SurfaceSpec] = None
    translating: Optional[SurfaceSpec] = None
    offset_stationary: Tuple[float, float] = (0.0, 0.0)
    offset_translating: Tuple[float, float] = (0.0, 0.0)
    reuse_same_field: bool = False

    def __post_init__(self):
        if self.stationary is None and self.translating is None:
            raise ValueError("a surface arrangement needs at least one rough surface")
        if (self.stationary is not None and self.translating is not None
                and self.stationary.to_dict() == self.translating.to_dict()
                and not self.reuse_same_field):
            raise ValueError("both members have the same recipe; set reuse_same_field=True "
                             "if the same field is meant to be on both bodies")
        for k in ("offset_stationary", "offset_translating"):
            v = tuple(float(c) for c in getattr(self, k))
            if len(v) != 2:
                raise ValueError(f"{k} must be (sx, sy)")
            object.__setattr__(self, k, v)

    @property
    def arrangement(self) -> str:
        if self.translating is None:
            return "stationary_only"
        if self.stationary is None:
            return "translating_only"
        return "stationary_and_translating"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SurfacePairSpec",
            "arrangement": self.arrangement,
            "stationary": None if self.stationary is None else self.stationary.to_dict(),
            "translating": None if self.translating is None else self.translating.to_dict(),
            "offset_stationary": list(self.offset_stationary),
            "offset_translating": list(self.offset_translating),
            "reuse_same_field": self.reuse_same_field,
        }
