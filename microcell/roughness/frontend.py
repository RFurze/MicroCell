"""``Roughness`` -- the front end of the cell's roughness (new in MicroCell, WP1).

Four constructors (plan D2):

    Roughness.generated(Sa, corr_length, seed)      one generated Gaussian surface (IAAFT)
    Roughness.array(z, Lx, Ly)                      a height array supplied by the caller
    Roughness.pair(stationary, translating)         two of the above: one stationary, one translating
    Roughness.cosine(Ah, kx, ky)                    the analytic cosine cell

Sign convention (the parent's): arrays are GAP perturbations ``z[j_y, i_x]`` [m],
``h = H0 + HT*t + z``; ``z < 0`` is an asperity.

The parent loads a generated surface from a prepared artefact on disk through a case
descriptor (its ``roughness/case.py`` and ``roughness/runtime.py``).  Here the same
``SurfaceField`` object is built in memory from the arrays; the solver reads the field's
attributes only, never a file.  The parent's ``roughness/case.py``, its artefact store and
the ``EHL_SURFACE_STORE`` read are not in this repo (removed in WP3).
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np

from microcell.roughness.evaluator import Bandwidth, FourierSurface
from microcell.roughness.runtime import SurfaceField

#: the oversampling of the envelope estimate (the parent's roughness/case.py ENVELOPE_OVERSAMPLE)
ENVELOPE_OVERSAMPLE = 8
ROLES = ("stationary", "translating")
#: acceptance of a generated surface's Sa: ``|got - want| <= max(atol, rtol * want)`` (the parent's
#: ``SurfaceTargets`` defaults)
SA_RTOL = 1e-4
SA_ATOL = 1e-13


def _mean_abs(z: np.ndarray) -> float:
    """Sa of ``z`` about its own mean (the parent's ``targets.realised_amplitudes``)."""
    zc = np.asarray(z, dtype=float) - float(np.mean(z))
    return float(np.mean(np.abs(zc)))


class ArrayField(SurfaceField):
    """A ``SurfaceField`` built from arrays in memory.

    ``members``: role -> ``(z, (off_x, off_y))`` with role in ``("stationary", "translating")``.
    Sets exactly the attributes the parent's ``SurfaceField.__init__`` set, with the same arithmetic
    (the envelope as the parent's ``roughness/case.py`` computes it), without the descriptor or the store."""

    def __init__(self, members: Dict[str, Tuple[np.ndarray, Tuple[float, float]]], Lx: float, Ly: float,
                 bandwidth: Optional[Bandwidth] = None):
        if not members or any(r not in ROLES for r in members):
            raise ValueError(f"members must be keyed by {ROLES}, got {sorted(members)}")
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.bandwidth = bandwidth if bandwidth is not None else Bandwidth()
        self.members = {}
        self.member_ids = {}
        envelope = 0.0
        source_n = 0
        ident = {}
        for role in ROLES:                      # the parent's member order
            if role not in members:
                continue
            z, off = members[role]
            z = np.array(z, dtype=float)
            if z.ndim != 2 or not np.all(np.isfinite(z)):
                raise ValueError(f"the {role} surface must be a finite 2-D array z[j_y, i_x]")
            fs = FourierSurface(z, self.Lx, self.Ly, self.bandwidth)
            lo, _hi = fs.extrema(oversample=ENVELOPE_OVERSAMPLE)
            envelope += max(0.0, -float(lo))
            self.members[role] = (fs, (float(off[0]), float(off[1])))
            sha = hashlib.sha256(np.ascontiguousarray(z, dtype="<f8").tobytes()).hexdigest()
            self.member_ids[role] = sha
            ident[role] = {"array_sha256": sha, "shape": list(z.shape), "offset_m": [float(off[0]), float(off[1])]}
            source_n = max(source_n, *z.shape)
        self.envelope_m = float(envelope)
        self.source_n = int(source_n)
        self.translating = "translating" in self.members
        self.arrangement = ("stationary_and_translating" if len(self.members) == 2
                            else "translating_only" if self.translating else "stationary_only")
        self.descriptor_json = json.dumps({"format": "microcell-array-field/1", "arrangement": self.arrangement,
                                           "cell_m": [self.Lx, self.Ly], "members": ident,
                                           "bandwidth": self.bandwidth.to_dict()}, sort_keys=True)
        self._static_cache = {}
        self._moving_cache = {}


class Roughness:
    """The roughness of one cell.  Build it with one of the constructors below."""

    def __init__(self, kind: str, **data):
        self.kind = kind                    # "surface" | "cosine"
        self._data = data
        self._field: Optional[ArrayField] = None

    # -- constructors ---------------------------------------------------------------------
    @classmethod
    def array(cls, z, Lx: float, Ly: float, moving: bool = False, offset=(0.0, 0.0),
              bandwidth: Optional[Bandwidth] = None) -> "Roughness":
        """A height array ``z[j_y, i_x]`` [m] (gap convention) on the periodic cell ``Lx x Ly`` [m].
        ``moving=False``: the surface is stationary in the cell frame (the parent's "cup");
        ``moving=True``: it translates at the full surface speed ``2 U`` (the parent's "ball").
        ``offset``: the surface's initial displacement [m]."""
        role = "translating" if moving else "stationary"
        return cls("surface", members={role: (np.array(z, dtype=float), (float(offset[0]), float(offset[1])))},
                   Lx=float(Lx), Ly=float(Ly), bandwidth=bandwidth)

    @classmethod
    def generated(cls, Sa: float, corr_length: Optional[float] = None, seed: int = 1000, Lx: float = 5e-6,
                  Ly: float = 5e-6, n: int = 64, moving: bool = False, n_iter: int = 150) -> "Roughness":
        """One generated surface: Gaussian autocorrelation of length ``corr_length`` [m] (default: the
        parent's ``HDS_CORR_LEN``), Gaussian height distribution, scaled to ``Sa`` [m]; the parent's
        recipe ``ehl/library/roughness.py:gaussian_generated``."""
        from microcell.roughness import generation as G

        # [MicroCell WP3] The parent's path spec -> prepare.resolve -> targets.apply_policy, written
        # out for this one recipe (IAAFT, take "Zh", fit "scale" on Sa) with the same arithmetic in the
        # same order, so the array is the one that path returned, bit for bit.
        Sa, Lx, Ly, n, seed, n_iter = float(Sa), float(Lx), float(Ly), int(n), int(seed), int(n_iter)
        corr_len = G.HDS_CORR_LEN if corr_length is None else float(corr_length)
        if not corr_len > 0.0:
            raise ValueError(f"corr_len must be > 0, got {corr_len!r}")
        if not Sa > 0.0:
            raise ValueError(f"target Sa must be > 0, got {Sa!r}")
        for name, v in (("Lx", Lx), ("Ly", Ly)):
            if not v > 0.0:
                raise ValueError(f"{name} must be > 0, got {v!r}")
        if n < 4:
            raise ValueError("nx must be an integer >= 4")
        if n_iter < 0:
            raise ValueError("n_iter must be >= 0")
        amp = G.psd_gaussian_acf(G.wavenumber_grid(n, n, Ly, Lx), corr_len=corr_len, q1=None)
        vec = G.hpd_sorted_sample(n * n, "gaussian", None)
        res = G.iaaft(amp, vec, np.random.default_rng(seed), n_iter=n_iter)
        zc = np.asarray(res.Zh, dtype=float) - float(np.mean(res.Zh))       # centred once
        now = _mean_abs(zc)
        if not now > 0.0:
            raise ValueError("cannot scale a zero field")
        zc = zc * (Sa / now)
        got = _mean_abs(zc)
        if not abs(got - Sa) <= max(SA_ATOL, SA_RTOL * Sa):
            raise ValueError(f"the generated surface did not meet its target (Sa {got!r} for {Sa!r})")
        return cls.array(zc, Lx, Ly, moving=moving)

    @classmethod
    def pair(cls, stationary: "Roughness", translating: "Roughness") -> "Roughness":
        """A stationary and a translating surface (the parent's ``two_surface_split`` arrangement).
        Both must be one-surface ``array`` / ``generated`` roughness on the same cell; the second
        is used as the translating member whatever its own ``moving`` flag."""
        for r in (stationary, translating):
            if r.kind != "surface" or len(r._data["members"]) != 1:
                raise ValueError("pair() takes two one-surface array / generated roughnesses")
        a, b = stationary._data, translating._data
        if (a["Lx"], a["Ly"]) != (b["Lx"], b["Ly"]):
            raise ValueError("the two surfaces of a pair must share one cell")
        if a["bandwidth"] != b["bandwidth"]:
            raise ValueError("the two surfaces of a pair must share one bandwidth")
        return cls("surface", members={"stationary": next(iter(a["members"].values())),
                                       "translating": next(iter(b["members"].values()))},
                   Lx=a["Lx"], Ly=a["Ly"], bandwidth=a["bandwidth"])

    @classmethod
    def cosine(cls, Ah: float, kx: float = 1, ky: float = 1, phi_y: float = 0.0,
               moving: bool = False) -> "Roughness":
        """The analytic cosine cell: amplitude ``Ah`` [m], ``kx`` and ``ky`` periods per cell.
        ``moving`` as in :meth:`array` (False: the parent's "cup"; True: "ball")."""
        return cls("cosine", Ah=float(Ah), kx=float(kx), ky=float(ky), phi_y=float(phi_y), moving=bool(moving))

    # -- what the solver call needs ---------------------------------------------------------
    @property
    def is_pair(self) -> bool:
        return self.kind == "surface" and len(self._data["members"]) == 2

    @property
    def velocity_ratio(self) -> float:
        """The parent's ``roughness_velocity_ratio``: 0 for a stationary pattern, 1 when a
        member translates (the parent's ``roughness/pair.py`` LEGACY_ROUGHNESS_MODES)."""
        if self.kind == "cosine":
            return 1.0 if self._data["moving"] else 0.0
        return 1.0 if "translating" in self._data["members"] else 0.0

    @property
    def cell(self) -> Optional[Tuple[float, float]]:
        """``(Lx, Ly)`` [m] of a surface; None for the cosine (it takes the settings' cell)."""
        return (self._data["Lx"], self._data["Ly"]) if self.kind == "surface" else None

    def field(self) -> ArrayField:
        """The in-memory ``SurfaceField`` (built once)."""
        if self.kind != "surface":
            raise TypeError("the cosine cell has no surface field")
        if self._field is None:
            d = self._data
            self._field = ArrayField(d["members"], d["Lx"], d["Ly"], d["bandwidth"])
        return self._field

    def cosine_parameters(self) -> dict:
        if self.kind != "cosine":
            raise TypeError("not a cosine cell")
        return dict(self._data)

    def __repr__(self) -> str:
        if self.kind == "cosine":
            d = self._data
            return f"Roughness.cosine(Ah={d['Ah']:.4g}, kx={d['kx']:g}, ky={d['ky']:g}, moving={d['moving']})"
        return f"Roughness({'+'.join(self._data['members'])}, cell {self._data['Lx']:.4g} x {self._data['Ly']:.4g} m)"
