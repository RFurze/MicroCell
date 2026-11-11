"""Recipe -> resolved array -> prepared artefact (prepare once, run the bytes).

:func:`resolve` turns a per-surface recipe (:mod:`ehl.roughness.spec`) into a
centred gap-perturbation array plus reports; :func:`prepare` wraps it into an
immutable :class:`~ehl.roughness.assets.PreparedSurface` and, given a root
directory, publishes it.  A surface whose targets are not accepted is never
published (it can still be resolved for inspection with
``allow_rejected=True``).

Recipe identity (``recipe_id``) covers the spec, the generator revision and
the **content hashes** of every input file, so editing a PSD table, a sample
file or a surface file at the same path gives a new recipe id (gate G4).

Generated surfaces need an amplitude: ``fit.mode`` must be ``scale`` (anchor
Sa or Sq) or ``ratio``.  The stored array is the scaled field; Sa/Sq targets
are exact to round-off on the master array, Sz is checked only (M3).

Only NumPy is imported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from . import assets as A
from . import generation as G
from .evaluator import FourierSurface
from .spec import FileSurface, GeneratedSurface, ModeSurface, SurfaceSpec
from .stats import surface_stats
from .targets import (TargetInfeasibleError, TargetReport, apply_policy,
                      fit_gennorm_ratio, screen_request)

__all__ = ["resolve", "prepare", "recipe_for", "TargetNotMetError"]


class TargetNotMetError(ValueError):
    """The resolved surface does not meet its hard targets; not published."""

    def __init__(self, report: TargetReport):
        super().__init__(f"surface targets not accepted ({report.status}): {report.reason}")
        self.report = report


def _path(p: str, base_dir: Optional[Union[str, Path]]) -> Path:
    q = Path(p)
    if not q.is_absolute() and base_dir is not None:
        q = Path(base_dir) / q
    if not q.is_file():
        raise FileNotFoundError(f"surface input not found: {q}")
    return q


def _check_hash(path: Path, want: Optional[str]) -> str:
    got = A.file_sha256(path)
    if want is not None and want != got:
        raise A.SurfaceFileError(f"{path}: sha256 {got} != declared {want}")
    return got


def recipe_for(spec: SurfaceSpec, base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """The canonical recipe of ``spec``: its dictionary, the generator
    revision and the content hashes of its input files."""
    inputs: Dict[str, str] = {}
    if isinstance(spec, GeneratedSurface):
        if spec.psd.kind == "radial_table":
            inputs["psd_table"] = _check_hash(_path(spec.psd.table, base_dir), spec.psd.table_sha256)
        if spec.hpd.family == "samples":
            inputs["hpd_samples"] = _check_hash(_path(spec.hpd.samples, base_dir),
                                                spec.hpd.samples_sha256)
    elif isinstance(spec, FileSurface):
        inputs["surface_file"] = _check_hash(_path(spec.path, base_dir), spec.sha256)
    elif not isinstance(spec, ModeSurface):
        raise TypeError(f"not a surface recipe: {type(spec).__name__}")
    return {"surface": spec.to_dict(), "generator": G.GENERATOR_REVISION,
            "input_sha256": inputs}


def _amplitude(spec: GeneratedSurface, base_dir) -> np.ndarray:
    p = spec.psd
    if p.kind == "radial_table":
        with np.load(_path(p.table, base_dir), allow_pickle=False) as d:
            if "f" not in d.files or "P" not in d.files:
                raise A.SurfaceFileError(f"{p.table}: radial PSD table needs arrays 'f' and 'P'")
            f, P = np.array(d["f"], dtype=float), np.array(d["P"], dtype=float)
        return G.amplitude_from_radial_table(f, P, spec.ny, spec.nx, spec.Ly, spec.Lx,
                                             gamma_tilt=p.gamma_tilt, out_of_band=p.out_of_band)
    q = G.wavenumber_grid(spec.ny, spec.nx, spec.Ly, spec.Lx)
    if p.kind == "gaussian_acf":
        return G.psd_gaussian_acf(q, corr_len=p.corr_len, q1=p.q1)
    if p.kind == "exponential_acf":
        return G.psd_exponential_acf(q, corr_len=p.corr_len, q1=p.q1)
    return G.psd_self_affine(q, H=p.H, q0=p.q0, q1=p.q1)


def _target_vector(spec: GeneratedSurface, base_dir, fit_info: Dict[str, Any]) -> np.ndarray:
    M = spec.nx * spec.ny
    h = spec.hpd
    if spec.fit.mode == "ratio":
        t = spec.targets
        if t.Sa is None or t.Sq is None:
            raise ValueError("ratio fit needs both Sa and Sq targets")
        rf = fit_gennorm_ratio(t.Sa / t.Sq, M)
        fit_info.update(rf.to_dict())
        vec = G.hpd_sorted_sample(M, "gennorm", rf.beta)
    elif h.family == "samples":
        samples = np.load(_path(h.samples, base_dir), allow_pickle=False)
        vec = G.heights_from_samples(samples, M)
    else:
        vec = G.hpd_sorted_sample(M, h.family, h.shape)
    if h.orientation == "protrusion":
        vec = np.sort(-vec)
    return vec


def _resolve_generated(spec: GeneratedSurface, base_dir) -> Tuple[np.ndarray, Dict[str, Any]]:
    if spec.fit.mode == "fixed":
        raise ValueError("a generated surface needs an amplitude: use fit mode 'scale' or 'ratio'")
    bad = screen_request(spec.targets)
    if bad:
        raise TargetNotMetError(TargetReport(mode=spec.fit.mode, anchor=spec.fit.anchor,
                                             status="invalid_request", reason=bad,
                                             requested=spec.targets.requested()))
    gen: Dict[str, Any] = {"generator": spec.generator, "revision": G.GENERATOR_REVISION}
    fit_info: Dict[str, Any] = {}
    if spec.generator == "gaussian_filter":
        z = G.gaussian_filtered_surface(N=spec.nx, Lx=spec.Lx, corr_len=spec.psd.corr_len,
                                        seed=spec.seed)
        amp = None
    else:
        amp = _amplitude(spec, base_dir)
        try:
            vec = _target_vector(spec, base_dir, fit_info)
        except TargetInfeasibleError as exc:
            raise TargetNotMetError(exc.report) from None
        res = G.iaaft(amp, vec, np.random.default_rng(int(spec.seed)), n_iter=int(spec.n_iter))
        z = res.Zh if spec.take == "Zh" else res.Zs
        zs_std = np.sort((res.Zs - res.Zs.mean()).ravel())
        zs_std = zs_std / np.sqrt(np.mean(zs_std ** 2))
        gen.update(err_h_legacy=res.err_h, err_s_legacy=res.err_s,
                   spectral_energy_error_Zh=G.spectral_energy_error(res.Zh, amp),
                   spectral_energy_error_Zs=G.spectral_energy_error(res.Zs, amp),
                   hpd_max_quantile_error_Zs=float(np.max(np.abs(zs_std - vec))),
                   n_iter=res.n_iter, take=spec.take)
    zc, rep = apply_policy(z, spec.targets, spec.fit)
    rep.fit.update(fit_info)
    return zc, {"generation": gen, "targets": rep}


def _resolve_file(spec: FileSurface, base_dir) -> Tuple[np.ndarray, Dict[str, Any], float, float]:
    sa = A.read_surface_file(_path(spec.path, base_dir), expected_sha256=spec.sha256)
    if spec.fit.mode == "fixed" and not spec.targets.requested():
        rep = TargetReport(mode="fixed", anchor=None, status="accepted",
                           realised={k: v for k, v in surface_stats(sa.z, sa.Lx, sa.Ly).items()
                                     if k in ("Sa", "Sq", "Sz")})
        return np.array(sa.z), {"source": sa.provenance, "targets": rep}, sa.Lx, sa.Ly
    zc, rep = apply_policy(sa.z, spec.targets, spec.fit)
    return zc, {"source": sa.provenance, "targets": rep}, sa.Lx, sa.Ly


def _resolve_modes(spec: ModeSurface) -> Tuple[np.ndarray, Dict[str, Any]]:
    f = FourierSurface.from_modes(spec.modes, spec.Lx, spec.Ly)
    rep = TargetReport(mode="fixed", anchor=None, status="accepted")
    return np.array(f.source), {"targets": rep, "source": {"kind": "modes",
                                                         "grid": list(f.source_shape)}}


def resolve(spec: SurfaceSpec, base_dir: Optional[Union[str, Path]] = None
            ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """``(z, Lx, Ly, reports)`` for a recipe; ``z`` centred, gap convention.
    ``reports['targets']`` is a :class:`TargetReport` (not yet a dict)."""
    if isinstance(spec, GeneratedSurface):
        z, rep = _resolve_generated(spec, base_dir)
        Lx, Ly = spec.Lx, spec.Ly
    elif isinstance(spec, FileSurface):
        z, rep, Lx, Ly = _resolve_file(spec, base_dir)
    elif isinstance(spec, ModeSurface):
        z, rep = _resolve_modes(spec)
        Lx, Ly = spec.Lx, spec.Ly
    else:
        raise TypeError(f"not a surface recipe: {type(spec).__name__}")
    # every branch above returns a field centred exactly once; centring again
    # would perturb the last bits of historical arrays (WTC26 parity)
    st = surface_stats(z, Lx, Ly)
    fs = FourierSurface(z, Lx, Ly)
    lo, hi = fs.extrema(oversample=4)
    rep["statistics"] = st
    rep["reconstruction"] = {**fs.identity(), **fs.variances(),
                             "extrema_oversample4": [lo, hi],
                             "overshoot_below_source_min": float(st["z_min"] - lo),
                             "overshoot_above_source_max": float(hi - st["z_max"])}
    return z, float(Lx), float(Ly), rep


def prepare(spec: SurfaceSpec, root: Optional[Union[str, Path]] = None,
            base_dir: Optional[Union[str, Path]] = None,
            allow_rejected: bool = False) -> A.PreparedSurface:
    """Resolve ``spec`` and wrap it as a :class:`PreparedSurface`; publish it
    under ``root`` when given (only if its targets are accepted)."""
    recipe = recipe_for(spec, base_dir)
    z, Lx, Ly, rep = resolve(spec, base_dir)
    trep: TargetReport = rep["targets"]
    if not trep.accepted and not allow_rejected:
        raise TargetNotMetError(trep)
    reports = dict(rep)
    reports["targets"] = trep.to_dict()
    reports["accepted"] = trep.accepted
    prepared = A.make_prepared(recipe, z, Lx, Ly, reports=reports)
    if root is not None:
        if not trep.accepted:
            raise TargetNotMetError(trep)
        A.publish_artefact(prepared, root)
    return prepared
