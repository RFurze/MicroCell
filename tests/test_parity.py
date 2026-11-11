"""Parity gate: every reference cell through ``microsolver()`` against the parent's values.

The references (``tests/references/*.npz``) were made by ``tools/make_references.py`` from the
parent ``MultiscaleMixedBallinSocket`` at 2aaccad, one process per cell, at full double precision.
``c5_cosine_fullfilm`` was re-made on 11 Nov 2026 with ``k_spring = 0`` (WP2, Rob's decision): the
case's ``1e20`` "off" value still ran the parent's cell-average spring in that cell.

The rule (plan D6; written before the first run, not fitted to it)
------------------------------------------------------------------
* Each cell is solved at the reference's ``mesh_used``, with ``MicroSettings`` built from the
  reference's recorded case and settings, the roughness from the reference's stored surface
  arrays (or cosine parameters), ``Zeta.from_task`` of its task tuple, its time window, its
  ``n_steps`` and its offset.
* Homogenised outputs (every field of the parent's ``MicroTaskResult``, read from where a
  ``MicroResult`` holds it now: ``RESULT_NOW``, WP4) and the field diagnostics: floats and each component of a vector pass when
  ``|a - b| <= 1e-12 * max(|a|, |b|)``; a NaN must be a NaN on both sides; integers and
  strings must be equal.
  The eleven global-x shear statistics of the parent's result (``REMOVED_RESULT_FIELDS``, WP4) are
  not in a ``MicroResult`` and are not compared; a reference that held a finite one, or a result
  that still carried one, fails.  With them the parent's ``no_frame`` status is gone: where a
  reference has it, the result must hold the status the reference's own support fractions give
  (``STATUS_WITHOUT_FRAME``).
* Fields (the ``_extract_fields`` arrays in SI, and the DG0 fluid indicator): same shape, the
  same NaN pattern, and ``max|a - b| <= 1e-12 * max|reference|``.
  ``p_fluid_masked`` is not held (``DERIVED_FIELDS``, WP4): the gate forms it from the held
  ``p_fluid`` and ``chi_fluid`` and compares that with the reference's array.
* Every field of the parent's settings dataclass that the reference stores must equal the value
  the solver reads now, read from where it reads it (``SETTINGS_NOW``, WP4: a ``MicroSettings``
  field, a fixed value of the solver module, or the call).  The ripple fields of a reference
  made without the ripple were never read by the parent; they pass when the settings carry no
  ripple (``RIPPLE_SETTINGS``).
  A setting the pruned solver no longer has (``REMOVED_SETTINGS``, WP2) is not skipped: the
  reference's stored value must equal the value the code now hard-wires, and ``SETTINGS_NOW``
  must not carry it.  A setting carried under another name and type
  (``REPLACED_SETTINGS``: the parent's ``phase_average`` tag is the flag ``pair_branch``) is
  checked value for value through its stated map.
* The cell-average spring is removed (WP2).  ``case.k_spring`` passes when the parent's spring
  did not act on the reference cell (``spring_acted``), whatever value the case stored.

Not compared, by construction:
  ``wall_time``, ``contact_eval_s``, ``pid``, ``task_id``, ``task``     timings and bookkeeping
  ``h_geom_min``, ``escalated``                                      the two-tier predictor is not in the API (D9)
  ``pasp_table``, ``pasp_coupled``, ``pasp_contact_only``, ``kasp_table``, ``amech_table``,
  ``offset_x_m``, ``offset_y_m``, ``phase_average``                  the contact table is not in the API (D4)
For the pair cell (a reference made with the contact table) ``Pasp_hom`` and ``k_asp_hom`` are
compared with the solver's coupled values before table composition (``coupled_json``), and
``contact_iters`` is not compared (the parent's count includes the table route's contact-only solve).

Script form:  python tests/test_parity.py [--cell ID ...] [--out report.tsv]
exit 0 = PASS, 1 = FAIL, 2 = INCOMPLETE (no references, or a cell could not be run).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCES = ROOT / "tests" / "references"
RTOL = 1e-12

NOT_COMPARED = ("wall_time", "contact_eval_s", "pid", "task_id", "task", "h_geom_min", "escalated",
                "pasp_table", "pasp_coupled", "pasp_contact_only", "kasp_table", "amech_table",
                "offset_x_m", "offset_y_m", "phase_average")
PAIR_FROM_COUPLED = {"Pasp_hom": "Pasp_hom", "k_asp_hom": "k_asp_hom"}
PAIR_NOT_COMPARED = ("contact_iters",)

#: WP2: settings removed from the copied dataclasses, with the value the pruned code hard-wires.
#: Every reference that stores the setting must store exactly this value (a reference that used
#: the removed option fails the gate here).
REMOVED_SETTINGS = {
    "spring_pointwise": False,          # group 2: the pointwise spring mode
    "use_reduced_modulus": False,       # group 3: fft_elastic_v1 refused True
    "overclosure_measure": "patch_max",  # group 3: fft_elastic_v1 refused anything else
    "gauge_lift": False,                # group 5: ignored by the solver (True only warned)
}

#: WP2 group 3: settings that only the removed ``legacy_patch_v1`` law read, so there is no value
#: to freeze.  The check is that the reference's own contact law is the kept one.
REMOVED_UNREAD_SETTINGS = {
    "contact_E": "legacy_patch_v1 only (Young's modulus of its closure)",
    "contact_nu": "legacy_patch_v1 only (with use_reduced_modulus)",
    "k_asp_exponent": "legacy_patch_v1 only (its closure tangent; fft_elastic_v1 uses contact_tangent)",
}
KEPT_CONTACT_MODEL = "fft_elastic_v1"

#: WP2 group 8: a reference setting the pruned code carries under another name and type.
#: reference key -> (built key, {reference value: built value}).  The built settings must carry the
#: new key and not the old one, and the built value must be the image of the reference's.
REPLACED_SETTINGS = {
    "phase_average": ("pair_branch", {"off": False, "contact_table_v1": True}),
}
_REPLACEMENT_KEYS = {new for new, _ in REPLACED_SETTINGS.values()}

#: WP4: the parent's settings dataclass, field by field, and where the solver reads that value now.
#:   "settings.<path>"  a field of the call's ``MicroSettings`` (``contact.`` and ``ripple.`` nested)
#:   "fixed.<NAME>"     a fixed value (module constant) of ``microcell.micro_phase_neutral_class``
#:   "contact law"      ``microcell.contact.FFT``, the one contact law
#:   "ripple on"        "ripple_halfspace" when ``MicroSettings.ripple`` is set, else "off"
#:   "pair"             the call's roughness is a stationary + translating pair
SETTINGS_NOW = {
    "relaxation_parameter": "settings.relaxation_parameter",
    "max_iterations": "settings.max_iterations",
    "abs_tolerance": "settings.abs_tolerance",
    "rel_tolerance": "settings.rel_tolerance",
    "delta_h": "settings.delta_h",
    "eps_solid": "settings.eps_solid",
    "eps_smooth": "settings.eps_smooth",
    "print_newton": "fixed.PRINT_NEWTON",
    "print_timing": "fixed.PRINT_TIMING",
    "spos_form": "settings.spos_form",
    "beta": "fixed.BETA",
    "jacobian_reg_eps": "settings.jacobian_reg_eps",
    "adaptive_damping": "fixed.ADAPTIVE_DAMPING",
    "omega_min": "fixed.OMEGA_MIN",
    "newton_backtrack_max": "fixed.NEWTON_BACKTRACK_MAX",
    "newton_backtrack_tol": "fixed.NEWTON_BACKTRACK_TOL",
    "newton_backtrack_warmup_iters": "fixed.NEWTON_BACKTRACK_WARMUP_ITERS",
    "omega_recovery_factor": "fixed.OMEGA_RECOVERY_FACTOR",
    "omega_cap_shrink_factor": "fixed.OMEGA_CAP_SHRINK_FACTOR",
    "fluid_deformation": "ripple on",
    "fluid_E_star": "settings.E_star",
    "ripple_tol": "settings.ripple.tol",
    "ripple_maxpass": "settings.ripple.maxpass",
    "ripple_omega": "settings.ripple.omega",
    "level_constraint": "settings.level_constraint",
    "patch_pin": "fixed.PATCH_PIN",
    "patch_pin_kappa": "fixed.PATCH_PIN_KAPPA",
    # the mixed and contact classes
    "pattern_flux_contact": "settings.pattern_flux_contact",
    "sigma_yield": "settings.contact.sigma_yield",
    "k_asp_mode": "settings.contact.k_asp_mode",
    "contact_model": "contact law",
    "contact_E_star": "settings.E_star",
    "contact_tol": "settings.contact.tol",
    "contact_maxit": "settings.contact.maxit",
    "mu_material": "settings.contact.mu",
    "pair_branch": "pair",
}
#: the parent read these only with the ripple on; a reference made without it stores the
#: parent's dataclass defaults, which nothing read
RIPPLE_SETTINGS = ("fluid_E_star", "ripple_tol", "ripple_maxpass", "ripple_omega")
#: fields of the parent's contact settings class (a full-film reference does not store them)
CONTACT_CLASS_SETTINGS = ("pattern_flux_contact", "sigma_yield", "k_asp_mode", "contact_model", "contact_E_star",
                          "contact_tol", "contact_maxit", "mu_material", "pair_branch")

#: WP2 group 2: the cell-average spring (``k_spring`` of the parent's case) is removed.  The parent
#: ran it in a transient cell of the full-film class with ``k_spring > 0``; its steady full-film
#: solve was raw, and its contact class never ran the spring and refused a stiffness between 0 and
#: this value.
SPRING_OFF_IN_CONTACT_CLASS = 1e19

#: WP2 group 3: ``contact_model`` of a full-film cell's result.  The parent's result carried its
#: default tag there; that law is gone and the API reports "off".
FULL_FILM_CONTACT_TAG = {"legacy_patch_v1": "off"}

#: WP4: where a ``MicroResult`` holds each field of the parent's ``MicroTaskResult``.  A value is a
#: named output of ``MicroResult`` or ``info.<key>``; a field not listed is ``info[<the same name>]``.
#: Four of the parent's fields are no longer written because, with one phase per call, they are a
#: second copy of an extremum (the parent wrote ``p_max`` twice, and its mean over the phases of
#: the phase extrema is the extremum of the one phase); they are still compared, against the one
#: place the number is held.
RESULT_NOW = {
    "delta_Q": "dQ", "delta_P": "dP", "Q": "Q", "Pst": "Pst", "P_f": "P_f", "Fst": "Fst",
    "tau_res": "tau_fluid",
    "shear_asperity_mean_local_Pa": "tau_asperity", "shear_total_mean_local_Pa": "tau_total",
    "Pasp_hom": "P_asp", "contact_frac": "contact_frac", "A_mech_frac": "A_mech_frac", "k_asp_hom": "k_asp",
    "p_max": "p_max", "p_min": "p_min", "h_max": "h_max", "h_min": "h_min", "yield_flag": "yield_flag",
    "p_fluid_sample_max_Pa": "p_max", "p_max_phase_mean": "p_max",     # second copies, one phase per call
    "p_fluid_sample_min_Pa": "p_min", "p_min_phase_mean": "p_min",
    "failure_reason": "reason",
    "mesh_used": "info.mesh_n",
}
#: ``info`` entries that are not a field of the parent's result: the call, and the field diagnostics
INFO_OF_THE_CALL = ("steady", "t_end", "n_steps", "offset", "roughness_velocity_ratio", "t_start", "dt",
                    "fields", "paraview")

#: WP4 (Rob, 11 Nov 2026): fields of the parent's result that left the result.  They are the
#: global-x envelopes of the shear tractions, which need the macro frame ``R_inv``; a standalone
#: cell has none, so each is NaN in every reference.  Not compared.  The gate fails a reference
#: that holds a finite one (it would be an output this solver can no longer give) and a result
#: that still carries one.
REMOVED_RESULT_FIELDS = (
    "shear_fluid_gx_min_Pa", "shear_fluid_gx_max_Pa", "shear_fluid_gx_sample_mean_Pa",
    "shear_fluid_support_gx_min_Pa", "shear_fluid_support_gx_max_Pa",
    "shear_asperity_gx_min_Pa", "shear_asperity_gx_max_Pa", "shear_asperity_gx_sample_mean_Pa",
    "shear_total_gx_min_Pa", "shear_total_gx_max_Pa", "shear_total_gx_sample_mean_Pa",
)
#: WP4: the parent's status ``no_frame`` ("the local mean is computed, the global-x statistics are
#: NaN") went with those statistics.  Where a reference stores it, the result holds the status of
#: the local mean, which the reference's own support fractions fix: ``ok`` when one of the listed
#: fractions is positive (the contribution is present at a sample), else ``no_support``.  Any
#: other reference status is compared as it is.
STATUS_WITHOUT_FRAME = {
    "shear_extrema_status": ("shear_fluid_support_fraction",),
    "shear_asperity_status": ("shear_asperity_support_fraction",),
    "shear_total_status": ("shear_fluid_support_fraction", "shear_asperity_support_fraction"),
}
#: WP4: reference fields a ``MicroResult`` does not hold, formed here from the fields it holds
DERIVED_FIELDS = {
    "p_fluid_masked": lambda f: np.where(f["chi_fluid"] > 0.5, f["p_fluid"], np.nan),
}
_MISSING = object()


def result_now(out, key: str):
    """The parent's result field ``key`` as the ``MicroResult`` holds it (``_MISSING`` if it does not)."""
    where = RESULT_NOW.get(key, "info." + key)
    if where.startswith("info."):
        return out.info.get(where[len("info."):], _MISSING)
    return getattr(out, where)


def status_without_frame(result: dict, key: str) -> str:
    """The reference's status ``result[key]`` with ``no_frame`` replaced (``STATUS_WITHOUT_FRAME``)."""
    want = result[key]
    if key in STATUS_WITHOUT_FRAME and want == "no_frame":
        return "ok" if any(float(result[f]) > 0.0 for f in STATUS_WITHOUT_FRAME[key]) else "no_support"
    return want


def spring_acted(ref: dict) -> bool:
    """True when the parent's cell-average spring changed this reference cell."""
    k = float(ref["micro_case"]["k_spring"])
    if bool(ref["micro_case"]["mixed_enabled"]):
        return 0.0 < k < SPRING_OFF_IN_CONTACT_CLASS
    return k > 0.0 and bool(ref["meta"]["transient"])


def reference_cells():
    return sorted(p.stem for p in REFERENCES.glob("*.npz"))


def load_reference(cell: str) -> dict:
    d = np.load(REFERENCES / f"{cell}.npz")
    ref = {k[:-5]: json.loads(str(d[k])) for k in d.files if k.endswith("_json")}
    ref["task"] = np.asarray(d["task"], dtype=float)
    ref["fields"] = {k[len("field__"):]: np.asarray(d[k]) for k in d.files if k.startswith("field__")}
    ref["fields"]["chi_fluid_dg0"] = np.asarray(d["chi_fluid_dg0"])
    ref["surfaces"] = {k.split("__")[1]: np.asarray(d[k]) for k in d.files if k.startswith("surface__")}
    return ref


def settings_from_reference(ref: dict):
    """``MicroSettings`` carrying exactly what the reference's case and settings recorded.  The one
    modulus is the contact law's when the reference has contact, else the ripple's (the gate compares
    both of the reference's moduli with it).  ``beta`` and ``rho0`` of the case are not passed: the
    first is a fixed value the gate checks, the second was never read."""
    from microcell import ContactSettings, MicroSettings, RippleSettings

    mc, sj, meta = ref["micro_case"], ref["settings"], ref["meta"]
    mixed = bool(mc["mixed_enabled"])
    contact = ContactSettings(
        tol=sj["contact_tol"], maxit=sj["contact_maxit"],
        sigma_yield=mc["sigma_yield"], mu=mc["mu_material"], k_asp_mode=mc["k_asp_mode"]) if mixed else None
    ripple = RippleSettings(tol=sj["ripple_tol"], maxpass=sj["ripple_maxpass"],
                            omega=sj["ripple_omega"]) if mc["fluid_deformation"] != "off" else None
    modulus = {} if not (mixed or ripple is not None) else {
        "E_star": mc["contact_E_star"] if mixed else mc["fluid_E_star"]}
    return MicroSettings(
        mesh_n=int(meta["mesh_used"]), xmax=mc["xmax"], ymax=mc["ymax"], eta0=mc["eta0"],
        penalty_gamma=mc["penalty_gamma"], h_min=mc["h_min"], spos_form=mc["spos_form"],
        level_constraint=mc["level_constraint"], pattern_flux_contact=mc["pattern_flux_contact"],
        jacobian_reg_eps=mc["jacobian_reg_eps"], relaxation_parameter=mc["relaxation_parameter"],
        max_iterations=mc["max_iterations"], abs_tolerance=mc["abs_tolerance"], rel_tolerance=mc["rel_tolerance"],
        contact=contact, ripple=ripple, cfl_target=float(meta["cfl_target"]),
        roughness_velocity_ratio=None, delta_h=mc["delta_h"], eps_solid=mc["eps_solid"],
        eps_smooth=mc["eps_smooth"], **modulus)


def settings_now(st, pair: bool) -> dict:
    """The parent's settings dataclass by field name, each value read from where the solver reads it
    now (``SETTINGS_NOW``).  A ripple field is absent when the settings carry no ripple, and a field
    of the parent's contact class when they carry no contact (the parent's full-film class had none)."""
    import microcell.micro_phase_neutral_class as solver_module
    from microcell.contact import FFT

    out = {}
    for name, where in SETTINGS_NOW.items():
        if name in CONTACT_CLASS_SETTINGS and st.contact is None:
            continue
        if name in RIPPLE_SETTINGS and st.ripple is None:
            continue
        if where == "contact law":
            out[name] = FFT
        elif where == "ripple on":
            out[name] = "ripple_halfspace" if st.ripple is not None else "off"
        elif where == "pair":
            out[name] = bool(pair)
        elif where.startswith("fixed."):
            out[name] = getattr(solver_module, where[len("fixed."):])
        else:
            obj = st
            for part in where[len("settings."):].split("."):
                obj = getattr(obj, part)
            out[name] = obj
    return out


def roughness_from_reference(ref: dict):
    """The reference's roughness: its stored surface arrays, or its cosine parameters."""
    from microcell import Roughness
    from microcell.roughness.evaluator import Bandwidth

    mc = ref["micro_case"]
    if not ref["surfaces"]:
        if mc["roughness_mode"] not in ("cup", "ball"):
            raise ValueError(f"cosine reference with roughness_mode {mc['roughness_mode']!r}")
        return Roughness.cosine(mc["Ah"], mc["kx"], mc["ky"], phi_y=mc["phi_y"],
                                moving=(mc["roughness_mode"] == "ball"))
    sj = ref["surface"]
    Lx, Ly = (float(v) for v in sj["cell_m"])
    bw = Bandwidth(**sj["bandwidth"])
    one = {role: Roughness.array(ref["surfaces"][role], Lx, Ly, moving=(role == "translating"),
                                 offset=sj["members"][role]["offset_m"], bandwidth=bw)
           for role in ref["surfaces"]}
    if len(one) == 2:
        return Roughness.pair(one["stationary"], one["translating"])
    return next(iter(one.values()))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """max over components of |a - b| / max(|a|, |b|) (0 where equal)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.maximum(np.abs(a), np.abs(b))
        r = np.where(a == b, 0.0, np.abs(a - b) / scale)
    return float(np.max(r)) if r.size else 0.0


def compare_scalar(name: str, want, got) -> dict:
    """One homogenised output: string, integer, float or short vector."""
    rec = {"name": name, "kind": "scalar", "rel": 0.0, "ok": True, "note": ""}
    if isinstance(want, str) or want is None:
        rec["ok"] = (got == want)
        rec["note"] = "" if rec["ok"] else f"{got!r} != {want!r}"
        return rec
    if isinstance(want, (bool, int)) and not isinstance(want, float):
        rec["ok"] = isinstance(got, (bool, int, np.integer)) and int(got) == int(want)
        rec["note"] = "" if rec["ok"] else f"{got!r} != {want!r}"
        return rec
    a = np.asarray(want, dtype=float).ravel()
    b = np.asarray(got, dtype=float).ravel()
    if a.shape != b.shape:
        rec.update(ok=False, note=f"size {b.shape} != {a.shape}")
        return rec
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        rec.update(ok=False, rel=float("inf"), note=f"NaN pattern differs: {b.tolist()} vs {a.tolist()}")
        return rec
    m = ~np.isnan(a)
    rec["rel"] = _rel(a[m], b[m])
    rec["ok"] = bool(rec["rel"] <= RTOL)
    if not rec["ok"]:
        rec["note"] = f"{b.tolist()} vs {a.tolist()}"
    return rec


def compare_field(name: str, want: np.ndarray, got) -> dict:
    rec = {"name": name, "kind": "field", "rel": 0.0, "ok": True, "note": ""}
    if got is None:
        rec.update(ok=False, rel=float("inf"), note="missing")
        return rec
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if got.shape != want.shape:
        rec.update(ok=False, rel=float("inf"), note=f"shape {got.shape} != {want.shape}")
        return rec
    if not np.array_equal(np.isnan(want), np.isnan(got)):
        rec.update(ok=False, rel=float("inf"), note="NaN pattern differs")
        return rec
    m = ~np.isnan(want)
    diff = float(np.max(np.abs(got[m] - want[m]))) if m.any() else 0.0
    scale = float(np.max(np.abs(want[m]))) if m.any() else 0.0
    rec["rel"] = 0.0 if diff == 0.0 else (diff / scale if scale > 0.0 else float("inf"))
    rec["ok"] = bool(rec["rel"] <= RTOL)
    if not rec["ok"]:
        rec["note"] = f"max|diff| {diff:.3e}, max|ref| {scale:.3e}"
    return rec


def run_cell(cell: str) -> dict:
    """Solve one reference cell through the API and compare.  Returns
    ``{"cell", "records": [...], "status", "wall_s"}``."""
    from microcell import Zeta, microsolver

    ref = load_reference(cell)
    meta = ref["meta"]
    if meta["status"] != "OK":
        raise RuntimeError(f"{cell}: the reference itself is {meta['status']}")
    st = settings_from_reference(ref)
    rough = roughness_from_reference(ref)
    records = []

    # the parent's settings dataclass, field by field, against what the solver reads now
    built = settings_now(st, rough.is_pair)
    want = {k: v for k, v in ref["settings"].items() if k != "__class__"}
    for k in sorted(set(want) | set(built)):
        if k in REMOVED_SETTINGS:
            if k in built:
                records.append({"name": f"settings.{k}", "kind": "setting", "rel": float("inf"), "ok": False,
                                "note": "listed as removed but SETTINGS_NOW still carries it"})
                continue
            r = compare_scalar(f"settings.{k}", REMOVED_SETTINGS[k], want[k])
            if r["rel"] != 0.0:
                r["ok"] = False
            if not r["ok"]:
                r["note"] = f"removed setting: the reference stores {want[k]!r}, the code hard-wires {REMOVED_SETTINGS[k]!r}"
            r["kind"] = "setting"
            records.append(r)
            continue
        if k in REMOVED_UNREAD_SETTINGS:
            used = want.get("contact_model")
            ok = k not in built and used == KEPT_CONTACT_MODEL
            records.append({"name": f"settings.{k}", "kind": "setting", "rel": 0.0 if ok else float("inf"), "ok": ok,
                            "note": "" if ok else (
                                "listed as removed but SETTINGS_NOW still carries it" if k in built else
                                f"removed setting read by {REMOVED_UNREAD_SETTINGS[k]}; the reference's contact_model "
                                f"is {used!r}")})
            continue
        if k in REPLACED_SETTINGS:
            new, image = REPLACED_SETTINGS[k]
            ok = (k not in built and new in built and k in want and want[k] in image
                  and type(built[new]) is type(image[want[k]]) and built[new] == image[want[k]])
            records.append({"name": f"settings.{k}", "kind": "setting", "rel": 0.0 if ok else float("inf"), "ok": ok,
                            "note": "" if ok else (f"replaced by {new!r}: the reference stores {want.get(k)!r}, "
                                                   f"the built settings carry {new}={built.get(new)!r}"
                                                   + (" and still the old key" if k in built else ""))})
            continue
        if k in _REPLACEMENT_KEYS and k not in want:
            continue                                # checked with the reference key it replaces
        if k in RIPPLE_SETTINGS and k in want and k not in built:
            ok = want.get("fluid_deformation") == "off" and st.ripple is None
            records.append({"name": f"settings.{k}", "kind": "setting", "rel": 0.0 if ok else float("inf"), "ok": ok,
                            "note": "" if ok else ("a ripple setting with no ripple in the settings, but the "
                                                   f"reference's fluid_deformation is {want.get('fluid_deformation')!r}")})
            continue
        if k not in built or k not in want:
            records.append({"name": f"settings.{k}", "kind": "setting", "rel": float("inf"), "ok": False,
                            "note": "only on one side"})
            continue
        r = compare_scalar(f"settings.{k}", want[k], built[k])
        if r["rel"] != 0.0:                         # settings are inputs: exact
            r["ok"] = False
        r["kind"] = "setting"
        records.append(r)
    acted = spring_acted(ref)
    records.append({"name": "case.k_spring", "kind": "setting", "rel": float("inf") if acted else 0.0,
                    "ok": not acted,
                    "note": "" if not acted else (
                        f"removed option: the reference was made with k_spring = {ref['micro_case']['k_spring']!r}, "
                        "under which the parent's cell-average spring acted on this cell")})
    if rough.kind == "surface":
        records.append(dict(compare_scalar("surface.envelope_m", ref["surface"]["envelope_m"],
                                           rough.field().envelope_m), kind="setting"))

    task = ref["task"]
    transient = bool(meta["transient"])
    out = microsolver(
        Zeta.from_task(task), t_end=(float(task[15]) if transient else None), roughness=rough, settings=st,
        n_steps=int(meta["n_steps"]), offset=tuple(meta["offset"]) if meta["offset"] else (0.0, 0.0),
        steady=not transient)
    records.append({"name": "status", "kind": "scalar", "rel": 0.0, "ok": out.status == "OK",
                    "note": "" if out.status == "OK" else f"{out.status}: {out.reason}"})

    pair = rough.is_pair
    result = dict(ref["result"])
    if st.contact is None:                          # full film: the parent's default tag of a dropped law
        result["contact_model"] = FULL_FILM_CONTACT_TAG.get(result["contact_model"], result["contact_model"])
    if pair:
        for key, src in PAIR_FROM_COUPLED.items():
            result[key] = ref["coupled"][src]
    for key in sorted(result):
        if key in NOT_COMPARED or (pair and key in PAIR_NOT_COMPARED):
            continue
        if key in REMOVED_RESULT_FIELDS:            # not a comparison: a record only when it does not hold
            finite = bool(np.any(np.isfinite(np.asarray(result[key], dtype=float))))
            if finite or key in out.info:
                records.append({"name": key, "kind": "scalar", "rel": float("inf"), "ok": False,
                                "note": (f"removed output: the reference holds {result[key]!r}" if finite
                                         else "removed output still in info")})
            continue
        got = result_now(out, key)
        if got is _MISSING:
            records.append({"name": key, "kind": "scalar", "rel": float("inf"), "ok": False, "note": "missing"})
            continue
        records.append(compare_scalar(key, status_without_frame(result, key), got))
    # each output once: nothing in info but the parent's fields (compared or listed as not) and the call
    held = {RESULT_NOW.get(k, "info." + k) for k in result}
    unknown = sorted(k for k in out.info if "info." + k not in held and k not in INFO_OF_THE_CALL)
    if unknown:
        records.append({"name": "info", "kind": "scalar", "rel": float("inf"), "ok": False,
                        "note": f"info entries that are neither a field of the parent's result nor the call: {unknown}"})
    for key in sorted(ref["info"]):
        got = out.info.get("fields", {}).get(key)
        records.append(dict(compare_scalar(f"info.{key}", ref["info"][key], got), kind="diagnostic")
                       if got is not None else
                       {"name": f"info.{key}", "kind": "diagnostic", "rel": float("inf"), "ok": False,
                        "note": "missing"})
    for key in sorted(ref["fields"]):
        if key in DERIVED_FIELDS:
            if key in out.fields:
                records.append({"name": key, "kind": "field", "rel": float("inf"), "ok": False,
                                "note": "listed as derived but the result still holds it"})
                continue
            records.append(compare_field(key, ref["fields"][key], DERIVED_FIELDS[key](out.fields)))
            continue
        records.append(compare_field(key, ref["fields"][key], out.fields.get(key)))
    extra = sorted(set(out.fields) - set(ref["fields"]) - {"x", "y"})
    if extra:
        records.append({"name": "fields", "kind": "field", "rel": float("inf"), "ok": False,
                        "note": f"fields the reference does not have: {extra}"})
    return {"cell": cell, "records": records, "status": "PASS" if all(r["ok"] for r in records) else "FAIL",
            "wall_s": float(out.info.get("wall_time", float("nan")))}


def summarise(res: dict) -> dict:
    """One row per cell: counts and the largest relative differences."""
    row = {"cell": res["cell"], "status": res["status"], "wall_s": res["wall_s"]}
    for kind in ("setting", "scalar", "diagnostic", "field"):
        recs = [r for r in res["records"] if r["kind"] == kind]
        worst = max(recs, key=lambda r: r["rel"]) if recs else None
        row[f"n_{kind}"] = len(recs)
        row[f"n_{kind}_exact"] = sum(1 for r in recs if r["ok"] and r["rel"] == 0.0)
        row[f"max_rel_{kind}"] = worst["rel"] if worst else 0.0
        row[f"worst_{kind}"] = worst["name"] if worst and worst["rel"] > 0.0 else ""
    row["failed"] = ";".join(r["name"] for r in res["records"] if not r["ok"])
    return row


# ---------------------------------------------------------------------------------------
# pytest form
# ---------------------------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    if "cell" in metafunc.fixturenames:
        metafunc.parametrize("cell", reference_cells())


def test_references_present():
    assert len(reference_cells()) >= 6, f"expected the reference cells in {REFERENCES}"


def test_parity(cell):
    res = run_cell(cell)
    bad = [f"{r['name']}: rel {r['rel']:.3e} {r['note']}" for r in res["records"] if not r["ok"]]
    assert not bad, f"{cell}: {len(bad)} of {len(res['records'])} comparisons outside 1e-12:\n" + "\n".join(bad)


def test_generated_surface_reproduces_reference():
    """``Roughness.generated`` returns the array the parent prepared for the GenSa40 case
    (the reference's stored stationary surface)."""
    from microcell import Roughness

    ref = load_reference("c1_gensa40_h200_ripple")
    z_ref = ref["surfaces"]["stationary"]
    fld = Roughness.generated(Sa=40e-9, seed=1000).field()
    z = fld.members["stationary"][0].source
    r = compare_field("generated_z", z_ref, np.asarray(z))
    assert r["ok"], r


def test_comparator_sees_a_perturbed_solve(monkeypatch):
    """Negative control: with the film of one cell raised by 1e-9 (relative) the gate must FAIL,
    on homogenised outputs and on fields."""
    original = load_reference

    def perturbed(cell):
        ref = original(cell)
        ref["task"] = ref["task"].copy()
        ref["task"][2] *= (1.0 + 1e-9)
        return ref

    monkeypatch.setattr(sys.modules[__name__], "load_reference", perturbed)
    res = run_cell("c2_gensa40_h130_ripple")
    bad = {kind: [r for r in res["records"] if not r["ok"] and r["kind"] == kind]
           for kind in ("setting", "scalar", "field")}
    assert res["status"] == "FAIL"
    assert not bad["setting"] and bad["scalar"] and bad["field"]


def test_removed_setting_is_checked(monkeypatch):
    """Negative control of ``REMOVED_SETTINGS``: when a reference's stored value of a removed
    setting is not the hard-wired one, the gate must FAIL on that setting and on nothing else."""
    monkeypatch.setitem(REMOVED_SETTINGS, "spring_pointwise", True)
    res = run_cell("c8_cosine_fullfilm_steady")
    assert [r["name"] for r in res["records"] if not r["ok"]] == ["settings.spring_pointwise"]


def test_removed_spring_is_checked(monkeypatch):
    """Negative control of the removed spring: a transient full-film reference whose case carries a
    positive ``k_spring`` (as c5 did before it was re-made) must FAIL on ``case.k_spring`` and on
    nothing else."""
    original = load_reference

    def with_spring(cell):
        ref = original(cell)
        ref["micro_case"] = dict(ref["micro_case"], k_spring=1e20)
        return ref

    monkeypatch.setattr(sys.modules[__name__], "load_reference", with_spring)
    res = run_cell("c5_cosine_fullfilm")
    assert [r["name"] for r in res["records"] if not r["ok"]] == ["case.k_spring"]


# ---------------------------------------------------------------------------------------
# script form
# ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", action="append", help="reference id (default: all)")
    ap.add_argument("--out", help="write the per-cell summary (TSV) and the per-comparison detail beside it")
    args = ap.parse_args(argv)
    cells = args.cell or reference_cells()
    if not cells:
        print(f"INCOMPLETE: no reference cells in {REFERENCES}")
        return 2
    rows, detail, incomplete = [], [], []
    for cell in cells:
        try:
            res = run_cell(cell)
        except Exception:                                    # could not be run: INCOMPLETE, not FAIL
            traceback.print_exc()
            incomplete.append(cell)
            continue
        rows.append(summarise(res))
        detail += [dict(cell=cell, **r) for r in res["records"]]
        print(f"{cell}: {res['status']}  ({len(res['records'])} comparisons, "
              f"{sum(1 for r in res['records'] if not r['ok'])} outside {RTOL:g}, {res['wall_s']:.1f} s)", flush=True)
        for r in res["records"]:
            if not r["ok"]:
                print(f"    {r['kind']} {r['name']}: rel {r['rel']:.3e}  {r['note']}", flush=True)
    if args.out and rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0])
        out.write_text("\n".join(["\t".join(cols)] + ["\t".join(_fmt(r[c]) for c in cols) for r in rows]) + "\n")
        dcols = ["cell", "kind", "name", "rel", "ok", "note"]
        out.with_name(out.stem + "_detail.tsv").write_text(
            "\n".join(["\t".join(dcols)] + ["\t".join(_fmt(r[c]) for c in dcols) for r in detail]) + "\n")
    n_fail = sum(1 for r in rows if r["status"] != "PASS")
    verdict = "INCOMPLETE" if incomplete else ("FAIL" if n_fail else "PASS")
    print(f"parity: {verdict} -- {len(rows) - n_fail} of {len(cells)} cells PASS, {n_fail} FAIL, "
          f"{len(incomplete)} not run {incomplete if incomplete else ''}")
    return 2 if incomplete else (1 if n_fail else 0)


def _fmt(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return str(v).replace("\t", " ").replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())
