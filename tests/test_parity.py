"""Parity gate: every reference cell through ``microsolver()`` against the parent's values.

The references (``tests/references/*.npz``) hold the inputs and results of single cells computed
by the parent repository (``MultiscaleMixedBallinSocket``, commit ``2aaccad``) at full double
precision.  ``tools/make_references.py`` writes them.

Each cell is solved with the mesh, settings, roughness, macro state (``Zeta.from_task``), time
window, ``n_steps`` and offset stored in its reference.  Compared:

* Homogenised outputs and field diagnostics.  Each reference result field is read from where a
  ``MicroResult`` holds it (``RESULT_NOW``).  Floats and vector components pass when
  ``|a - b| <= 1e-12 * max(|a|, |b|)``.  A NaN must be a NaN on both sides.  Integers and
  strings must be equal.
* Fields (SI arrays and the DG0 fluid indicator): same shape, same NaN pattern and
  ``max|a - b| <= 1e-12 * max|reference|``.  ``p_fluid_masked`` is formed here from ``p_fluid``
  and ``chi_fluid`` (``DERIVED_FIELDS``).
* Settings.  Every setting stored in the reference must equal the value the solver reads
  (``SETTINGS_NOW``), exactly.  A setting the solver does not have must be stored at the value
  fixed in the code (``REMOVED_SETTINGS``) or map onto its replacement (``REPLACED_SETTINGS``).
  The ripple settings of a reference without ripple pass when the settings have no ripple.
* ``case.k_spring``.  The solver has no cell-average spring.  The check passes when the spring
  did not act on the reference cell (``spring_acted``).

Not compared:

* timings and bookkeeping: ``wall_time``, ``contact_eval_s``, ``pid``, ``task_id``, ``task``;
* outputs of the two-tier mesh predictor and of the contact table, which the API does not have:
  ``h_geom_min``, ``escalated``, ``pasp_table``, ``pasp_coupled``, ``pasp_contact_only``,
  ``kasp_table``, ``amech_table``, ``offset_x_m``, ``offset_y_m``, ``phase_average``;
* the eleven global-x shear statistics (``REMOVED_RESULT_FIELDS``), which need the macro frame.
  A reference that holds a finite one fails.  A reference status ``no_frame`` is replaced by the
  status that its support fractions give (``STATUS_WITHOUT_FRAME``).

For the pair cell, ``Pasp_hom`` and ``k_asp_hom`` are compared with the coupled values stored in
the reference (``coupled_json``), and ``contact_iters`` is not compared.

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

#: Reference settings the solver does not have -> the value fixed in the code.  A reference that
#: stores the setting must store this value.
REMOVED_SETTINGS = {
    "spring_pointwise": False,          # pointwise spring mode
    "use_reduced_modulus": False,       # the FFT contact law takes False only
    "overclosure_measure": "patch_max",  # the FFT contact law takes "patch_max" only
    "gauge_lift": False,                # not read by the solver
}

#: Reference settings read only by a contact law the solver does not have.  The check is that the
#: contact law of the reference is ``KEPT_CONTACT_MODEL``.
REMOVED_UNREAD_SETTINGS = {
    "contact_E": "legacy_patch_v1 only (Young's modulus of its closure)",
    "contact_nu": "legacy_patch_v1 only (with use_reduced_modulus)",
    "k_asp_exponent": "legacy_patch_v1 only (its closure tangent; fft_elastic_v1 uses contact_tangent)",
}
KEPT_CONTACT_MODEL = "fft_elastic_v1"

#: Reference settings held under another name and type:
#: reference key -> (built key, {reference value: built value}).
REPLACED_SETTINGS = {
    "phase_average": ("pair_branch", {"off": False, "contact_table_v1": True}),
}
_REPLACEMENT_KEYS = {new for new, _ in REPLACED_SETTINGS.values()}

#: Reference setting -> where the solver reads the value:
#:   "settings.<path>"  a field of ``MicroSettings`` (``contact.`` and ``ripple.`` nested)
#:   "fixed.<NAME>"     a module constant of ``microcell.solver_base``
#:   "contact law"      ``microcell.contact.FFT``
#:   "ripple on"        "ripple_halfspace" when ``MicroSettings.ripple`` is set, else "off"
#:   "pair"             True when the roughness is a stationary + translating pair
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
    # mixed and contact solver classes
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
#: read only with ripple on; a reference without ripple stores unused defaults
RIPPLE_SETTINGS = ("fluid_E_star", "ripple_tol", "ripple_maxpass", "ripple_omega")
#: stored only by a reference with contact
CONTACT_CLASS_SETTINGS = ("pattern_flux_contact", "sigma_yield", "k_asp_mode", "contact_model", "contact_E_star",
                          "contact_tol", "contact_maxit", "mu_material", "pair_branch")

#: Cell-average spring of the reference case (``k_spring``).  In a cell with contact the spring is
#: off at 0 and at this value or above.  In a full-film cell it acts in a transient solve with
#: ``k_spring > 0``.
SPRING_OFF_IN_CONTACT_CLASS = 1e19

#: ``contact_model`` of a full-film result: reference tag -> the tag the API reports
FULL_FILM_CONTACT_TAG = {"legacy_patch_v1": "off"}

#: Reference result field -> where a ``MicroResult`` holds it: a named output or ``info.<key>``.
#: A field not listed is ``info[<same name>]``.  With one phase per call the sample and phase-mean
#: pressure extrema equal ``p_max`` and ``p_min`` and are compared with them.
RESULT_NOW = {
    "delta_Q": "dQ", "delta_P": "dP", "Q": "Q", "Pst": "Pst", "P_f": "P_f", "Fst": "Fst",
    "tau_res": "tau_fluid",
    "shear_asperity_mean_local_Pa": "tau_asperity", "shear_total_mean_local_Pa": "tau_total",
    "Pasp_hom": "P_asp", "contact_frac": "contact_frac", "A_mech_frac": "A_mech_frac", "k_asp_hom": "k_asp",
    "p_max": "p_max", "p_min": "p_min", "h_max": "h_max", "h_min": "h_min", "yield_flag": "yield_flag",
    "p_fluid_sample_max_Pa": "p_max", "p_max_phase_mean": "p_max",     # equal to p_max with one phase per call
    "p_fluid_sample_min_Pa": "p_min", "p_min_phase_mean": "p_min",
    "failure_reason": "reason",
    "mesh_used": "info.mesh_n",
}
#: ``info`` entries that are not reference result fields: the call and the field diagnostics
INFO_OF_THE_CALL = ("steady", "t_end", "n_steps", "offset", "roughness_velocity_ratio", "t_start", "dt",
                    "fields", "paraview")

#: Reference result fields a ``MicroResult`` does not have: the global-x envelopes of the shear
#: tractions, which need the macro frame.  Each is NaN in every reference.  Not compared; a
#: reference that holds a finite one, or a result that holds one, fails.
REMOVED_RESULT_FIELDS = (
    "shear_fluid_gx_min_Pa", "shear_fluid_gx_max_Pa", "shear_fluid_gx_sample_mean_Pa",
    "shear_fluid_support_gx_min_Pa", "shear_fluid_support_gx_max_Pa",
    "shear_asperity_gx_min_Pa", "shear_asperity_gx_max_Pa", "shear_asperity_gx_sample_mean_Pa",
    "shear_total_gx_min_Pa", "shear_total_gx_max_Pa", "shear_total_gx_sample_mean_Pa",
)
#: Status key -> support fractions of the reference result.  A reference status ``no_frame`` is
#: compared as ``ok`` when one of the fractions is positive, else as ``no_support``.  Any other
#: reference status is compared unchanged.
STATUS_WITHOUT_FRAME = {
    "shear_extrema_status": ("shear_fluid_support_fraction",),
    "shear_asperity_status": ("shear_asperity_support_fraction",),
    "shear_total_status": ("shear_fluid_support_fraction", "shear_asperity_support_fraction"),
}
#: Reference fields a ``MicroResult`` does not hold, formed here from the fields it holds
DERIVED_FIELDS = {
    "p_fluid_masked": lambda f: np.where(f["chi_fluid"] > 0.5, f["p_fluid"], np.nan),
}
_MISSING = object()


def result_now(out, key: str):
    """Reference result field ``key`` as the ``MicroResult`` ``out`` holds it (``_MISSING`` when absent)."""
    where = RESULT_NOW.get(key, "info." + key)
    if where.startswith("info."):
        return out.info.get(where[len("info."):], _MISSING)
    return getattr(out, where)


def status_without_frame(result: dict, key: str) -> str:
    """Reference status ``result[key]``, with ``no_frame`` replaced as ``STATUS_WITHOUT_FRAME`` states."""
    want = result[key]
    if key in STATUS_WITHOUT_FRAME and want == "no_frame":
        return "ok" if any(float(result[f]) > 0.0 for f in STATUS_WITHOUT_FRAME[key]) else "no_support"
    return want


def spring_acted(ref: dict) -> bool:
    """True when the cell-average spring of the reference case acted on the cell."""
    k = float(ref["micro_case"]["k_spring"])
    if bool(ref["micro_case"]["mixed_enabled"]):
        return 0.0 < k < SPRING_OFF_IN_CONTACT_CLASS
    return k > 0.0 and bool(ref["meta"]["transient"])


def reference_cells():
    """Ids of the reference files in ``tests/references``, sorted."""
    return sorted(p.stem for p in REFERENCES.glob("*.npz"))


def load_reference(cell: str) -> dict:
    """Reads ``<cell>.npz``.  Returns the JSON blocks by name, the ``task`` vector and the
    ``fields`` and ``surfaces`` arrays."""
    d = np.load(REFERENCES / f"{cell}.npz")
    ref = {k[:-5]: json.loads(str(d[k])) for k in d.files if k.endswith("_json")}
    ref["task"] = np.asarray(d["task"], dtype=float)
    ref["fields"] = {k[len("field__"):]: np.asarray(d[k]) for k in d.files if k.startswith("field__")}
    ref["fields"]["chi_fluid_dg0"] = np.asarray(d["chi_fluid_dg0"])
    ref["surfaces"] = {k.split("__")[1]: np.asarray(d[k]) for k in d.files if k.startswith("surface__")}
    return ref


def settings_from_reference(ref: dict):
    """``MicroSettings`` from the case and settings stored in the reference.

    ``E_star`` is the contact modulus when the reference has contact, else the ripple modulus;
    the gate compares both reference moduli with it.  ``beta`` is not passed: it is a fixed
    value of the solver, checked through ``SETTINGS_NOW``.
    """
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
    """Reference setting name -> the value the solver reads (``SETTINGS_NOW``).

    st: the ``MicroSettings`` of the call.  pair: True for a stationary + translating pair.
    Ripple settings are absent without ripple, and contact settings without contact.
    """
    import microcell.solver_base as solver_module
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
    """``Roughness`` from the surface arrays stored in the reference, or from its cosine parameters."""
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
    """Largest ``|a - b| / max(|a|, |b|)`` over the components (0 where equal)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.maximum(np.abs(a), np.abs(b))
        r = np.where(a == b, 0.0, np.abs(a - b) / scale)
    return float(np.max(r)) if r.size else 0.0


def compare_scalar(name: str, want, got) -> dict:
    """Compares one output (string, integer, float or short vector) with the reference value ``want``.

    Returns a record ``{"name", "kind", "rel", "ok", "note"}``; ``rel`` is the relative difference.
    """
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
    """Compares one field array with the reference array ``want``: shape, NaN pattern and
    ``max|got - want| / max|want|``.  Returns a record as ``compare_scalar`` does."""
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
    """Solves one reference cell through ``microsolver()`` and compares it with the reference.

    Returns ``{"cell", "records", "status", "wall_s"}``: one record per comparison, and
    ``status`` "PASS" or "FAIL".
    """
    from microcell import Zeta, microsolver

    ref = load_reference(cell)
    meta = ref["meta"]
    if meta["status"] != "OK":
        raise RuntimeError(f"{cell}: the reference itself is {meta['status']}")
    st = settings_from_reference(ref)
    rough = roughness_from_reference(ref)
    records = []

    # settings stored in the reference against the values the solver reads
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
                r["note"] = f"removed setting: the reference stores {want[k]!r}, the code fixes {REMOVED_SETTINGS[k]!r}"
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
    if st.contact is None:                          # full film: map the reference tag
        result["contact_model"] = FULL_FILM_CONTACT_TAG.get(result["contact_model"], result["contact_model"])
    if pair:
        for key, src in PAIR_FROM_COUPLED.items():
            result[key] = ref["coupled"][src]
    for key in sorted(result):
        if key in NOT_COMPARED or (pair and key in PAIR_NOT_COMPARED):
            continue
        if key in REMOVED_RESULT_FIELDS:            # recorded only on failure
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
    # info may hold only reference result fields and the entries of the call
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
    """One summary row for a cell: comparison counts and the largest relative difference of each kind."""
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


# ---- pytest form ----
def pytest_generate_tests(metafunc):
    """Parametrises ``cell`` over the reference cells."""
    if "cell" in metafunc.fixturenames:
        metafunc.parametrize("cell", reference_cells())


def test_references_present():
    assert len(reference_cells()) >= 6, f"expected the reference cells in {REFERENCES}"


def test_parity(cell):
    """Every comparison of the cell passes the rule of the module docstring."""
    res = run_cell(cell)
    bad = [f"{r['name']}: rel {r['rel']:.3e} {r['note']}" for r in res["records"] if not r["ok"]]
    assert not bad, f"{cell}: {len(bad)} of {len(res['records'])} comparisons outside 1e-12:\n" + "\n".join(bad)


def test_generated_surface_reproduces_reference():
    """``Roughness.generated(Sa=40e-9, seed=1000)`` matches the stationary surface stored in
    reference c1, to a relative 1e-12 of its largest height."""
    from microcell import Roughness

    ref = load_reference("c1_gensa40_h200_ripple")
    z_ref = ref["surfaces"]["stationary"]
    fld = Roughness.generated(Sa=40e-9, seed=1000).field()
    z = fld.members["stationary"][0].source
    r = compare_field("generated_z", z_ref, np.asarray(z))
    assert r["ok"], r


def test_comparator_sees_a_perturbed_solve(monkeypatch):
    """Negative control: with the film of one cell raised by a relative 1e-9, the gate fails on
    homogenised outputs and on fields, and on no setting."""
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
    """Negative control of ``REMOVED_SETTINGS``: when the value stored in a reference differs from
    the fixed one, the gate fails on that setting only."""
    monkeypatch.setitem(REMOVED_SETTINGS, "spring_pointwise", True)
    res = run_cell("c8_cosine_fullfilm_steady")
    assert [r["name"] for r in res["records"] if not r["ok"]] == ["settings.spring_pointwise"]


def test_removed_spring_is_checked(monkeypatch):
    """Negative control of ``case.k_spring``: a transient full-film reference with a positive
    ``k_spring`` fails on ``case.k_spring`` only."""
    original = load_reference

    def with_spring(cell):
        ref = original(cell)
        ref["micro_case"] = dict(ref["micro_case"], k_spring=1e20)
        return ref

    monkeypatch.setattr(sys.modules[__name__], "load_reference", with_spring)
    res = run_cell("c5_cosine_fullfilm")
    assert [r["name"] for r in res["records"] if not r["ok"]] == ["case.k_spring"]


# ---- script form ----
def main(argv=None) -> int:
    """Runs the gate on the cells of ``--cell`` (default: all), prints one line per cell and
    writes the TSV reports of ``--out``.  Returns the exit code."""
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
        except Exception:                                    # not run: INCOMPLETE, not FAIL
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
    """Text of one TSV cell: ``repr`` of a float, else the string without tabs or newlines."""
    if isinstance(v, float):
        return repr(v)
    return str(v).replace("\t", " ").replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())
