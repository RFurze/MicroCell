#!/usr/bin/env python
"""Reference cells for the MicroCell parity gate, made from the parent repository.

Each reference is one micro cell solved by the parent's production worker
(``scripts/run/run_microscale_phase_neutral.run_micro_task_phase_neutral``), called the
way ``scripts/run/run_micro_cells.run_cell`` calls it, and saved at full double precision
as ``<out>/<cell>.npz``.  ``cells.tsv`` of ``run_micro_cells.py`` holds six significant
figures, which cannot carry a 1e-12 parity.

Run it from the PARENT's root (the parent resolves ``config/`` from the working
directory), in WSL with the parent's environment:

    cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MultiscaleMixedBallinSocket
    source scripts/ehl_env.sh
    python ../MicroCell/tools/make_references.py --list
    python ../MicroCell/tools/make_references.py --all --procs 8

Nothing is written inside the parent: the references go to ``--out`` (default
``MicroCell/tests/references``), the prepared surfaces and the contact table to
``--store`` (default ``RefactorRefactor/_to_delete/micro_standalone_2026-11-11/surface_store``,
used through the parent's own ``EHL_SURFACE_STORE``), and bytecode is not written.

One cell per process: a cached solver re-targeted by ``reset_task`` cannot enter a
reference.  ``--all`` starts one child process per cell.

What a reference holds (JSON strings keep every float exactly):

    meta_json        cell id, case file, variant, in-memory changes, parent commit, status,
                     failure reason, transient flag, offset, n_steps, cfl_target, the NaN audit
    task             the task tuple (20 values transient, 13 steady)
    micro_case_json  the MicroCase fields the worker read
    env_json         the MICRO_* environment of the process
    settings_json    the solver settings the worker built (dataclass fields + class name)
    phys_json        the solver's MicroPhysicalParameters
    result_json      every field of the worker's MicroTaskResult, by name
    coupled_json     the solver's own Pasp_hom / k_asp_hom after the coupled solve (for the
                     pair, the values before the contact-table composition)
    field__<name>    the solver's final state on the (n+1) x (n+1) vertex grid, SI units
    chi_fluid_dg0    the DG0 fluid indicator
    info_json        run_micro_cells._extract_fields' consistency diagnostics
    surface_json, surface__<role>__z
                     the member surfaces of a generated-roughness cell (source arrays [m],
                     cell, bandwidth, member offsets), so the parity test needs no store

A NaN in a reference must come with a status (``nan_audit``); an uncovered NaN makes the
reference a failure.  Exit status: 0 = every cell OK, 1 = a cell failed, 2 = incomplete.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
MICROCELL = HERE.parents[1]
WORK = MICROCELL.parent
DEFAULT_OUT = MICROCELL / "tests" / "references"
DEFAULT_STORE = WORK / "_to_delete" / "micro_standalone_2026-11-11" / "surface_store"

PARENT_COMMIT_EXPECTED = "2aaccad"

GENROUGH = "cases/mixed_transfull_cup90_genrough_ripple.py"
WANGPAIR = "cases/wang2008_mixed_genpair_ripple.py"
BASELINE = "cases/refactor_baseline.py"
CUP90 = "cases/mixed_transfull_cup90.py"

#: non-zero gradients and rates of the cells that carry them
RATES = dict(Uy=0.005, gx=2.0e9, gy=-1.0e9, Hdot=-1.0e-7, Pdot=1.0e7, UxT=0.05, UyT=0.01,
             dpdx_dot=1.0e9, dpdy_dot=-5.0e8)

#: id -> the cell.  ``changes``: dataclasses.replace() of the case's MicroCase, in memory
#: ("@gensa40:<field>" takes the value from the GenSa40-ripple MicroCase).
CELLS = {
    "c1_gensa40_h200_ripple": dict(
        case=GENROUGH, variant="GenSa40-ripple", H=200e-9, P=40.8e6,
        covers="generated surface, ripple on, no contact"),
    "c2_gensa40_h130_ripple": dict(
        case=GENROUGH, variant="GenSa40-ripple", H=130e-9, P=40.8e6,
        covers="generated surface, ripple on, contact"),
    "c3_gensa40_h130_noripple": dict(
        case=GENROUGH, variant="GenSa40-ripple", H=130e-9, P=40.8e6,
        changes={"fluid_deformation": "off", "fluid_E_star": None},
        covers="generated surface, ripple off, contact"),
    "c4_wangpair10_h25_offset": dict(
        case=WANGPAIR, variant="Wang2008-GenPair10-ripple-table", H=25e-9, P=40.8e6,
        offset=(1.25e-6, 0.625e-6), rates=True,
        covers="generated pair at a set offset, ripple on, fluid_mean_v1, gradients and rates"),
    "c5_cosine_fullfilm": dict(
        case=BASELINE, variant="ms", H=120e-9, P=40.8e6, rates=True,
        # WP2 (Rob, 11 Nov 2026): the case's k_spring = 1e20 "off" value still ran the
        # cell-average spring loop in a transient full-film cell; the reference is made
        # with the spring really off, and the standalone has no spring
        changes={"k_spring": 0.0},
        covers="cosine, full film, gradients and rates"),
    "c6_cosine_fft_sqrt": dict(
        case=CUP90, variant="base", H=None, P=40.8e6,
        changes={"contact_model": "fft_elastic_v1", "contact_E_star": "@gensa40:contact_E_star",
                 "spos_form": "sqrt"},
        H_rule="Ah + h_min - 5 nm",
        covers="cosine with fft_elastic_v1 contact, spos_form sqrt"),
    "c7_gensa40_h130_ripple_steady": dict(
        case=GENROUGH, variant="GenSa40-ripple", H=130e-9, P=40.8e6, steady=True,
        covers="steady solve: generated surface, ripple on, contact"),
    "c8_cosine_fullfilm_steady": dict(
        case=BASELINE, variant="ms", H=120e-9, P=40.8e6, steady=True, gradients=True,
        covers="steady solve: cosine, full film, gradients"),
    # open item 8 (Rob, 11 Nov 2026): added, the eight above are not re-made
    "c9_gensa10_h26_rates": dict(
        case=GENROUGH, variant="GenSa10-ripple-tdfc", H=26e-9, P=40.8e6, rates=True,
        covers="one generated surface (Sa 10 nm, the pendulum production recipe), ripple on, contact, "
               "gradients and rates"),
}

U_DEFAULT = 0.02
DT_DEFAULT = 0.02


# ---------------------------------------------------------------------------------------
# the parent tree
# ---------------------------------------------------------------------------------------
def _parent_root() -> Path:
    root = Path.cwd()
    if not ((root / "ehl").is_dir() and (root / "config").is_dir() and (root / "scripts" / "run").is_dir()):
        sys.exit("ERROR: run this from the parent repository's root (MultiscaleMixedBallinSocket)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _parent_commit(root: Path) -> str:
    """The parent's HEAD, read from its .git files (git itself must not run in WSL here)."""
    try:
        git = root / ".git"
        if git.is_file():                        # a worktree: "gitdir: <path>"
            return "worktree"
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            f = git / ref
            if f.is_file():
                return f.read_text().strip()
            for line in (git / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
            return ""
        return head
    except Exception:
        return ""


def _stray_micro_env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith("MICRO_")}


# ---------------------------------------------------------------------------------------
# the surface store (outside the parent)
# ---------------------------------------------------------------------------------------
def _fill_store(root: Path, store: Path) -> None:
    """Copy the parent's prepared surfaces and contact tables into ``store`` (never the
    other way), so the pair's members are the bytes its table was built on."""
    src = root / "data" / "surfaces"
    (store / "tables").mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")) + sorted(src.glob("*.npz")):
        if not (store / f.name).exists():
            shutil.copy2(f, store / f.name)
    for f in sorted((src / "tables").glob("*")):
        if f.is_file() and not (store / "tables" / f.name).exists():
            shutil.copy2(f, store / "tables" / f.name)


def _load_case(spec: dict):
    from ehl import caseio
    cf = caseio.load(spec["case"])
    return cf, cf.get(spec["variant"])


def _prepare_surfaces(store: Path) -> None:
    """Prepare (idempotent) every generated surface the cells need, into ``store``."""
    from ehl.roughness.case import prepare_arrangement
    done = set()
    for spec in CELLS.values():
        key = (spec["case"], spec["variant"])
        if key in done:
            continue
        done.add(key)
        _cf, case = _load_case(spec)
        if getattr(case.roughness, "kind", None) == "surface":
            prepare_arrangement(case.roughness, store)


def _micro_case(spec: dict):
    """(case, MicroCase as the case has it, MicroCase of the cell, the changes applied)."""
    from ehl.micro_case import MicroCase
    _cf, case = _load_case(spec)
    mc0 = MicroCase.from_case(case)
    changes = {}
    for k, v in (spec.get("changes") or {}).items():
        if isinstance(v, str) and v.startswith("@gensa40:"):
            _c, gen = _load_case(CELLS["c1_gensa40_h200_ripple"])
            v = getattr(MicroCase.from_case(gen), v.split(":", 1)[1])
        changes[k] = v
    mc = dataclasses.replace(mc0, **changes) if changes else mc0
    return case, mc0, mc, changes


def _film(spec: dict, mc) -> float:
    if spec["H"] is not None:
        return float(spec["H"])
    return float(mc.Ah) + float(mc.h_min) - 5e-9          # H_rule "Ah + h_min - 5 nm"


def _task(spec: dict, mc, index: int) -> tuple:
    r = dict(Uy=0.0, gx=0.0, gy=0.0, Hdot=0.0, Pdot=0.0, UxT=0.0, UyT=0.0, dpdx_dot=0.0, dpdy_dot=0.0)
    if spec.get("rates"):
        r.update(RATES)
    if spec.get("gradients"):
        r.update(Uy=RATES["Uy"], gx=RATES["gx"], gy=RATES["gy"])
    H = _film(spec, mc)
    # the 20-element transient task of coupling_helper_fns.build_task_list_transient
    # (run_micro_cells._ladder_task), with the rate slots filled
    t = (index, index, float(H), float(spec["P"]), float(U_DEFAULT), float(r["Uy"]), 0.0,
         float(r["gx"]), float(r["gy"]), 0.0, 0.0, 0.0, 0.0, float(r["Hdot"]), float(r["Pdot"]),
         float(DT_DEFAULT), float(r["UxT"]), float(r["UyT"]), float(r["dpdx_dot"]), float(r["dpdy_dot"]))
    return t[:13] if spec.get("steady") else t


# ---------------------------------------------------------------------------------------
# NaN audit
# ---------------------------------------------------------------------------------------
_NAN_COVER = (
    # (field prefix, status field, values of the status under which the field is measured)
    ("shear_fluid_", "shear_extrema_status", ("ok",)),
    ("shear_asperity_", "shear_asperity_status", ("ok",)),
    ("shear_total_", "shear_total_status", ("ok",)),
    ("shear_contact_", "shear_asperity_status", ("ok",)),
    ("p_fluid_sample_", "pressure_extrema_status", ("ok",)),
    ("p_max_phase_mean", "pressure_extrema_status", ("ok",)),
    ("p_min_phase_mean", "pressure_extrema_status", ("ok",)),
    ("pasp_", "phase_average", ("contact_table_v1",)),
    ("kasp_table", "phase_average", ("contact_table_v1",)),
    ("amech_table", "phase_average", ("contact_table_v1",)),
    ("offset_", "phase_average", ("contact_table_v1",)),
)


def nan_audit(result: dict) -> tuple[list, list]:
    """(covered, uncovered): the NaN floats of a result and the status that explains each."""
    covered, uncovered = [], []
    for name, v in result.items():
        vals = np.ravel(np.asarray(v, dtype=float)) if isinstance(v, (float, list, tuple)) else None
        if vals is None or not np.isnan(vals).any():
            continue
        why = None
        for prefix, status_field, measured in _NAN_COVER:
            if name.startswith(prefix) and str(result.get(status_field)) not in measured:
                why = f"{status_field}={result.get(status_field)}"
                break
        (covered if why else uncovered).append((name, why))
    return covered, uncovered


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(q) for q in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(q) for k, q in v.items()}
    return v


def _dc(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return dict(vars(obj))


# ---------------------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------------------
def run_cell(cell_id: str, out: Path, store: Path) -> int:
    root = _parent_root()
    spec = CELLS[cell_id]
    stray = _stray_micro_env()
    if stray:
        print(f"ERROR: MICRO_* variables are set in this shell: {stray}; unset them (a reference "
              "records only what this script sets)")
        return 2
    os.environ["EHL_SURFACE_STORE"] = str(store)

    from ehl import launch
    case, mc0, mc, changes = _micro_case(spec)

    # ---- the environment the worker reads: the case's own exports, then run_micro_cells' ----
    exports = launch.env_exports(case)
    for k, v in exports.items():
        if k.startswith("MICRO_") and k not in ("MICRO_NPROCS", "MICRO_PAIR_XWARM"):
            os.environ[k] = str(v)
    os.environ["MICRO_SOLVER_REUSE"] = "1"
    os.environ["MICRO_SKIP_FAILURES"] = "1"
    if mc.contact_model == "fft_elastic_v1" and bool(mc.mixed_enabled):
        os.environ.setdefault("MICRO_CONTACT_TOL", "1e-11")
    env_rec = dict(sorted(_stray_micro_env().items()))
    for k in ("MICRO_NSTEPS", "MICRO_CFL_TARGET", "MICRO_TEND", "MICRO_MESH_N", "MICRO_MESH_N_M",
              "MICRO_MESH_N_H", "MICRO_MESH_N_CONTACT", "MICRO_DPHI_SAMPLES"):
        if k in env_rec:
            print(f"ERROR: the case exports {k}={env_rec[k]}; this script does not model it")
            return 2

    from scripts.run import run_micro_cells as rmc
    from scripts.run import run_microscale_phase_neutral as pn

    rmc._init_worker(True)                  # macro process settings + the micro quadrature pin
    pn.set_micro_case(mc, workers_get_macro_settings=True)
    if mc.surface is not None:
        pn._surface_field(mc)

    offset = spec.get("offset")
    table_rec = None
    if str(getattr(mc, "phase_average", "off")) == "contact_table_v1":
        from ehl import contact_table as CT
        table = CT.for_case(case, store=store)
        pn.set_contact_table(str(table.npz), table.key)
        table_rec = {"key": table.key, "npz": str(table.npz), "floor_m": table.floor_m,
                     "first_contact_m": table.first_contact_m}
        if offset is None:
            print("ERROR: a contact-table cell needs an offset")
            return 2

    transient = not bool(spec.get("steady"))
    task = _task(spec, mc, list(CELLS).index(cell_id))
    settings = pn._build_phase_neutral_settings()

    t0 = time.time()
    res = pn.run_micro_task_phase_neutral(task, transient=transient, macro_dt=None, export_vtk=False,
                                          force_fine=True, offset=offset)
    wall = time.time() - t0
    result = {k: _jsonable(v) for k, v in res._asdict().items()}
    ok = (not res.failure_reason) and bool(np.isfinite(res.delta_P))
    covered, uncovered = nan_audit(result) if ok else ([], [])
    status = "OK" if ok else "FAIL"
    reason = str(res.failure_reason)
    if ok and uncovered:
        status, reason = "FAIL", "NaN without a status: " + ", ".join(n for n, _ in uncovered)

    arrays = {"task": np.asarray(task, dtype=float)}
    coupled, phys, info = {}, {}, {}
    if ok:
        s = rmc._cached_solver(pn, mc, res.mesh_used)
        fields, cell_fields, info = rmc._extract_fields(s, mc)
        for name, G in fields.items():
            if name.endswith("_MPa"):
                arrays["field__" + name[:-4]] = G / rmc.MPA
            elif name.endswith("_nm"):
                arrays["field__" + name[:-3]] = G / rmc.NM
            else:
                arrays["field__" + name] = np.asarray(G, dtype=float)
        bad = [k for k, G in arrays.items() if k.startswith("field__") and k != "field__p_fluid_masked"
               and np.isnan(G).any()]
        if bad:
            status, reason = "FAIL", "NaN in fields: " + ", ".join(bad)
        arrays["chi_fluid_dg0"] = np.asarray(cell_fields["chi_fluid_dg0"], dtype=float)
        phys = _jsonable(_dc(s.params))
        if bool(mc.mixed_enabled):
            coupled = {"Pasp_hom": float(s.Pasp_hom), "k_asp_hom": float(s.k_asp_hom),
                       "A_mech_frac": float(getattr(s, "A_mech_frac", float("nan"))),
                       "contact_frac": float(s.contact_frac)}
            if table_rec is not None and not (coupled["Pasp_hom"] == float(res.pasp_coupled)):
                status = "FAIL"
                reason = (f"the solver's Pasp_hom {coupled['Pasp_hom']!r} is not the worker's "
                          f"pasp_coupled {float(res.pasp_coupled)!r}")

    surface_rec = None
    if mc.surface is not None:
        from ehl.roughness import assets as A
        from ehl.roughness.case import _abs
        d = json.loads(mc.surface)
        surface_rec = {"arrangement": d["arrangement"], "cell_m": d["cell_m"],
                       "bandwidth": d.get("reconstruction", {}).get("bandwidth", {}),
                       "evaluator": d.get("reconstruction", {}).get("evaluator"),
                       "envelope_m": d["envelope_m"], "members": {}}
        for role in ("stationary", "translating"):
            m = d["members"].get(role)
            if m is None:
                continue
            p = A.load_artefact(_abs(m["artefact"]))
            arrays[f"surface__{role}__z"] = np.asarray(p.z, dtype=float)
            surface_rec["members"][role] = {"offset_m": m["offset_m"], "Lx": float(p.Lx), "Ly": float(p.Ly),
                                            "shape": list(p.z.shape), "array_sha256": m["array_sha256"],
                                            "recipe_id": m["recipe_id"]}

    meta = {
        "cell": cell_id, "covers": spec["covers"], "case_file": spec["case"], "variant": spec["variant"],
        "case_name": case.name, "changes": _jsonable(changes), "parent_commit": _parent_commit(root),
        "status": status, "failure_reason": reason, "transient": transient,
        "offset": None if offset is None else [float(offset[0]), float(offset[1])],
        "n_steps": int(mc.n_steps), "cfl_target": float(mc.cfl_target), "mesh_used": int(res.mesh_used),
        "force_fine": True, "wall_s": wall, "contact_table": table_rec,
        "nan_covered": covered, "nan_uncovered": uncovered,
        "case_env_exports": {k: str(v) for k, v in exports.items()},
        "made": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "numpy": np.__version__, "python": sys.version.split()[0],
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{cell_id}.npz",
        meta_json=json.dumps(meta), micro_case_json=json.dumps(_jsonable(mc.fields())),
        env_json=json.dumps(env_rec),
        settings_json=json.dumps({"__class__": type(settings).__name__, **_jsonable(_dc(settings))}),
        phys_json=json.dumps(phys), result_json=json.dumps(result), coupled_json=json.dumps(coupled),
        info_json=json.dumps(_jsonable(info)), surface_json=json.dumps(surface_rec), **arrays)
    print(f"[reference] {cell_id}: {status}  mesh {int(res.mesh_used)}  wall {wall:.1f} s  "
          f"dP {float(res.delta_P)!r}  Pasp {float(res.Pasp_hom)!r}  contact_frac {float(res.contact_frac)!r}"
          + (f"  reason {reason[:200]}" if status != "OK" else ""), flush=True)
    return 0 if status == "OK" else 1


# ---------------------------------------------------------------------------------------
# list / all
# ---------------------------------------------------------------------------------------
def list_cells(store: Path) -> int:
    root = _parent_root()
    os.environ["EHL_SURFACE_STORE"] = str(store)
    _fill_store(root, store)
    _prepare_surfaces(store)
    cols = ("cell", "case", "variant", "mode", "H_nm", "P_MPa", "U", "dt", "offset_m", "mesh", "roughness",
            "Ah_nm", "h_min_m", "contact", "ripple", "spos", "level", "phase_avg", "n_steps")
    print("\t".join(cols))
    for cid, spec in CELLS.items():
        _case, _mc0, mc, _ch = _micro_case(spec)
        mixed = bool(mc.mixed_enabled)
        row = (cid, spec["case"], spec["variant"], "steady" if spec.get("steady") else "transient",
               f"{_film(spec, mc) * 1e9:.4g}", f"{spec['P'] * 1e-6:.4g}", U_DEFAULT,
               0 if spec.get("steady") else DT_DEFAULT, spec.get("offset", "-"),
               int(mc.mesh_n_contact) if (mixed and float(mc.h_min) > 0.0) else int(mc.mesh_n),
               ("surface:" + json.loads(mc.surface)["arrangement"]) if mc.surface else f"cosine:{mc.roughness_mode}",
               f"{float(mc.Ah) * 1e9:.4g}", f"{float(mc.h_min):.5g}" if mixed else "-",
               mc.contact_model if mixed else "full film", mc.fluid_deformation, mc.spos_form,
               mc.level_constraint, mc.phase_average, int(mc.n_steps))
        print("\t".join(str(v) for v in row))
    print(f"parent commit {_parent_commit(root)[:7]} (expected {PARENT_COMMIT_EXPECTED})")
    return 0


def run_all(out: Path, store: Path, procs: int, only: list | None) -> int:
    root = _parent_root()
    commit = _parent_commit(root)
    if not commit.startswith(PARENT_COMMIT_EXPECTED):
        print(f"ERROR: the parent is at {commit[:7] or '?'}, not {PARENT_COMMIT_EXPECTED}")
        return 2
    os.environ["EHL_SURFACE_STORE"] = str(store)
    _fill_store(root, store)
    _prepare_surfaces(store)
    ids = only or list(CELLS)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    pending, running, codes = list(ids), {}, {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < max(1, min(int(procs), 8)):
            cid = pending.pop(0)
            log = open(logs / f"{cid}.log", "w")
            p = subprocess.Popen([sys.executable, str(HERE), "--cell", cid, "--out", str(out), "--store", str(store)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(root))
            running[cid] = (p, log)
            print(f"[reference] started {cid}", flush=True)
        for cid, (p, log) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                codes[cid] = rc
                del running[cid]
                tail = [ln for ln in (logs / f"{cid}.log").read_text(errors="replace").splitlines()
                        if ln.startswith("[reference]") or ln.startswith("ERROR")]
                print(f"[reference] finished {cid}: exit {rc}  " + (tail[-1] if tail else "(no summary line)"),
                      flush=True)
        time.sleep(0.5)
    n_bad = sum(1 for c in codes.values() if c == 1)
    n_inc = sum(1 for c in codes.values() if c not in (0, 1))
    print(f"[reference] {len(codes)} cells in {time.time() - t0:.0f} s: {len(codes) - n_bad - n_inc} OK, "
          f"{n_bad} failed, {n_inc} incomplete -> {out}")
    return 2 if n_inc else (1 if n_bad else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--list", action="store_true", help="print the cell table (no solve)")
    ap.add_argument("--all", action="store_true", help="every cell, one process each")
    ap.add_argument("--cell", action="append", help="one cell id (repeat with --all to run a subset)")
    ap.add_argument("--procs", type=int, default=8, help="processes at once with --all (at most 8)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    a = ap.parse_args()
    out, store = Path(a.out).resolve(), Path(a.store).resolve()
    if a.list:
        return list_cells(store)
    if a.all:
        return run_all(out, store, a.procs, a.cell)
    if a.cell and len(a.cell) == 1:
        if a.cell[0] not in CELLS:
            sys.exit(f"ERROR: no cell {a.cell[0]!r}; one of {list(CELLS)}")
        return run_cell(a.cell[0], out, store)
    ap.error("give --list, --all or one --cell")
    return 2


if __name__ == "__main__":
    sys.exit(main())
