"""WP4 step 1: the measured part of the three inventories, at the base (no code edit).

It reads the base tree (68bc829: ``MicroResult.raw``, ``api._failed_raw``, the settings dataclasses),
which groups 2 and 3 removed; to repeat it, run it in ``_to_delete/wp4_2026-11-11/base_source_68bc829/``.

Pytest form so it runs through ``_claude/wsltest`` from ``MicroCell/``:

    ../_claude/wsltest docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/inventory.py -q -p no:cacheprovider

Writes, beside this file:
  inventory_settings.tsv         every field of the nine settings containers and each LEGACY_DEFAULTS key
  inventory_settings_other.tsv   attribute reads of a field NAME on a receiver the scan could not classify
  inventory_results.tsv          every ``raw`` key (and every reference result key)
  inventory_nan_observed.tsv     every NaN seen in a result: the 8 reference cells and forced failures
The condition-and-status table (inventory_nan_paths.tsv) is written by hand from the code, with
this file's observations as its evidence.
"""
import ast
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MICROCELL = HERE.parents[2]
for _p in (MICROCELL, MICROCELL / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCAN = sorted((MICROCELL / "microcell").rglob("*.py")) + sorted((MICROCELL / "tests").glob("*.py")) \
    + sorted((MICROCELL / "examples").glob("*.py"))

#: receiver text (after local-alias resolution) -> which object it is
SOLVER_SETTINGS = {"self.settings", "solver.settings", "s.settings", "solver_settings", "self.solver_settings"}
PARAMS = {"self.params", "solver.params", "s.params", "physical_params", "phys"}
API_SETTINGS = {"st", "settings", "MicroSettings()"}
API_CONTACT = {"st.contact", "settings.contact"}
API_RIPPLE = {"st.ripple", "settings.ripple"}
API_ZETA = {"zeta"}
KNOWN = SOLVER_SETTINGS | PARAMS | API_SETTINGS | API_CONTACT | API_RIPPLE | API_ZETA
#: a function ARGUMENT that is the settings or the params object (the caller passes it):
#: DampingPolicy(s, omega) is called with self.settings (micro_phase_neutral_class.py:1983);
#: SurfaceMotion.set_from_params(p) with the solver's params
ARGUMENT_RECEIVERS = {
    ("microcell/micro_phase_neutral_class.py", "__init__"): {"s": "self.settings"},
    ("microcell/roughness/runtime.py", "set_from_params"): {"p": "self.params"},
}


def _fmt(v):
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return "(" + ", ".join(_fmt(float(x)) if isinstance(x, (float, np.floating)) else str(x)
                               for x in np.asarray(v, dtype=object).ravel()) + ")"
    return str(v).replace("\t", " ").replace("\n", " ")


def _write(name, cols, rows):
    (HERE / name).write_text("\n".join(["\t".join(cols)] + ["\t".join(_fmt(r.get(c, "")) for c in cols)
                                                            for r in rows]) + "\n")


def scan_reads():
    """(file, line, receiver, attr, how) of every ``<recv>.<attr>`` load and ``getattr(<recv>, "<attr>")``
    in the scanned files, with one-letter local aliases (``s = self.settings``) resolved per function,
    and every ``legacy_value("<key>")``."""
    out = []
    for path in SCAN:
        rel = path.relative_to(MICROCELL).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        owner = {}
        for fn in scopes:
            alias = {}
            args = {a.arg for a in fn.args.args}
            for arg, what in ARGUMENT_RECEIVERS.get((rel, fn.name), {}).items():
                if arg in args:
                    alias[arg] = what
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    src = ast.unparse(n.value)
                    if src in KNOWN:
                        alias[n.targets[0].id] = src
            for n in ast.walk(fn):
                owner.setdefault(id(n), alias)          # the outermost function wins (walk order)
        for n in ast.walk(tree):
            alias = owner.get(id(n), {})
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                recv = ast.unparse(n.value)
                out.append((rel, n.lineno, alias.get(recv, recv), n.attr, "attr"))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
                  and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
                recv = ast.unparse(n.args[0])
                dflt = ast.unparse(n.args[2]) if len(n.args) > 2 else ""
                out.append((rel, n.lineno, alias.get(recv, recv), n.args[1].value, f"getattr default {dflt}"))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "legacy_value"
                  and n.args and isinstance(n.args[0], ast.Constant)):
                out.append((rel, n.lineno, "legacy", n.args[0].value, "legacy_value"))
    return out


def _nan_items(prefix, obj):
    """(name, n_nan, n_total) of every float / array / tuple leaf of ``obj`` that holds a NaN."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _nan_items(f"{prefix}{k}" if not prefix or prefix.endswith(".") else f"{prefix}.{k}", v)
        return found
    if isinstance(obj, (str, bytes, bool, type(None))):
        return found
    try:
        a = np.asarray(obj, dtype=float)
    except (TypeError, ValueError):
        return found
    n = int(np.isnan(a).sum())
    if n:
        found.append((prefix, n, int(a.size)))
    return found


def test_inventory(monkeypatch):
    import test_parity as tp
    from microcell import api
    from microcell import ContactSettings, MicroSettings, RippleSettings, Zeta, microsolver
    from microcell import micro_phase_neutral_class as base
    from microcell import micro_phase_neutral_class_contact as con
    from microcell import micro_phase_neutral_class_mixed as mix
    from microcell import ripple as _ripple
    from microcell.contact import ContactResolutionError

    cells = tp.reference_cells()
    short = [c.split("_")[0] for c in cells]

    # ---- run every reference cell once: its settings, its params, its raw --------------------
    captured = {}
    real_params = base.MicroPhysicalParameters

    def run(cell):
        ref = tp.load_reference(cell)
        st = tp.settings_from_reference(ref)
        rough = tp.roughness_from_reference(ref)
        task = ref["task"]
        transient = bool(ref["meta"]["transient"])
        box = {}

        def capture(*a, **k):
            box["phys"] = real_params(*a, **k)
            return box["phys"]

        monkeypatch.setattr(base, "MicroPhysicalParameters", capture)
        try:
            out = microsolver(Zeta.from_task(task), t_end=(float(task[15]) if transient else None),
                              roughness=rough, settings=st, n_steps=int(ref["meta"]["n_steps"]),
                              offset=tuple(ref["meta"]["offset"]) if ref["meta"]["offset"] else (0.0, 0.0),
                              steady=not transient)
        finally:
            monkeypatch.setattr(base, "MicroPhysicalParameters", real_params)
        return dict(ref=ref, st=st, rough=rough, out=out, phys=box["phys"], zeta=Zeta.from_task(task),
                    built=api._solver_settings(st, st.contact is not None, rough.is_pair),
                    transient=transient, task=task)

    for cell in cells:
        captured[cell] = run(cell)
        assert captured[cell]["out"].status == "OK", cell

    # ---- 1. settings ------------------------------------------------------------------------
    reads = scan_reads()
    containers = [
        ("Zeta", Zeta, API_ZETA, lambda c: c["zeta"]),
        ("ContactSettings", ContactSettings, API_CONTACT, lambda c: c["st"].contact),
        ("RippleSettings", RippleSettings, API_RIPPLE, lambda c: c["st"].ripple),
        ("MicroSettings", MicroSettings, API_SETTINGS, lambda c: c["st"]),
        ("MicroPhysicalParameters", base.MicroPhysicalParameters, PARAMS, lambda c: c["phys"]),
        ("MicroSolverSettings", base.MicroSolverSettings, SOLVER_SETTINGS, lambda c: c["built"]),
        ("PhaseNeutralSettings(base)", base.PhaseNeutralSettings, SOLVER_SETTINGS, lambda c: c["built"]),
        ("PhaseNeutralSettings(mixed)", mix.PhaseNeutralSettings, SOLVER_SETTINGS, lambda c: c["built"]),
        ("PhaseNeutralContactSettings", con.PhaseNeutralContactSettings, SOLVER_SETTINGS, lambda c: c["built"]),
    ]
    api_src = (MICROCELL / "microcell" / "api.py").read_text().splitlines()

    def set_at(pattern):
        hits = [str(i + 1) for i, line in enumerate(api_src) if pattern in line]
        return "api.py:" + ",".join(hits) if hits else ""

    own_fields = {}
    seen = set()
    for name, cls, _recv, _get in containers:
        mine = [f for f in dataclasses.fields(cls) if (f.name, "solver" if _recv is SOLVER_SETTINGS else name) not in seen]
        if _recv is SOLVER_SETTINGS:                       # an inherited field belongs to the class that declares it
            mine = [f for f in dataclasses.fields(cls) if f.name not in seen]
            seen.update(f.name for f in mine)
        own_fields[name] = mine

    def default_of(f):
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f"{f.default_factory.__name__}()"
        return "(required)"

    # every default a field NAME has anywhere, to flag two containers that disagree
    defaults_by_name = {}
    for name, cls, _recv, _get in containers:
        for f in own_fields[name]:
            defaults_by_name.setdefault(f.name, []).append((name, default_of(f)))
    for k, v in api.LEGACY_DEFAULTS.items():
        defaults_by_name.setdefault(k, []).append(("LEGACY_DEFAULTS", v))
    #: the same setting under two names (API name -> solver-side name)
    same = {"ContactSettings.E_star": "contact_E_star", "ContactSettings.tol": "contact_tol",
            "ContactSettings.maxit": "contact_maxit", "ContactSettings.mu": "mu_material",
            "ContactSettings.sigma_yield": "sigma_yield", "ContactSettings.k_asp_mode": "k_asp_mode",
            "RippleSettings.E_star": "fluid_E_star", "RippleSettings.tol": "ripple_tol",
            "RippleSettings.maxpass": "ripple_maxpass", "RippleSettings.omega": "ripple_omega",
            "Zeta.H": "H0", "Zeta.P": "p0", "Zeta.Hdot": "HT", "Zeta.Pdot": "PT", "Zeta.Uxdot": "UxT",
            "Zeta.Uydot": "UyT"}

    rows, other = [], []
    classified = set()
    for name, cls, recv, get in containers:
        for f in own_fields[name]:
            hits = [(rel, ln, how) for rel, ln, r, attr, how in reads if attr == f.name and r in recv]
            if name == "MicroSettings":                    # check() reads its own fields as self.<name>
                hits += [(rel, ln, how) for rel, ln, r, attr, how in reads
                         if attr == f.name and r == "self" and rel == "microcell/api.py" and 148 <= ln <= 162]
            for h in hits:
                classified.add((h[0], h[1], f.name))
            row = {"container": name, "field": f.name, "default": default_of(f),
                   "set_by": set_at(f"{f.name}=") if name not in ("Zeta", "ContactSettings", "RippleSettings",
                                                                  "MicroSettings") else "the caller",
                   "n_reads": len(hits),
                   "readers": "; ".join(sorted({f"{rel}:{ln}" + (f" ({how})" if how != "attr" else "")
                                                for rel, ln, how in hits}))}
            vals = []
            for cell, sh in zip(cells, short):
                obj = get(captured[cell])
                v = "(None)" if obj is None else (getattr(obj, f.name, "(absent)"))
                row[sh] = v
                vals.append(_fmt(v))
            row["constant_over_refs"] = "yes" if len(set(vals)) == 1 else "no"
            key = f"{name}.{f.name}"
            twin = same.get(key, f.name)
            others = [(c, d) for c, d in defaults_by_name.get(twin, []) if c != name]
            if twin == f.name:
                others = [(c, d) for c, d in defaults_by_name.get(f.name, []) if c != name]
            differ = [f"{c}.{twin} = {_fmt(d)}" for c, d in others if _fmt(d) != _fmt(default_of(f))]
            row["other_default"] = "; ".join(differ)
            rows.append(row)
    for k, v in api.LEGACY_DEFAULTS.items():
        hits = [(rel, ln) for rel, ln, r, attr, how in reads if how == "legacy_value" and attr == k]
        row = {"container": "LEGACY_DEFAULTS", "field": k, "default": v, "set_by": "the caller (MicroSettings.legacy)",
               "n_reads": len(hits), "readers": "; ".join(f"{rel}:{ln} (legacy_value)" for rel, ln in hits)}
        vals = []
        for cell, sh in zip(cells, short):
            row[sh] = captured[cell]["st"].legacy_value(k)
            row[sh + "_given"] = k in captured[cell]["st"].legacy
            vals.append(_fmt(row[sh]))
        row["constant_over_refs"] = "yes" if len(set(vals)) == 1 else "no"
        differ = [f"{c}.{k} = {_fmt(d)}" for c, d in defaults_by_name.get(k, []) if c != "LEGACY_DEFAULTS"
                  and _fmt(d) != _fmt(v)]
        row["other_default"] = "; ".join(differ)
        rows.append(row)
    _write("inventory_settings.tsv",
           ["container", "field", "default", "other_default", "set_by", "n_reads", "readers"] + short
           + ["constant_over_refs"], rows)

    names = {f.name for fs in own_fields.values() for f in fs} | set(api.LEGACY_DEFAULTS)
    for rel, ln, r, attr, how in reads:
        if attr in names and (rel, ln, attr) not in classified and how != "legacy_value":
            other.append({"file": rel, "line": ln, "receiver": r, "attr": attr, "how": how})
    _write("inventory_settings_other.tsv", ["file", "line", "receiver", "attr", "how"], other)

    # ---- 2. results -------------------------------------------------------------------------
    raws = {c: captured[c]["out"].raw for c in cells}
    refs = {c: captured[c]["ref"]["result"] for c in cells}
    keys = sorted(set().union(*[set(r) for r in raws.values()], *[set(r) for r in refs.values()]))
    result_src = "\n".join(api_src[340:360])               # _result()
    info_keys = ("pressure_extrema_status", "shear_extrema_status", "shear_asperity_status", "shear_total_status",
                 "deformation_model", "contact_model", "level_constraint")
    info_direct = {"newton_iters": "newton_iters", "contact_iters": "contact_iters", "ripple_passes": "ripple_passes",
                   "contact_residual": "contact_residual", "contact_resolution": "contact_resolution",
                   "lambda_nd": "lambda_nd", "spos_mean": "spos_mean", "wall_time": "wall_time",
                   "mesh_used": "mesh_n"}
    # the failed result, for each (mixed, ripple) kind the references hold
    failed = {}
    for cell in cells:
        st = captured[cell]["st"]
        kind = (st.contact is not None, st.ripple is not None)
        lc = {} if st.level_constraint == "whole_cell" else dict(level_constraint=st.level_constraint,
                                                                 n_whole_cell=0, n_fluid_mean=0, n_no_fluid=0)
        failed.setdefault(kind, api._failed_raw(st, kind[0], kind[1], lc, "forced",
                                                dict(newton=0, contact=0, ripple=0, contact_s=0.0), 0.0))
    rrows = []
    for k in keys:
        in_raw = [sh for c, sh in zip(cells, short) if k in raws[c]]
        in_ref = [sh for c, sh in zip(cells, short) if k in refs[c]]
        vals = [raws[c][k] for c in cells if k in raws[c]]
        fv = [_fmt(np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for v in vals]
        nan_cells = [sh for c, sh in zip(cells, short) if k in raws[c] and _nan_items("x", raws[c][k])]
        writers = [str(i + 1) for i, line in enumerate(api_src)
                   if (f"{k}=" in line or f'"{k}"' in line) and not (340 <= i < 360)]
        helper = ("fluid_shear.shear_fields / failed_shear_fields" if k.startswith("shear_fluid") or k.startswith("shear_extrema")
                  else "contact_shear.contact_shear_fields / failed_contact_shear_fields"
                  if k.startswith(("shear_asperity", "shear_total", "shear_contact")) else
                  "pressure.PRESSURE_FIELDS (api.py:598)" if k in ("P_f", "Pst_fluid", "P_patch", "P_fluid_pos", "P_fluid_neg")
                  else "")
        named = f'raw["{k}"]' in result_src or f'vec("{k}")' in result_src
        if not in_raw:
            read = "not in raw"
        else:
            read = ", ".join(x for x in ("MicroResult field" if named else "",
                                         f"info[{k!r}]" if k in info_keys else "",
                                         f"info[{info_direct[k]!r}] (written beside raw)" if k in info_direct else "")
                             if x) or "nothing but the gate"
        compared = (k not in tp.NOT_COMPARED)
        rrows.append({
            "key": k, "in_raw": f"{len(in_raw)}/8", "in_reference": f"{len(in_ref)}/8",
            "written_at": ("api.py:" + ",".join(writers) if writers else "") + (f" {helper}" if helper else ""),
            "read_by": read,
            "gate": ("compared" if compared and in_ref else "not compared (NOT_COMPARED)" if in_ref else "no reference value")
                    + (" (pair cell: not compared)" if k in tp.PAIR_NOT_COMPARED else "")
                    + (" (pair cell: against coupled_json)" if k in tp.PAIR_FROM_COUPLED else ""),
            "type": type(vals[0]).__name__ if vals else "",
            "constant_over_refs": ("yes: " + fv[0] if len(set(fv)) == 1 else "no") if vals else "",
            "nan_in_ok_cells": ",".join(nan_cells),
            "nan_when_failed": ",".join(f"mixed={m} ripple={r}" for (m, r), raw in sorted(failed.items())
                                        if k in raw and _nan_items("x", raw[k])),
            "absent_when_failed": ",".join(f"mixed={m} ripple={r}" for (m, r), raw in sorted(failed.items())
                                           if k not in raw),
        })
    _write("inventory_results.tsv", ["key", "in_raw", "in_reference", "written_at", "read_by", "gate", "type",
                                     "constant_over_refs", "nan_in_ok_cells", "nan_when_failed",
                                     "absent_when_failed"], rrows)

    # ---- 3. observed NaN: OK cells, then forced failures ---------------------------------------
    named_fields = [f.name for f in dataclasses.fields(api.MicroResult) if f.name not in ("fields", "info", "raw")]
    nrows = []

    def observe(label, out):
        statuses = {k: out.info.get(k) for k in info_keys}
        for f in named_fields:
            for nm, n, tot in _nan_items(f, getattr(out, f)):
                nrows.append({"case": label, "status": out.status, "where": "named", "name": nm, "n_nan": n,
                              "n_values": tot, **statuses})
        for nm, n, tot in _nan_items("", {k: v for k, v in out.raw.items()}):
            nrows.append({"case": label, "status": out.status, "where": "raw", "name": nm, "n_nan": n,
                          "n_values": tot, **statuses})
        for nm, n, tot in _nan_items("", out.info):
            nrows.append({"case": label, "status": out.status, "where": "info", "name": nm, "n_nan": n,
                          "n_values": tot, **statuses})
        for nm, n, tot in _nan_items("", out.fields):
            nrows.append({"case": label, "status": out.status, "where": "fields", "name": nm, "n_nan": n,
                          "n_values": tot, **statuses})
        if not out.fields:
            nrows.append({"case": label, "status": out.status, "where": "fields", "name": "(no fields: empty dict)",
                          "n_nan": 0, "n_values": 0, **statuses})

    for cell, sh in zip(cells, short):
        observe(sh, captured[cell]["out"])

    def forced(cell, label, where, exc):
        c = captured[cell]
        cls = con.MicroPhaseNeutralContactSolver if c["st"].contact is not None else base.MicroPhaseNeutralSolver
        method = {"run": "run_steady" if not c["transient"] else "run", "asp": "compute_asperity_load"}[where]

        def boom(self, *a, **k):
            raise exc

        monkeypatch.setattr(cls, method, boom)
        try:
            out = microsolver(c["zeta"], t_end=(float(c["task"][15]) if c["transient"] else None),
                              roughness=c["rough"], settings=c["st"], n_steps=int(c["ref"]["meta"]["n_steps"]),
                              offset=tuple(c["ref"]["meta"]["offset"]) if c["ref"]["meta"]["offset"] else (0.0, 0.0),
                              steady=not c["transient"])
        finally:
            monkeypatch.undo()
        observe(label, out)
        return out

    mixed_cell = next(c for c in cells if c.startswith("c2"))
    full_cell = next(c for c in cells if c.startswith("c5"))
    assert forced(mixed_cell, "c2 forced FAILED_NEWTON", "run", RuntimeError("forced")).status == "FAILED_NEWTON"
    assert forced(mixed_cell, "c2 forced FAILED_RIPPLE", "run",
                  _ripple.RippleNotConvergedError("forced")).status == "FAILED_RIPPLE"
    assert forced(mixed_cell, "c2 forced FAILED_CONTACT (in run)", "run",
                  ContactResolutionError("forced")).status == "FAILED_CONTACT"
    assert forced(mixed_cell, "c2 forced FAILED_CONTACT (in compute_asperity_load)", "asp",
                  ContactResolutionError("forced")).status == "FAILED_CONTACT"
    assert forced(full_cell, "c5 forced FAILED_NEWTON", "run", RuntimeError("forced")).status == "FAILED_NEWTON"
    _write("inventory_nan_observed.tsv", ["case", "status", "where", "name", "n_nan", "n_values"] + list(info_keys),
           nrows)

    # ---- a plain dump of the info keys and field names per cell (for the results tables) --------
    irows = []
    for cell, sh in zip(cells, short):
        out = captured[cell]["out"]
        for k, v in out.info.items():
            if k == "fields":
                for kk, vv in v.items():
                    irows.append({"cell": sh, "part": "info.fields", "key": kk, "value": vv})
            else:
                irows.append({"cell": sh, "part": "info", "key": k, "value": v})
        for k, v in out.fields.items():
            irows.append({"cell": sh, "part": "fields", "key": k, "value": f"shape {np.shape(v)}"})
    _write("inventory_info_fields.tsv", ["cell", "part", "key", "value"], irows)
