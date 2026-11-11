"""WP4: the settings, results and observed-NaN tables of the tree as it stands (the "after" tables).

Pytest form so it runs through ``_claude/wsltest`` from ``MicroCell/``; the one line of
``tables_tag.txt`` beside this file names the state (an environment variable does not reach WSL):

    ../_claude/wsltest docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/tables.py -q -p no:cacheprovider

Writes, beside this file:
  settings_<tag>.tsv       every input of a solve: where it is defined (file:line), its one default,
                           the parent settings field(s) it carries (tests/test_parity.SETTINGS_NOW)
  results_<tag>.tsv        every output of a MicroResult: where it is held, the parent result field(s)
                           it carries (RESULT_NOW), its kind, the reference cells in which it holds a
                           NaN on an OK result, and what a failed result holds
  nan_observed_<tag>.tsv   every NaN seen: the 8 reference cells (OK) and the five forced failures of
                           tests/test_results.py, with the overall and the four per-quantity statuses
  counts_<tag>.tsv         the counts the log quotes
"""
import dataclasses
import inspect
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()
for p in (TREE, TREE / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
TAG = (HERE / "tables_tag.txt").read_text().strip() if (HERE / "tables_tag.txt").exists() else "current"
STATUS_KEYS = ("pressure_extrema_status", "shear_extrema_status", "shear_asperity_status", "shear_total_status")


def _fmt(v):
    return "" if v is None else str(v).replace("\t", " ").replace("\n", " ")


def _write(name, cols, rows):
    (HERE / name).write_text("\n".join(["\t".join(cols)] + ["\t".join(_fmt(r.get(c, "")) for c in cols)
                                                            for r in rows]) + "\n")


def _nan_items(prefix, obj):
    """(name, n_nan, n_total) of every float / array / tuple leaf of ``obj`` that holds a NaN."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _nan_items(f"{prefix}.{k}" if prefix else str(k), v)
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


def _rel(path):
    return Path(path).resolve().relative_to(TREE).as_posix()


def test_tables(monkeypatch):
    import test_parity as tp
    import test_results as tr
    from microcell import api
    import microcell.micro_phase_neutral_class as solver_module

    assert Path(tp.__file__).resolve().parent == TREE / "tests"
    counts = []

    # ---- settings ---------------------------------------------------------------------------
    carried = {}                                    # where -> the parent's settings fields
    for parent, where in getattr(tp, "SETTINGS_NOW", {}).items():
        carried.setdefault(where, []).append(parent)
    srows = []
    prefix = {"MicroSettings": "settings.", "ContactSettings": "settings.contact.",
              "RippleSettings": "settings.ripple."}
    for cls in (api.Zeta, api.MicroSettings, api.ContactSettings, api.RippleSettings):
        src, start = inspect.getsourcelines(cls)
        for f in dataclasses.fields(cls):
            line = start + next(i for i, text in enumerate(src) if re.match(rf"\s+{f.name}\s*:", text))
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                default = f"{f.default_factory.__name__}()"
            else:
                default = "(required)"
            srows.append({"container": cls.__name__, "name": f.name, "default": default,
                          "defined_at": f"{_rel(inspect.getsourcefile(cls))}:{line}",
                          "parent_settings_field": ", ".join(carried.get(prefix.get(cls.__name__, "?") + f.name, []))})
    if hasattr(api, "LEGACY_DEFAULTS"):
        line = next(i for i, text in enumerate(Path(api.__file__).read_text().splitlines(), 1)
                    if text.startswith("LEGACY_DEFAULTS"))
        for k, v in api.LEGACY_DEFAULTS.items():
            srows.append({"container": "LEGACY_DEFAULTS", "name": k, "default": repr(v),
                          "defined_at": f"{_rel(api.__file__)}:{line}",
                          "parent_settings_field": ", ".join(carried.get("legacy." + k, []))})
    text = Path(solver_module.__file__).read_text().splitlines()
    for i, row in enumerate(text, 1):
        m = re.match(r"([A-Z][A-Z0-9_]+)(?:\s*:\s*[^=]+)?\s*=\s*(.+?)\s*(#.*)?$", row)
        if m and ("fixed." + m.group(1) in carried or m.group(1) == "TRACE_NEWTON"):
            srows.append({"container": "fixed value (solver module)", "name": m.group(1), "default": m.group(2),
                          "defined_at": f"{_rel(solver_module.__file__)}:{i}",
                          "parent_settings_field": ", ".join(carried.get("fixed." + m.group(1), []))})
    sig = inspect.signature(api.microsolver)
    for name, par in sig.parameters.items():
        if name in ("zeta", "settings", "roughness"):
            continue
        srows.append({"container": "microsolver argument", "name": name,
                      "default": "(required)" if par.default is inspect.Parameter.empty else repr(par.default),
                      "defined_at": f"{_rel(api.__file__)}:{inspect.getsourcelines(api.microsolver)[1]}",
                      "parent_settings_field": ""})
    _write(f"settings_{TAG}.tsv", ["container", "name", "default", "defined_at", "parent_settings_field"], srows)
    for c in sorted({r["container"] for r in srows}):
        counts.append({"what": f"settings: {c}", "n": sum(1 for r in srows if r["container"] == c)})
    counts.append({"what": "settings: all rows", "n": len(srows)})
    counts.append({"what": "parent settings fields carried (SETTINGS_NOW)", "n": len(getattr(tp, "SETTINGS_NOW", {}))})

    # ---- results: the 8 cells and the forced failures ----------------------------------------
    cells = tp.reference_cells()
    short = [c.split("_")[0] for c in cells]
    outs = {sh: tr.solve(c) for c, sh in zip(cells, short)}
    assert all(o.status == "OK" for o in outs.values())
    forced = {case: tr.solve_forced(case, monkeypatch) for case in sorted(tr.FORCED)}
    refs = {sh: tp.load_reference(c) for c, sh in zip(cells, short)}
    parent_fields = sorted(set().union(*[set(r["result"]) for r in refs.values()]))
    held = {}                                       # where -> the parent's result fields
    for k in parent_fields:
        held.setdefault(tp.RESULT_NOW.get(k, "info." + k), []).append(k)
    named = tr.named_outputs()

    def places(out):
        """(where, name, value) of every output of a result."""
        rows = [("status", "status", out.status), ("status", "reason", out.reason)]
        rows += [("named", n, getattr(out, n)) for n in named]
        rows += [("info", k, v) for k, v in out.info.items() if k != "fields"]
        rows += [("info.fields", k, v) for k, v in out.info["fields"].items()]
        rows += [("fields", k, v) for k, v in out.fields.items()]
        return rows

    def kind(v):
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return "array"
        if isinstance(v, list):
            return "list"
        return tr.kind(v)

    order, seen = [], {}
    for sh in short:
        for where, name, value in places(outs[sh]):
            if (where, name) not in seen:
                order.append((where, name))
                seen[(where, name)] = {"kinds": set(), "cells": [], "nan_cells": []}
            rec = seen[(where, name)]
            rec["kinds"].add(kind(value))
            rec["cells"].append(sh)
            if _nan_items("x", value):
                rec["nan_cells"].append(sh)
    fail_mixed, fail_full = forced["newton"], forced["newton_full_film"]

    def failed_value(out, where, name):
        got = {(w, n): v for w, n, v in places(out)}
        if (where, name) not in got:
            return "absent"
        v = got[(where, name)]
        return "NaN" if _nan_items("x", v) else repr(v) if not isinstance(v, (float, np.floating)) else repr(float(v))

    not_compared = set(tp.NOT_COMPARED)
    rrows = []
    for where, name in order:
        rec = seen[(where, name)]
        key = name if where == "named" else ("reason" if name == "reason" else f"info.{name}") if where in ("info", "status") else None
        parents = held.get(key, []) if key else []
        if where == "status" and name == "status":
            gate = "compared (must be OK)"
        elif where in ("info.fields", "fields"):
            gate = "compared" if name not in ("x", "y") else "not compared (the grid)"
        elif parents:
            gate = "compared" if any(p not in not_compared for p in parents) else "not compared"
        else:
            gate = "not compared (the call)"
        rrows.append({"where": where, "name": name, "parent_result_field": ", ".join(parents),
                      "kind": "/".join(sorted(rec["kinds"])), "in_cells": f"{len(rec['cells'])}/8", "gate": gate,
                      "nan_in_ok_cells": ",".join(rec["nan_cells"]),
                      "failed_mixed_ripple": failed_value(fail_mixed, where, name) if where in ("named", "info", "status") else "absent" if where != "info.fields" else "absent",
                      "failed_full_film": failed_value(fail_full, where, name) if where in ("named", "info", "status") else "absent"})
    _write(f"results_{TAG}.tsv", ["where", "name", "parent_result_field", "kind", "in_cells", "gate",
                                 "nan_in_ok_cells", "failed_mixed_ripple", "failed_full_film"], rrows)
    for w in ("status", "named", "info", "info.fields", "fields"):
        counts.append({"what": f"results: {w}", "n": sum(1 for r in rrows if r["where"] == w)})
    counts.append({"what": "results: all rows", "n": len(rrows)})
    counts.append({"what": "parent result fields in the references", "n": len(parent_fields)})
    counts.append({"what": "parent result fields not compared (NOT_COMPARED)",
                   "n": sum(1 for k in parent_fields if k in not_compared)})
    counts.append({"what": "parent result fields held under another name or place (RESULT_NOW)",
                   "n": sum(1 for k in parent_fields if k in tp.RESULT_NOW)})
    unheld = [k for k in parent_fields if k not in not_compared
              and (tp.RESULT_NOW.get(k, "info." + k)) not in {(n if w == "named" else "reason" if n == "reason" else f"info.{n}")
                                                              for w, n in order if w in ("named", "info", "status")}]
    counts.append({"what": "compared parent result fields with no place in a MicroResult", "n": len(unheld)})

    # ---- observed NaN -------------------------------------------------------------------------
    nrows = []

    def observe(label, out):
        statuses = {k: out.info.get(k) for k in STATUS_KEYS}
        for where, name, value in places(out):
            for nm, n, tot in _nan_items(name, value):
                nrows.append({"case": label, "status": out.status, "where": where, "name": nm, "n_nan": n,
                              "n_values": tot, **statuses})
        if not out.fields:
            nrows.append({"case": label, "status": out.status, "where": "fields", "name": "(no fields: empty dict)",
                          "n_nan": 0, "n_values": 0, **statuses})

    for sh in short:
        observe(sh, outs[sh])
    for case, out in forced.items():
        observe(f"{tr.FORCED[case][0].split('_')[0]} forced {case}", out)
    _write(f"nan_observed_{TAG}.tsv", ["case", "status", "where", "name", "n_nan", "n_values"] + list(STATUS_KEYS),
           nrows)
    ok_rows = [r for r in nrows if r["status"] == "OK"]
    for w in ("named", "info", "info.fields", "fields"):
        counts.append({"what": f"NaN on OK results: distinct names in {w}",
                       "n": len({r["name"] for r in ok_rows if r["where"] == w})})
    for case, out in forced.items():
        label = f"{tr.FORCED[case][0].split('_')[0]} forced {case}"
        counts.append({"what": f"NaN in {label} ({out.status}): names in named + info",
                       "n": sum(1 for r in nrows if r["case"] == label and r["where"] in ("named", "info"))})
    _write(f"counts_{TAG}.tsv", ["what", "n"], counts)
