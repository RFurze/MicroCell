"""The layout of a ``MicroResult`` (WP4): an OK and a failed result of the same cell.

The parity gate (``test_parity.py``) only reaches OK results.  Here a failure is forced in a
reference cell by replacing one solver method with one that raises, and the failed result is
checked against the OK result of the same cell:

* the status is the one the exception maps to;
* every named homogenised output is NaN, ``yield_flag`` is 0 and there are no fields;
* ``info`` carries the same entries as the OK result's, each of the same kind
  (string, integer, float, pair, dict or None).

And the rule of WP4 group 4 (the second half of this file): every NaN of a result -- homogenised
outputs, ``info``, fields -- has a status that is not OK and says why: the result's own
``status``, or the per-quantity status that the one table ``microcell.api.NAN_STATUS`` ties the
output to.  It is checked on every reference cell, on the forced failures, and on a NaN put into
a converged solve (``FAILED_NONFINITE``).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_parity as tp  # noqa: E402

MIXED_CELL = "c2_gensa40_h130_ripple"       # fft_elastic_v1 with the ripple, transient
FULL_FILM_CELL = "c5_cosine_fullfilm"       # no contact law, no ripple, transient

#: (cell, the method replaced, the exception it raises, the status expected)
FORCED = {
    "newton": (MIXED_CELL, "run", "RuntimeError", "FAILED_NEWTON"),
    "ripple": (MIXED_CELL, "run", "RippleNotConvergedError", "FAILED_RIPPLE"),
    "contact_in_run": (MIXED_CELL, "run", "ContactResolutionError", "FAILED_CONTACT"),
    "contact_in_load": (MIXED_CELL, "compute_asperity_load", "ContactResolutionError", "FAILED_CONTACT"),
    "newton_full_film": (FULL_FILM_CELL, "run", "RuntimeError", "FAILED_NEWTON"),
}
_OK = {}


def solve(cell: str):
    """The reference cell through ``microsolver()``, as the parity gate calls it."""
    from microcell import Zeta, microsolver

    ref = tp.load_reference(cell)
    meta, task = ref["meta"], ref["task"]
    transient = bool(meta["transient"])
    return microsolver(
        Zeta.from_task(task), t_end=(float(task[15]) if transient else None),
        roughness=tp.roughness_from_reference(ref), settings=tp.settings_from_reference(ref),
        n_steps=int(meta["n_steps"]), offset=tuple(meta["offset"]) if meta["offset"] else (0.0, 0.0),
        steady=not transient)


def solve_ok(cell: str):
    if cell not in _OK:
        _OK[cell] = solve(cell)
        assert _OK[cell].status == "OK", _OK[cell].reason
    return _OK[cell]


def solve_forced(case: str, monkeypatch):
    """The cell of ``FORCED[case]`` with the named solver method raising the named exception."""
    from microcell import solver_base as base
    from microcell import solver_contact as con
    from microcell import ripple
    from microcell.contact import ContactResolutionError

    cell, method, exc_name, _ = FORCED[case]
    exc = {"RuntimeError": RuntimeError, "RippleNotConvergedError": ripple.RippleNotConvergedError,
           "ContactResolutionError": ContactResolutionError}[exc_name]("forced")
    mixed = tp.settings_from_reference(tp.load_reference(cell)).contact is not None
    cls = con.MicroPhaseNeutralContactSolver if mixed else base.MicroPhaseNeutralSolver

    def boom(self, *args, **kwargs):
        raise exc

    with monkeypatch.context() as m:
        m.setattr(cls, method, boom)
        return solve(cell)


def named_outputs():
    from microcell import MicroResult

    return [f.name for f in dataclasses.fields(MicroResult) if f.name not in ("status", "reason", "fields", "info")]


def kind(value) -> str:
    if value is None or isinstance(value, (str, dict)):
        return type(value).__name__
    if isinstance(value, (bool, int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    return f"{len(value)} floats"


@pytest.mark.parametrize("case", sorted(FORCED))
def test_failed_result_layout(case, monkeypatch):
    cell, _, _, status = FORCED[case]
    ok = solve_ok(cell)
    out = solve_forced(case, monkeypatch)
    assert out.status == status and not out.ok and "forced" in out.reason
    for name in named_outputs():
        value = getattr(out, name)
        if name == "yield_flag":
            assert value == 0
        else:
            assert np.all(np.isnan(np.asarray(value, dtype=float))), f"{name} = {value!r} in a failed result"
            assert np.shape(value) == np.shape(getattr(ok, name)), name
    assert out.fields == {} and out.info["fields"] == {}
    assert sorted(out.info) == sorted(ok.info), (sorted(set(out.info) ^ set(ok.info)))
    different = {k: (kind(out.info[k]), kind(ok.info[k])) for k in ok.info if kind(out.info[k]) != kind(ok.info[k])}
    assert not different, different


# ---------------------------------------------------------------------------------------
# WP4 group 4: every NaN has a status that says why (``microcell.api.NAN_STATUS``)
# ---------------------------------------------------------------------------------------
#: a NaN put into an OK solve: (cell, what is replaced, the output the reason must name)
NONFINITE = {
    "number": (MIXED_CELL, "spatial_homogenisation", "Fst"),
    "field": (MIXED_CELL, "extract_fields", "fields.film"),
    "number_full_film": (FULL_FILM_CELL, "spatial_homogenisation", "Fst"),
}
PER_QUANTITY_STATUSES = ("pressure_extrema_status", "shear_extrema_status", "shear_asperity_status",
                         "shear_total_status")
#: the per-quantity statuses under which the numbers are physical: never an explanation of a NaN
PHYSICAL = ("ok", "no_support")


def solve_nonfinite(case: str, monkeypatch):
    """The cell of ``NONFINITE[case]`` with one NaN put into a converged solve."""
    from microcell import fields as fields_module
    from microcell import solver_base as base
    from microcell import solver_contact as con

    cell, what, _ = NONFINITE[case]
    with monkeypatch.context() as m:
        if what == "spatial_homogenisation":
            mixed = tp.settings_from_reference(tp.load_reference(cell)).contact is not None
            cls = con.MicroPhaseNeutralContactSolver if mixed else base.MicroPhaseNeutralSolver
            original = cls.spatial_homogenisation

            def homogenise(self, *args, **kwargs):
                values = list(original(self, *args, **kwargs))
                values[3] = float("nan")                   # Fst
                return tuple(values)

            m.setattr(cls, "spatial_homogenisation", homogenise)
        else:
            original = fields_module.extract_fields

            def extract(*args, **kwargs):
                shown, cell_fields, info = original(*args, **kwargs)
                shown["film_nm"] = shown["film_nm"].copy()
                shown["film_nm"][0, 0] = float("nan")
                return shown, cell_fields, info

            m.setattr(fields_module, "extract_fields", extract)
        return solve(cell)


def nan_outputs(out) -> list:
    """(where, name) of every output of ``out`` that holds a NaN or an infinity: named outputs,
    ``info`` entries, ``info["fields"]`` entries and ``fields`` arrays."""
    places = [("named", n, getattr(out, n)) for n in named_outputs()]
    places += [("info", k, v) for k, v in out.info.items() if k != "fields"]
    places += [("info.fields", k, v) for k, v in out.info["fields"].items()]
    places += [("fields", k, v) for k, v in out.fields.items()]
    found = []
    for where, name, value in places:
        if value is None or isinstance(value, (str, bool, dict)) or (
                isinstance(value, list) and value and isinstance(value[0], str)):
            continue
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            found.append((where, name))
    return found


def check_every_nan_has_a_status(out):
    """The rule: a NaN is explained by a status that is not OK -- the result's own ``status``, or
    the per-quantity status the table ``NAN_STATUS`` ties the output to."""
    from microcell import NAN_STATUS, OK

    found = nan_outputs(out)
    if out.status != OK:
        assert out.status.startswith("FAILED_")
        return found
    for where, name in found:
        assert where in ("named", "info"), f"{where}.{name} is not finite in an OK result"
        assert name in NAN_STATUS, f"{name} is not finite in an OK result and is not in NAN_STATUS"
        key, explains = NAN_STATUS[name]
        assert out.info[key] in explains, f"{name} is not finite with {key} = {out.info[key]!r}"
        assert out.info[key] not in PHYSICAL
    return found


def test_table_names_outputs_and_statuses():
    """The table's keys are outputs of a result, its statuses are the four per-quantity ones, and
    no status under which the numbers are physical explains a NaN."""
    from microcell import NAN_STATUS

    out = solve_ok(MIXED_CELL)
    outputs = set(named_outputs()) | set(out.info)
    for name, (key, explains) in NAN_STATUS.items():
        assert name in outputs, name
        assert key in PER_QUANTITY_STATUSES and key in out.info, (name, key)
        assert explains and not set(explains) & set(PHYSICAL) and "OK" not in explains, (name, explains)


@pytest.mark.parametrize("cell", tp.reference_cells())
def test_every_nan_has_a_status_reference_cell(cell):
    out = solve_ok(cell)
    found = check_every_nan_has_a_status(out)
    mixed = tp.settings_from_reference(tp.load_reference(cell)).contact is not None
    # a full-film cell has no contact law: its asperity and total shear are NaN, "unavailable"
    assert bool(found) == (not mixed), found


@pytest.mark.parametrize("case", sorted(FORCED))
def test_every_nan_has_a_status_forced_failure(case, monkeypatch):
    out = solve_forced(case, monkeypatch)
    assert out.status == FORCED[case][3]
    assert check_every_nan_has_a_status(out)


@pytest.mark.parametrize("case", sorted(NONFINITE))
def test_unexplained_nan_fails_the_result(case, monkeypatch):
    """A NaN the table does not explain, in a converged solve: ``FAILED_NONFINITE``, laid out as
    every failed result."""
    from microcell import FAILED_NONFINITE

    cell, _, name = NONFINITE[case]
    ok = solve_ok(cell)
    out = solve_nonfinite(case, monkeypatch)
    assert out.status == FAILED_NONFINITE and not out.ok, (out.status, out.reason)
    assert name in out.reason.split(": ")[-1].split(", "), out.reason
    for n in named_outputs():
        if n != "yield_flag":
            assert np.all(np.isnan(np.asarray(getattr(out, n), dtype=float))), n
    assert out.yield_flag == 0 and out.fields == {} and out.info["fields"] == {}
    assert sorted(out.info) == sorted(ok.info)
    assert check_every_nan_has_a_status(out)


def test_listed_output_needs_its_status():
    """A listed output is explained only by its own status: NaN with ``ok`` is unexplained, NaN with
    ``unavailable`` (a full-film cell's asperity shear) is explained."""
    from microcell.api import unexplained_nonfinite

    nan2 = np.array([float("nan"), float("nan")])
    mixed = solve_ok(MIXED_CELL)
    assert mixed.info["shear_asperity_status"] == "ok" and unexplained_nonfinite(mixed) == []
    assert unexplained_nonfinite(dataclasses.replace(mixed, tau_asperity=nan2)) == ["tau_asperity"]
    assert unexplained_nonfinite(dataclasses.replace(mixed, P_asp=float("inf"))) == ["P_asp"]
    full = solve_ok(FULL_FILM_CELL)
    assert full.info["shear_asperity_status"] == "unavailable" and np.all(np.isnan(full.tau_asperity))
    assert unexplained_nonfinite(full) == []
