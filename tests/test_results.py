"""The layout of a ``MicroResult`` (WP4): an OK and a failed result of the same cell.

The parity gate (``test_parity.py``) only reaches OK results.  Here a failure is forced in a
reference cell by replacing one solver method with one that raises, and the failed result is
checked against the OK result of the same cell:

* the status is the one the exception maps to;
* every named homogenised output is NaN, ``yield_flag`` is 0 and there are no fields;
* ``info`` carries the same entries as the OK result's, each of the same kind
  (string, integer, float, pair, dict or None).
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
    from microcell import micro_phase_neutral_class as base
    from microcell import micro_phase_neutral_class_contact as con
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
