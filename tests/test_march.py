"""``n_steps=None``: the march from t = 0 to ``t_end`` (plan D3; WP5 group 2).

The cell: a translating cosine in full film, mesh 32, with a film rate, so the film changes at
every step and the end state depends on where the window starts.  ``t_end`` is 8.5 CFL steps.

What is expected, and where each expectation comes from:

* the step count ``N = ceil(t_end / dt_cfl) = 9``, with ``dt_cfl = cfl_target * min(xmax, ymax) /
  mesh_n / (2 |Ux|)`` worked out here from the settings, not read from the solver;
* ``t_start = 0`` and ``N * dt = t_end``: the march covers the whole window;
* the result is bitwise the result of ``n_steps=N`` and of ``n_steps=N + 3``.  Those are the same
  problem by construction: a window of ``n`` CFL steps ending at ``t_end`` with ``n >= N`` does not
  fit in ``[0, t_end]``, and the solver then starts at t = 0 and takes ``N`` equal steps.  The
  integer path is the one the parity gate covers; ``n_steps=None`` reaches it by its own count;
* the result differs from the default 3-step window (which starts at ``t_end - 3 dt_cfl > 0``), so
  the comparison above is not passing because the window does not matter.

Run from the repo root::

    python -m pytest tests/test_march.py -q
"""
from __future__ import annotations

import math
from dataclasses import fields as dc_fields

import numpy as np
import pytest

from microcell import MicroResult, MicroSettings, Roughness, Zeta, microsolver

SETTINGS = dict(mesh_n=32, contact=None, ripple=None)
ZETA = dict(H=200e-9, P=1.0e6, Ux=0.02, Hdot=-1.0e-4)
N_CFL = 8.5                      # the window in CFL steps


def _cell():
    st = MicroSettings(**SETTINGS)
    zeta = Zeta(**ZETA)
    dt_cfl = float(st.cfl_target) * min(st.xmax, st.ymax) / int(st.mesh_n) / (2.0 * abs(zeta.Ux))
    return st, zeta, Roughness.cosine(Ah=50e-9, moving=True), N_CFL * dt_cfl, dt_cfl


def _same(a: MicroResult, b: MicroResult) -> bool:
    """Every named output and every field of the two results is bitwise equal."""
    for f in dc_fields(MicroResult):
        if f.name in ("status", "reason", "fields", "info"):
            continue
        if not np.array_equal(np.asarray(getattr(a, f.name)), np.asarray(getattr(b, f.name)), equal_nan=True):
            return False
    if sorted(a.fields) != sorted(b.fields):
        return False
    return all(np.array_equal(a.fields[k], b.fields[k], equal_nan=True) for k in a.fields)


@pytest.fixture(scope="module")
def march():
    st, zeta, rough, t_end, _dt_cfl = _cell()
    return microsolver(zeta, t_end=t_end, roughness=rough, settings=st, n_steps=None)


def test_march_covers_the_whole_window(march):
    _st, _zeta, _rough, t_end, dt_cfl = _cell()
    n = math.ceil(t_end / dt_cfl)
    assert n == 9
    assert march.ok, march.reason
    assert march.info["n_steps"] == n
    assert march.info["t_start"] == 0.0
    assert march.info["dt"] * n == pytest.approx(t_end, rel=1e-14, abs=0.0)
    assert march.info["dt"] < dt_cfl                     # 8.5 CFL steps in 9 equal steps


@pytest.mark.parametrize("extra", [0, 3])
def test_march_equals_the_integer_window_that_does_not_fit(march, extra):
    st, zeta, rough, t_end, dt_cfl = _cell()
    n = math.ceil(t_end / dt_cfl) + extra
    out = microsolver(zeta, t_end=t_end, roughness=rough, settings=st, n_steps=n)
    assert out.ok, out.reason
    assert out.info["t_start"] == 0.0 and out.info["dt"] == march.info["dt"]
    assert out.info["newton_iters"] == march.info["newton_iters"]
    assert out.info["n_steps"] == math.ceil(t_end / dt_cfl)      # the count marched, not the count asked for
    assert _same(march, out)


def test_march_differs_from_the_three_step_window(march):
    st, zeta, rough, t_end, dt_cfl = _cell()
    out = microsolver(zeta, t_end=t_end, roughness=rough, settings=st, n_steps=3)
    assert out.ok, out.reason
    assert out.info["t_start"] == pytest.approx(t_end - 3 * dt_cfl, rel=1e-12)
    assert out.info["n_steps"] == 3
    assert out.info["dt"] == pytest.approx(dt_cfl, rel=1e-12)
    assert not _same(march, out)
    assert np.max(np.abs(out.fields["p_fluid"] - march.fields["p_fluid"])) > 0.0


def test_a_steady_solve_ignores_n_steps():
    st, zeta, rough, _t_end, _dt_cfl = _cell()
    a = microsolver(zeta, roughness=rough, settings=st, steady=True, n_steps=None)
    b = microsolver(zeta, roughness=rough, settings=st, steady=True)
    assert a.ok and b.ok and a.info["n_steps"] == 0 == b.info["n_steps"]
    assert _same(a, b)


def test_a_failed_newton_solve_reports_its_iterations():
    """Tolerances no solve can meet: the result is FAILED_NEWTON and counts the iterations it made."""
    _st, zeta, rough, _t_end, _dt_cfl = _cell()
    st = MicroSettings(**SETTINGS, max_iterations=3, abs_tolerance=1e-300, rel_tolerance=1e-300)
    out = microsolver(zeta, roughness=rough, settings=st, steady=True)
    assert out.status == "FAILED_NEWTON", (out.status, out.reason)
    assert 0 < out.info["newton_iters"] <= 3
    assert np.isnan(out.dP) and not out.fields
