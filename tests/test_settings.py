"""``MicroSettings.check()``: the values refused before a mesh is built.

No solve runs.  Each refused case changes the default settings and ``check()`` must raise
``ValueError``.  Each accepted case must pass.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import ContactSettings, MicroSettings, RippleSettings  # noqa: E402

NAN, INF = float("nan"), float("inf")

#: case name -> changed fields; a "contact." or "ripple." key changes the nested settings
REFUSED = {
    "spos_form": {"spos_form": "cubic"},
    "level_constraint": {"level_constraint": "fluid_mean_v2"},
    "mesh_n": {"mesh_n": 1},
    "jacobian_reg_eps nan": {"jacobian_reg_eps": NAN},
    "delta_h negative": {"delta_h": -1e-9},
    "eps_solid nan": {"eps_solid": NAN},
    "eps_smooth inf": {"eps_smooth": INF},
    "E_star zero": {"E_star": 0.0},
    "E_star nan": {"E_star": NAN},
    "E_star nan, ripple only": {"E_star": NAN, "contact": None},
    "E_star nan, contact only": {"E_star": NAN, "ripple": None},
    "h_min zero with contact": {"h_min": 0.0},
    "h_min nan with contact": {"h_min": NAN},
    "contact.k_asp_mode": {"contact.k_asp_mode": "secant"},
    "ripple.tol": {"ripple.tol": 0.0},
    "ripple.maxpass": {"ripple.maxpass": 0},
    "ripple.omega zero": {"ripple.omega": 0.0},
    "ripple.omega above one": {"ripple.omega": 1.5},
}
#: accepted cases: a value of a part that is off is not read
ACCEPTED = {
    "defaults": {},
    "full film, no ripple": {"contact": None, "ripple": None},
    "h_min zero without contact": {"h_min": 0.0, "contact": None},
    "E_star nan with contact and ripple off": {"E_star": NAN, "contact": None, "ripple": None},
    "delta_h zero": {"delta_h": 0.0},
}


def build(changes: dict) -> MicroSettings:
    top = {k: v for k, v in changes.items() if "." not in k}
    st = MicroSettings(**top)
    for k, v in changes.items():
        if "." in k:
            part, name = k.split(".")
            st = dataclasses.replace(st, **{part: dataclasses.replace(getattr(st, part), **{name: v})})
    return st


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_check_refuses(case):
    with pytest.raises(ValueError):
        build(REFUSED[case]).check()


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_check_accepts(case):
    build(ACCEPTED[case]).check()


def test_one_modulus():
    """``E_star`` is a field of ``MicroSettings`` only: the contact and ripple settings have none."""
    names = lambda cls: {f.name for f in dataclasses.fields(cls)}  # noqa: E731
    assert "E_star" in names(MicroSettings)
    assert "E_star" not in names(ContactSettings) and "E_star" not in names(RippleSettings)
    assert not {"legacy", "rho0"} & names(MicroSettings)
