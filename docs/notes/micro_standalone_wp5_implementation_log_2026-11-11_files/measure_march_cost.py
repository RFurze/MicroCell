"""WP5 step 1, inventory 3: what a march from t = 0 costs, measured on short windows.  Edits nothing.

The solver already marches from t = 0 when the requested window is longer than ``t_end``
(``_run_once``: ``T_start < 0`` -> ``T_start = 0``, ``n_steps = ceil(t_end / dt_cfl)``,
``dt = t_end / n_steps``).  A large ``n_steps`` on a short ``t_end`` therefore runs the march that
``n_steps=None`` is to run, with the code as it stands at the base commit.

Run from ``MicroCell/``:

    <RefactorRefactor>/_claude/wsltest <path to this file> -q -p no:cacheprovider

Writes ``march_cost_<tag>.tsv`` beside this file (tag from ``inventory_tag.txt``).
"""
import math
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

from microcell import MicroResult, MicroSettings, Roughness, Zeta, microsolver

HERE = Path(__file__).resolve().parent
BIG = 10 ** 6            # a window far longer than t_end: the solver clamps it to a march from 0


def _cfl_dt(st: MicroSettings, zeta: Zeta) -> float:
    """The solver's CFL step for a constant velocity: cfl * min(Lx, Ly) / mesh_n / |2 U|."""
    speed = 2.0 * max(abs(zeta.Ux), abs(zeta.Uy))
    return float(st.cfl_target) * min(st.xmax, st.ymax) / int(st.mesh_n) / speed


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


def test_measure_march_cost():
    tag = (HERE / "inventory_tag.txt").read_text().strip() if (HERE / "inventory_tag.txt").exists() else "base"
    rows = ["\t".join(["cell", "mesh_n", "contact", "ripple", "t_end_s", "n_steps_asked", "steps_from_cfl_rule",
                       "status", "t_start_s", "dt_s", "steps_marched", "newton_iters", "contact_iters",
                       "ripple_passes", "wall_s", "wall_per_step_s", "dP_Pa", "P_asp_Pa"])]

    def run(cell, zeta, t_end, rough, st, n_steps):
        dt_cfl = _cfl_dt(st, zeta)
        n_rule = max(1, math.ceil(t_end / dt_cfl))
        t0 = time.time()
        out = microsolver(zeta, t_end=t_end, roughness=rough, settings=st, n_steps=n_steps)
        wall = time.time() - t0
        i = out.info
        marched = round((t_end - i["t_start"]) / i["dt"]) if out.ok else -1
        rows.append("\t".join(str(v) for v in [
            cell, st.mesh_n, st.contact is not None, st.ripple is not None, f"{t_end:.9e}", n_steps, n_rule,
            out.status, f"{i['t_start']:.9e}", f"{i['dt']:.9e}", marched, i["newton_iters"], i["contact_iters"],
            i["ripple_passes"], f"{wall:.2f}", f"{wall / max(marched, 1):.3f}", f"{out.dP:.9e}",
            f"{out.P_asp:.9e}"]))
        return out

    # A: the example cell (examples/run_cell.py): stationary generated surface, contact + ripple
    zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)
    st = MicroSettings()
    rough = Roughness.generated(Sa=40e-9, seed=1000)
    run("A_gensa40_h130_stationary", zeta, 0.02, rough, st, 3)              # the example's own window
    for t_end in (1.3e-5, 2.6e-5):                                            # 11 and 21 steps from 0
        run("A_gensa40_h130_stationary", zeta, t_end, rough, st, BIG)

    # B: the same surface translating (the film changes every step)
    rough_m = Roughness.generated(Sa=40e-9, seed=1000, moving=True)
    run("B_gensa40_h130_translating", zeta, 0.02, rough_m, st, 3)
    for t_end in (1.3e-5, 2.6e-5):
        run("B_gensa40_h130_translating", zeta, t_end, rough_m, st, BIG)

    # C: a small full-film cosine cell, translating, with a film rate: the candidate cell of the
    #    n_steps=None test.  8.5 CFL steps -> 9 steps from 0.
    st_c = MicroSettings(mesh_n=32, contact=None, ripple=None)
    zeta_c = Zeta(H=200e-9, P=1.0e6, Ux=0.02, Hdot=-1.0e-4)
    rough_c = Roughness.cosine(Ah=50e-9, moving=True)
    t_c = 8.5 * _cfl_dt(st_c, zeta_c)
    big = run("C_cosine_fullfilm_translating", zeta_c, t_c, rough_c, st_c, BIG)
    n9 = run("C_cosine_fullfilm_translating", zeta_c, t_c, rough_c, st_c, 9)
    n12 = run("C_cosine_fullfilm_translating", zeta_c, t_c, rough_c, st_c, 12)
    n3 = run("C_cosine_fullfilm_translating", zeta_c, t_c, rough_c, st_c, 3)
    rows.append("")
    rows.append("\t".join(["check", "result"]))
    rows.append(f"C: n_steps=10**6 bitwise equal to n_steps=9\t{_same(big, n9)}")
    rows.append(f"C: n_steps=10**6 bitwise equal to n_steps=12\t{_same(big, n12)}")
    rows.append(f"C: n_steps=10**6 bitwise equal to n_steps=3 (a 3-step window; expected False)\t{_same(big, n3)}")
    rows.append(f"C: t_start of n_steps=10**6\t{big.info['t_start']!r}")
    rows.append(f"C: dt * 9 - t_end of n_steps=10**6\t{big.info['dt'] * 9 - t_c!r}")
    rows.append(f"C: dP n_steps=10**6 / n_steps=3\t{big.dP!r} / {n3.dP!r}")
    (HERE / f"march_cost_{tag}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert big.ok and n9.ok and n12.ok and n3.ok
