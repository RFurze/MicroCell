"""Open item 1 probe (11 Nov 2026): generated surfaces in contact at the roughness and films of the
Wang 2008 and Cup90 pendulum cases, stationary / translating / pair, at the micro Newton
abs_tolerance 1e-5 (the production Sa 8 / Sa 10 / Wang value) and 1e-6 (the MicroSettings default),
with the Sa 40 nm cell of the WP5 probe repeated as the last two cells.

Run from the MicroCell root (pytest form, so the wsltest wrapper runs it):

    _claude/wsltest docs/notes/micro_standalone_open_items_implementation_log_2026-11-11_files/probe_translating_realistic.py -q -p no:cacheprovider -s

Writes ``probe_translating_realistic.tsv`` beside this file, one row per solve, flushed per row.
A NaN in a row comes with a status that is not OK.  The test passes when every solve returned a result.
"""
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import (ContactSettings, MicroSettings, RippleSettings, Roughness, Zeta,    # noqa: E402
                       microsolver)

OUT = HERE.with_suffix(".tsv")

#: Cup90 pendulum (PendClean) generated-roughness cases: MicroSettings defaults for h_min and E*
PEND = dict(h_min=1.0331e-11, pattern_flux_contact=False, ripple=RippleSettings(tol=1e-4))
#: Wang 2008 cases (examples/pair_cell.py, parity cell c4)
WANG = dict(h_min=1.7825e-11, E_star=115384615384.61537, pattern_flux_contact=False,
            contact=ContactSettings(sigma_yield=450e6, mu=0.4), ripple=RippleSettings(tol=1e-4))
#: the Sa 40 cell of the WP5 probe: MicroSettings defaults
DEFAULT = dict()


def _rough(sa_nm: float, arrangement: str) -> Roughness:
    if arrangement == "pair":
        sa = sa_nm * 1e-9 / math.sqrt(2.0)
        return Roughness.pair(stationary=Roughness.generated(Sa=sa, seed=1000),
                              translating=Roughness.generated(Sa=sa, seed=1001))
    return Roughness.generated(Sa=sa_nm * 1e-9, seed=1000, moving=(arrangement == "translating"))


#: (family, settings, level constraint, Sa nm (composite for a pair), arrangement, H nm)
CELLS = [
    ("pend", PEND, "whole_cell", 8, "stationary", 22),
    ("pend", PEND, "whole_cell", 8, "stationary", 26),
    ("pend", PEND, "whole_cell", 8, "translating", 22),
    ("pend", PEND, "whole_cell", 8, "translating", 26),
    ("pend", PEND, "whole_cell", 10, "stationary", 26),
    ("pend", PEND, "whole_cell", 10, "translating", 26),
    ("pend", PEND, "fluid_mean_v1", 8, "pair", 22),
    ("pend", PEND, "fluid_mean_v1", 10, "pair", 25),
    ("wang", WANG, "fluid_mean_v1", 10, "pair", 25),
    ("wang", WANG, "fluid_mean_v1", 12, "pair", 25),
    ("wang", WANG, "fluid_mean_v1", 12, "pair", 30),
    ("wang", WANG, "whole_cell", 8, "stationary", 22),
    ("wang", WANG, "whole_cell", 8, "translating", 22),
    ("sa40", DEFAULT, "whole_cell", 40, "stationary", 130),
    ("sa40", DEFAULT, "whole_cell", 40, "translating", 130),
]
TOLS = (1e-5, 1e-6)
COLS = ("family", "Sa_nm", "arrangement", "H_nm", "level_constraint", "abs_tolerance", "status",
        "newton_iters", "contact_iters", "ripple_passes", "wall_s", "dP_Pa", "P_asp_Pa", "contact_frac",
        "final_F", "reason")


def test_probe():
    rows = 0
    with OUT.open("w", newline="\n") as fh:
        fh.write("\t".join(COLS) + "\n")
        for family, base, level, sa_nm, arrangement, h_nm in CELLS:
            rough = _rough(sa_nm, arrangement)
            zeta = Zeta(H=h_nm * 1e-9, P=40.8e6, Ux=0.02)
            for tol in TOLS:
                st = MicroSettings(**base, level_constraint=level, abs_tolerance=tol)
                t0 = time.time()
                out = microsolver(zeta, t_end=0.02, roughness=rough, settings=st, n_steps=3)
                wall = time.time() - t0
                m = re.search(r"\|\|F\|\| = ([0-9.eE+-]+)", out.reason or "")
                assert out.status == "OK" or math.isnan(out.dP)
                fh.write("\t".join(str(v) for v in (
                    family, sa_nm, arrangement, h_nm, level, f"{tol:g}", out.status,
                    out.info.get("newton_iters"), out.info.get("contact_iters"),
                    out.info.get("ripple_passes"), f"{wall:.2f}", f"{out.dP:.6e}", f"{out.P_asp:.6e}",
                    f"{out.contact_frac:.6f}", m.group(1) if m else "", (out.reason or "").replace("\t", " "),
                )) + "\n")
                fh.flush()
                rows += 1
    assert rows == len(CELLS) * len(TOLS)
