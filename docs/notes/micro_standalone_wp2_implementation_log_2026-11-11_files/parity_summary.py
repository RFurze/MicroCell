"""WP2 per-group parity summary (the numbers of the log's parity table).

The gate itself is ``pytest tests/`` (pass / fail at 1e-12).  This file runs the same
``tests/test_parity.run_cell`` on every reference cell and writes one row per cell with the
comparison counts and the largest relative difference, to ``parity_current.tsv`` beside it
(the gate script renames it ``parity_g<N>.tsv``).  Run from ``MicroCell/``:

    ../_claude/wsltest docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/parity_summary.py -q -p no:cacheprovider
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MICROCELL = HERE.parents[2]
for p in (MICROCELL, MICROCELL / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_parity_summary():
    import test_parity as tp

    cols = ["cell", "status", "n_total", "n_exact", "n_outside", "max_rel", "worst", "n_setting", "n_scalar",
            "n_diagnostic", "n_field"]
    rows = ["\t".join(cols)]
    total = exact = outside = 0
    worst_all = 0.0
    for cell in tp.reference_cells():
        res = tp.run_cell(cell)
        recs = res["records"]
        worst = max(recs, key=lambda r: r["rel"])
        n_exact = sum(1 for r in recs if r["ok"] and r["rel"] == 0.0)
        n_out = sum(1 for r in recs if not r["ok"])
        counts = [sum(1 for r in recs if r["kind"] == k) for k in ("setting", "scalar", "diagnostic", "field")]
        rows.append("\t".join(str(v) for v in [cell, res["status"], len(recs), n_exact, n_out, repr(worst["rel"]),
                                               worst["name"] if worst["rel"] > 0.0 else ""] + counts))
        total += len(recs); exact += n_exact; outside += n_out
        worst_all = max(worst_all, worst["rel"])
    rows.append("\t".join(str(v) for v in ["All", "PASS" if outside == 0 else "FAIL", total, exact, outside,
                                           repr(worst_all), "", "", "", "", ""]))
    (HERE / "parity_current.tsv").write_text("\n".join(rows) + "\n")
    assert outside == 0, f"{outside} comparisons outside 1e-12"
