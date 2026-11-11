"""WP4 per-group parity summary (the numbers of the log's parity table and of the count account).

The gate itself is ``pytest tests/`` (pass / fail at 1e-12).  This file runs the same
``tests/test_parity.run_cell`` on every reference cell and writes, beside this file:

  parity_current.tsv          one row per cell: counts by kind, exact, outside 1e-12, largest relative difference
  parity_current_detail.tsv   one row per comparison: cell, kind, name, rel, ok

The gate script renames them ``parity_<tag>.tsv`` and ``parity_<tag>_detail.tsv``.  Run from the
tree under test (``MicroCell/`` or an archive of a commit):

    <RefactorRefactor>/_claude/wsltest <path to this file> -q -p no:cacheprovider

The tree under test is the current working directory, not this file's location, so the same
script measures the base commit from an archive.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()
for p in (TREE, TREE / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_parity_summary():
    import test_parity as tp

    assert Path(tp.__file__).resolve().parent == TREE / "tests", f"test_parity imported from {tp.__file__}"
    cols = ["cell", "status", "n_total", "n_exact", "n_outside", "max_rel", "worst", "n_setting", "n_scalar",
            "n_diagnostic", "n_field"]
    rows = ["\t".join(cols)]
    detail = ["\t".join(["cell", "kind", "name", "rel", "ok"])]
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
        detail += ["\t".join([cell, r["kind"], r["name"], repr(r["rel"]), str(r["ok"])]) for r in recs]
        total += len(recs); exact += n_exact; outside += n_out
        worst_all = max(worst_all, worst["rel"])
    rows.append("\t".join(str(v) for v in ["All", "PASS" if outside == 0 else "FAIL", total, exact, outside,
                                           repr(worst_all), "", "", "", "", ""]))
    (HERE / "parity_current.tsv").write_text("\n".join(rows) + "\n")
    (HERE / "parity_current_detail.tsv").write_text("\n".join(detail) + "\n")
    assert outside == 0, f"{outside} comparisons outside 1e-12"
