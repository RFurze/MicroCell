"""Every c5 comparison outside the tolerance with k_spring forced to 0, by name (WP2 log section 2).

diag_spring_k0.py counted the records that fail but listed only those with a non-zero relative
difference, so a failing integer or string record was counted and not named.  This lists all of
them.  Run as a pytest file from MicroCell/ (the wsltest wrapper):

    _claude/wsltest docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/diag_spring_k0_names.py -q -p no:cacheprovider

Writes diag_spring_k0_names.tsv beside this file.  The test passes when the file is written.

Runs only on a tree before b18879b: that commit removed the spring and the ``k_spring`` key
this script sets (the c5 reference was re-made with k_spring = 0 at a1395f4).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

CELL = "c5_cosine_fullfilm"


def test_spring_k0_names():
    import test_parity as tp

    original = tp.settings_from_reference

    def forced(r, _orig=original):
        st = _orig(r)
        st.legacy["k_spring"] = 0.0
        return st

    tp.settings_from_reference = forced
    try:
        res = tp.run_cell(CELL)
    finally:
        tp.settings_from_reference = original
    bad = [r for r in res["records"] if not r["ok"]]
    rows = ["cell\tkind\tname\trel\tnote"]
    for r in sorted(bad, key=lambda r: -r["rel"]):
        rows.append(f"{CELL}\t{r['kind']}\t{r['name']}\t{r['rel']!r}\t{r['note']}")
    rows.append(f"# {len(res['records'])} comparisons, {len(bad)} outside, status {res['status']}")
    (HERE / "diag_spring_k0_names.tsv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
