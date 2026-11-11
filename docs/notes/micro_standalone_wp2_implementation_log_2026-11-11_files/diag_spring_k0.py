"""WP2 diagnostic (not a gate): do the cosine reference cells, whose stored k_spring is 1e20, use the
spring?  Each is solved through the API with k_spring forced to 0.0 and compared with its reference.
Writes diag_spring_k0.tsv beside this file.  Run from MicroCell/:
    ../_claude/wsltest ../_to_delete/micro_standalone_2026-11-11/wp2_logs/diag_spring_k0.py -q -p no:cacheprovider -s
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MICROCELL = HERE.parents[2] / "MicroCell"
for p in (MICROCELL, MICROCELL / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CELLS = ("c5_cosine_fullfilm", "c6_cosine_fft_sqrt", "c8_cosine_fullfilm_steady")


def test_spring_k0():
    import test_parity as tp

    original = tp.settings_from_reference
    rows = ["cell\tk_spring_reference\tk_spring_run\tn_comparisons\tn_outside_1e-12\tmax_rel\tworst\tstatus"]
    try:
        for cell in CELLS:
            ref = tp.load_reference(cell)
            k_ref = ref["micro_case"]["k_spring"]

            def forced(r, _orig=original):
                st = _orig(r)
                st.legacy["k_spring"] = 0.0
                return st

            tp.settings_from_reference = forced
            res = tp.run_cell(cell)
            recs = res["records"]
            finite = [r for r in recs if r["rel"] != float("inf")]
            worst = max(recs, key=lambda r: r["rel"])
            rows.append("\t".join(str(v) for v in (
                cell, repr(k_ref), "0.0", len(recs), sum(1 for r in recs if not r["ok"]),
                repr(worst["rel"]), worst["name"] if worst["rel"] > 0 else "", res["status"])))
            for r in sorted(recs, key=lambda r: -r["rel"])[:12]:
                if r["rel"] > 0:
                    rows.append(f"#\t{cell}\t{r['kind']}\t{r['name']}\t{r['rel']!r}\t{r['note']}")
    finally:
        tp.settings_from_reference = original
    (HERE / "diag_spring_k0.tsv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
