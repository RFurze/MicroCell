"""What each reference cell stored for the options WP2 removed (stdlib only, no numpy).

    python -I docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/reference_removed_options.py

Reads tests/references/*.npz (the JSON members) and writes reference_removed_options.tsv beside
this file: one row per (option, cell) with the stored value, the value the pruned code hard-wires,
and whether they agree.  exit 0 = every row agrees except the ones marked KEPT (an option a
reference uses, which was therefore not removed), 1 = a removed option is used by a reference.
"""
import ast
import json
import pathlib
import struct
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REFS = HERE.parents[2] / "tests" / "references"
ENV_READS = ("MICRO_DEBUG_LOG", "MICRO_NONDIM_DIAG", "MICRO_UREF_FLOOR_OMEGA", "MICRO_EPSP_ND_CAP", "MICRO_SPOS_FORM")


def load(path):
    z = zipfile.ZipFile(path)
    d = {}
    for name in z.namelist():
        if not name.endswith("_json.npy"):
            continue
        b = z.read(name)
        assert b[:6] == b"\x93NUMPY"
        if b[6] == 1:
            hl, off = struct.unpack("<H", b[8:10])[0], 10
        else:
            hl, off = struct.unpack("<I", b[8:12])[0], 12
        hdr = ast.literal_eval(b[off:off + hl].decode("latin1"))
        assert hdr["descr"].startswith("<U"), hdr
        d[name[:-9]] = json.loads(b[off + hl:].decode("utf-32-le").rstrip("\x00"))
    return d


def rows_for(cell, d):
    st, mc, env = d["settings"], d["micro_case"], d["env"]
    mixed = bool(mc["mixed_enabled"])
    pair = bool(d["meta"].get("offset")) or str(st.get("phase_average")) == "contact_table_v1"
    out = []

    def row(group, option, stored, frozen, ok=None, note=""):
        if ok is None:
            ok = (stored == frozen)
        out.append((group, option, cell, repr(stored), repr(frozen), "same" if ok else "DIFFERENT", note))

    row("2", "settings.spring_pointwise", st["spring_pointwise"], False)
    k = float(mc["k_spring"])
    row("2", "micro_case.k_spring", k, 0.0,
        note="" if k == 0.0 else "KEPT: the cell-average spring is not removed (log section 2)")
    if mixed:
        row("3", "settings.contact_model", st["contact_model"], "fft_elastic_v1")
        row("3", "settings.use_reduced_modulus", st["use_reduced_modulus"], False)
        row("3", "settings.overclosure_measure", st["overclosure_measure"], "patch_max")
    else:
        row("3", "result.contact_model (full film)", d["result"]["contact_model"], "off",
            ok=(d["result"]["contact_model"] == "legacy_patch_v1"),
            note="the parent's default tag on a cell with no contact; reported as 'off' now")
    row("5", "settings.gauge_lift", st["gauge_lift"], False)
    row("5", "settings.spos_form is not c2", st["spos_form"], "spline or sqrt", ok=st["spos_form"] in ("spline", "sqrt"))
    if mixed:
        row("8", "settings.phase_average -> pair_branch", st["phase_average"], bool(pair),
            ok=({"off": False, "contact_table_v1": True}[st["phase_average"]] == bool(d["meta"].get("offset"))),
            note="pair cell" if d["meta"].get("offset") else "")
    for v in ENV_READS:
        row("9", "env." + v, env.get(v), None, note="reference made with the variable unset" if v not in env else "")
    return out


def main():
    rows = []
    for p in sorted(REFS.glob("*.npz")):
        rows += rows_for(p.stem, load(p))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    head = ("group", "option", "cell", "reference_value", "frozen_value", "agree", "note")
    with open(HERE / "reference_removed_options.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(head) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    bad = [r for r in rows if r[5] != "same" and not r[6].startswith("KEPT")]
    kept = [r for r in rows if r[6].startswith("KEPT")]
    print(f"{len(rows)} rows; {len(kept)} kept (used by a reference, not removed); {len(bad)} removed-but-used")
    for r in kept + bad:
        print("  ", *r[:5], r[6])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
