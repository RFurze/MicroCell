#!/usr/bin/env bash
# WP3 gate, run from MicroCell/ in Git Bash:  bash <this file> <tag>      (tag: g0_base_c89915a, g1, ...)
# 1. the gate itself (pytest tests/), log to logs/gate_<tag>.log, count line appended to gate_counts.tsv
# 2. the per-cell parity summary (parity_summary.py), written to parity_<tag>.tsv
# exit 0 = PASS (gate passed and no comparison outside 1e-12), 1 = FAIL, 2 = INCOMPLETE (no count line)
set -u
tag="$1"
F="docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files"
W="/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest"
"$W" tests/ -q -p no:cacheprovider -rfE --tb=short > "$F/logs/gate_$tag.log" 2>&1
rc=$?
line=$(grep -E '(passed|failed|error).* in [0-9.]+s' "$F/logs/gate_$tag.log" | tail -1)
[ -f "$F/gate_counts.tsv" ] || printf 'gate\tcount_line\n' > "$F/gate_counts.tsv"
printf '%s\t%s\n' "$tag" "$line" >> "$F/gate_counts.tsv"
echo "gate $tag: exit $rc: $line"
[ -n "$line" ] || exit 2
rm -f "$F/parity_current.tsv"
"$W" "$F/parity_summary.py" -q -p no:cacheprovider > "$F/logs/parity_$tag.log" 2>&1
rc2=$?
[ -f "$F/parity_current.tsv" ] || { echo "no parity summary"; exit 2; }
mv "$F/parity_current.tsv" "$F/parity_$tag.tsv"
tail -1 "$F/parity_$tag.tsv"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && exit 0
exit 1
