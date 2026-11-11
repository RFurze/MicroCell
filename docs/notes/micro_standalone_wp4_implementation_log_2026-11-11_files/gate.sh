#!/usr/bin/env bash
# WP4 gate, run in Git Bash:  bash <this file> <tag> [tree]
#   tag   g0_base_68bc829, g1, g2a, ...
#   tree  the tree under test (default: MicroCell/, the repo this file is in); an archive of a
#         commit for the base measurement
# 1. the gate itself (pytest tests/), log to logs/gate_<tag>.log, count line appended to gate_counts.tsv
# 2. the per-cell parity summary and the per-comparison detail (parity_summary.py), written to
#    parity_<tag>.tsv and parity_<tag>_detail.tsv
# exit 0 = PASS (gate passed and no comparison outside 1e-12), 1 = FAIL, 2 = INCOMPLETE (no count line
# or no summary).  Nothing is deleted: an existing parity_current*.tsv is moved to logs/ first.
set -u
tag="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="${2:-$(cd "$here/../../.." && pwd)}"
W="/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest"
mkdir -p "$here/logs"
cd "$tree" || exit 2
"$W" tests/ -q -p no:cacheprovider -rfE --tb=short > "$here/logs/gate_$tag.log" 2>&1
rc=$?
line=$(grep -E '(passed|failed|error).* in [0-9.]+s' "$here/logs/gate_$tag.log" | tail -1)
[ -f "$here/gate_counts.tsv" ] || printf 'gate\tcount_line\n' > "$here/gate_counts.tsv"
printf '%s\t%s\n' "$tag" "$line" >> "$here/gate_counts.tsv"
echo "gate $tag: exit $rc: $line"
[ -n "$line" ] || exit 2
for f in parity_current.tsv parity_current_detail.tsv; do
  [ -f "$here/$f" ] && mv "$here/$f" "$here/logs/stale_${tag}_$f"
done
# the summary runs with the tree under test as the working directory; the script is given by its
# path relative to that tree when it is inside it, else by its WSL path
case "$here" in
  "$tree"/*) script="${here#$tree/}/parity_summary.py" ;;
  *) script="$(echo "$here/parity_summary.py" | sed -E 's|^/([a-zA-Z])/|/mnt/\L\1/|')" ;;
esac
"$W" "$script" -q -p no:cacheprovider > "$here/logs/parity_$tag.log" 2>&1
rc2=$?
[ -f "$here/parity_current.tsv" ] || { echo "no parity summary (see logs/parity_$tag.log)"; exit 2; }
mv "$here/parity_current.tsv" "$here/parity_$tag.tsv"
mv "$here/parity_current_detail.tsv" "$here/parity_${tag}_detail.tsv"
tail -1 "$here/parity_$tag.tsv"
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && exit 0
exit 1
