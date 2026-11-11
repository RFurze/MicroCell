#!/usr/bin/env bash
# WP4 accounts, run in Git Bash from MicroCell/:  bash <this file> <base sha> <head sha> <base tag> <head tag>
#   e.g.  bash .../accounts.sh 68bc829 7c41fc8 g0_base_68bc829 g4_final
# Writes, beside this file (read-only on the repo: git show / git diff / git log only):
#   lines_per_file_<head>.tsv   lines of every changed file under microcell/ and tests/ at base and head,
#                               their sum, and the package total (microcell/**/*.py)
#   commits.tsv                 the code commits base..head that touch microcell/, tests/ or examples/
#   count_account_<head>.tsv    per cell: compared at base = compared at head + no longer compared
#                               (from parity_<base tag>_detail.tsv and parity_<head tag>_detail.tsv)
# exit 0 = written and the account closes in every cell, 1 = the account does not close, 2 = INCOMPLETE.
set -u
base="$1"; head="$2"; btag="$3"; htag="$4"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$here/parity_${btag}_detail.tsv" ] && [ -f "$here/parity_${htag}_detail.tsv" ] || { echo "no parity detail"; exit 2; }
nl() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }

out="$here/lines_per_file_$head.tsv"
printf 'file\tlines_%s\tlines_%s\tchange\n' "$base" "$head" > "$out"
tb=0; th=0
for f in $(git diff --name-only "$base" "$head" -- microcell tests | sort); do
  a=$(nl "$base" "$f"); b=$(nl "$head" "$f"); tb=$((tb + a)); th=$((th + b))
  printf '%s\t%s\t%s\t%s\n' "$f" "$a" "$b" "$((b - a))" >> "$out"
done
printf 'changed files\t%s\t%s\t%s\n' "$tb" "$th" "$((th - tb))" >> "$out"
pb=0; for f in $(git ls-tree -r --name-only "$base" microcell | grep '\.py$'); do pb=$((pb + $(nl "$base" "$f"))); done
ph=0; for f in $(git ls-tree -r --name-only "$head" microcell | grep '\.py$'); do ph=$((ph + $(nl "$head" "$f"))); done
printf 'microcell/**/*.py (the package, all files)\t%s\t%s\t%s\n' "$pb" "$ph" "$((ph - pb))" >> "$out"

out="$here/commits.tsv"
printf 'commit\tfiles_changed\tinsertions\tdeletions\tsubject\n' > "$out"
for c in $(git log --reverse --format=%h "$base..$head" -- microcell tests examples); do
  stat=$(git show --shortstat --format= "$c" | tail -1)
  nf=$(echo "$stat" | sed -E 's/^ *([0-9]+) file.*/\1/')
  ins=$(echo "$stat" | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+'); del=$(echo "$stat" | grep -oE '[0-9]+ deletion' | grep -oE '[0-9]+')
  printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$nf" "${ins:-0}" "${del:-0}" "$(git log -1 --format=%s "$c" | sed 's/; log docs.*//')" >> "$out"
done

out="$here/count_account_$head.tsv"
printf 'cell\tcompared_%s\tcompared_%s\tno_longer_compared\tnewly_compared\tcloses\tnames_no_longer_compared\n' "$base" "$head" > "$out"
rc=0; sb=0; sh=0; sg=0; sn=0
for cell in $(tail -n +2 "$here/parity_${btag}_detail.tsv" | cut -f1 | sort -u); do
  tail -n +2 "$here/parity_${btag}_detail.tsv" | awk -F'\t' -v c="$cell" '$1==c {print $2 "\t" $3}' | sort > "$here/.acc_b"
  tail -n +2 "$here/parity_${htag}_detail.tsv" | awk -F'\t' -v c="$cell" '$1==c {print $2 "\t" $3}' | sort > "$here/.acc_h"
  nb=$(wc -l < "$here/.acc_b" | tr -d ' '); nh=$(wc -l < "$here/.acc_h" | tr -d ' ')
  gone=$(comm -23 "$here/.acc_b" "$here/.acc_h" | cut -f2 | paste -sd, -); ng=$(comm -23 "$here/.acc_b" "$here/.acc_h" | wc -l | tr -d ' ')
  nn=$(comm -13 "$here/.acc_b" "$here/.acc_h" | wc -l | tr -d ' ')
  ok=yes; [ "$nb" -eq $((nh + ng - nn)) ] || { ok=no; rc=1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$nb" "$nh" "$ng" "$nn" "$ok" "$gone" >> "$out"
  sb=$((sb + nb)); sh=$((sh + nh)); sg=$((sg + ng)); sn=$((sn + nn))
done
mkdir -p "$here/logs"; mv "$here/.acc_b" "$here/logs/acc_last_base.txt"; mv "$here/.acc_h" "$here/logs/acc_last_head.txt"
ok=yes; [ "$sb" -eq $((sh + sg - sn)) ] || { ok=no; rc=1; }
printf 'All\t%s\t%s\t%s\t%s\t%s\t\n' "$sb" "$sh" "$sg" "$sn" "$ok" >> "$out"
tail -1 "$out"
exit $rc
