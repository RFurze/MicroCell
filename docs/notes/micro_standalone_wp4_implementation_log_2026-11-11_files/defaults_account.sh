#!/usr/bin/env bash
# WP4: what became of every caller-side default of the base.  Run in Git Bash:
#   bash <this file> <settings tag>      e.g. g4_7c41fc8
# Reads inventory_settings.tsv (the base, 68bc829) and settings_<tag>.tsv (now); writes
# defaults_account_<tag>.tsv: one row per base row of Zeta, MicroSettings, ContactSettings,
# RippleSettings and LEGACY_DEFAULTS, with where the value is now and whether the default is the
# same text.  exit 0 = no default differs, 1 = one differs.
set -u
tag="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/defaults_account_$tag.tsv"
awk -F'\t' -v OFS='\t' '
  function norm(v) { gsub(/\047/, "", v); return v }      # the base inventory wrote strings unquoted
  FNR == 1 { next }
  FILENAME ~ /settings_[^\/]*$/ && FILENAME !~ /inventory_settings/ { now[$1 "." $2] = $3; next }
  $1 == "Zeta" || $1 == "MicroSettings" || $1 == "ContactSettings" || $1 == "RippleSettings" || $1 == "LEGACY_DEFAULTS" {
    key = $1 "." $2; to = key
    if ($1 == "LEGACY_DEFAULTS") {
      to = "MicroSettings." $2
      if ($2 == "beta") to = "fixed value (solver module).BETA"
      if ($2 == "patch_pin") to = "fixed value (solver module).PATCH_PIN"
      if ($2 == "patch_pin_kappa") to = "fixed value (solver module).PATCH_PIN_KAPPA"
    }
    if (($1 == "ContactSettings" || $1 == "RippleSettings") && $2 == "E_star") to = "MicroSettings.E_star"
    if (to in now) { what = (norm(now[to]) == norm($3)) ? "same default" : "DEFAULT DIFFERS"; print key, $3, to, now[to], what; seen[to] = 1 }
    else print key, $3, "", "", "removed"
  }
  END { for (k in now) if (!(k in seen) && k !~ /^(fixed value|microsolver argument)/) print "", "", k, now[k], "new" }
' "$here/settings_$tag.tsv" "$here/inventory_settings.tsv" > "$here/.defaults_body"
{ printf 'base\tbase_default\tnow\tnow_default\twhat\n'; cat "$here/.defaults_body"; } > "$out"
mkdir -p "$here/logs"; mv "$here/.defaults_body" "$here/logs/defaults_last_body.txt"
cut -f5 "$out" | tail -n +2 | sort | uniq -c
grep -q "DEFAULT DIFFERS" "$out" && exit 1
exit 0
