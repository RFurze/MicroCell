#!/usr/bin/env bash
# WP5 group 1, part 1: build the four merged module texts (no file under microcell/ is touched).
# Run from Git Bash on the tree at d3b88ae.  Writes _to_delete/wp5_2026-11-11/group1_build/*.py.
# Part 2 (the git mv / git rm, the copies into microcell/ and the import edits) is group1_apply.txt.
# Code moves verbatim.  What changes: module docstrings (the old ones are kept inside the new one),
# import lines (re-pointed; the imports between merged files are dropped), and __all__
# (samplers.py: dropped; contact.py: FFT and ContactResolutionError added).
# The line numbers below are those of the files at d3b88ae.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
M=microcell
B=_to_delete/wp5_2026-11-11/group1_build
mkdir -p "$B"

doc()  { sed -n "1,$(( $2 - 1 ))p" "$1" | sed '1s/^"""//'; }          # file, line of the closing quotes
body() { f=$1; n=$2; shift 2; sed -e "1,${n}d" -e "$(( n + 1 ))d" "$@" "$f"; }   # after docstring and __future__
part() { printf '\n``%s`` (the parent'"'"'s ``%s``)\n' "$1" "$2"; printf '%s\n' "------------------------------------------------------------------------------------------"; }

# ---------------------------------------------------------------- shear.py
{
  printf '%s\n' '"""Shear traction of a micro cell: the reduction of sampled traction, the fluid statistics, and the'
  printf '%s\n' 'asperity and total statistics.'
  printf '%s\n' ''
  printf '%s\n' 'MicroCell WP5 (11 Nov 2026): ``shear_reduce.py``, ``fluid_shear.py`` and ``contact_shear.py`` in one'
  printf '%s\n' 'file, in that order.  The code is moved verbatim; the imports between the three are gone.  Their'
  printf '%s\n' 'module docstrings follow.  Where one names one of the three files, read this file.'
  part shear_reduce.py ehl/shear_reduce.py;   doc $M/shear_reduce.py 39
  part fluid_shear.py ehl/fluid_shear.py;     doc $M/fluid_shear.py 42
  part contact_shear.py ehl/contact_shear.py; doc $M/contact_shear.py 43
  printf '%s\n' '"""' 'from __future__ import annotations'
  body $M/shear_reduce.py 39
  printf '\n\n'
  body $M/fluid_shear.py 42 -e '50,51d'
  printf '\n\n'
  body $M/contact_shear.py 43 -e '51d'
} > "$B/shear.py"

# ---------------------------------------------------------------- samplers.py
{
  printf '%s\n' '"""Samplers of a solved micro cell: the fluid pressure and traction samples, and the contact pressure'
  printf '%s\n' 'and traction at the same sample points.'
  printf '%s\n' ''
  printf '%s\n' 'MicroCell WP5 (11 Nov 2026): ``micro_diagnostics.py`` and ``contact_sample.py`` in one file, in that'
  printf '%s\n' 'order.  The code is moved verbatim; the imports between the two are gone and the others point at'
  printf '%s\n' 'the merged modules (``film``, ``shear``).  Their module docstrings follow.  Where one names one of'
  printf '%s\n' 'the two files, read this file.'
  part micro_diagnostics.py microscale/src/functions/micro_diagnostics.py; doc $M/micro_diagnostics.py 49
  part contact_sample.py microscale/src/functions/contact_sample.py;       doc $M/contact_sample.py 98
  printf '%s\n' '"""' 'from __future__ import annotations'
  body $M/micro_diagnostics.py 49 -e '59,61d' -e '358d' -e '384d' \
       -e '255s/microcell\.ideal_ht/microcell.film/' \
       -e '330s/microcell\.fluid_shear/microcell.shear/' \
       -e '331s/microcell\.shear_reduce/microcell.shear/' \
       -e '382s/import contact_shear as cs/import shear as cs/' \
       -e '383s/microcell\.shear_reduce/microcell.shear/' \
       -e '429s/import contact_shear as cs/import shear as cs/'
  printf '\n\n'
  body $M/contact_sample.py 98 -e '106s/microcell\.shear_reduce/microcell.shear/'
} > "$B/samplers.py"

# ---------------------------------------------------------------- film.py
{
  printf '%s\n' '"""The film callbacks ``ht`` of the micro solver: the analytic cosine cell and the generated surfaces.'
  printf '%s\n' ''
  printf '%s\n' 'MicroCell WP5 (11 Nov 2026): ``ideal_ht.py`` and ``surface_ht.py`` in one file, in that order.  The'
  printf '%s\n' 'code is moved verbatim.  Their module docstrings follow.  Where one names one of the two files,'
  printf '%s\n' 'read this file.'
  part ideal_ht.py microscale/src/functions/ideal_ht.py;     doc $M/ideal_ht.py 56
  part surface_ht.py microscale/src/functions/surface_ht.py; doc $M/surface_ht.py 28
  printf '%s\n' '"""' 'from __future__ import annotations'
  body $M/ideal_ht.py 56
  printf '\n\n'
  body $M/surface_ht.py 28
} > "$B/film.py"

# ---------------------------------------------------------------- contact.py
{
  printf '%s\n' '"""Micro contact: the contact identity and error, and the periodic elastic (FFT) contact solve.'
  printf '%s\n' ''
  printf '%s\n' 'MicroCell WP5 (11 Nov 2026): ``contact.py`` and ``elastic_contact.py`` in one file, in that order.'
  printf '%s\n' 'The code is moved verbatim; the import between the two is gone.  Their module docstrings follow.'
  printf '%s\n' 'Where one names ``elastic_contact``, read this file.'
  part contact.py ehl/contact.py;                 doc $M/contact.py 5
  part elastic_contact.py ehl/elastic_contact.py; doc $M/elastic_contact.py 41
  printf '%s\n' '"""' 'from __future__ import annotations'
  sed -n '6,$p' $M/contact.py
  printf '\n\n'
  body $M/elastic_contact.py 41 -e '48d' -e '50s/$/\n    "FFT", "ContactResolutionError",/'
} > "$B/contact.py"

