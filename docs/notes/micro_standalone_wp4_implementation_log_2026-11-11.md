# Standalone micro-cell solver: WP4 implementation log, settings and results collapse (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp4_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (work package WP4).
Base: `MicroCell` `main` at 68bc829. Parent `MultiscaleMixedBallinSocket` not touched.
`F/` below is `docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/`.

## 0. Summary

**State: WP4 is part built and waits on Rob.** Built and committed on `main` (Table 1): group 1,
the answer-independent part of group 2 (2a) and of group 3 (3a). Not built: group 2b, the rest
of group 3 and group 4, which depend on the eight questions of §10. Nothing is pushed. WP5 is
not started.

**Gate (measured).** After every commit all eight reference cells PASS at relative 1e-12 and
every comparison is exact (Table 5). The comparison count is unchanged, per cell and by name
(Table 6). No reference was regenerated and no tolerance changed. The test count rose by the
five tests of the new `tests/test_results.py`.

**Settings (measured, Table 2).** The five solver-side settings and parameter containers and
`api._solver_settings` are gone. My call, as the brief allowed: no solver-side container is
kept; the solver classes read the caller's `Zeta` and `MicroSettings` and the call's values
directly (§2). No default changed (§2). The six `legacy` keys still travel in
`MicroSettings.legacy` until question 1 is answered.

**Results (measured, Table 3).** `MicroResult.raw`, `api._RAW_DEFAULTS` and `api._failed_raw`
are gone. Each output is in one place: a named output or `info`. A failed result is built in
one function and carries the same `info` entries as an OK result (§3).

**NaN and status.** Group 4 is not built. Table 4 is the NaN measured in the tree as it stands.

**Changed test expectations.** Two, both in how the gate reads the code, neither in a number
(§7).

**Departures.** Four, numbered in §8. The one Rob may want to reverse: 8.2, a failed
result's `info` now has every entry of an OK result, which changes three values a caller of a
failed full-film or `whole_cell` solve could have read.

**For Rob.** The eight questions (§10; recommendation A in each). Then, in a fresh session:
groups 2b, 3b and 4, the final script runs, the README and the second pass of this log (§11).
Push `main` when he chooses.

**Open items** §11; candidates left in place §9.

## 1. What was built

Table 1. Commits, in the order made. Source: `F/commits.tsv` (git `--shortstat`).

| Group | Commit | Files changed | Insertions | Deletions | What |
|---|---|---|---|---|---|
| 1 decided removals | f77fd1d | 2 | 3 | 6 | `ContactSettings.nu`, `.k_asp_exponent` removed |
| 2a settings, the 1:1 part | 9f719fb | 12 | 451 | 501 | solver classes read `Zeta` and `MicroSettings`; five containers and `api._solver_settings` removed |
| 3a results, the answer-independent part | 7b57abb | 3 | 259 | 132 | `MicroResult.raw`, `_RAW_DEFAULTS`, `_failed_raw` removed; one failed-result builder; `RESULT_NOW`; `tests/test_results.py` |

Author and committer of each: `RFurze <py07rf@leeds.ac.uk>`; no attribution lines.

Step 1 (no edits) wrote the three inventories at the base: `F/inventory_settings.tsv`,
`F/inventory_results.tsv`, `F/inventory_nan_paths.tsv` (with `F/inventory_nan_observed.tsv`,
`F/inventory_info_fields.tsv`, `F/inventory_settings_other.tsv`), made by `F/inventory.py`.
That script reads the base tree and no longer runs in the current one; to repeat it, run it
in `_to_delete/wp4_2026-11-11/base_source_68bc829/` (a `git archive` of the base).

## 2. Settings: before and after

Table 2. Rows per container. Sources: base `F/inventory_settings.tsv`; now
`F/settings_g3a_7b57abb.tsv` and `F/counts_g3a_7b57abb.tsv` (made by `F/tables.py`).

| Container | Rows at 68bc829 | Rows at 7b57abb |
|---|---|---|
| `Zeta` | 12 | 12 |
| `MicroSettings` | 20 | 20 |
| `ContactSettings` | 8 | 6 |
| `RippleSettings` | 4 | 4 |
| `LEGACY_DEFAULTS` (through `MicroSettings.legacy`) | 6 | 6 |
| `MicroPhysicalParameters` | 20 | gone |
| `MicroSolverSettings` | 10 | gone |
| `PhaseNeutralSettings` (base class) | 17 | gone |
| `PhaseNeutralSettings` (mixed class) | 1 | gone |
| `PhaseNeutralContactSettings` | 8 | gone |
| fixed values of the solver module | not a container | 10 |
| `microsolver` arguments other than `zeta`, `roughness`, `settings` | not listed | 6 |

At the base 50 of the 106 rows named a quantity that a second container also defined
(column `other_default` of `F/inventory_settings.tsv`), and nine of those pairs disagreed in
value (for example `level_constraint`: `whole_cell` in `MicroSettings`, `fluid_mean_v1` in
`PhaseNeutralSettings`). `api._solver_settings` copied one into the other on every call.

**What the solver reads now (my call: no solver-side container).** The constructor is
`SolverCls(mesh_m, zeta, t_end, settings, ht, velocity_ratio, pair, auto_solve)`. The classes
read `self.zeta` (a `Zeta` that `microsolver` builds, rates zeroed when steady), `self.t_end`,
`self.settings` (the caller's `MicroSettings`, with `.contact` and `.ripple`),
`self.velocity_ratio`, `self.pair` and `self.h_min`. `self.h_min` is 0.0 in the full-film
class and `settings.h_min` in the mixed class (`_contact_onset_film()`), which is what the
removed remapping did. Keeping one internal container would have kept a second name for every
setting; reading the API objects leaves one.

**One default per setting.** Column `defined_at` of `F/settings_g3a_7b57abb.tsv` gives the
file and line of each definition. The values did not change: of the 50 base rows of `Zeta`,
`MicroSettings`, `ContactSettings`, `RippleSettings` and `LEGACY_DEFAULTS`, 48 are present with
the same default and two are the removed `nu` and `k_asp_exponent` (compare column `default`
of the two files).

**Fixed values.** The seven damping knobs and `print_newton`, `print_timing` were dataclass
fields the API could not set; they are module constants in
`microcell/micro_phase_neutral_class.py`. The tenth, `TRACE_NEWTON`, replaces a hidden switch:
the solver read `getattr(settings, "trace_newton", False)`, which no dataclass defined.
`pressure.LEVEL_CONSTRAINT_RUN_DEFAULT` (a second default of `level_constraint`) is removed.
The contact-law tag `fft_elastic_v1` had four definitions; it has one, `microcell.contact.FFT`.

**`MicroSettings.check()`** now refuses, before a mesh is built, what the solver classes
refused later: a non-finite `jacobian_reg_eps`; a ripple `E_star` that is not finite and
positive, `tol` <= 0, `maxpass` < 1, `omega` outside (0, 1]; a contact `E_star` that is not
finite and positive, a `k_asp_mode` other than `off` or `closure`; `h_min` not finite and
positive with contact on; ripple and contact moduli that differ by more than 1e-12 relative.
It does not refuse a non-finite `contact.mu`: the solver returns `unavailable` for the
asperity shear (path N12 of `F/inventory_nan_paths.tsv`); that belongs to group 4.

**Still two layers (waits on question 1).** `LEGACY_DEFAULTS`, `MicroSettings.legacy` and
`legacy_value()`, and the six `legacy.*` rows of `SETTINGS_NOW`.

## 3. Results: before and after

Table 3. Where the outputs of a result are held. Sources: base `F/inventory_results.tsv`
(columns `in_raw`, `read_by`) and `F/inventory_info_fields.tsv`; now
`F/results_g3a_7b57abb.tsv` and `F/counts_g3a_7b57abb.tsv`.

| Place | Entries at 68bc829 | Entries at 7b57abb |
|---|---|---|
| named outputs of `MicroResult` | 18 | 18 |
| `raw` | 83 | gone |
| of which also a named output | 18 | |
| of which also written to `info` | 16 | |
| of which read by nothing but the gate | 49 | |
| `info`, without `info["fields"]` | 23 | 67 |
| `info["fields"]` (field diagnostics; 10 in the full-film cells c5, c8) | 12 | 12 |
| `fields` (names over the 8 cells) | 18 | 18 |

**Layout.** The 18 named outputs are unchanged. Everything else of the parent's
`MicroTaskResult` is in `info` under the parent's name: statuses and tags, counters, the
pressure ledger, ripple and contact numbers, the sampled shear statistics. `info` also holds
the call (`steady`, `t_end`, `n_steps`, `offset`, `roughness_velocity_ratio`, `t_start`, `dt`),
`mesh_n`, `wall_time`, `contact_eval_s` and the `info["fields"]` diagnostics.

**Not written any more.** Four of the parent's fields were second copies of an extremum:
`p_fluid_sample_max_Pa` and `p_max_phase_mean` (both `p_max` with one phase per call),
`p_fluid_sample_min_Pa` and `p_min_phase_mean` (both `p_min`). `failure_reason` was a second
copy of `out.reason`. The gate still compares each of the five against the one place the
value is held (`RESULT_NOW` in `tests/test_parity.py`), so the count did not fall.

**The name map.** `RESULT_NOW` lists 24 of the 96 parent result fields
(`F/counts_g3a_7b57abb.tsv`): 11 are named outputs of the same name and 13 are held under
another name or in another place. A field not listed is `info[<the same name>]`. The gate
now also fails if `info` holds an entry that is neither a parent result field nor one of the
call's entries (`INFO_OF_THE_CALL`); no comparison compared at the base lacks a place.

**The failed result.** `api._failed_result()` is the one place it is built: every named output
NaN, `yield_flag` 0, no fields, and `info` with the entries of an OK result (NaN numbers,
`failed` statuses). `tests/test_results.py` forces five failures and checks this against the
OK result of the same cell (Table 4). What changed for a caller is departure 8.2.

**Left as they are until questions 3 and 4 are answered:** the 11 global-x (`gx_*`) statistics
and the field `p_fluid_masked`.

## 4. NaN and status, as measured at 7b57abb

Group 4 (one table in code, and its test) is not built. Table 4 is what the tree returns now.
Source: `F/nan_observed_g3a_7b57abb.tsv` (the 8 reference cells and the five forced failures of
`tests/test_results.py`); counts in `F/counts_g3a_7b57abb.tsv`.

| NaN | Where | Cases | Overall status | Status that reports it today |
|---|---|---|---|---|
| `tau_asperity`, `tau_total` | named | c5, c8 (full film) | OK | `shear_asperity_status`, `shear_total_status` = `unavailable` |
| `shear_asperity_support_fraction` and the four `shear_contact_*` numbers | `info` | c5, c8 | OK | the same |
| the 11 global-x (`gx_*`) statistics | `info` | all 8 cells | OK | `no_frame` (c5, c8: `unavailable` for the asperity and total ones) |
| `p_fluid_masked` at the contact vertices | `fields` | c2, c3, c4, c6, c7 | OK | none |
| every named output but `yield_flag`; the numbers of `info` | named, `info` | the five forced failures | `FAILED_*` | the overall status; the four per-quantity statuses are `failed` (full film: asperity and total stay `unavailable`) |

On OK results the NaN are in 2 named outputs, 16 `info` entries and 1 field (distinct names).
The failed results hold NaN in 50 to 55 named and `info` entries; the two extra of the three
failures forced in `run` are `t_start` and `dt`, which the solver had not set.
Paths not seen in any cell, by code reading only: N7 to N18 of `F/inventory_nan_paths.tsv`.

## 5. Parity per commit

Table 5. Sources: `F/gate_counts.tsv` (the pytest count line of
`tests/ -q -p no:cacheprovider -rfE --tb=short`), `F/parity_<tag>.tsv` (row `All`).

| State | Tag | Tests | Cells PASS | Comparisons | Exact | Outside 1e-12 | Largest relative difference |
|---|---|---|---|---|---|---|---|
| base 68bc829 | `g0_base_68bc829` | 13 passed in 9.44s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 1, f77fd1d | `g1` | 13 passed in 8.28s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 2a, 9f719fb | `g2a_final` | 13 passed in 8.41s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 3a, 7b57abb | `g3a_final` | 18 passed in 11.42s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |

`F/gate_counts.tsv` has three more rows. `g2a` twice: the first, `11 failed, 2 passed in
1.07s`, was a syntax error of mine in an f-string (Python 3.11), no solve ran, and its log was
overwritten by the second run (`13 passed in 8.90s`) after the fix; `g2a_final` followed the
removal of `LEVEL_CONSTRAINT_RUN_DEFAULT`. `g3a` (`13 passed in 8.60s`) is the tree before
`tests/test_results.py` was added. The cause of the first `g2a` row is my account; no file
records it.

The gate run before the first edit (the brief's command, run directly) printed
`13 passed in 8.95s`, which `F/questions_2026-11-11.md` quotes. That run kept no log. The base
row of Table 5 is a second run, through `F/gate.sh` on the archive of 68bc829.

Table 6. Count account, base against 7b57abb. Source: `F/count_account_7b57abb.tsv` (from the
two `_detail.tsv` files, by cell, kind and name).

| Cell | Compared at base | Compared now | No longer compared | Newly compared |
|---|---|---|---|---|
| c1 | 155 | 155 | 0 | 0 |
| c2 | 155 | 155 | 0 | 0 |
| c3 | 154 | 154 | 0 | 0 |
| c4 | 154 | 154 | 0 | 0 |
| c5 | 131 | 131 | 0 | 0 |
| c6 | 153 | 153 | 0 | 0 |
| c7 | 155 | 155 | 0 | 0 |
| c8 | 131 | 131 | 0 | 0 |

The three negative controls and `test_generated_surface_reproduces_reference` pass unchanged
(they are among the 13).

## 6. Lines per file

Table 7. Source: `F/lines_per_file_7b57abb.tsv` (`git show <commit>:<file> | wc -l`).

| File | 68bc829 | 7b57abb | Change |
|---|---|---|---|
| `microcell/api.py` | 638 | 581 | -57 |
| `microcell/micro_phase_neutral_class.py` | 2804 | 2732 | -72 |
| `microcell/micro_phase_neutral_class_contact.py` | 360 | 333 | -27 |
| `microcell/micro_phase_neutral_class_mixed.py` | 718 | 706 | -12 |
| `microcell/fields.py` | 196 | 195 | -1 |
| `microcell/micro_diagnostics.py` | 459 | 458 | -1 |
| `microcell/pressure.py` | 102 | 101 | -1 |
| `microcell/roughness/runtime.py` | 211 | 212 | 1 |
| `microcell/contact_sample.py`, `contact_shear.py`, `surface_ht.py` | 395, 269, 88 | the same | 0 |
| the package, all `microcell/**/*.py` | 8672 | 8502 | -170 |
| `tests/test_parity.py` | 474 | 602 | 128 |
| `tests/test_results.py` | new | 116 | 116 |

## 7. Changed test expectations

7.1 **Settings are compared with what the solver reads now.** At the base the gate compared
each field of the parent's settings dataclass with the same field of the copied dataclass
that `api._solver_settings` built. That dataclass is gone. `SETTINGS_NOW` in
`tests/test_parity.py` maps each of the 36 parent settings fields to where the solver reads
it (a `MicroSettings` field, a `legacy` key, a fixed value of the solver module, the contact
law, "ripple on", the pair flag), and `settings_now()` reads the values there.

7.2 **Ripple fields of a reference made without the ripple.** Such a reference stores the
parent's dataclass defaults of `fluid_E_star`, `ripple_tol`, `ripple_maxpass`, `ripple_omega`,
which the parent never read. At the base they were compared with the copied dataclass's
equal defaults. Now they pass when the reference's `fluid_deformation` is `off` and the
settings carry no ripple. The number of comparisons is the same.

Results are read through `RESULT_NOW` (§3); every value compared at the base is still
compared, with the same tolerance.

## 8. Departures from the brief

8.1 **Groups 2 and 3 are each split** into an answer-independent commit (2a, 3a) and a part
that waits (2b, 3b). The brief asked for one commit per group; it also said to carry on with
what does not depend on an answer.

8.2 **A failed result's `info` changed** (group 3a). At the base it held the call, four
counters and seven statuses and tags; the NaN numbers were in `raw`. Now it holds the entries
of an OK result. Three values differ from the base (measured: the forced-failure rows of
`F/inventory_nan_observed.tsv` against `F/nan_observed_g3a_7b57abb.tsv`; `level_constraint`
by reading `api.py` at 7b57abb): `shear_asperity_status` and `shear_total_status` of a failed
full-film solve were `None` and are `unavailable`; `level_constraint` of a failed
`whole_cell` solve was `None` and is `whole_cell`. No status code changed its meaning and none
was added.

8.3 **A test file was added in group 3**, `tests/test_results.py` (five tests). The brief
places the new test in group 4. This one checks only the layout of a failed result, which
the parity gate cannot reach; group 4's NaN-to-status test will extend it.

8.4 **This log is filed before WP4 is complete**, so that the commits' log reference exists
and a fresh session can finish from it. It will be revised in place (pre-edit copy to
`_to_delete/`) when the remaining groups are built.

## 9. Candidates left in place

- `reset_task()` of the base solver class: no caller in MicroCell.
- The branches under the constants `PRINT_NEWTON`, `PRINT_TIMING`, `TRACE_NEWTON` (all False)
  and `_print_timing_summary`: unreachable.
- `ripple.FLUID_DEFORMATION_MODES`, `ripple.deformation_model_tag`: unused.
- `micro_diagnostics.sample_contact_traction`: its `else: raise ValueError("unknown contact
  model")` cannot be reached with one contact law.
- The level-constraint counts `n_whole_cell`, `n_fluid_mean`, `n_no_fluid` are reported as 0
  with `whole_cell`, as the parent did, whatever the solver counted.
- `tools/make_references.py:34`: its docstring names `MicroPhysicalParameters` (it reads the
  parent, where the class exists).

## 10. Questions to Rob (asked 11 Nov 2026; not answered at the time of writing)

Full text and evidence tables: `F/questions_2026-11-11.md`. A is the recommendation in each.

| # | Question | A | Other options | Blocks |
|---|---|---|---|---|
| 1 | the six `legacy` keys | `delta_h`, `eps_solid`, `eps_smooth` become `MicroSettings` fields; `beta`, `patch_pin`, `patch_pin_kappa` fixed | B all fixed; C all fields | 2b |
| 2 | defaults | keep the GenSa40-ripple values | B `level_constraint = "fluid_mean_v1"` | 2b |
| 3 | NaN on an OK result | per-quantity status table; new overall `FAILED_NONFINITE` for an unexplained NaN; `p_fluid_masked` leaves `out.fields` | B keep `p_fluid_masked` with a mask rule; C no NaN on OK at all | 3b, 4 |
| 4 | global-x shear (`gx_*`) | remove the 11 outputs, `shear_frame` and the `rotation=` path | B keep | 3b, 4 |
| 5 | `n_steps=None` | leave for WP5 | B build now; C drop | nothing in WP4 if A |
| 6 | renames toward the target layout | first commit of WP5 | B end of WP4; C never | nothing in WP4 if A |
| 7 | `MicroSettings.rho0` (no reader) | remove | B keep | 2b |
| 8 | one modulus | a single `MicroSettings.E_star` | B keep both, `check()` refuses a difference | 2b |

Question 4, option A, lowers the comparison count by 11 per cell; every one of those values
is NaN in every reference (Table 4).

## 11. Open items: what is left of WP4

1. Rob's answers, recorded here and in `F/questions_2026-11-11.md` with their date.
2. Group 2b: the legacy keys, `rho0`, the modulus, any default (questions 1, 2, 7, 8).
3. Group 3b: the `gx_*` outputs and `p_fluid_masked` (questions 3, 4).
4. Group 4: the NaN-to-status table in code and its test over the 8 cells and the forced
   failures.
5. The two final script runs (§12), the README (call example, settings list, the `info`
   layout), the final tables with `F/tables.py`, the revised log and its second reading.
6. Rob pushes `main`.

## 12. Commands

The gate, from `MicroCell/` in Git Bash (about 12 s; expected `18 passed`):

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short
```

The gate with the per-cell summary and the per-comparison detail written to `F/` (about 25 s;
exit 0 = PASS, 1 = FAIL, 2 = INCOMPLETE):

```bash
bash docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/gate.sh <tag>
```

Not yet run in WP4 (they run once, at the end; each about 10 s; `wslrun` asks Rob):

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wslrun python tests/test_parity.py --out docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/parity_report.tsv
```

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wslrun python examples/run_cell.py
```

## 13. Files

Changed in git (68bc829 to 7b57abb): the 13 files of Table 7 (11 under `microcell/`,
`tests/test_parity.py`, `tests/test_results.py`).

`F/`: `inventory.py` and its six `inventory_*.tsv`; `questions_2026-11-11.md`; `gate.sh`,
`parity_summary.py`, `gate_counts.tsv`, `parity_<tag>.tsv` and `parity_<tag>_detail.tsv` for the
tags `g0_base_68bc829`, `g1`, `g2a`, `g2a_final`, `g3a`, `g3a_final`; `tables.py`,
`tables_tag.txt`, `settings_g3a_7b57abb.tsv`, `results_g3a_7b57abb.tsv`,
`nan_observed_g3a_7b57abb.tsv`, `counts_g3a_7b57abb.tsv`; `count_account_7b57abb.tsv`,
`lines_per_file_7b57abb.tsv`, `commits.tsv`; `logs/`.

Outside git: `_to_delete/wp4_2026-11-11/base_source_68bc829/`, the `git archive` of the base
used for the base parity measurement. Nothing was deleted.

## 14. Second-reader corrections

Second reading of 11 Nov 2026 (the `second-reader` agent; git, the TSV files and code reading,
no Python run). No table entry disagreed. Two wording corrections, both made in §3:

- "24 of the 96 parent result fields under another name or place": 11 of the 24 keep their
  name; the text now says 11 and 13.
- "Five second copies": four are second copies of an extremum and the fifth,
  `failure_reason`, is `out.reason`; the text now says so (the code comment above
  `RESULT_NOW` counts the four).

It also noted that `F/questions_2026-11-11.md` gives the base gate as 8.95s and Table 5 as
9.44s: two runs, now stated under Table 5.

Not checkable by the second reader: the cause of the first `g2a` row (no file records it); the
timings and the expected count of §12 (nothing was run); the "unreachable" and "no caller"
statements of §9 beyond a grep; the "Blocks" column of §10, which is my dependency view.
