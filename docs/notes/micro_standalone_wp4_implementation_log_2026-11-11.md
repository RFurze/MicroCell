# Standalone micro-cell solver: WP4 implementation log, settings and results collapse (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp4_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (work package WP4).
Base: `MicroCell` `main` at 68bc829. Parent `MultiscaleMixedBallinSocket` not touched.
`F/` below is `docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/`.
This text replaces the interim log of f014d6a (pre-edit copies: §13).

## 0. Summary

**State: WP4 is built and committed on `main`, head 7c41fc8 (Table 1). Nothing is pushed. WP5
is not started.** Rob answered the eight questions on 11 Nov 2026: A on all eight (Table 2).

**Gate (measured, Table 8).** After every group all eight reference cells PASS at relative
1e-12 and every comparison is exact. The comparison count fell once, in group 3b, by the
eleven global-x outputs in each cell; the account closes in every cell and nothing else left
(Table 9). No reference was regenerated and no tolerance changed. The negative controls and
`test_generated_surface_reproduces_reference` pass. The script-form gate and the example ran
once at the head, both clean (§6).

**Settings (measured, Tables 3, 4).** One layer: the solver classes read the caller's `Zeta`
and `MicroSettings`. The five solver-side containers, `api._solver_settings`, `LEGACY_DEFAULTS`
and `MicroSettings.legacy` are gone. No default changed; four settings were removed.

**Results (measured, Table 5).** `MicroResult.raw` is gone; each output is in one place (a
named output, `fields`, or `info`). The global-x shear outputs, `shear_frame`, the rotation
path and the `no_frame` status are gone. `p_fluid_masked` is no longer in `out.fields`.

**NaN and status (Tables 6, 7).** One table in code, `api.NAN_STATUS`, ties each output that
may be NaN on an OK result to a per-quantity status. A NaN or infinity it does not explain
turns the result into the new status `FAILED_NONFINITE`. Measured on the eight cells: the only
NaN on OK results are the asperity and total shear of the two full-film cells, `unavailable`.

**Changed test expectations:** six, all in how the gate reads the code, none in a number (§8).

**Departures:** six (§9). Two were Rob's to accept or reverse, 9.2 (a failed result's `info` has
every entry of an OK result) and 9.5 (`check()` refuses three values the solver accepted):
he kept both, 11 Nov 2026 ("Keep both").

**Not measured, by code reading only:** the statuses `no_fluid` and `phase_invalid` explain a
NaN in `NAN_STATUS` but no reference cell or test reaches them (§5).

**For Rob.** Push `main` when he chooses.
WP5 in a fresh session: it starts with the renames (answer 6) and builds `n_steps=None`
(answer 5).

**Open items** §11; candidates left in place §10; commands §12.

## 1. What was built

Table 1. Code commits, in the order made. Source: `F/commits.tsv` (git `--shortstat`).

| Group | Commit | Files changed | Insertions | Deletions | What |
|---|---|---|---|---|---|
| 1 decided removals | f77fd1d | 2 | 3 | 6 | `ContactSettings.nu`, `.k_asp_exponent` removed |
| 2a settings, the 1:1 part | 9f719fb | 12 | 451 | 501 | solver classes read `Zeta` and `MicroSettings`; five containers and `api._solver_settings` removed |
| 3a results, answer-independent | 7b57abb | 3 | 259 | 132 | `MicroResult.raw`, `_RAW_DEFAULTS`, `_failed_raw` removed; one failed-result builder; `RESULT_NOW`; `tests/test_results.py` |
| 2b settings, after the answers | 8635eb2 | 6 | 135 | 69 | `legacy`, `LEGACY_DEFAULTS`, `rho0` removed; `delta_h`, `eps_solid`, `eps_smooth` settings; one `E_star`; `tests/test_settings.py` |
| 3b results, after the answers | 2e1419f | 9 | 127 | 210 | global-x outputs, `shear_frame`, rotation path, `no_frame` removed; `p_fluid_masked` out of `out.fields` |
| 4 NaN and status | 7c41fc8 | 3 | 223 | 8 | `api.NAN_STATUS`, `FAILED_NONFINITE`, the NaN-to-status tests |

Between 7b57abb and 8635eb2 the interim log was committed (f014d6a, documents only). Author
and committer of every commit: `RFurze <py07rf@leeds.ac.uk>`; no attribution lines.

Step 1 (no edits) wrote the six `F/inventory_*.tsv` at the base, with `F/inventory.py`. That
script reads the base tree only; to repeat it, run it in
`_to_delete/wp4_2026-11-11/base_source_68bc829/` (a `git archive` of the base).

## 2. Rob's answers (11 Nov 2026)

"A on all eight", given in the WP4 session after the interim log. Full question text and
evidence: `F/questions_2026-11-11.md`.

Table 2.

| # | Question | Answer A | Built in |
|---|---|---|---|
| 1 | the six `legacy` keys | `delta_h`, `eps_solid`, `eps_smooth` are `MicroSettings` fields; `beta`, `patch_pin`, `patch_pin_kappa` fixed values | 2b |
| 2 | defaults | the GenSa40-ripple values stay | no change |
| 3 | NaN on an OK result | per-quantity status table; new overall `FAILED_NONFINITE`; `p_fluid_masked` leaves `out.fields` | 3b, 4 |
| 4 | global-x shear | the outputs, `shear_frame` and the `rotation=` path removed | 3b |
| 5 | `n_steps=None` | left for WP5 | not in WP4 |
| 6 | renames toward the target layout | first commit of WP5 | not in WP4 |
| 7 | `MicroSettings.rho0` (no reader) | removed | 2b |
| 8 | one modulus | a single `MicroSettings.E_star` | 2b |

## 3. Settings: before and after

Table 3. Rows per container. Sources: base `F/inventory_settings.tsv`; now
`F/settings_g4_7c41fc8.tsv` (every setting with its default and the file and line that defines
it) and `F/counts_g4_7c41fc8.tsv`, both made by `F/tables.py`.

| Container | Rows at 68bc829 | Rows at 7c41fc8 |
|---|---|---|
| `Zeta` | 12 | 12 |
| `MicroSettings` | 20 | 22 |
| `ContactSettings` | 8 | 5 |
| `RippleSettings` | 4 | 3 |
| `LEGACY_DEFAULTS` (through `MicroSettings.legacy`) | 6 | gone |
| `MicroPhysicalParameters` | 20 | gone |
| `MicroSolverSettings` | 10 | gone |
| `PhaseNeutralSettings` (base class) | 17 | gone |
| `PhaseNeutralSettings` (mixed class) | 1 | gone |
| `PhaseNeutralContactSettings` | 8 | gone |
| fixed values of the solver module | not a container | 13 |
| `microsolver` arguments other than `zeta`, `roughness`, `settings` | not listed | 6 |
| all rows | 106 | 61 |

Table 4. What became of the caller-side defaults of the base (the rows of `Zeta`,
`MicroSettings`, `ContactSettings`, `RippleSettings`, `LEGACY_DEFAULTS`). Source:
`F/defaults_account_g4_7c41fc8.tsv` (made by `F/defaults_account.sh`).

| Outcome | Rows | Which |
|---|---|---|
| present, same default | 46 | of these, 8 moved: the two `E_star` to `MicroSettings.E_star`; `delta_h`, `eps_solid`, `eps_smooth` to `MicroSettings`; `beta`, `patch_pin`, `patch_pin_kappa` to fixed values |
| removed | 4 | `ContactSettings.nu`, `.k_asp_exponent`, `MicroSettings.rho0`, `MicroSettings.legacy` |
| present, default differs | 0 | |
| all | 50 | |

**What the solver reads (my call in 2a: no solver-side container).** The classes read
`self.zeta`, `self.t_end`, `self.settings` (the caller's `MicroSettings`, with `.contact` and
`.ripple`), `self.velocity_ratio`, `self.pair` and `self.h_min` (0.0 in the full-film class,
`settings.h_min` in the mixed class, as the removed remapping did).

**Fixed values.** Module constants in `microcell/micro_phase_neutral_class.py`: the seven
damping knobs, `PRINT_NEWTON`, `PRINT_TIMING`, `TRACE_NEWTON` (2a; the API could not set them)
and `BETA`, `PATCH_PIN`, `PATCH_PIN_KAPPA` (2b, answer 1). The contact-law tag
`fft_elastic_v1` has one definition, `microcell.contact.FFT`.

**One modulus (2b, answer 8).** `MicroSettings.E_star` serves the contact law and the ripple.
The solver's own check that the two moduli agree is removed with the second modulus.

**`MicroSettings.check()`** refuses, before a mesh is built: a `spos_form`, `level_constraint`
or `mesh_n` outside its range (as at the base); a non-finite `jacobian_reg_eps`; `delta_h`,
`eps_solid` or `eps_smooth` not finite or negative (departure 9.5); with contact or the ripple
on, an `E_star` that is not finite and positive; ripple `tol` <= 0, `maxpass` < 1, `omega`
outside (0, 1]; a contact `k_asp_mode` other than `off` or `closure`; `h_min` not finite and
positive with contact on. A value of a part that is off is not read. It does not refuse a
non-finite `contact.mu` (candidate, §10). `tests/test_settings.py` covers each refusal.

## 4. Results: before and after

Table 5. Where the outputs of a result are held, and the gate's name maps. Sources: base
`F/inventory_results.tsv`, `F/inventory_info_fields.tsv`; now `F/results_g4_7c41fc8.tsv`
(every output with its place) and `F/counts_g4_7c41fc8.tsv`.

| Place or map | At 68bc829 | At 7c41fc8 |
|---|---|---|
| named outputs of `MicroResult` | 18 | 18 |
| `raw` | 83 | gone |
| `info`, without `info["fields"]` | 23 | 56 |
| `info["fields"]` (field diagnostics; 10 in the full-film cells c5, c8) | 12 | 12 |
| `fields` (names over the 8 cells) | 18 | 17 |
| parent result fields in the references | 96 | 96 |
| of these, not compared (`NOT_COMPARED`, unchanged) | 15 | 15 |
| held under another name or place (`RESULT_NOW`) | no map | 24 |
| removed from the result (`REMOVED_RESULT_FIELDS`) | none | 11 |
| compared, with no place in a `MicroResult` | 0 | 0 |
| reference fields the gate forms from held fields (`DERIVED_FIELDS`) | none | 1 |

**Layout.** `status`, `reason`, the named outputs (unchanged), `fields`, and `info`. `info`
holds everything else of the parent's `MicroTaskResult` under the parent's name (statuses and
tags, counters, the pressure ledger, ripple and contact numbers, the sampled shear
statistics), the call (`steady`, `t_end`, `n_steps`, `offset`, `roughness_velocity_ratio`,
`t_start`, `dt`), `mesh_n`, `wall_time`, `contact_eval_s` and `info["fields"]`.

**Second copies not written (3a).** `p_fluid_sample_max_Pa`, `p_max_phase_mean` (both `p_max`),
`p_fluid_sample_min_Pa`, `p_min_phase_mean` (both `p_min`) and `failure_reason` (`out.reason`).
The gate compares each against the one place the value is held (`RESULT_NOW`).

**Global-x shear removed (3b, answer 4).** The removed outputs are the `shear_fluid_gx_*`,
`shear_fluid_support_gx_*`, `shear_asperity_gx_*` and `shear_total_gx_*` statistics (names in
`F/count_account_7c41fc8.tsv`). Each was NaN in every reference (`F/inventory_nan_observed.tsv`);
the gate fails if a reference holds a finite one or `info` still carries one. With them went
`solver.shear_frame`, the `rotation` argument of `shear_reduce`, `union_phases` and the
`no_frame` status. The shear samples are in the micro's local frame.

**`p_fluid_masked` (3b, answer 3).** Not in `out.fields`; it was NaN at the contact vertices of
five cells. The ParaView files still carry it. The gate forms it from the held `p_fluid` and
`chi_fluid` and compares that with the reference's array, so it is still compared.

**The failed result.** `api._failed_result()` is the one place it is built: every named output
NaN, `yield_flag` 0, no fields, `info` with the entries of an OK result (NaN numbers, `failed`
statuses; a full-film cell's asperity and total shear statuses stay `unavailable`).

## 5. NaN and status

Overall statuses: `OK`, `FAILED_NEWTON`, `FAILED_RIPPLE`, `FAILED_CONTACT` and, new in group 4
(answer 3), `FAILED_NONFINITE`. Per-quantity statuses, in `info`: `pressure_extrema_status`
(`ok`, `no_fluid`, `phase_invalid`, `failed`); `shear_extrema_status`, `shear_asperity_status`,
`shear_total_status` (`ok`, `no_support`, `phase_invalid`, `unavailable`, `failed`).

**The rule in code.** `api.NAN_STATUS` (Table 6) is the one table. At the end of `microsolver`,
`api.unexplained_nonfinite(out)` lists every named output, `info` number, `info["fields"]`
number and field array that holds a NaN or an infinity and is not explained: not listed, or
listed with its status entry at a value the table does not name. If the list is not empty the
result is `FAILED_NONFINITE`, with the names in `reason`, laid out as every failed result.
`ok` and `no_support` never explain a NaN. Fields are never NaN on an OK result.

Table 6. `api.NAN_STATUS`. Source: `F/nan_status_g4_7c41fc8.tsv` (read from the code by
`F/tables.py`; 12 outputs).

| Output (where) | Status entry | Statuses that explain a NaN | Seen NaN on an OK reference cell |
|---|---|---|---|
| `p_max`, `p_min` (named) | `pressure_extrema_status` | `no_fluid`, `phase_invalid` | no |
| `Pst_fluid` (`info`) | `pressure_extrema_status` | `no_fluid` | no |
| `shear_fluid_mean_local_Pa`, `shear_fluid_support_fraction` (`info`) | `shear_extrema_status` | `phase_invalid` | no |
| `tau_asperity` (named); `shear_asperity_support_fraction`, `shear_contact_overlap_fraction`, `shear_contact_force_on_fluid_fraction`, `shear_contact_obstruction_without_pressure_fraction`, `shear_contact_mean_check` (`info`) | `shear_asperity_status` | `unavailable`, `phase_invalid` | c5, c8 (`unavailable`) |
| `tau_total` (named) | `shear_total_status` | `unavailable`, `phase_invalid` | c5, c8 (`unavailable`) |

Table 7. NaN measured at 7c41fc8, distinct names. Sources: `F/nan_observed_g4_7c41fc8.tsv`,
`F/counts_g4_7c41fc8.tsv` (the 8 reference cells; the five forced failures and the three
forced NaN of `tests/test_results.py`).

| Case | Overall status | Named | `info` | `info["fields"]`, `fields` |
|---|---|---|---|---|
| c1, c2, c3, c4, c6, c7 | OK | 0 | 0 | 0 |
| c5, c8 (full film) | OK | 2 | 5 | 0 |
| c2 forced: Newton, ripple, contact in `run` | `FAILED_NEWTON`, `_RIPPLE`, `_CONTACT` | 17 | 27 | no fields |
| c2 forced: contact in the load evaluation | `FAILED_CONTACT` | 17 | 25 | no fields |
| c5 forced: Newton | `FAILED_NEWTON` | 17 | 22 | no fields |
| c2 with a NaN put into `Fst`, or into the `film` field | `FAILED_NONFINITE` | 17 | 25 | no fields |
| c5 with a NaN put into `Fst` | `FAILED_NONFINITE` | 17 | 20 | no fields |

The 17 are every named output but `yield_flag`. The two extra `info` names of the failures
forced in `run` are `t_start` and `dt`, which the solver had not set. NaN on OK results that
`NAN_STATUS` does not list: 0 (`F/counts_g4_7c41fc8.tsv`).

**Tests (group 4, in `tests/test_results.py`).** The table names real outputs and statuses;
every NaN of each reference cell, forced failure and forced NaN has a status that says why;
a listed output is explained only by its own status.

**Measured against inferred.** Measured: `unavailable` (c5, c8) and the three forced NaN. By
code reading only: `no_fluid` and `phase_invalid` as explanations (paths N7 to N14 of
`F/inventory_nan_paths.tsv`, written at the base). No reference cell or test reaches them. `Pst_fluid` is NaN when the DG0
fluid area is zero, while `no_fluid` is set from the vertex samples; if the two ever disagree
the result is `FAILED_NONFINITE`, not a silent NaN.

## 6. Parity per group

Table 8. Sources: `F/gate_counts.tsv` (the pytest count line of
`tests/ -q -p no:cacheprovider -rfE --tb=short`), `F/parity_<tag>.tsv` (row `All`).

| State | Tag | Tests | Cells PASS | Comparisons | Exact | Outside 1e-12 | Largest relative difference |
|---|---|---|---|---|---|---|---|
| base 68bc829 | `g0_base_68bc829` | 13 passed in 9.44s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 1, f77fd1d | `g1` | 13 passed in 8.28s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 2a, 9f719fb | `g2a_final` | 13 passed in 8.41s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 3a, 7b57abb | `g3a_final` | 18 passed in 11.42s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 2b, 8635eb2 | `g2b_final` | 42 passed in 11.24s | 8 of 8 | 1188 | 1188 | 0 | 0.0 |
| group 3b, 2e1419f | `g3b` | 42 passed in 10.93s | 8 of 8 | 1100 | 1100 | 0 | 0.0 |
| group 4, 7c41fc8 | `g4_final` | 60 passed in 17.59s | 8 of 8 | 1100 | 1100 | 0 | 0.0 |

`g4_final` ran on the committed tree. `g2b_final` and `g3b` ran on the working tree before
their commits (log times against commit times). The test count rose with the new test files:
`tests/test_results.py` (3a, extended in 4) and `tests/test_settings.py` (2b).

`F/gate_counts.tsv` has five more rows. `g2a` twice: the first, `11 failed, 2 passed in 1.07s`,
was a syntax error of mine in an f-string, no solve ran, and its log was overwritten by the
second (`13 passed in 8.90s`); the cause is my account, no file records it. `g3a`
(`13 passed in 8.60s`), `g2b` (`18 passed in 10.78s`) and `g4` (`60 passed in 17.38s`) are
earlier runs within their groups.

The gate run before the first edit (the brief's command, run directly) printed
`13 passed in 8.95s`, quoted in `F/questions_2026-11-11.md`; it kept no log. The base row of
Table 8 is a second run, through `F/gate.sh` on the archive of 68bc829.

Table 9. Count account, base against 7c41fc8. Source: `F/count_account_7c41fc8.tsv` (made by
`F/accounts.sh` from the two `_detail.tsv` files, by cell, kind and name).

| Cell | Compared at base | Compared now | No longer compared | Newly compared | Closes |
|---|---|---|---|---|---|
| c1 | 155 | 144 | 11 | 0 | yes |
| c2 | 155 | 144 | 11 | 0 | yes |
| c3 | 154 | 143 | 11 | 0 | yes |
| c4 | 154 | 143 | 11 | 0 | yes |
| c5 | 131 | 120 | 11 | 0 | yes |
| c6 | 153 | 142 | 11 | 0 | yes |
| c7 | 155 | 144 | 11 | 0 | yes |
| c8 | 131 | 120 | 11 | 0 | yes |
| All | 1188 | 1100 | 88 | 0 | yes |

The names no longer compared are the same in every cell: the removed global-x outputs of §4.
Reason: answer 4; every one was NaN in every reference.

**End-of-WP runs at 7c41fc8 (each once).** `python tests/test_parity.py --out
F/parity_report.tsv`: every cell PASS with the per-cell counts of Table 9
(`F/parity_report.tsv`, `F/logs/parity_report_7c41fc8.log`). `python examples/run_cell.py`:
the transient and the steady solve both `OK` (`F/logs/run_cell_7c41fc8.log`). Exit codes: 0 for these two
runs, `F/accounts.sh` and `F/defaults_account.sh`, as seen in the session; no file records
them, the outputs are consistent with them.

## 7. Lines per file

Table 10. Source: `F/lines_per_file_7c41fc8.tsv` (`git show <commit>:<file> | wc -l`).

| File | 68bc829 | 7c41fc8 | Change |
|---|---|---|---|
| `microcell/api.py` | 638 | 628 | -10 |
| `microcell/contact_shear.py` | 269 | 247 | -22 |
| `microcell/fluid_shear.py` | 148 | 127 | -21 |
| `microcell/micro_diagnostics.py` | 459 | 435 | -24 |
| `microcell/micro_phase_neutral_class.py` | 2804 | 2735 | -69 |
| `microcell/micro_phase_neutral_class_contact.py` | 360 | 328 | -32 |
| `microcell/micro_phase_neutral_class_mixed.py` | 718 | 705 | -13 |
| `microcell/shear_reduce.py` | 253 | 186 | -67 |
| six more under `microcell/`, each changed by one line or none (in the source file) | | | |
| the package, all `microcell/**/*.py` | 8672 | 8413 | -259 |
| `tests/test_parity.py` | 474 | 659 | 185 |
| `tests/test_results.py` | new | 267 | 267 |
| `tests/test_settings.py` | new | 79 | 79 |

## 8. Changed test expectations

8.1 **Settings are compared with what the solver reads now** (2a). `SETTINGS_NOW` in
`tests/test_parity.py` maps each parent settings field to where the solver reads it (a
`MicroSettings` field, a fixed value of the solver module, the contact law, "ripple on", the
pair flag); its size is in `F/counts_g4_7c41fc8.tsv`.

8.2 **Ripple fields of a reference made without the ripple** (2a). The parent never read
them. They pass when the reference's `fluid_deformation` is `off` and the settings carry no
ripple. The number of comparisons is the same.

8.3 **One modulus, no `beta`, no `rho0`** (2b). `settings_from_reference` passes one `E_star`
and neither `beta` nor `rho0`; the reference's `beta` and the patch-pin fields are compared
with the fixed values.

8.4 **The removed global-x outputs are not compared** (3b); Table 9.

8.5 **`no_frame` in a reference** (3b). The result must hold `ok` when a support fraction of
the reference listed for that status is above zero, otherwise `no_support`
(`STATUS_WITHOUT_FRAME`). Table 11 is what the eight cells give.

8.6 **`p_fluid_masked` is derived by the gate** (3b); §4.

Table 11. Per-quantity statuses, reference against result at 7c41fc8. Source:
`F/statuses_g4_7c41fc8.tsv`.

| Status entry | Reference | Result | Cells |
|---|---|---|---|
| `pressure_extrema_status` | `ok` | `ok` | 8 |
| `shear_extrema_status` | `no_frame` | `ok` | 8 |
| `shear_asperity_status` | `no_frame` | `ok` | 5 |
| `shear_asperity_status` | `no_frame` | `no_support` | 1 (c1) |
| `shear_asperity_status` | `unavailable` | `unavailable` | 2 |
| `shear_total_status` | `no_frame` | `ok` | 6 |
| `shear_total_status` | `unavailable` | `unavailable` | 2 |

## 9. Departures from the brief

9.1 **Groups 2 and 3 were each split** into an answer-independent commit (2a, 3a) and one
after the answers (2b, 3b): six code commits, not four.

9.2 **A failed result's `info` changed** (3a). At the base it held the call, four counters
and seven statuses and tags; the NaN numbers were in `raw`. Now it holds the entries of an OK
result. Three values differ from the base (measured: the forced-failure rows of
`F/inventory_nan_observed.tsv` against `F/nan_observed_g3a_7b57abb.tsv`; `level_constraint` by
reading `api.py`): `shear_asperity_status` and `shear_total_status` of a failed full-film solve
were `None` and are `unavailable`; `level_constraint` of a failed `whole_cell` solve was `None`
and is `whole_cell`. Not put to Rob as a question. To reverse: set the three in
`api._failed_result()`. **Kept by Rob, 11 Nov 2026.**

9.3 **A test file was added in group 3**, `tests/test_results.py`; the brief places the new
test in group 4. Group 4 extended it.

9.4 **An interim log was committed** (f014d6a) while the questions waited; this text
replaces it.

9.5 **`check()` refuses more than the solver did** (2b): `delta_h`, `eps_solid`, `eps_smooth`
not finite or negative. The solver accepted any value of the three legacy keys. None of the
eight cells or the example is affected (they pass). Not put to Rob as a question. To reverse:
three lines of `check()` and three cases of `tests/test_settings.py`. **Kept by Rob, 11 Nov 2026.**

9.6 **An infinity is treated as a NaN** by `FAILED_NONFINITE`. Answer 3 spoke of NaN.

## 10. Candidates left in place

- `reset_task()` of the base solver class: no caller in MicroCell.
- The branches under the constants `PRINT_NEWTON`, `PRINT_TIMING`, `TRACE_NEWTON` (all False),
  `_print_timing_summary`, and the branch under `PATCH_PIN` (None): unreachable.
- `ripple.FLUID_DEFORMATION_MODES`, `ripple.deformation_model_tag`: unused.
- `micro_diagnostics.sample_contact_traction`: its `else: raise ValueError("unknown contact
  model")` cannot be reached with one contact law.
- The level-constraint counts `n_whole_cell`, `n_fluid_mean`, `n_no_fluid` are reported as 0
  with `whole_cell`, as the parent did, whatever the solver counted.
- `fluid_shear.SHEAR_EXTREMA_DEFINITION` and `contact_shear.CONTACT_SHEAR_DEFINITION` still say
  `global_x` and `phase_union`. They are compared with the references, so changing the text is
  a changed expectation.
- `ShearReduction`'s minimum, maximum, sample mean and support extrema, and `direction=GLOBAL_X`,
  are still computed; nothing outputs them now. `ContactTotalReduction.as_dict`: unused.
- A non-finite `contact.mu` is accepted by `check()`; the asperity shear is then `unavailable`.
- `tools/make_references.py:34`: its docstring names `MicroPhysicalParameters` (it reads the
  parent, where the class exists).

## 11. Open items

1. Rob pushes `main`.
2. Departures 9.2 and 9.5: closed, Rob kept both (11 Nov 2026).
3. WP5, fresh session: renames as the first commit (answer 6); `n_steps=None` with its own test
   (answer 5); the candidates of §10.
4. No test reaches `no_fluid` or `phase_invalid` (§5).

## 12. Commands

From `MicroCell/` in Git Bash. The gate (about 20 s; expected `60 passed`):

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short
```

The gate with the per-cell summary and per-comparison detail written to `F/` (under a minute;
exit 0 = PASS, 1 = FAIL, 2 = INCOMPLETE; expected 1100 comparisons, all exact):

```bash
bash docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/gate.sh <tag>
```

The tables of §3 to §5 and §8 come from `F/tables.py` (through `wsltest`, tag in
`F/tables_tag.txt`); the accounts of Tables 1, 4, 9, 10 from `F/accounts.sh` and
`F/defaults_account.sh` (git only). The head of each file gives its command and exit codes.

The two end-of-WP runs, already run once at 7c41fc8 (each about 10 s; `wslrun` asks Rob):

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wslrun python tests/test_parity.py --out docs/notes/micro_standalone_wp4_implementation_log_2026-11-11_files/parity_report.tsv
```

```bash
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wslrun python examples/run_cell.py
```

## 13. Files

Changed in git, 68bc829 to 7c41fc8: the files of Table 10 (14 under `microcell/`, three under
`tests/`). With this log: `README.md` (state, call example, settings table, result layout,
running, provenance).

`F/`: the inventories; `questions_2026-11-11.md` (with the answers); `gate.sh`,
`parity_summary.py`, `gate_counts.tsv`, `parity_<tag>.tsv` and `parity_<tag>_detail.tsv` for
every tag; `parity_report.tsv` and its `_detail`; `tables.py`, `tables_tag.txt` and its tables
for the tags `g3a_7b57abb` and `g4_7c41fc8`; `accounts.sh`, `defaults_account.sh` and their
tables for 7b57abb and 7c41fc8; `commits.tsv`; `logs/`.

Outside git, nothing deleted: `_to_delete/wp4_2026-11-11/base_source_68bc829/` (the archive of
the base), `_to_delete/wp4_2026-11-11/pre_edit/` (the interim log, the log before this
revision, the README, the questions file and `commits.tsv` before their edits),
`_to_delete/wp4_2026-11-11/scratch/`.

## 14. Second-reader corrections

**Interim log, 11 Nov 2026** (the `second-reader` agent; git, the TSV files and code reading).
No table entry disagreed. Two wording corrections in the results section (the split of the
`RESULT_NOW` entries; four second copies of an extremum plus `failure_reason`), and the two
base gate timings, now stated under Table 8.

**This log, 11 Nov 2026** (the `second-reader` agent; git, the TSV files and code reading, no
Python run). No table or prose number disagreed. One correction: "exit 0" was stated for
`F/defaults_account.sh`, `F/accounts.sh` and the two end-of-WP runs with no file that records
an exit code; §6 now says they were seen in the session only.

Not checked by the second reader: the base side of departure 9.2 item by item; that
`g2b_final` and `g3b` ran on the working tree (log times against commit times only); the
timings of §12; the "unreachable" and "unused" statements of §10 beyond a grep.
