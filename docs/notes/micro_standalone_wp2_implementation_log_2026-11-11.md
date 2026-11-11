# Standalone micro-cell solver: WP2 implementation log, prune dead physics (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp2_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (decisions D1, D5, D6, D10; work package WP2).
Base: `MicroCell` `main` at c96ad39 (WP1). Parent `MultiscaleMixedBallinSocket` not touched.
`F/` below is `docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/`.

## 0. Summary

**State.** WP2 is built: all nine removal groups are done, at least one commit per group, on
`main` of `MicroCell` (Table 1). The commits up to 7d64e1e are on `origin/main`; the ones after
it are not pushed (Table 7). WP3 is not started.

**Rob's decision (11 Nov 2026): "Yes, re-make c5".** The first pass left the cell-average
spring in place, because reference c5 used it (§2). On Rob's answer the c5 reference was
re-made from the parent at 2aaccad with `k_spring = 0`, and the spring was then removed. The
other seven references are unchanged.

**Gate (measured).** After every group all eight reference cells PASS at relative 1e-12 and
every comparison is exact: the largest relative difference is zero at every gate, including
the two that changed a UFL form (groups 4 and 2; Table 4). No tolerance changed. The prediction
written before the c5 re-make held (§2). The final script run and the example agree with WP1 (§4).

**Size (measured).** Table 3: the five pruned solver files fall from 5699 to 4367 lines.

**Departures.** Eleven, numbered in §6. The ones a caller or a reader of the gate can see:
- 6.1 one reference (c5) was regenerated, on Rob's decision; the brief said never;
- 6.2 a full-film result's `contact_model` tag is `off` (was the parent's default `legacy_patch_v1`);
- 6.5 the pair branch is selected by a plain flag `pair_branch` on the contact settings, in
  place of the `phase_average = "contact_table_v1"` tag (allowed by the brief if results are
  unchanged; they are);
- 6.7 with `MICRO_DEBUG_LOG` gone, the import still disables logging for the process, as the
  parent did with the variable unset;
- 6.10 `k_spring` is no longer an input: the `legacy` key is refused as unknown.

**Changed test expectations.** §9 (Table 6): removed settings and the removed spring are checked
against the reference, not skipped; two tests added (Table 4 shows the counts). **Open items**
§8; candidates for further pruning §7, left in place.

**Decisions for Rob.**
1. `ContactSettings.nu` and `ContactSettings.k_asp_exponent` are now unread API inputs (§6.3).
   Recommended: remove them at WP4 with the settings collapse. Alternative: remove them now.
2. Accept or reject departures 6.2, 6.5 and 6.7.
3. Push the commits after 7d64e1e, and the go-ahead for WP3 as its own session.

## 1. What was removed

Table 1. One row per group. Commit sizes are in `F/commits.tsv`.

| # | Group | Status | Commit | What went |
|---|---|---|---|---|
| 1 | Dead imports, `__main__` blocks | done | 921c077 | The `__main__` shim of the base class; the mixed file's harness (`_main_params`, `_main_settings` with its `from config import micro_solver`, `__main__`). |
| 2 | Spring deformation, pointwise mode | done | 763a074 | `spring_pointwise`, `hd_nd_func`, `_target_hd_nd_field`, `_spring_solve_pointwise`, `spring_relax`, its branch in `_solve_with_spring_once`, its refusal in `_ripple_setup`, its part of `_cg1_film_parts` (diagnostics). |
| 2 | Spring deformation, cell-average spring | done | a1395f4 (reference), b18879b (code) | `MicroPhysicalParameters.k_spring` and the `legacy` key; `hd_nd_const` (the film is `h_nd = h_base_nd`); `_spring_enabled`, `_cell_avg_positive_pressure_dim` (base and mixed), `_target_hd_nd_from_Pst`, `_spring_solve_cell_avg`, `_solve_with_spring_once`; `spring_max_iter`, `spring_rtol`, `_area_total`; the spring timer and counter; the two `k_spring` refusals (§6.10). `_solve_with_spring` is `_solve_level` and `_pair_solve_with_spring` is `_pair_solve` (§6.11). |
| 3 | `legacy_patch_v1`, `hertz_summit_v1` | done | 90413fd | Contact class: `_summit_geometry`, `_summit_pressure`, `_find_contact_patches`, `_asperity_pressure`, every non-fft branch, settings `contact_E`, `contact_nu`, `use_reduced_modulus`, `overclosure_measure`, `k_asp_exponent`. `contact.py`: `LEGACY`, `HERTZ`, `hertz_response`, `CosineSummits`, `cosine_summits`. Diagnostics: the unsupported-model branches. API: `legacy` keys `contact_E`, `contact_model`, `use_reduced_modulus`, `overclosure_measure`. |
| 4 | Roelands, compressibility stubs | done | 7cedddb | `roelands_eta_dim` (base; its callers in base, mixed, diagnostics now read `params.eta0`), `rho_nd` and `eta_nd` (base, mixed), both `Constant(1.0)`. |
| 5 | `gauge_lift`, `c2` form | done | 16baf1e | The `gauge_lift` setting and its warning; `smooth_pos_c2`, `smooth_pos_deriv_c2` and the `c2` value of `spos_form`; the then-unused `import warnings`. |
| 6 | Export hooks, `mesh_h`, `k` | done | 3eb8ae2 | Constructor arguments `mesh_h`, `k`, `export_vtk`, `output_dir`; `_setup_export`, `_open_xdmf_series`, `_write_xdmf_step`, `_close_xdmf_series`, `post_process`; the XDMF calls and stride in `_run_once`; the XDMF timers; the then-unused `Optional` import. |
| 7 | Warm-start plumbing | done | feb3e1d | Constructor argument `warm_start_state`, `_apply_warm_start`, `warm_start_snapshot`; `set_pair_warm`, `pair_warm_record`, `_pair_warm_begin`, `_pair_warm_check`, the `_pair_warm*` attributes, the warm branch of `_ripple_write_field`, and `_pair_level` (§6.4). |
| 8 | Table-only code | done | c7f799e | `contact_only_load`; `PHASE_AVERAGE_OFF`, `PHASE_AVERAGE_TABLE`, `PHASE_AVERAGES`; the `phase_average` setting, replaced by `pair_branch` (§6.5). The pair branch and the warm-start contact rescue are kept. |
| 9 | Environment reads | done | b482975 | The five reads, each frozen at its unset value: debug log off, nondimensionalisation print off, Uref floor off (its block removed), smoothing cap 1.0 (now the module constant `_EPSP_ND_CAP`), `spos_form` from the settings only. The then-unused `import os`. |
| - | README, final report | - | 68028f5 | The provenance note marks the five pruned files; `F/parity_report.tsv` and the example log. |

Files not touched: `microcell/roughness/`, the shear modules, `ripple.py`, `elastic_contact.py`,
`fields.py`, `examples/`, and seven of the eight references. `tools/make_references.py` gained
the c5 change (§2).

For groups 4, 6 and 7 there is no stored setting to check. Group 4's stubs were constants. For
groups 6 and 7 the evidence is the API at c96ad39 (`git show c96ad39:microcell/api.py`): it
passed `export_vtk=False`, `mesh_h` and `k` that the solver ignored, no `warm_start_state`, and
never called `set_pair_warm`, and it matched the references exactly at WP1.

## 2. Group 2: the spring that a reference used

**Finding (first pass).** The base class enabled the spring when `k_spring > 0`. The cosine
parent case carries `k_spring = 1e20` as its "off" value. The contact class never ran the spring
(and refused a stiffness between 0 and 1e19); in the base class, used for full film, the
cell-average loop ran on every transient step, and the steady solve was raw. So of the three
cells that stored 1e20, only the transient full-film one, c5, felt it.

Table 2. The standalone at commit 763a074 with `k_spring` forced to 0, against the references
as they then were (measured; `F/diag_spring_k0.tsv`, script `F/diag_spring_k0.py`, log
`F/diag_spring_k0.log`).

| Cell | `k_spring` stored then | Comparisons | Outside 1e-12 | Largest relative difference | Where |
|---|---|---|---|---|---|
| c5_cosine_fullfilm | 1e20 | 130 | 9 | 9.997e-06 | `delta_Q` |
| c6_cosine_fft_sqrt | 1e20 | 152 | 0 | 0 | - |
| c8_cosine_fullfilm_steady | 1e20 | 130 | 0 | 0 | - |

The nine c5 comparisons are listed by name in `F/diag_spring_k0_names.tsv` (script
`F/diag_spring_k0_names.py`, run on the tree of b482975): `delta_Q`, the film minimum and
maximum, the film field, `tau_res`, the fluid shear mean, `Q`, `info.film_min_nm`, and the
Newton iteration count.

**Decision.** Three options were put to Rob: A re-make c5 with the spring off and remove the
spring (recommended), B keep the spring until WP4, C remove it and accept the 1e-5 difference.
Rob chose A on 11 Nov 2026 ("Yes, re-make c5"). The prediction written before the re-make: the
standalone with the spring removed reproduces the new reference on all c5 comparisons at 1e-12.

**What was done.**

1. `tools/make_references.py`: c5 gained `changes={"k_spring": 0.0}`, applied in memory to the
   case's `MicroCase` as the tool does for c3 and c6. The parent was at 2aaccad with a clean
   tree before and after; nothing was written in it.
2. c5 alone was re-made (command in §10; exit 0, status OK). The old file is in git at 7d64e1e
   and in `MicroCell/_to_delete/wp2_c5_spring_2026-11-11/reference_before/`.
3. Gate g2b_ref: the code with the spring still in it, given `k_spring = 0`, against the new c5.
4. The spring was removed (commit b18879b). Gate g2b.

Table 2b. The old c5 reference against the new one (measured; `F/c5_reference_old_vs_new.tsv`,
script `F/c5_reference_old_vs_new.py`, stdlib only).

| Entries of the file | Number | What |
|---|---|---|
| Members compared | 19 | every member of the npz |
| Entries that differ | 16 | |
| of which results | 9 | the same nine quantities as Table 2 (film field, `h_min`, `h_max`, `info.film_min_nm`, `Q`, `delta_Q`, `tau_res`, `shear_fluid_mean_local_Pa`, `newton_iters` 8 to 5) |
| of which the input | 2 | `k_spring` 1e20 to 0 in `micro_case` and `phys` |
| of which bookkeeping | 5 | `changes`, `made`, `wall_s`, `wall_time`, `pid` |

**Outcome of the prediction (measured, Table 4).** Held. At g2b_ref and at g2b every c5
comparison is exact, not only within 1e-12, and so is every comparison of the other seven cells.

**c6 and c8 were not re-made.** They still store `k_spring = 1e20`. The parent's spring did not
act on them (Table 2), and the parity test now checks that for every cell instead of passing
the value on (`case.k_spring`, Table 6).

## 3. Lines per file

Table 3. `wc -l` at c96ad39 (`F/lines_before_c96ad39.tsv`) and at b18879b
(`F/lines_after_b18879b.tsv`). `F/lines_after_68028f5.tsv` holds the counts before the spring
was removed.

| File | Before | After | Change |
|---|---|---|---|
| `microcell/micro_phase_neutral_class.py` | 3501 | 2804 | -697 |
| `microcell/micro_phase_neutral_class_mixed.py` | 859 | 718 | -141 |
| `microcell/micro_phase_neutral_class_contact.py` | 770 | 360 | -410 |
| `microcell/micro_diagnostics.py` | 476 | 459 | -17 |
| `microcell/contact.py` | 93 | 26 | -67 |
| Five pruned solver files | 5699 | 4367 | -1332 |
| `microcell/api.py` | 649 | 638 | -11 |
| `tests/test_parity.py` | 357 | 474 | +117 |
| `README.md` | 110 | 120 | +10 |
| `tools/make_references.py` | 521 | 525 | +4 |

## 4. Parity per group

The gate after each group: `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from
`MicroCell/`, log `F/logs/gate_<tag>.log`, count lines collected in `F/gate_counts.tsv`. The
gate prints no differences when it passes, so `F/parity_summary.py` (a pytest file run the same
way) runs `tests/test_parity.run_cell` on every cell and writes `F/parity_<tag>.tsv`.

Table 4. Measured, in the order run. "Exact" counts comparisons with relative difference zero.

| Tag | After | Commit | pytest | Cells PASS | Comparisons exact / total | Largest relative difference | Source |
|---|---|---|---|---|---|---|---|
| g0 | base | c96ad39 | 11 passed, 8.73 s | 8 of 8 | not tabulated | not tabulated | `F/logs/gate_g0_base_c96ad39.log` |
| g1 | group 1 | 921c077 | 11 passed, 8.66 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g1.tsv` |
| g2a | group 2, pointwise mode | 763a074 | 12 passed, 8.76 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g2a.tsv` |
| g3 | group 3 | 90413fd | 12 passed, 8.85 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g3.tsv` |
| g4 | group 4 | 7cedddb | 12 passed, 23.06 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g4.tsv` |
| g5 | group 5 | 16baf1e | 12 passed, 8.95 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g5.tsv` |
| g6 | group 6 | 3eb8ae2 | 12 passed, 9.68 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g6.tsv` |
| g7 | group 7 | feb3e1d | 12 passed, 9.00 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g7.tsv` |
| g8 | group 8 | c7f799e | 12 passed, 9.02 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g8.tsv` |
| g9 | group 9 | b482975 | 12 passed, 8.90 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g9.tsv` |
| g2b_ref | the new c5 reference, spring still in the code | a1395f4 | 12 passed, 9.16 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g2b_ref.tsv` |
| g2b | group 2, cell-average spring | b18879b | 13 passed, 32.84 s | 8 of 8 | 1188 / 1188 | 0 | `F/parity_g2b.tsv` |

Notes on Table 4:

- The per-cell comparison counts are in each `parity_<tag>.tsv`. They did not change from g1 to
  g2b_ref, because a removed setting still yields one record. At g2b each cell has one more,
  `case.k_spring` (Table 6), which is the step from 1180 to 1188.
- g0 was run before the summary script existed; the WP1 log's table 3 holds the base values.
- Two groups changed UFL forms, so the forms recompiled, which is the longer g4 and g2b time.
  Group 4: `K_nd` and `F_adv_nd` lost their factors of `Constant(1.0)`. Group 2: the film lost
  its `+ hd_nd_const` term, a `Constant(0.0)` in every cell. The results did not move in either,
  so there is no round-off to attribute.
- The first g8 run failed with a `TypeError` because the edit script had stopped half-way
  (my error: a wrong end marker). The edits were completed and the gate rerun; `gate_g8.log`
  is the rerun. No result was compared in the failed run.

End-of-work runs, on the tree of b18879b:

- `_claude/wslrun python tests/test_parity.py --out F/parity_report.tsv`: exit 0, eight cells
  PASS, every category's largest relative difference zero (`F/parity_report.tsv`, per-record
  `F/parity_report_detail.tsv`, console `F/logs/parity_report_run.log`).
- `_claude/wslrun python examples/run_cell.py`: exit 0, both solves OK
  (`F/logs/example_run_cell.log`). Its printed `dP`, `P_asp`, `contact_frac` and Newton counts
  are the same digits as the WP1 run (`micro_standalone_wp0_wp1_implementation_log_2026-11-11_files/logs/example_run1.log`).

Both runs end with PETSc's "5 unused database options" warning. It is in the base gate log too
(`F/logs/gate_g0_base_c96ad39.log`), so it predates WP2.

## 5. The references do not use what was removed

`F/reference_removed_options.py` reads the eight `tests/references/*.npz` and writes
`F/reference_removed_options.tsv`: one row per option and cell, with the stored value and the
value the pruned code hard-wires. It exits 0: no removed option is used by a reference.

Table 5. Stored values, from `F/reference_removed_options.tsv` (c5 as re-made).

| Group | Option | Stored in the references | Hard-wired now |
|---|---|---|---|
| 2 | `spring_pointwise` | False in all eight | False (removed) |
| 2 | `k_spring` | 0 in six; 1e20 in c6 and c8, where the parent's spring did not act (marked "unused") | no spring (removed) |
| 3 | `contact_model`, mixed cells | `fft_elastic_v1` in all six | `fft_elastic_v1` |
| 3 | `use_reduced_modulus`, mixed cells | False in all six | False (removed) |
| 3 | `overclosure_measure`, mixed cells | `patch_max` in all six | `patch_max` (removed) |
| 3 | result `contact_model`, full-film cells | `legacy_patch_v1` in both (the parent's default tag; no contact ran) | `off` (§6.2) |
| 5 | `gauge_lift` | False in all eight | False (removed) |
| 5 | `spos_form` | `sqrt` in six, `spline` in two, `c2` in none | `c2` removed |
| 8 | `phase_average`, mixed cells | `contact_table_v1` in the pair cell c4, `off` in five | `pair_branch` True for c4, False for five |
| 9 | the five environment variables | none set when the references were made (their `env` record) | each read removed |

`contact_E`, `contact_nu` and `k_asp_exponent` were read only by `legacy_patch_v1`; the check
for them is that the reference's own law is `fft_elastic_v1` (Table 6).

## 6. Departures from the brief, and clarifications

1. **One reference was regenerated.** The brief says the references are never regenerated. c5
   was re-made with `k_spring = 0` on Rob's decision of 11 Nov 2026 (§2). The other seven are
   the WP1 files.
2. **Full-film `contact_model` tag (group 3).** The parent's result carried its settings
   default `legacy_patch_v1` on a cell with no contact. That law is gone, so the API reports
   `off` (`CONTACT_OFF` in `api.py`). The parity test maps the reference's tag
   (`FULL_FILM_CONTACT_TAG`). This changes a result string a caller can read.
3. **`ContactSettings.nu` and `ContactSettings.k_asp_exponent` kept (group 3).** They are API
   inputs that only the removed law read. The brief's rule names the `legacy` keys, which are
   removed; these two are named fields of the plan's `ContactSettings`, so they are left, marked
   "not read since WP2", for Rob to decide.
4. **`_pair_level` and `_apply_warm_start` removed (group 7).** Neither is named in the table;
   each was read only by the removed warm-start code.
5. **`pair_branch` replaces the tag (group 8).** `PhaseNeutralContactSettings.phase_average`
   (a string) is now `pair_branch` (a bool); `_phase_table_on()` is `_pair_branch_on()`. The API
   sets it from `roughness.is_pair`, as it set the tag. No input was added to `microsolver` or
   `MicroSettings`. The flag stays a setting, not a test on the geometry, so that a pair on the
   full-film class is still refused with the ripple on. c4, the pair cell, is exact at g8.
6. **`auto_solve` kept (group 6).** Not in the table. It now only calls `run_steady()`, and the
   API passes False.
7. **Logging stays disabled at import (group 9).** With `MICRO_DEBUG_LOG` unset the parent
   called `logging.disable(logging.CRITICAL)` when the module was imported. That is the frozen
   behaviour, so the call stays. It is a process-wide side effect of importing the solver.
8. **Flags removed with their only reader (group 9).** `_uref_floored`, `_gamma_nd_capped` and
   the `sys.stdout.flush()` after the removed print. `_epsP_nd_capped` is left (§7).
9. **Occurrence counts.** The brief's table gives one group-4 occurrence in the contact class.
   At c96ad39 there is none, so nothing was changed there for group 4.
10. **`k_spring` is no longer an input (group 2).** `MicroSettings.legacy` refuses the key as
    unknown. Two refusals that read it went with it: the ripple's "needs `k_spring = 0`" and the
    contact class's "0 < `k_spring` < 1e19". The contact class's `_spring_enabled` also called
    `_fft_validate`; that validation still runs in `_refresh_contact`, before every solve, and
    in `_fft_asperity_load`.
11. **Two methods renamed (group 2).** `_solve_with_spring` is `_solve_level` and
    `_pair_solve_with_spring` is `_pair_solve`: with no spring left the old names were wrong.
    The mixed class keeps its `_steady_solve` and `_seed_solve` overrides, which now call
    `_solve_level`, so the pair branch is entered where it was.

## 7. Candidates for further pruning (left in place)

12 items that look dead but are not in the brief's table, or are out of WP2's scope, are
listed with where they are and why in `F/pruning_candidates.md`. None was removed. The ones
that bear on a decision: `ContactSettings.nu` and `.k_asp_exponent` (decision 1), `auto_solve`
(§6.6), the logging switch (§6.7), and `h_base_nd`, the same object as `h_nd` since the
spring went.

## 8. Open items and commits

Open:

1. The three solver class files read no environment variable now (open item 6.1 of the WP1
   log). The one read left in `microcell/` is `EHL_SURFACE_STORE` in `roughness/case.py`, which
   the brief gives to WP3.
2. `docs/notes/provenance.tsv` still lists the parent's line counts for the pruned files; it
   records where they came from and was left as it is. The README says which files are pruned.
3. The gate was run on this machine only (WSL). Nothing was run on Calder.
4. `F/diag_spring_k0.py` and `F/diag_spring_k0_names.py` set the `k_spring` key, so they run
   only on a tree before b18879b. Their docstrings say so. They are the record of Table 2.
5. `tests/references/logs/c5_cosine_fullfilm.log` is byte-identical after the re-make (git
   shows no change): the tool's log line prints `dP`, which the spring did not move.

Table 7. Commits on `main` of `MicroCell`. Author `RFurze <py07rf@leeds.ac.uk>`. Sizes:
`F/commits.tsv`. `origin/main` is at 7d64e1e (`git reflog show origin/main`: "update by push",
11 Nov 2026 11:39; not pushed by this session), so the last three rows are not on the remote.

| Commit | Content |
|---|---|
| 921c077 | group 1; the session prompt filed; `F/` started |
| 763a074 | group 2, pointwise mode; removed-settings check and its negative control in the test |
| 90413fd | group 3 |
| 7cedddb | group 4 |
| 16baf1e | group 5 |
| 3eb8ae2 | group 6 |
| feb3e1d | group 7 |
| c7f799e | group 8 |
| b482975 | group 9 |
| 68028f5 | README provenance note; final parity report and example log |
| 7d64e1e | this log as it stood before Rob's decision, and the files it cited |
| a1395f4 | the c5 reference re-made; `tools/make_references.py`; `F/c5_reference_old_vs_new.*`; gate g2b_ref |
| b18879b | group 2, cell-average spring; `case.k_spring` check and its negative control; gate g2b |
| (next) | this log revised, the final report and example log rerun, the evidence files updated |

## 9. Changed test expectations

Table 6 is `F/changed_test_expectations.md`: 9 rows, each with the commit group and the
reason. The comparison of results is unchanged: same fields, same tolerance. In short:

- a setting WP2 removed is checked against the reference's stored value, not skipped
  (`REMOVED_SETTINGS`, `REMOVED_UNREAD_SETTINGS`, `REPLACED_SETTINGS`, `FULL_FILM_CONTACT_TAG`);
- the removed spring is checked per cell by a new record `case.k_spring` (`spring_acted`): it
  fails for a transient full-film cell with `k_spring > 0`, or a mixed cell with `k_spring`
  between 0 and 1e19;
- two negative controls were added, `test_removed_setting_is_checked` (763a074) and
  `test_removed_spring_is_checked` (b18879b);
- `settings_from_reference` no longer passes the removed `legacy` keys;
- the c5 reference was re-made (a1395f4, Table 2b).

## 10. Commands

The gate, from WSL (about 10 s; about 30 s the first time after a form change):

```
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
source scripts/ehl_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/ -q -p no:cacheprovider -rfE --tb=short
python tests/test_parity.py --out docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/parity_report.tsv; echo "exit $?"
python examples/run_cell.py; echo "exit $?"
```

Expected: `13 passed`; the script prints `parity: PASS -- 8 of 8 cells PASS` and exits 0 (about
10 s); the example prints two `OK` rows and exits 0 (about 10 s).

The c5 re-make, as it was run through `_claude/wslrun` from the parent's root (about 5 s; it
needs no rerun): `python ../MicroCell/tools/make_references.py --all --cell c5_cosine_fullfilm --procs 1`.

The reference check of §5, from Git Bash in `MicroCell/` (under a second, no FEniCS):

```
python -I docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/reference_removed_options.py
```

Table 2b was made the same way with `F/c5_reference_old_vs_new.py <old npz> <new npz> <out tsv>`,
the old file being the copy in `_to_delete/` (§11).

## 11. File list

Changed: `microcell/micro_phase_neutral_class.py`, `microcell/micro_phase_neutral_class_mixed.py`,
`microcell/micro_phase_neutral_class_contact.py`, `microcell/micro_diagnostics.py`,
`microcell/contact.py`, `microcell/api.py`, `tests/test_parity.py`, `README.md`,
`tools/make_references.py`, `tests/references/c5_cosine_fullfilm.npz`.

Added: `docs/notes/micro_standalone_wp2_session_prompt_2026-11-11.md`, this log, and in `F/`:

- `lines_before_c96ad39.tsv`, `lines_after_68028f5.tsv`, `lines_after_b18879b.tsv`, `commits.tsv`,
  `gate_counts.tsv`, `pruning_candidates.md`, `changed_test_expectations.md`;
- `parity_summary.py`, `parity_g1.tsv` ... `parity_g9.tsv` (g2 is `parity_g2a.tsv`),
  `parity_g2b_ref.tsv`, `parity_g2b.tsv`;
- `parity_report.tsv`, `parity_report_detail.tsv`;
- `reference_removed_options.py`, `reference_removed_options.tsv`;
- `c5_reference_old_vs_new.py`, `c5_reference_old_vs_new.tsv`;
- `diag_spring_k0.py`, `diag_spring_k0.tsv`, `diag_spring_k0.log`, `diag_spring_k0_names.py`,
  `diag_spring_k0_names.tsv`;
- `logs/gate_<tag>.log`, `logs/summary_<tag>.log`, `logs/parity_report_run.log`, `logs/example_run_cell.log`.

Nothing was deleted outside git. `MicroCell/_to_delete/wp2_c5_spring_2026-11-11/` (not in git)
holds the old c5 reference and its log, and the pre-edit copies of `tools/make_references.py`
and of this log. Copies of the spring diagnostic are also in
`RefactorRefactor/_to_delete/micro_standalone_2026-11-11/wp2_logs/`.

## 12. Second-reader corrections

**First check (the log at 7d64e1e).** The `second-reader` agent (a fresh context) recomputed the
tables and checked the code claims against the files and git: about 135 items checked,
1 disagreement, 1 not checkable.

1. **Disagreement, corrected.** §2 cited `F/diag_spring_k0.tsv` for the c5 rows outside the
   tolerance, but that file names eight of the nine it counts. `F/diag_spring_k0_names.py` was
   added; it names all nine. The count in Table 2 was right.
2. **Not checkable.** The base gate g0 has no per-comparison table; "every comparison exact" is
   stated for the gates after a group. The base values are in the WP1 log's table 3.

Not checked by the reader: the commits' author field (`git log --format='%h %an <%ae>'
c96ad39..HEAD` shows it) and that `_epsP_nd_capped` is never read (grep: two assignments).
The full text of this section as first written is in git at 7d64e1e.

**Second check (this revision, after the c5 re-make and the spring removal).**

A fresh `second-reader` checked this revision against the files and git: about 150 items,
2 disagreements, about 10 not checkable.

1. **Disagreement, corrected.** The log said the commits were "not pushed". `git status -sb`
   shows `main...origin/main [ahead 2]` with `origin/main` at 7d64e1e. §0, decision 3 and
   Table 7 now say which commits are on the remote.
2. **Placeholder.** This section was still empty when the reader saw it; it is now filled.

Not checkable from the files it read: the failed first g8 run (no log of it was kept); the
parent's behaviour behind §6.7 and §6.8 (the parent source was not read); the brief's wording
behind §6.1, §6.5 and §6.9 (the session prompt was not opened). Table 1's lists of removed
names were spot-checked by grep on the current tree, not diffed commit by commit, and Table 2b
was read from the kept tsv, not rerun.
