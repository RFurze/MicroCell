# Standalone micro-cell solver: WP2 implementation log, prune dead physics (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp2_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (decisions D1, D5, D6, D10; work package WP2).
Base: `MicroCell` `main` at c96ad39 (WP1). Parent `MultiscaleMixedBallinSocket` not touched.
`F/` below is `docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/`.

## 0. Summary

**State.** WP2 is built except for one part of group 2, which waits on Rob. Eight of the nine
removal groups are done in full and group 2 in part, one commit per group, on `main` of
`MicroCell`, not pushed (Table 1, §8). WP3 is not started.

**Gate (measured).** After every group all eight reference cells PASS at relative 1e-12 and
every comparison is exact: the largest relative difference is zero in every group, including
group 4, which changed a UFL form (Table 4). No reference was regenerated and no tolerance
changed. The final script run and the example agree with WP1 (§4).

**Size (measured).** Table 3: the five pruned solver files fall from 5699 to 4518 lines.

**Not removed: the cell-average spring (group 2).** The brief's rule is "if a reference uses
it, stop and ask". Reference c5 uses it: three cosine cells store `k_spring = 1e20`, the
parent's "spring off" convention, but in the full-film class that value still runs the
cell-average spring loop on transient steps. Forcing `k_spring = 0` moves c5 by up to 1e-5
relative (Table 2). The pointwise mode, which no cell uses, is removed. The rest of the spring
(`k_spring`, `_solve_with_spring`, the loop and its constants) is still in the code.

**Departures.** Nine, numbered in §6. The three that change something a caller can see:
- 6.2 a full-film result's `contact_model` tag is `off` (was the parent's default `legacy_patch_v1`);
- 6.5 the pair branch is selected by a plain flag `pair_branch` on the contact settings, in
  place of the `phase_average = "contact_table_v1"` tag (the brief allowed this if results are
  unchanged; they are);
- 6.7 with `MICRO_DEBUG_LOG` gone, the import still disables logging for the process, as the
  parent did with the variable unset.

**Changed test expectations.** Table 6: removed settings are checked against the reference, not
skipped; one test added (Table 4 shows the count). **Open items** §8; candidates for further
pruning §7, left in place.

**Decisions for Rob.**
1. The spring and c5 (§2). Recommended: A, re-make the c5 reference from the parent at 2aaccad
   with `k_spring = 0`, then remove the spring. B: keep the cell-average spring until WP4.
   C: remove it and accept an attributed 1e-5 difference on c5 (not recommended: it loosens the gate).
2. `ContactSettings.nu` and `ContactSettings.k_asp_exponent` are now unread API inputs (§6.3).
   Recommended: remove them at WP4 with the settings collapse. Alternative: remove them now.
3. Accept or reject departures 6.2, 6.5 and 6.7.
4. Push `main` of `MicroCell`, and the go-ahead for WP3 as its own session.

## 1. What was removed

Table 1. One row per group. Commit sizes are in `F/commits.tsv`.

| # | Group | Status | Commit | What went |
|---|---|---|---|---|
| 1 | Dead imports, `__main__` blocks | done | 921c077 | The `__main__` shim of the base class; the mixed file's harness (`_main_params`, `_main_settings` with its `from config import micro_solver`, `__main__`). |
| 2 | Spring deformation | part | 763a074 | The pointwise mode: `spring_pointwise`, `hd_nd_func`, `_target_hd_nd_field`, `_spring_solve_pointwise`, `spring_relax`, its branch in `_solve_with_spring_once`, its refusal in `_ripple_setup`, its part of `_cg1_film_parts` (diagnostics). Kept: everything of the cell-average spring (§2). |
| 3 | `legacy_patch_v1`, `hertz_summit_v1` | done | 90413fd | Contact class: `_summit_geometry`, `_summit_pressure`, `_find_contact_patches`, `_asperity_pressure`, every non-fft branch, settings `contact_E`, `contact_nu`, `use_reduced_modulus`, `overclosure_measure`, `k_asp_exponent`. `contact.py`: `LEGACY`, `HERTZ`, `hertz_response`, `CosineSummits`, `cosine_summits`. Diagnostics: the unsupported-model branches. API: `legacy` keys `contact_E`, `contact_model`, `use_reduced_modulus`, `overclosure_measure`. |
| 4 | Roelands, compressibility stubs | done | 7cedddb | `roelands_eta_dim` (base; its callers in base, mixed, diagnostics now read `params.eta0`), `rho_nd` and `eta_nd` (base, mixed), both `Constant(1.0)`. |
| 5 | `gauge_lift`, `c2` form | done | 16baf1e | The `gauge_lift` setting and its warning; `smooth_pos_c2`, `smooth_pos_deriv_c2` and the `c2` value of `spos_form`; the then-unused `import warnings`. |
| 6 | Export hooks, `mesh_h`, `k` | done | 3eb8ae2 | Constructor arguments `mesh_h`, `k`, `export_vtk`, `output_dir`; `_setup_export`, `_open_xdmf_series`, `_write_xdmf_step`, `_close_xdmf_series`, `post_process`; the XDMF calls and stride in `_run_once`; the XDMF timers; the then-unused `Optional` import. |
| 7 | Warm-start plumbing | done | feb3e1d | Constructor argument `warm_start_state`, `_apply_warm_start`, `warm_start_snapshot`; `set_pair_warm`, `pair_warm_record`, `_pair_warm_begin`, `_pair_warm_check`, the `_pair_warm*` attributes, the warm branch of `_ripple_write_field`, and `_pair_level` (§6.4). |
| 8 | Table-only code | done | c7f799e | `contact_only_load`; `PHASE_AVERAGE_OFF`, `PHASE_AVERAGE_TABLE`, `PHASE_AVERAGES`; the `phase_average` setting, replaced by `pair_branch` (§6.5). The pair branch and the warm-start contact rescue are kept. |
| 9 | Environment reads | done | b482975 | The five reads, each frozen at its unset value: debug log off, nondimensionalisation print off, Uref floor off (its block removed), smoothing cap 1.0 (now the module constant `_EPSP_ND_CAP`), `spos_form` from the settings only. The then-unused `import os`. |
| - | README, final report | - | 68028f5 | The provenance note marks the five pruned files; `F/parity_report.tsv` and the example log. |

Files not touched: `microcell/roughness/`, the shear modules, `ripple.py`, `elastic_contact.py`,
`fields.py`, `tests/references/`, `tools/`, `examples/`.

For groups 4, 6 and 7 there is no stored setting to check. Group 4's stubs were constants. For
groups 6 and 7 the evidence is the API at c96ad39 (`git show c96ad39:microcell/api.py`): it
passed `export_vtk=False`, passed `mesh_h` and `k` that the solver ignored, passed no
`warm_start_state` and never called `set_pair_warm`, and with those calls it already matched
the references exactly at WP1.

## 2. Group 2: the spring that a reference uses

The base class enables the spring when `k_spring > 0`. The cosine parent case carries
`k_spring = 1e20` as its "off" value. The contact class never runs the spring (and refuses a
stiffness between 0 and 1e19); in the base class, used for full film, the cell-average loop
runs on every transient step. So of the three cells that store 1e20, only the transient
full-film one feels it.

Table 2. The standalone at commit 763a074 with `k_spring` forced to 0, against the unchanged
references (measured; `F/diag_spring_k0.tsv`, script `F/diag_spring_k0.py`, log `F/diag_spring_k0.log`).

| Cell | `k_spring` stored | Comparisons | Outside 1e-12 | Largest relative difference | Where |
|---|---|---|---|---|---|
| c5_cosine_fullfilm | 1e20 | 130 | 9 | 9.997e-06 | `delta_Q` |
| c6_cosine_fft_sqrt | 1e20 | 152 | 0 | 0 | - |
| c8_cosine_fullfilm_steady | 1e20 | 130 | 0 | 0 | - |

All the c5 comparisons outside the tolerance are listed by name, with their values, in
`F/diag_spring_k0_names.tsv` (script `F/diag_spring_k0_names.py`, run on the tree of b482975;
its numbers are the same as the 763a074 run's). They are `delta_Q`, the film minimum and
maximum, the film field, `tau_res`, the fluid shear mean, `Q`, and the Newton iteration count,
an integer that differs without a relative difference. The other five cells store
`k_spring = 0` (`F/reference_removed_options.tsv`).

Cause (inferred from the code, consistent with the table): `_spring_enabled()` in the base
class is `k_spring > 0`, and `_steady_solve` of the base class is a raw solve, which is why the
steady cell c8 does not move.

Options for Rob:

- **A (recommended).** Re-make the c5 reference from the parent at 2aaccad with `k_spring = 0`
  (one `changes={"k_spring": 0.0}` entry in `tools/make_references.py`), then remove the spring.
  Prediction to check, written here before any such run: the standalone with the spring removed
  reproduces the new reference on all c5 comparisons at 1e-12. This needs Rob's go-ahead because
  the brief says the references are never regenerated.
- **B.** Keep the cell-average spring in the base class until WP4. Group 2 stays "part".
- **C.** Remove it and record the Table 2 difference as attributed. Not recommended: the gate
  rule is 1e-12 and this is 1e-5.

## 3. Lines per file

Table 3. `wc -l` at c96ad39 (`F/lines_before_c96ad39.tsv`) and at 68028f5
(`F/lines_after_68028f5.tsv`; the solver files and the test are unchanged since b482975).

| File | Before | After | Change |
|---|---|---|---|
| `microcell/micro_phase_neutral_class.py` | 3501 | 2924 | -577 |
| `microcell/micro_phase_neutral_class_mixed.py` | 859 | 741 | -118 |
| `microcell/micro_phase_neutral_class_contact.py` | 770 | 368 | -402 |
| `microcell/micro_diagnostics.py` | 476 | 459 | -17 |
| `microcell/contact.py` | 93 | 26 | -67 |
| Five pruned solver files | 5699 | 4518 | -1181 |
| `microcell/api.py` | 649 | 638 | -11 |
| `tests/test_parity.py` | 357 | 435 | +78 |
| `README.md` | 110 | 120 | +10 |

## 4. Parity per group

The gate after each group: `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from
`MicroCell/`, log `F/logs/gate_<tag>.log`, count lines collected in `F/gate_counts.tsv`. The
gate prints no differences when it passes, so `F/parity_summary.py` (a pytest file run the same
way) runs `tests/test_parity.run_cell` on every cell and writes `F/parity_<tag>.tsv`.

Table 4. Measured. "Exact" counts comparisons with relative difference zero.

| Tag | After group | Commit | pytest | Cells PASS | Comparisons exact / total | Largest relative difference | Source |
|---|---|---|---|---|---|---|---|
| g0 | base | c96ad39 | 11 passed, 8.73 s | 8 of 8 | not tabulated | not tabulated | `F/logs/gate_g0_base_c96ad39.log` |
| g1 | 1 | 921c077 | 11 passed, 8.66 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g1.tsv` |
| g2a | 2 (part) | 763a074 | 12 passed, 8.76 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g2a.tsv` |
| g3 | 3 | 90413fd | 12 passed, 8.85 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g3.tsv` |
| g4 | 4 | 7cedddb | 12 passed, 23.06 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g4.tsv` |
| g5 | 5 | 16baf1e | 12 passed, 8.95 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g5.tsv` |
| g6 | 6 | 3eb8ae2 | 12 passed, 9.68 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g6.tsv` |
| g7 | 7 | feb3e1d | 12 passed, 9.00 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g7.tsv` |
| g8 | 8 | c7f799e | 12 passed, 9.02 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g8.tsv` |
| g9 | 9 | b482975 | 12 passed, 8.90 s | 8 of 8 | 1180 / 1180 | 0 | `F/parity_g9.tsv` |

Notes on Table 4:

- The per-cell comparison counts are in each `parity_<tag>.tsv`; they did not change between
  groups, because a removed setting still yields one record (Table 6).
- g0 was run before the summary script existed; the WP1 log's table 3 holds the base values.
- Group 4 changed two UFL forms (`K_nd` and `F_adv_nd` lost their factors of `Constant(1.0)`),
  so the forms recompiled, which is the longer g4 time. The results did not move, so there is
  no round-off to attribute.
- The first g8 run failed with a `TypeError` because the edit script had stopped half-way
  (my error: a wrong end marker). The edits were completed and the gate rerun; `gate_g8.log`
  is the rerun. No result was compared in the failed run.

End-of-work runs, on the tree of b482975 (the solver files are the same at 68028f5):

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

Table 5. Stored values, from `F/reference_removed_options.tsv`.

| Group | Option | Stored in the references | Hard-wired now |
|---|---|---|---|
| 2 | `spring_pointwise` | False in all eight | False (removed) |
| 2 | `k_spring` | 0 in five, 1e20 in c5, c6, c8 | kept as an input (§2) |
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

1. **Group 2 is part done.** The cell-average spring stays because reference c5 uses it (§2).
2. **Full-film `contact_model` tag (group 3).** The parent's result carried its settings
   default `legacy_patch_v1` on a cell with no contact. That law is gone, so the API reports
   `off` (`CONTACT_OFF` in `api.py`). The parity test maps the reference's tag
   (`FULL_FILM_CONTACT_TAG`). This changes a result string a caller can read.
3. **`ContactSettings.nu` and `ContactSettings.k_asp_exponent` kept (group 3).** They are API
   inputs that only the removed law read. The brief's rule names the `legacy` keys, which are
   removed; these two are named fields of the plan's `ContactSettings`, so they are left, marked
   "not read since WP2", for Rob to decide.
4. **`_pair_level` and `_apply_warm_start` removed (group 7).** Neither is named in the table.
   `_apply_warm_start` was called only for `warm_start_state`; `_pair_level` was read only by
   the warm branch and the warm record, so both are dead after the group.
5. **`pair_branch` replaces the tag (group 8).** `PhaseNeutralContactSettings.phase_average`
   (a string) is now `pair_branch` (a bool); `_phase_table_on()` is `_pair_branch_on()`. The API
   sets it from `roughness.is_pair`, as it set the tag. No input was added to `microsolver` or
   `MicroSettings`. The flag stays a setting, not a test on the geometry, so that a pair on the
   full-film class is still refused with the ripple on, as before. Evidence that results are
   unchanged: c4, the pair cell with the ripple on, is exact after the group (Table 4, g8).
6. **`auto_solve` kept (group 6).** It is not in the table. With `post_process` gone,
   `auto_solve=True` now only calls `run_steady()`. The API passes False.
7. **Logging stays disabled at import (group 9).** With `MICRO_DEBUG_LOG` unset the parent
   called `logging.disable(logging.CRITICAL)` when the module was imported. That is the frozen
   behaviour, so the call stays. It is a process-wide side effect of importing the solver.
8. **Flags removed with their only reader (group 9).** `_uref_floored` went with the Uref floor
   and `_gamma_nd_capped` with the nondimensionalisation print that read it; the
   `sys.stdout.flush()` after that print went too. `_epsP_nd_capped` was unread before WP2 and
   is left (§7).
9. **Occurrence counts.** The brief's table gives one group-4 occurrence in the contact class.
   At c96ad39 there is none (`git show c96ad39:microcell/micro_phase_neutral_class_contact.py`
   has no `roelands`, `rho_nd` or `eta_nd`); nothing was changed there for group 4.

## 7. Candidates for further pruning (left in place)

Not in the brief's table, or out of WP2's scope. None was removed.

| Candidate | Where | Why it looks dead |
|---|---|---|
| The cell-average spring and `k_spring` | base, mixed, contact, `api.py` (`legacy` key) | pending decision 1 |
| `ContactSettings.nu`, `.k_asp_exponent` | `api.py` | unread since group 3 |
| `contact_model` settings field | contact settings | single-valued; the shear modules read it |
| `UNSUPPORTED_MODELS` and its branches | `contact_shear.py`, `contact_sample.py` | name the two removed laws (WP3 scope) |
| Hertz descriptor entries | `ideal_ht.py` | fed the removed summit law (WP3 scope; not checked in detail) |
| `effective_modulus`, `CONTACT_MODELS` | `contact.py` | no reader in the package |
| `MicroPhaseNeutralSolver = MicroPhaseNeutralMixedSolver` | end of the mixed file | back-compat alias; the API imports the classes by their own names |
| `auto_solve` | base constructor | the API always passes False |
| `logging.disable` at import, `logger` | base | §6.7; one `logger.warning` call remains and can never print |
| `_epsP_nd_capped` | base | written, never read |
| `_contact_restarts`, `_contact_restart_kinds` | contact class | counted by the rescue; the worker that printed them is not in this repo |
| `print_timing` and the timing summary | base | a settings field; WP4 |

## 8. Open items and commits

Open:

1. Group 2 remainder (decision 1).
2. The three solver class files read no environment variable now (open item 6.1 of the WP1
   log). The one read left in `microcell/` is `EHL_SURFACE_STORE` in `roughness/case.py`, which
   the brief gives to WP3.
3. `docs/notes/provenance.tsv` still lists the parent's line counts for the pruned files; it
   records where they came from and was left as it is. The README says which files are pruned.
4. The gate was run on this machine only (WSL). Nothing was run on Calder.

Table 7. Commits on `main` of `MicroCell`, not pushed. Author `RFurze <py07rf@leeds.ac.uk>`.
Sizes: `F/commits.tsv`.

| Commit | Content |
|---|---|
| 921c077 | group 1; the session prompt filed; `F/` started |
| 763a074 | group 2, part; removed-settings check and its negative control in the test |
| 90413fd | group 3 |
| 7cedddb | group 4 |
| 16baf1e | group 5 |
| 3eb8ae2 | group 6 |
| feb3e1d | group 7 |
| c7f799e | group 8 |
| b482975 | group 9 |
| 68028f5 | README provenance note; final parity report and example log |
| (next) | this log and the files it cites |

## 9. Changed test expectations

Table 6. All in `tests/test_parity.py`. The comparison of results is unchanged: same fields,
same tolerance, same references.

| Change | Since | Why |
|---|---|---|
| `REMOVED_SETTINGS` (`spring_pointwise`, `use_reduced_modulus`, `overclosure_measure`, `gauge_lift`): the built settings must not carry the key, and the reference's stored value must equal the frozen value exactly | groups 2, 3, 5 | the brief: a removed setting is checked, not skipped |
| `REMOVED_UNREAD_SETTINGS` (`contact_E`, `contact_nu`, `k_asp_exponent`): the built settings must not carry the key, and the reference's `contact_model` must be `fft_elastic_v1` | group 3 | only the removed law read them, so there is no value to freeze |
| `FULL_FILM_CONTACT_TAG`: a full-film reference's result tag `legacy_patch_v1` is compared as `off` | group 3 | §6.2 |
| `REPLACED_SETTINGS` (`phase_average` -> `pair_branch`, `off` -> False, `contact_table_v1` -> True): the built settings must carry the new key with the mapped value and type, and not the old key | group 8 | §6.5 |
| `settings_from_reference` no longer passes the `legacy` keys `contact_E`, `contact_model`, `use_reduced_modulus`, `overclosure_measure` | group 3 | the API no longer has them |
| New test `test_removed_setting_is_checked`: with the frozen value of `spring_pointwise` set wrong, cell c8 must fail on exactly `settings.spring_pointwise` | group 2 | a negative control for the new check |

## 10. Commands

The gate, from WSL (about 10 s; about 25 s the first time after a form change):

```
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
source scripts/ehl_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/ -q -p no:cacheprovider -rfE --tb=short
python tests/test_parity.py --out docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/parity_report.tsv; echo "exit $?"
python examples/run_cell.py; echo "exit $?"
```

Expected: `12 passed`; the script prints `parity: PASS -- 8 of 8 cells PASS` and exits 0 (about
10 s); the example prints two `OK` rows and exits 0 (about 10 s).

The reference check of §5, from Git Bash in `MicroCell/` (under a second, no FEniCS):

```
python -I docs/notes/micro_standalone_wp2_implementation_log_2026-11-11_files/reference_removed_options.py
```

## 11. File list

Changed: `microcell/micro_phase_neutral_class.py`, `microcell/micro_phase_neutral_class_mixed.py`,
`microcell/micro_phase_neutral_class_contact.py`, `microcell/micro_diagnostics.py`,
`microcell/contact.py`, `microcell/api.py`, `tests/test_parity.py`, `README.md`.

Added: `docs/notes/micro_standalone_wp2_session_prompt_2026-11-11.md`, this log, and in `F/`:

- `lines_before_c96ad39.tsv`, `lines_after_68028f5.tsv`, `commits.tsv`, `gate_counts.tsv`;
- `parity_summary.py`, `parity_g1.tsv` ... `parity_g9.tsv` (g2 is `parity_g2a.tsv`);
- `parity_report.tsv`, `parity_report_detail.tsv`;
- `reference_removed_options.py`, `reference_removed_options.tsv`;
- `diag_spring_k0.py`, `diag_spring_k0.tsv`, `diag_spring_k0.log`, `diag_spring_k0_names.py`,
  `diag_spring_k0_names.tsv`;
- `logs/gate_<tag>.log`, `logs/summary_<tag>.log`, `logs/parity_report_run.log`, `logs/example_run_cell.log`.

Nothing was deleted outside git. Copies of the spring diagnostic are also in
`RefactorRefactor/_to_delete/micro_standalone_2026-11-11/wp2_logs/`.

## 12. Second-reader corrections

The `second-reader` agent (a fresh context) recomputed the tables and checked the code claims
against the files and git: about 135 items checked, 1 disagreement, 1 not checkable.

1. **Disagreement, corrected.** §2 said "the other eight c5 rows ... are listed with their
   values in `F/diag_spring_k0.tsv`". That file counts nine comparisons outside the tolerance
   but lists eight by name: its script printed only records with a non-zero relative
   difference. The ninth was not named anywhere. `F/diag_spring_k0_names.py` was added and run;
   it lists all nine, and the ninth is the Newton iteration count. §2 now cites that file. The
   count in Table 2 was right.
2. **Not checkable from the WP2 files.** The base gate g0 has no per-comparison table (Table 4
   says "not tabulated"). The statement that every comparison is exact is made for the nine
   gates after a group, g1 to g9, each with its `parity_<tag>.tsv`. The base values are in the
   WP1 log's table 3.

The reader also noted what it did not check: the author field of the commits (they were made
with `--author="RFurze <py07rf@leeds.ac.uk>"`; `git log --format='%h %an <%ae>' c96ad39..HEAD`
shows it), and that `_epsP_nd_capped` is never read (grep: two assignments, no read).
