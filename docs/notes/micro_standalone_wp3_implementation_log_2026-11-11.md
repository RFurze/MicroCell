# Standalone micro-cell solver: WP3 implementation log, prune the helpers (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp3_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (decisions D2, D5, D10; work package WP3).
Base: `MicroCell` `main` at c89915a. Parent `MultiscaleMixedBallinSocket` not touched.
`F/` below is `docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/`.

## 0. Summary

**State.** WP3 is built: the five removal groups are done, one commit each, on `main` of
`MicroCell` (Table 1). Nothing is pushed. WP4 and WP5 are not started.

**Gate (measured).** After every group all eight reference cells PASS at relative 1e-12 and
every comparison is exact; the test count did not change (Table 3). No reference was
regenerated, no tolerance changed, no test was edited. The final script run and the example
agree with the gate and with WP2's example (§4).

**Size (measured).** Table 2: `microcell/roughness/` and the nine other in-scope modules, by
file and in total. Eight roughness files left git; they are in
`_to_delete/wp3_removed_2026-11-11/roughness/`.

**Rewiring (measured).** `Roughness.generated` now calls the generator directly. Its arrays
are bit for bit those of the old path on every argument set tried (Table 5), recorded before
the rewiring; its arguments are unchanged.

**Not removable.** The descriptor tuple entries in `ideal_ht.py` that the brief listed under
group 2 are still read (§1, group 2).

**Departures.** Seven, numbered in §6. The ones Rob may want to reverse:
- 6.3 `roughness/stats.py` is removed whole and `roughness/pair.py` is cut to one function:
  the brief's "kept by the plan" list named `pair` and "the parts of stats they call", but
  after the cut nothing kept calls `SurfacePair` or `stats`;
- 6.4 the generator keeps only the Gaussian autocorrelation and Gaussian height distribution
  (what `Roughness.generated` calls): the other PSD models, HPD families, the Sm fit and the
  filtered Gaussian generator are gone, and `quantiles.py` keeps only `norm_ppf`;
- 6.5 three things a caller could see changed, none on a path the API takes (§6).

**Changed test expectations.** None (§9).

**Rob's answer (11 Nov 2026): "Accept"**, to departures 6.3, 6.4 and 6.5 as built. Nothing is
restored: `SurfacePair`, `stats.py` and the non-Gaussian generator code stay out (they are in
the parent and in git history), and the three caller-visible changes stand.

**Still for Rob.** Push `main` (the WP3 commits and the commits of this log); the WP4
go-ahead, as its own session.

**Open items** §8; candidates for further pruning §7, left in place.

## 1. What was removed

Table 1. Commits, in the order made. Source: `F/commits.tsv` (git `--shortstat`),
`F/removed_names.tsv` (top-level names in `F/inventory_base.tsv` and absent from the next
inventory).

| Group | Commit | Files changed | Insertions | Deletions | Top-level names removed |
|---|---|---|---|---|---|
| 1 shear macro-frame and batch code | 548ecaa | 2 | 9 | 550 | 75 |
| 2 remains of the removed contact laws | e47664c | 5 | 11 | 79 | 8 |
| 3 macro-side helpers of `pressure.py` | 05d034d | 1 | 6 | 77 | 11 |
| 5 surface store and its environment read | a9a056d | 3 | 18 | 289 | 23 |
| 4 roughness plumbing | 6c047c1 | 13 | 108 | 2878 | 146 |

The name-by-name list, with each name's class at the base and the group that removed it, is
`F/removed_names.tsv`. Group 4 also added three names to `frontend.py` (`SA_RTOL`, `SA_ATOL`,
`_mean_abs`).

**Group 1** (`fluid_shear.py`, `contact_shear.py`). Every candidate in the brief was dead and
went, with the constants only they read: the run_micro array names, the batch statuses
(`dedup_member`, `legacy`, `unavailable`), the XDMF status codes, the metadata records.
`applied_asperity_traction` (`contact_shear.py`) was not in the brief's list; the inventory
had it read by nothing. Kept: the per-phase dataclasses, `union_fluid_traction`,
`union_contact_traction`, `shear_fields`, `failed_shear_fields`, the statuses the worker returns.

**Group 2.** `UNSUPPORTED_MODELS`, its status and branches and `finish_unsupported`
(`contact_shear.py`, `contact_sample.py`); `effective_modulus`, `CONTACT_MODELS` (`contact.py`,
now `FFT` and `ContactResolutionError`); `SurfaceContactHeight.__getitem__`
(`roughness/runtime.py`), which existed only to refuse the removed laws' index reads.
`phase_from_reductions` lost its `n_samples` argument, read only by the removed branch.
**Not removable:** the Hertz and summit descriptor entries of `ideal_ht.py`. The tuple
`(Ah, kx, ky, two_surface)` is read by `height` and by `micro_phase_neutral_class.py:1345`;
only the docstring's Hertz wording changed there.

**Group 3** (`pressure.py`). `require_definition`, `level_constraint_of`,
`require_level_constraint`, `effective_level_constraint`, and the constants only they read
(`LEVEL_CONSTRAINT_DEFAULT`, `LEVEL_CONSTRAINTS`, the batch extrema statuses).
`LEVEL_CONSTRAINT_RUN_DEFAULT` stays: `micro_phase_neutral_class.py:57` imports it.

**Group 5** (before group 4, departure 6.1). `roughness/case.py` removed whole: the store, the
descriptor format and the `EHL_SURFACE_STORE` read. In `runtime.py`, `SurfaceField.__init__`
(the descriptor loader), `load_field` and its cache. The API never reached the store:
`frontend.ArrayField` has built the field in memory since WP1 and overrides the loader, so no
caller sees a change and nothing was measured or asked (inferred from the code; the gate after
the group is the check).

**Group 4.** Removed whole: `spec.py`, `prepare.py`, `targets.py`, `report.py`, `modes.py`,
`assets.py`, `stats.py`. Cut to what is called: `generation.py`, `quantiles.py`, `pair.py`,
`evaluator.py` (`nodal_grid`, `bilinear_nodal`), `__init__.py` (re-exports `Bandwidth` and
`FourierSurface` only). `frontend.py`: `Roughness.generated` rewired (§5).

## 2. Inventory

`F/inventory.py` (pure `ast`, pytest form) lists every top-level function, class and constant
of the in-scope modules and classes it by who reads it: `solver` (reachable from
`microsolver` through `api.py`, the solver classes, `micro_diagnostics.py`, `fields.py`),
`tests`, `examples`, `tools`, `dead_only` (read only by names that are themselves dead),
`none`. Attribute reads are matched by name, so the classing errs toward "used".

Table 4. Names by class at each stage. Source: `F/inventory_stages.tsv`, from
`F/inventory_<stage>_counts.tsv`; the names are in `F/inventory_<stage>.tsv`.

| Stage | Commit | Total | solver | tests | examples | tools | dead_only | none |
|---|---|---|---|---|---|---|---|---|
| base | c89915a | 428 | 282 | 0 | 0 | 0 | 85 | 61 |
| g1 | 548ecaa | 353 | 282 | 0 | 0 | 0 | 29 | 42 |
| g2 | e47664c | 345 | 278 | 0 | 0 | 0 | 29 | 38 |
| g3 | 05d034d | 334 | 278 | 0 | 0 | 0 | 25 | 31 |
| g5_store | a9a056d | 311 | 241 | 0 | 0 | 9 | 33 | 28 |
| g4 (final) | 6c047c1 | 168 | 166 | 0 | 0 | 0 | 0 | 2 |

Notes. (a) `solver` at the base means statically reachable, not executed: the names of that
class removed in groups 2, 5 and 4 were reached only through branches of the removed contact
laws, through the loader that `ArrayField` overrides, and through the `spec` -> `prepare` ->
`targets` path that the rewiring replaced. (b) The `tools` class at g5_store is a name match
from `tools/make_references.py`, which imports the parent's `ehl.*`, not `microcell`.
(c) The two names left in class `none` are candidates A1 and A2 (§7).

`F/check_names.py` (pure `ast`) was run with the inventory after every group: no undefined
bare name, no `from microcell.x import N` or `alias.N` without a definition. Its unused-import
lists are `F/unused_imports_<stage>.txt`.

## 3. Lines per file

Table 2. `git show <commit>:<file> | wc -l`. Source: `F/lines_changed.tsv` (files whose count
changed), `F/lines_totals.tsv`, from `F/lines_before_c89915a.tsv` and `F/lines_after_6c047c1.tsv`
(every `.py` under `microcell/`, `tests/`, `examples/`, `tools/`).

| File | c89915a | 6c047c1 |
|---|---|---|
| `microcell/contact.py` | 26 | 12 |
| `microcell/contact_sample.py` | 402 | 395 |
| `microcell/contact_shear.py` | 602 | 269 |
| `microcell/fluid_shear.py` | 397 | 148 |
| `microcell/pressure.py` | 173 | 102 |
| `microcell/roughness/__init__.py` | 44 | 29 |
| `microcell/roughness/assets.py` | 478 | removed |
| `microcell/roughness/case.py` | 208 | removed |
| `microcell/roughness/evaluator.py` | 372 | 327 |
| `microcell/roughness/frontend.py` | 181 | 207 |
| `microcell/roughness/generation.py` | 632 | 194 |
| `microcell/roughness/modes.py` | 145 | removed |
| `microcell/roughness/pair.py` | 257 | 29 |
| `microcell/roughness/prepare.py` | 214 | removed |
| `microcell/roughness/quantiles.py` | 389 | 107 |
| `microcell/roughness/report.py` | 159 | removed |
| `microcell/roughness/runtime.py` | 280 | 211 |
| `microcell/roughness/spec.py` | 349 | removed |
| `microcell/roughness/stats.py` | 179 | removed |
| `microcell/roughness/targets.py` | 264 | removed |

| Set | Files, c89915a | Lines, c89915a | Files, 6c047c1 | Lines, 6c047c1 |
|---|---|---|---|---|
| `microcell/roughness/` | 15 | 4151 | 7 | 1104 |
| the nine in-scope modules outside `roughness/` | 9 | 2581 | 9 | 1907 |
| `microcell/` (all) | 34 | 12393 | 26 | 8672 |
| `tests/`, `examples/`, `tools/` | 4 | 1070 | 4 | 1070 |

`ideal_ht.py` changed (docstring wording) with its line count unchanged, so it is not in the
first table. `shear_reduce.py`, `surface_ht.py` and `elastic_contact.py` are unchanged: the
inventory found nothing dead in them at the top level.

## 4. Parity per group

Table 3. The gate after each group: `_claude/wsltest tests/ -q -p no:cacheprovider -rfE
--tb=short`, then `F/parity_summary.py`. Source: `F/parity_by_group.tsv`, from
`F/gate_counts.tsv` (count lines; logs `F/logs/gate_<tag>.log`) and `F/parity_<tag>.tsv`.

| Group | Commit | Gate count line | Cells PASS | Comparisons exact of total | Outside 1e-12 | Largest relative difference | Source file |
|---|---|---|---|---|---|---|---|
| base | c89915a | 13 passed in 9.13s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g0_base_c89915a.tsv` |
| 1 | 548ecaa | 13 passed in 9.05s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g1.tsv` |
| 2 | e47664c | 13 passed in 8.95s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g2.tsv` |
| 3 | 05d034d | 13 passed in 9.02s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g3.tsv` |
| 5 | a9a056d | 13 passed in 9.25s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g5_store.tsv` |
| 4 | 6c047c1 | 13 passed in 8.70s | 8/8 | 1188 of 1188 | 0 | 0.0 | `parity_g4.tsv` |

The thirteen tests include the three negative controls and
`test_generated_surface_reproduces_reference`. No UFL form changed. There is no nonzero
difference to attribute.

End runs at 6c047c1, both exit 0 (through `_claude/wslrun`):
- `python tests/test_parity.py --out F/parity_report.tsv`: the same cells, comparisons and
  exact counts as the group 4 row of Table 3 (`F/parity_report.tsv`; log
  `F/logs/parity_report_6c047c1.log`).
- `python examples/run_cell.py`: Table 6.

Table 6. The example, WP2 against WP3. Source: `F/example_run_cell.tsv`, from WP2's
`…wp2…_files/logs/example_run_cell.log` and `F/logs/run_cell_6c047c1.log` (as printed, seven
significant figures).

| Run | Solve | Status | dP [Pa] | P_asp [Pa] | Contact fraction | Newton |
|---|---|---|---|---|---|---|
| WP2 | transient | OK | -7.415780e+04 | 6.026040e+07 | 0.001817 | 6 |
| WP2 | steady | OK | -7.415780e+04 | 6.026040e+07 | 0.001817 | 3 |
| WP3 6c047c1 | transient | OK | -7.415780e+04 | 6.026040e+07 | 0.001817 | 6 |
| WP3 6c047c1 | steady | OK | -7.415780e+04 | 6.026040e+07 | 0.001817 | 3 |

## 5. The rewiring of `Roughness.generated`

Before: `GeneratedSurface` spec -> `prepare.resolve` -> `targets.apply_policy`. After: the
same arithmetic in the same order, written out in `frontend.py` for the one recipe the
constructor exposes (Gaussian ACF, Gaussian HPD, IAAFT, `Zh`, scaled to `Sa`): argument checks,
`psd_gaussian_acf`, `hpd_sorted_sample`, `iaaft`, centre once, scale by `Sa / mean|z|`, check
`Sa` to the parent's `SurfaceTargets` defaults (`SA_RTOL`, `SA_ATOL`).

Check, as the brief requires, written before the rewiring: `F/generated_rewire_check.py` stored
the constructor's arrays at a9a056d (`F/generated_before_rewire.npz`,
`F/generated_before_rewire.tsv`), then compared after the rewiring and again after the group's
last edit, with `np.array_equal`.

Table 5. Source: `F/generated_rewire_check.tsv`.

| Argument set | Identical | Largest absolute difference |
|---|---|---|
| `default_sa40_seed1000` | True | 0 |
| `sa10_seed7_corr2e-7` | True | 0 |
| `sa25_seed3_cell4um_n32_iter20` | True | 0 |
| `sa12_seed1001_5x4um_n16_iter0_moving` | True | 0 |
| `sa7_seed1001_moving` | True | 0 |

Every argument of the constructor is away from its default in at least one set. The other
constructors (`array`, `pair`, `cosine`) were not rewired.

Finding, not a WP3 change: reference c4's two stored surfaces are reproduced by
`Roughness.generated` to round-off, not bit for bit.

Table 7. Source: `F/generated_c4_seeds.tsv`.

| c4 member | Seed | Identical | Largest relative difference |
|---|---|---|---|
| stationary | 1000 | False | 9.532e-14 |
| translating | 1001 | False | 8.579e-14 |

## 6. Departures from the brief, and clarifications

6.1 **Order.** Group 5 was done before group 4: `case.py` imports `prepare`, so the store had
to go before the plumbing. One commit each, as required.

6.2 **How files left git.** Not `git rm`: each removed file was moved to
`MicroCell/_to_delete/wp3_removed_2026-11-11/roughness/` (gitignored) and the deletion staged.
The folder holds the eight removed files and `pair_before_wp3_group4.py`, a copy of `pair.py`
before its cut.

6.3 **`stats.py` and `SurfacePair`.** The brief keeps "generation, evaluator, runtime, pair,
and the parts of stats they call". At the base the solver read one thing from `pair.py`:
`pattern_displacement` (`runtime.SurfaceMotion`). By `F/inventory_base.tsv`, `SurfacePair` was
read by `report.py` only and the legacy mode table and cross-term helper by nothing; `stats.py`
was read by `SurfacePair`, `prepare.py`, `report.py`, the package re-export and the Sm fit and
`generate_rough_surface` of `generation.py`. With those gone nothing kept calls them, so by
the brief's rule ("dead after the group") they went. In the old path `prepare.resolve` called
`surface_stats` for its report, not for the array (Table 5 is the check).

6.4 **Generator cut to the Gaussian recipe.** `generation.py` keeps `HDS_CORR_LEN`,
`wavenumber_grid`, `normalise_amplitude`, `psd_gaussian_acf`, `hpd_sorted_sample` (Gaussian
branch), `iaaft` with its helpers and its two legacy error metrics. Removed: the self-affine
and exponential PSD models, radial-PSD tables, the four other HPD families with their shape
fit and sample input, the Sm fit, `generate_rough_surface`, `gaussian_filtered_surface`,
`spectral_energy_error`, the revision and provenance constants. `quantiles.py` keeps
`norm_ppf`. Plan D2 keeps generated Gaussian surfaces only, and `Roughness.generated` has no
argument that reaches the removed code.

6.5 **Caller-visible changes off the API's path.**
- `sample_contact` with a model other than `fft_elastic_v1` raises `ValueError("unknown
  contact model …")`; it returned the status `unsupported_model` for the two removed laws.
- `Roughness.generated`'s unmet-target error is one `ValueError` with new text; the parent's
  `TargetNotMetError` and its report are gone. Not reachable in practice: the scaling meets
  `Sa` to round-off. The order of the argument checks is the parent's.
- `microcell.roughness` no longer re-exports `EVALUATOR_REVISION`, `GENERATOR_REVISION`,
  `bilinear_nodal`, `surface_stats`; `hpd_sorted_sample` refuses the four removed families.

6.6 **Names removed beyond the brief's candidate lists**, on the inventory:
`applied_asperity_traction`, the run_micro array-name and batch-status constants of both shear
modules, the batch extrema statuses and level-constraint constants of `pressure.py`,
`SurfaceContactHeight.__getitem__`, the `n_samples` argument, `runtime.load_field`.

6.7 **No permanent test added.** The rewire check is a script in `F/`, not a test in `tests/`,
so the gate's count stays as at the base.

## 7. Candidates for further pruning (left in place)

`F/pruning_candidates.md`, sections A to H: two top-level names still dead (A); unread methods
and branches, including the global-x shear path that is always `no_frame` (B); what could not
be removed (C); stale wording (D); unused imports (E); renames and merges toward the target
layout (F); WP2's list (G); a second surface test for WP5 (H).

## 8. Open items

1. Rob's push and his WP4 go-ahead (§0); departures 6.3 to 6.5 are accepted.
2. Push: `main` is ahead of `origin/main` by the WP3 commits, the commit of this log, and
   c89915a.
3. `docs/notes/provenance.tsv` is unchanged: it records what WP1 copied, including the eight
   files now removed. The README's table marks them.
4. The stale wording of candidates D was left, to keep the diff to removals.

## 9. Changed test expectations

None. `tests/` is byte for byte as at c89915a (Table 2, last row; `git diff c89915a 6c047c1
-- tests examples tools` is empty).

## 10. Commands

Repeat the gate, from `MicroCell/` in Git Bash (about 10 s each):

```
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short
bash docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/gate.sh <tag>
```

The second writes `F/logs/gate_<tag>.log`, a line in `F/gate_counts.tsv` and `F/parity_<tag>.tsv`
(exit 0 PASS, 1 FAIL, 2 INCOMPLETE). The inventory, name check and rewire check (about 2 s):

```
../_claude/wsltest docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/generated_rewire_check.py docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/check_names.py docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/inventory.py -q -p no:cacheprovider
```

The inventory's output tag is the one line of `F/inventory_tag.txt`. Rob ran nothing; the session
ran the two end runs of §4 through `_claude/wslrun`.

## 11. File list

Changed in `microcell/`: `contact.py`, `contact_sample.py`, `contact_shear.py`,
`fluid_shear.py`, `ideal_ht.py`, `pressure.py`, `roughness/__init__.py`,
`roughness/evaluator.py`, `roughness/frontend.py`, `roughness/generation.py`,
`roughness/pair.py`, `roughness/quantiles.py`, `roughness/runtime.py`.
Removed from git (in `_to_delete/wp3_removed_2026-11-11/roughness/`): `assets.py`, `case.py`,
`modes.py`, `prepare.py`, `report.py`, `spec.py`, `stats.py`, `targets.py`.
Docs: this log; `README.md` (provenance note and table; pre-edit copy in
`_to_delete/wp3_scratch/`).
`F/`: `inventory.py`, `inventory_tag.txt`, `inventory_<stage>.tsv` and `_counts.tsv` (six
stages), `inventory_stages.tsv`, `removed_names.tsv`; `check_names.py`,
`unused_imports_<stage>.txt`; `gate.sh`, `parity_summary.py`, `gate_counts.tsv`,
`parity_<tag>.tsv` (six), `parity_by_group.tsv`, `parity_report.tsv`; `lines_before_c89915a.tsv`,
`lines_after_6c047c1.tsv`, `lines_changed.tsv`, `lines_totals.tsv`; `commits.tsv`;
`generated_rewire_check.py`, `generated_before_rewire.npz`, `generated_before_rewire.tsv`,
`generated_rewire_check.tsv`, `generated_c4_seeds.tsv`; `example_run_cell.tsv`;
`pruning_candidates.md`; `logs/`.
Outside the repo: the "Standalone micro-cell solver" row of
`RefactorRefactor/_claude/memory/project_workstreams.md`.

## 12. Second-reader corrections

The `second-reader` agent (fresh context, 11 Nov 2026) recomputed the tables from git and the
source files in `F/`, the prose counts, the §11 file list and the line references: no
disagreement, nothing it could not check, so no correction was made. It read but did not run
the statements this log labels as inferred or untraced (group 5's loader override, the "no
caller found by grep" entries and B7 of `F/pruning_candidates.md`). Its scratch files are in
`_to_delete/wp3_second_reader/`.
