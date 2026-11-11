# Standalone micro-cell solver: open items, implementation log (11 Nov 2026)

Files folder: `micro_standalone_open_items_implementation_log_2026-11-11_files/` (called `F/` below).

## 0. Summary

**State.** The open items of the five WP logs were put to Rob as 16 numbered items (§1). He
accepted the recommendations ("Yes, as recommended", 11 Nov 2026): act on 1 to 5, 8, 12 and 13,
and add a getting-started guide to the README. All of that is done and committed on `main`,
not pushed. The gate passes; the parity gate passes on nine of nine reference cells.

**Decisions (Rob, 11 Nov 2026).**
- Items 1-5, 8, 12, 13 and the README guide: do. Items 6, 7, 11, 14, 15, 16: leave. Items 9, 10: his.
- Item 1: probe the translating generated surface at the roughness and films of the Wang and
  pendulum cases; Sa 40 nm may be an unphysical input.

**Numbers that matter** (each stated once, in the table it is cited from):

| What | Where |
|---|---|
| Commits | §2 |
| Gate counts | §3 |
| Probe of generated surfaces in contact (item 1) | §4 |
| The added reference cell c9 and its parity (item 8) | §5 |
| Lines of `microcell/` | §6 |

**Item 1 result.** Measured: every solve at Wang and pendulum roughness is `OK` at both Newton
tolerances; the only failure is the Sa 40 nm translating cell at `abs_tolerance` 1e-6, which
solves at 1e-5 (§4). Inferred, not measured: the failure is the stall just above tolerance that
the parent's Sa 8 case file records (21 Oct 2026) and that made 1e-5 the production value.
No solver change was made. The parent was not run on the Sa 40 cell (departure 7.2).

**Open, for Rob.**
1. Default `abs_tolerance`: stays 1e-6 (the parent's preset). The README now names 1e-5 as the
   production value. Decided (Rob, 11 Nov 2026): keep 1e-6.
2. Items 9 and 10 (run the full `n_steps=None` march once; open the ParaView files).
3. No LICENSE file; matters only if the repo is shared.
4. Push the five commits of §2 and this log's commit.

## 1. The items

| # | Item | Outcome |
|---|---|---|
| 1 | A translating generated surface in contact fails (Sa 40 nm, H 130 nm) | Probed at Wang and pendulum roughness: no failure there (§4); README note |
| 2 | A failed result reports `newton_iters = 0` | Fixed: the failed solve's iterations are counted before the raise |
| 3 | `info["n_steps"]` reports the count asked for, not the count marched | Fixed: reads the solver's `n_steps_run` |
| 4 | Importing `solver_base.py` disables logging for the caller's process | Fixed: the `logging.disable` call is gone |
| 5 | PETSc prints "unused database options" at exit | Fixed: the five macro KSP options are no longer set |
| 6 | A non-finite `contact.mu` passes `check()` | Left |
| 7 | No test reaches `phase_invalid` | Left |
| 8 | No mixed single-surface reference with non-zero rates and gradients | Added `c9_gensa10_h26_rates` (§5) |
| 9 | The full `n_steps=None` march was never run | Rob's |
| 10 | ParaView files written but never opened | Rob's |
| 11 | Gate run on this WSL only | Left |
| 12 | Comments and docstrings name the old files | Fixed, text only (departure 7.3) |
| 13 | Names nothing reads | Removed, except `_reset_timers` (departure 7.1) |
| 14 | Pruning candidates, shear extras, `global_x` definition strings | Left |
| 15 | `docs/notes/provenance.tsv` describes the WP1 copy | Left |
| 16 | The `solver=` reuse handle is not built | Left |

What each fix changed:

- **Item 2.** `solver_base.py`, Newton failure branch: `_n_newton_iters_total` and
  `_n_newton_backtracks_total` are incremented before the `RuntimeError`.
- **Item 3.** `api.py`, `n_steps_reported()`: 0 for a steady solve, else the solver's `n_steps_run`.
  README bullet reworded.
- **Item 4.** `solver_base.py`: the module keeps its own logger and no longer calls
  `logging.disable(logging.CRITICAL)`.
- **Item 5.** `process_settings.py`: `MACRO_PETSC_OPTIONS` and `apply_petsc_options()` removed;
  `apply_macro_process_settings()` applies the form-compiler flags only. The cell's solves are
  direct (MUMPS), so the KSP options were never read.
- **Item 12.** 15 files under `microcell/`; references to `micro_phase_neutral_class*.py`,
  `micro_diagnostics.py`, `contact_sample.py`, `ideal_ht.py`, `surface_ht.py`, `ehl/…` now name
  `solver_base.py`, `solver_contact.py`, `samplers.py`, `film.py`, `shear.py`, `contact.py`,
  `pressure.py`, `ripple.py`. Stale parent line numbers were dropped with the names.
- **Item 13.** `SHEAR_REDUCE_VERSION` and the timing accumulators (`_t_newton_solve`,
  `_t_history_update`, `_t_diagnostics`, `_t_step_overhead`, `_t_init`, `_t_asm_total`,
  `_t_lin_total`, `_t_upd_total`) with their `perf_counter` calls; `_assemble_newton_rhs` returns
  nothing and `_schur_step` returns `(d_lambda, d_mu)`; `import time` left `solver_base.py` and
  `solver_mixed.py`.
- **README.** "Getting started" (clone, create and start the environment, check, launch a case,
  own case); the "Running" block is relative to the repo root; the Newton row of the settings
  table names the production `abs_tolerance`.

## 2. Commits

All on `main` in `MicroCell`, author `RFurze <py07rf@leeds.ac.uk>`, on top of 4994bbf (pushed).
None is pushed.

| Commit | Content |
|---|---|
| f6ee886 | README getting-started guide |
| e696ebc | Items 2-5; `tests/test_march.py` extended |
| 4a83a4b | Items 12-13 |
| ef34679 | Item 8: `c9_gensa10_h26_rates` in `tools/make_references.py`, the reference, its logs |
| ebf11dc | Item 1: the probe and its table; README Newton row |

This log is one further commit.

## 3. Gate

Command, from `MicroCell/` (under a minute; minutes after a form change):

```
/c/Users/Robin/Documents/code/RefactorRefactor/_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short
```

| After | Result | Source |
|---|---|---|
| Items 2-5 | 110 passed in 27.02 s | `F/logs/gate_items2to5.log` |
| Items 12-13 | 110 passed in 27.01 s | `F/logs/gate_items12_13.log` |
| Item 8 (nine references) | 112 passed in 27.89 s | `F/logs/gate_item8_c9.log` |

No log has a FAILED or ERROR line. `F/logs/gate_items2to5.log` has no "unused database" line (item 5).

Changed test expectations:

- `tests/test_march.py`: `test_march_equals_the_integer_window_that_does_not_fit` also asserts
  `info["n_steps"] == ceil(t_end / dt_cfl)`; `test_march_differs_from_the_three_step_window` also
  asserts `info["n_steps"] == 3` (item 3).
- New `test_a_failed_newton_solve_reports_its_iterations` (item 2): tolerances of 1e-300 with
  `max_iterations=3` give `FAILED_NEWTON`, `0 < newton_iters <= 3`, NaN `dP`, no fields.
- The reference list is globbed, so c9 adds two parametrised tests (item 8).

## 4. Item 1: generated surfaces in contact at Wang and pendulum roughness

Probe `F/probe_translating_realistic.py`, run in pytest form through `wsltest`
(`F/logs/probe_translating_realistic.log`: 1 passed in 71.70 s); table
`F/probe_translating_realistic.tsv`, one row per solve. Every solve: `n_steps=3`, `t_end` 0.02 s,
`Ux` 0.02 m/s, `P` 40.8 MPa, zero rates and gradients, each cell at `abs_tolerance` 1e-5 and 1e-6.
A pair is seeds 1000 (stationary) + 1001 (translating), each at Sa/√2.

| Family (settings) | Cells | Solves | OK | Failed |
|---|---|---|---|---|
| pend (h_min 1.0331e-11 m, default E*, `pattern_flux_contact` False, ripple tol 1e-4) | Sa 8 stationary and translating at H 22 and 26 nm; Sa 10 stationary and translating at H 26 nm; pair Sa 8 at H 22 nm; pair Sa 10 at H 25 nm | 16 | 16 | 0 |
| wang (h_min 1.7825e-11 m, E* 115.38 GPa, mu 0.4, yield 450 MPa, ripple tol 1e-4) | pair Sa 10 at H 25 nm; pair Sa 12 at H 25 and 30 nm; Sa 8 stationary and translating at H 22 nm | 10 | 10 | 0 |
| sa40 (MicroSettings defaults) | Sa 40 stationary and translating at H 130 nm | 4 | 3 | 1 |

Each of the 29 `OK` solves is in contact (`contact_frac` from 0.001817 to
0.021083 in the table).

The Sa 40 nm translating cell at H 130 nm:

| `abs_tolerance` | Status | Newton iterations | Final field residual | Wall |
|---|---|---|---|---|
| 1e-5 | OK | 487 | not reported | 24.15 s |
| 1e-6 | FAILED_NEWTON | 294 | 1.0654e-06 | 14.33 s |

The same surface stationary is `OK` at both tolerances in 6 Newton iterations.

**Reading.**
- Measured: at the roughness and films of the Wang and pendulum cases a translating generated
  surface in contact solves, single or in a pair, at either tolerance. The failure of the WP5
  probe needs both Sa 40 nm and the 1e-6 tolerance.
- Measured: the failing solve ends with a field residual 6.5 % above its tolerance.
- Inferred: this is the stall the parent records in
  `cases/mixed_transfull_cup90_genrough_ripple.py` (comment of 21 Oct 2026, lines 275-283: the
  Sa 8 runs stalled at a residual of about 1e-6, "not blow-ups"), for which the Sa 8 and Sa 10
  recipes set `abs_tolerance=1e-5`. Not checked in the parent for this cell (departure 7.2).
- One cause: a large-amplitude translating surface at the default tolerance. One fix, if the
  case is ever wanted: `MicroSettings(abs_tolerance=1e-5)`, the production value. No code change.

## 5. Item 8: reference cell c9

`c9_gensa10_h26_rates`: case `cases/mixed_transfull_cup90_genrough_ripple.py`, variant
`GenSa10-ripple-tdfc`, one stationary generated surface (Sa 10 nm), H 26 nm, P 40.8 MPa, the
`RATES` set of `tools/make_references.py` (the one c4 and c5 carry). Made from the parent at
2aaccad (`git status` empty before and after the run; checked in the session, not kept in a file) with

```
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MultiscaleMixedBallinSocket
python ../MicroCell/tools/make_references.py --all --cell c9_gensa10_h26_rates --procs 1
```

run through `wslrun` (log `F/logs/make_reference_c9.log`). The Sa 10 surface was prepared into
the store under `RefactorRefactor/_to_delete/micro_standalone_2026-11-11/surface_store`, not in
the parent.

| Quantity | Value | Source |
|---|---|---|
| Status in the parent | OK | `tests/references/logs/c9_gensa10_h26_rates.log` |
| dP | -507121.6353639662 Pa | same |
| P_asp | 82689182.98564312 Pa | same |
| Contact fraction | 0.0139 | same |
| Parity comparisons on c9 | 144, 0 outside 1e-12 | `F/logs/parity_report_c9.log` |
| Largest relative difference on c9 (settings, scalars, diagnostics, fields) | 0.0 | `F/parity_report_c9.tsv` |
| Cells passing parity | 9 of 9 | `F/logs/parity_report_c9.log` |

The prediction was written before the reference was made (`F/c9_prediction.txt`): parent status
OK with contact, parity on c9 with no change to `microcell/` or the test, the eight existing
references untouched, no failing test. All four held. The eight existing references have the
sha256 of `F/references_sha256_before_c9.txt` (checked with `sha256sum -c` after the run).

## 6. Lines of `microcell/`

`wc -l` over the `.py` files under `microcell/` at each commit (`F/microcell_lines_by_commit.tsv`):

| Commit | Files | Lines |
|---|---|---|
| 4994bbf | 21 | 8298 |
| e696ebc | 21 | 8285 |
| 4a83a4b | 21 | 8236 |

## 7. Departures

1. **Item 13, `_reset_timers` kept.** The item listed it as unread. It zeroes the Newton
   counters the API reads and sets up the Newton trace index, and `tests/test_fluid_mean_fenics.py`
   calls it. Only its timing lines were removed. Five timing accumulators beyond the `_t_*_total`
   ones named in the item went with them, since nothing reads them either.
2. **Item 1, the parent was not run on the failing cell.** The recommendation was to run the Sa 40
   translating cell in the parent first. After Rob's note the probe went to realistic cells, where
   nothing fails; a translating-only single generated surface is not an arrangement of any parent
   case (the generated-roughness cases are stationary-only or a pair), so the parent run would
   have needed an in-memory change of the case. Whether the parent stalls on that cell is unknown.
3. **Item 12, provenance headers left.** The headers of the merged files (`film.py`,
   `contact.py`, `samplers.py`, `shear.py`) still name the parent files each part came from; that
   is their purpose. References to the parent's own documents and tests are also left.
4. **The probe ran through `wsltest`** as a pytest module (one core, 72 s), not through `wslrun`.
5. **The first `make_references.py --cell` call failed** (`SurfaceNotPreparedError`: the Sa 10
   surface was not in the store). `--all --cell` prepares the store first; the log holds both calls.

## 8. Not checked

- The getting-started steps on a clean machine (`conda env create -f environment.yml` has never
  been run); the 7-line own-case script in the README was not run.
- The parent on the Sa 40 translating cell (departure 7.2).
- Translating cells with non-zero rates: the probe has zero rates; c4 (pair) carries them.

## 9. Files

- Changed: `README.md`, `microcell/` (15 files: `api.py`, `contact.py`, `film.py`,
  `form_compiler.py`, `pressure.py`, `process_settings.py`, `ripple.py`, `samplers.py`, `shear.py`,
  `solver_base.py`, `solver_contact.py`, `solver_mixed.py`, `roughness/__init__.py`,
  `roughness/pair.py`, `roughness/runtime.py`), `tests/test_march.py`, `tools/make_references.py`.
- Added: `tests/references/c9_gensa10_h26_rates.npz`, `tests/references/logs/c9_gensa10_h26_rates.log`,
  this log and `F/`.
- Outside the repo: the prepared Sa 10 surface in the `_to_delete` store (§5); the workstreams row.

## 10. Second-reader corrections

The `second-reader` agent recomputed about 75 numbers and claims from the source files: no mismatch.
Two wordings changed after its notes: §4 said "every one of the 30 solves is in contact where it is OK"
(29 are `OK`; the failed one has no contact fraction), and §5 said the parent was "clean before and
after" without saying that no file records it. It could not check the negative claims of §8.
