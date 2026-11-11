# Standalone micro-cell solver: WP0 + WP1 implementation log (11 Nov 2026)

Brief: `docs/notes/micro_standalone_wp0_wp1_session_prompt_2026-11-11.md`.
Plan: `docs/notes/micro_standalone_plan_2026-11-11.md` (decisions D1-D13).
Parent: `MultiscaleMixedBallinSocket`, `main` at 2aaccad (read only; `git status` there is clean after the work).
`F/` below is `docs/notes/micro_standalone_wp0_wp1_implementation_log_2026-11-11_files/`.

## 0. Summary

**State.** WP0 and WP1 are built. The parity gate passes on every reference cell. The work is
committed on `main` of `MicroCell` and not pushed (§8). WP2 onward is not started.

**What exists.**
- WP0: eight reference cells made by the parent's own worker code at full double precision (§2).
- WP1: the solver closure copied verbatim (import paths only), and the public call
  `microsolver(zeta, t_end, roughness, settings, n_steps, offset, steady, save_paraview, output_dir)`
  with `Zeta`, `Roughness`, `MicroSettings`, `MicroResult` (§1).
- The parity test `tests/test_parity.py` (pytest and script form, exit 0 / 1 / 2) and one example.

**Gate (measured).** Table 3: all eight cells PASS at relative 1e-12; every one of the
comparisons is exact (largest relative difference zero). Source `F/parity_report.tsv`.

**Rob's decisions, 11 Nov 2026.**
- Repo `MicroCell` created; WSL free.
- Pair reference = Wang GenPair10 at a non-zero offset, parity on the coupled `P_asp` / `k_asp`.
- Cell 6 = cosine + fft contact + sqrt, built in memory.
- `roughness_velocity_ratio` default `None`.
- `MicroSettings` defaults at WP1 = the GenSa40-ripple values; revisit at WP4.
- Steady solves are needed: a flag on the function says steady or transient (added as `steady=`).
- Go-ahead for the first commits ("Yes, commit", after reading the summary of this log).

**Departures from the plan.** Twelve, numbered in §5. The ones that need a view from Rob:
- 5.7 `MicroSettings.legacy`, a pass-through for copied settings the plan does not list (goes at WP4);
- 5.8 `save_paraview` writes the end state only, not one set per step;
- 5.9 `n_steps=None` (march from 0) is not built.

**Open items.** §6. Main ones: environment reads remain in the copied solver (6.1; against D10
until WP4); shear statistics in the global-x frame are not available because `shear_frame` is not
an input; rates are exercised by two cells only (c4, c5).

**Waits on Rob.**
1. A yes or no on departures 5.7-5.9.
2. Go-ahead for WP2 (prune dead physics) as its own session.
3. Pushing `main` of `MicroCell`, when he wants it on a remote.

## 1. What was built

| Part | Files | Lines | Note |
|---|---|---|---|
| Copied solver closure | 30 files under `microcell/` | 12 698 | parent 2aaccad, import paths only; per-file parent path, blob and line count in `docs/notes/provenance.tsv` |
| Roughness front end | `microcell/roughness/frontend.py` | 181 | `Roughness.array / generated / pair / cosine`; in-memory `ArrayField` in place of the surface store |
| API | `microcell/api.py` | 649 | `microsolver`, `Zeta` (+ `Zeta.from_task`), `MicroSettings`, `ContactSettings`, `RippleSettings`, `MicroResult`, status codes |
| Fields and ParaView | `microcell/fields.py` | 196 | parent's `_extract_fields`, `_write_vts`, `_write_xdmf`; `to_si` is new |
| Package | `microcell/__init__.py` | 12 | exports |
| Reference maker | `tools/make_references.py`, `tools/show_references.py` | 521, 25 | runs in the parent tree; writes only under `MicroCell/tests/references/` |
| Parity test | `tests/test_parity.py` | 357 | rule in its docstring |
| Example | `examples/run_cell.py` | 46 | transient with `save_paraview=True`, then steady |
| Environment | `scripts/ehl_env.sh`, `environment.yml`, `pyproject.toml`, `.gitattributes`, `.gitignore` | 67, 19, 13, 10, 8 | `ehl_env.sh` is the parent's file |
| Readme | `README.md` | 110 | the call, running, layout, provenance |

How the call works:
- `api._ensure_process()` pins quadrature degree 2 and applies the parent's process settings
  before the solver modules are imported, so no form is compiled before the pin.
- Each call builds a new `UnitSquareMesh(mesh_n, mesh_n)` and a new solver. Nothing is cached.
- Transient: `run(t_end, n_steps, cfl_target)`. `zeta` is the state at t = 0 of the window and the
  outputs are the end state, `zeta + rate * t_end` (the parent's convention).
- Steady: `run_steady()`; the rates of `zeta` and `t_end` are not used.
- The result is assembled as the parent worker does it, with the one-phase reductions.
- Failure: a `RuntimeError` from the run maps to `FAILED_NEWTON`, `FAILED_RIPPLE` or
  `FAILED_CONTACT`; every homogenised output is NaN and `fields` is empty.
- No file is written unless `save_paraview=True`.

Status codes and NaN: a failed solve has a non-OK `status`. In an OK full-film cell the asperity
and total shear means are NaN with the sub-status `unavailable` in `info` (see 6.4).

## 2. Reference cells (WP0)

Made by `tools/make_references.py` in one run. Source of every value: `F/reference_table.tsv`
(written from `tests/references/*.npz`); run log `F/logs/make_references_run1.log`.
All cells: P = 40.8 MPa, Ux = 0.02 m/s, status OK. Transient cells: t_end = 0.02 s, n_steps = 3.

**Table 2. Reference cells.**

| Cell | Case, variant | Mode | Mesh | H (m) | Rates | delta_P (Pa) | Pasp_hom (Pa) | contact_frac | Newton / contact / ripple |
|---|---|---|---|---|---|---|---|---|---|
| c1_gensa40_h200_ripple | genrough ripple, `GenSa40-ripple` | transient | 100 | 2e-07 | no | 4.112720489501953e-06 | 0.0 | 0.0 | 5 / 0 / 1 |
| c2_gensa40_h130_ripple | same | transient | 100 | 1.3e-07 | no | -74157.8045212552 | 60260395.79441124 | 0.0018166666666666886 | 6 / 49 / 2 |
| c3_gensa40_h130_noripple | same, ripple off | transient | 100 | 1.3e-07 | no | -74157.8045212552 | 60184626.80087217 | 0.0018166666666666886 | 5 / 33 / 0 |
| c4_wangpair10_h25_offset | Wang genpair ripple, `Wang2008-GenPair10-ripple-table` | transient | 100 | 2.5e-08 | yes | -809749.9999958277 | 63714053.49474145 | 0.019749999999999823 | 24 / 736 / 4 |
| c5_cosine_fullfilm | refactor baseline, `ms` | transient | 60 | 1.2e-07 | yes | -4.366040229797363e-06 | 0.0 | 0.0 | 8 / 0 / 0 |
| c6_cosine_fft_sqrt | mixed cup90 `base`, changed in memory | transient | 100 | 7.73e-08 | no | -916640.0330708921 | 26471856.20692746 | 0.022466666666666524 | 5 / 47 / 0 |
| c7_gensa40_h130_ripple_steady | genrough ripple, `GenSa40-ripple` | steady | 100 | 1.3e-07 | – | -74157.80283036828 | 60260395.79438652 | 0.0018166666666666886 | 3 / 42 / 2 |
| c8_cosine_fullfilm_steady | refactor baseline, `ms` | steady | 60 | 1.2e-07 | – | -4.403293132781982e-06 | 0.0 | 0.0 | 2 / 0 / 0 |

Notes to Table 2:
- Case files: `cases/mixed_transfull_cup90_genrough_ripple.py`, `cases/wang2008_mixed_genpair_ripple.py`,
  `cases/refactor_baseline.py`, `cases/mixed_transfull_cup90.py`.
- "Rates = yes" is the task with Uy 0.005, dpdx 2e9, dpdy -1e9, Hdot -1e-7, Pdot 1e7, Uxdot 0.05,
  Uydot 0.01, dpdx_dot 1e9, dpdy_dot -5e8 (the `task` array of the npz). c8 carries the same Uy and
  gradients and no rates. The other cells have Ux only.
- c4 runs at the offset (1.25e-06, 6.25e-07) m. Its `Pasp_hom` in the table is the parent's
  table-composed value; the solver's own coupled value is in the npz key `coupled_json` (see 5.2).
- c6 is the `base` case (`MixedTransFull-Cup90-hmin4.0e-08`) with three settings changed in memory:
  `contact_model` to fft, its `contact_E_star`, and `spos_form` to sqrt (`meta_json["changes"]` of
  the npz). Its h_min and `k_asp_mode` off are the case's own. No case file was written.

## 3. Parity gate (WP1)

Rule (D6): scalars and vector components `|a - b| <= 1e-12 * max(|a|, |b|)`, the NaN pattern
identical, integers and strings exact; fields the same shape and NaN pattern with
`max|a - b| <= 1e-12 * max|ref|`. The test also compares the copied settings dataclasses
field by field and the field diagnostics (5.11).

Source: `F/parity_report.tsv` (per cell) and `F/parity_report_detail.tsv` (one row per comparison),
written by `python tests/test_parity.py --out F/parity_report.tsv`; log `F/logs/parity_script_run3.log`.

**Table 3. Parity, API against the references (measured).**

| Cell | Settings | Scalars | Diagnostics | Fields | Total | Exact | Max rel. diff. | Status |
|---|---|---|---|---|---|---|---|---|
| c1_gensa40_h200_ripple | 44 | 82 | 12 | 16 | 154 | 154 | 0.0 | PASS |
| c2_gensa40_h130_ripple | 44 | 82 | 12 | 16 | 154 | 154 | 0.0 | PASS |
| c3_gensa40_h130_noripple | 44 | 82 | 12 | 15 | 153 | 153 | 0.0 | PASS |
| c4_wangpair10_h25_offset | 44 | 81 | 12 | 16 | 153 | 153 | 0.0 | PASS |
| c5_cosine_fullfilm | 29 | 82 | 10 | 9 | 130 | 130 | 0.0 | PASS |
| c6_cosine_fft_sqrt | 43 | 82 | 12 | 15 | 152 | 152 | 0.0 | PASS |
| c7_gensa40_h130_ripple_steady | 44 | 82 | 12 | 16 | 154 | 154 | 0.0 | PASS |
| c8_cosine_fullfilm_steady | 29 | 82 | 10 | 9 | 130 | 130 | 0.0 | PASS |
| All | | | | | 1180 | 1180 | 0.0 | PASS, exit 0 |

Notes to Table 3:
- The settings count includes `surface.envelope_m` for cells with a surface; the scalar count
  includes the `status` record.
- Not compared, because they are not solver outputs or do not exist in a one-call cell:
  `wall_time, contact_eval_s, pid, task_id, task, h_geom_min, escalated, pasp_table, pasp_coupled,
  pasp_contact_only, kasp_table, amech_table, offset_x_m, offset_y_m, phase_average`.
- c4: `Pasp_hom` and `k_asp_hom` are compared against `coupled_json`; `contact_iters` is not
  compared (5.2). That is why c4 has one scalar fewer.
- All eight cells run in one process in the gate, so a pass also shows that no state leaks from
  one call to the next.

## 4. Other checks

**Table 4. Tests and smoke checks (measured).**

| Check | Result | Source |
|---|---|---|
| `pytest tests/` (8 parity cells, references present, generated surface reproduces c1's stored surface, negative control) | 11 passed in 8.25 s | `F/logs/parity_pytest_run4.log` |
| Reference run | 8 cells in 9 s: 8 OK, 0 failed, 0 incomplete | `F/logs/make_references_run1.log` |
| Example, transient | OK; dP -7.415780e+04 Pa, P_asp 6.026040e+07 Pa, contact_frac 0.001817, Newton 6 | `F/logs/example_run1.log` |
| Example, steady | OK; same dP, P_asp and contact_frac to the printed digits, Newton 3 | same |
| Example ParaView files | `cell.vts` and `cell.xdmf` written to `examples/output/gensa40_h130/` (gitignored); `cell.h5`, the XDMF's data file, is beside them | same for the first two; the folder itself for `cell.h5` |
| `Roughness.generated(Sa=10e-9/sqrt(2), seed 1000 / 1001)` against c4's two stored members | relative difference 1.2721415110886553e-17 each | `F/logs/smoke_run1.log` (`smoke.py`) |
| Moving cosine, full film, transient and steady | both OK | same |
| Unreachable tolerances (`max_iterations=3`, both tolerances 0), transient and steady | `FAILED_NEWTON`, all homogenised outputs NaN, no fields, extrema status `failed` | `F/logs/smoke_run1.log` (`smoke_fail.py`) |
| `max_iterations=1` alone on c2's inputs | still OK, so it does not exercise the failure path (cause not traced) | same, first line |

The negative control (`test_comparator_sees_a_perturbed_solve`) changes c2's H by a relative 1e-9
and requires the comparison to fail. It is one of the passing tests.

The smoke checks are not part of the gate. The example and the smoke checks ran before the last
edit to `api.py` and `README.md` (two docstring lines and one README line on the time convention);
the gate and pytest rows are from after it.

## 5. Departures from the plan and clarifications

1. **`process_settings.py` joins the copied closure.** The parent applies PETSc options and
   `cpp_optimize` per process; the copy needs the same to reproduce the numbers.
2. **Pair reference.** c4 is Wang GenPair10 with the contact table, as Rob agreed. The standalone
   call composes no table. Parity is on the solver's coupled `P_asp` / `k_asp` (`coupled_json`).
   `contact_iters` is not compared for c4: the parent's count includes the table route's
   contact-only solve. For a stationary + translating pair the API sets the `contact_table_v1`
   tag on the copied settings so the solver takes its pair branch; no table is read.
3. **Own surface store for the references.** `make_references.py` generated the surfaces into
   `RefactorRefactor/_to_delete/micro_standalone_2026-11-11/surface_store`. Nothing was written in
   the parent.
4. **Eight cells, not six.** Two steady cells (c7, c8) were added after Rob asked for steady solves.
5. **Film heights.** c5, c6 and c8 run at the heights in Table 2, not the heights shown to Rob
   before the reference run.
6. **`steady` flag.** Added to the call at Rob's request.
7. **`MicroSettings.legacy`.** A dict that passes through the copied settings the plan's API does
   not list (`beta, delta_h, eps_solid, eps_smooth, k_spring, contact_E, contact_model,
   use_reduced_modulus, overclosure_measure, patch_pin, patch_pin_kappa`). The reference cases need
   them for parity. The plan allows the WP1 API to wrap the copied dataclasses; the dict goes when
   WP2 / WP4 remove those settings. `ContactSettings` and `RippleSettings` are sub-dataclasses of
   `MicroSettings`.
8. **`save_paraview` writes the end state only.** The plan says one file set per solved step plus
   a PVD. The parent's writers were ported as they are (end state).
9. **`n_steps=None` is not built.** It raises `NotImplementedError`. The plan's D3 "march from 0"
   has no reference cell at WP1.
10. **`api.py` is longer than planned** (Table 1 against the plan's "~300 lines"): the result
    assembly and the failure path are written out in full to match the worker.
11. **Parity is stricter than D6.** It also compares the field diagnostics and the settings.
12. **`MicroResult.raw` and `fields["chi_fluid_dg0"]`** are additions for the parity test: `raw`
    holds the result under the parent's `MicroTaskResult` field names.

## 6. Open items

1. **Environment reads left in the copied solver** (D10 says none; removal is WP2 / WP4):
   `MICRO_DEBUG_LOG`, `MICRO_NONDIM_DIAG`, `MICRO_UREF_FLOOR_OMEGA`, `MICRO_EPSP_ND_CAP` and
   `MICRO_SPOS_FORM` (it overrides `spos_form`) in `microcell/micro_phase_neutral_class.py`; `EHL_SURFACE_STORE` in
   `microcell/roughness/case.py`. None is set by the wrappers.
2. **Three dead lazy imports** of parent modules in functions the API never calls:
   `micro_phase_neutral_class.py:3500`, `micro_phase_neutral_class_mixed.py:830` and `:857`.
3. **`shear_frame` is not an input.** The API sets it to `None`, so the global-x shear statistics
   have status `no_frame`.
4. **NaN paths.** In full-film cells the asperity and total shear means are NaN with the
   sub-status `unavailable`; sub-statuses live in `info`, not in `status`. A status code for every
   NaN path is WP4.
5. **`solver=` handle** (D10, optional) is not built.
6. **Defaults.** `MicroSettings()` is the GenSa40-ripple case (`level_constraint="whole_cell"`),
   not the current run default `fluid_mean_v1`. Rob agreed to revisit at WP4.
7. **Coverage of rates.** Only c4 and c5 carry non-zero rates, transverse velocity and pressure
   gradients (Table 2). No mixed single-surface cell has them.
8. **`tests/references/logs/`** holds the eight per-cell run logs. They are committed
   with the references (§8).
9. **Not run:** nothing longer than the checks in Table 4; no run outside WSL; no HPC.

## 7. Commands

Run by the session (WSL, through `_claude/wslrun` and `_claude/wsltest`, from `MicroCell/`):

| Command | Time | Result |
|---|---|---|
| `python tools/make_references.py` (in the parent tree's environment, output to `MicroCell/tests/references/`) | 9 s | Table 4 |
| `python tests/test_parity.py --out F/parity_report.tsv` | seconds (not timed as a whole; per-cell wall times are in `F/parity_report.tsv`) | Table 3 |
| `python -m pytest tests/ -q -p no:cacheprovider -rfE --tb=short` | Table 4 | Table 4 |
| `python examples/run_cell.py` | seconds (not timed as a whole; solve wall times are in `F/logs/example_run1.log`) | Table 4 |

For Rob, to repeat the gate and the example:

```
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
source scripts/ehl_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tests/test_parity.py; echo "exit $?"
python examples/run_cell.py; echo "exit $?"
```

Expected: `parity: PASS -- 8 of 8 cells PASS, 0 FAIL, 0 not run`, exit 0; the example prints two
OK rows and exits 0. Each takes seconds once the forms are compiled (the first run in a new
dijitso cache takes minutes).

Gate verdict: **PASS** (Table 3). Rob has not run it himself.

## 8. Commits

Made on 11 Nov 2026 after Rob's go-ahead, on `main` of `MicroCell`, author
`RFurze <py07rf@leeds.ac.uk>`, not pushed (source: `git log`).

| # | Commit | Message |
|---|---|---|
| 1 | 626f695 | `MicroCell: repo scaffold (env, gitattributes, pyproject, README, plan and session prompt)` |
| 2 | 5c66a57 | `MicroCell WP0: reference maker and eight reference cells from parent 2aaccad` |
| 3 | 92e7327 | `MicroCell WP1: solver closure copied from parent 2aaccad, import paths only` |
| 4 | 2e47d0d | `MicroCell WP1: microsolver API with steady flag, fields, roughness front end, example` |
| 5 | the commit that holds this log | `MicroCell WP1: parity test and implementation log` |

Each message ends `; log docs/notes/micro_standalone_wp0_wp1_implementation_log_2026-11-11.md`.
Commit 5's hash is not written here because it contains this file; it is the commit after 2e47d0d.

## 9. File list

New in `MicroCell/` (all in the commits of §8):
- `.gitattributes`, `.gitignore`, `README.md`, `environment.yml`, `pyproject.toml`
- `scripts/ehl_env.sh`
- `microcell/` : the copied files (list in `docs/notes/provenance.tsv`), `__init__.py`,
  `api.py`, `fields.py`, `roughness/frontend.py`
- `tools/make_references.py`, `tools/show_references.py`
- `tests/test_parity.py`, `tests/references/c1 ... c8 *.npz`, `tests/references/logs/*.log`
- `examples/run_cell.py`
- `docs/notes/`: the plan, the session prompt, `provenance.tsv`, this log, and `F/`
  (`parity_report.tsv`, `parity_report_detail.tsv`, `reference_table.tsv`, `logs/`)

Outside the repo, in `RefactorRefactor/_to_delete/micro_standalone_2026-11-11/`: the surface store
used for the references, `wp0_logs/`, `wp1_logs/` (all runs, including the earlier ones).

Changed elsewhere: the "Standalone micro-cell solver" row of
`_claude/memory/project_workstreams.md`. Nothing in the parent clone or any worktree.

## 10. Second-reader corrections

The `second-reader` agent recomputed the tables from the source files on 11 Nov 2026. It found no
disagreeing number in Tables 1-3 or in the provenance totals. Corrections made after its report:

1. Table 2 note on c6: the note said h_min and k_asp off were set in memory. Only `contact_model`,
   `contact_E_star` and `spos_form` were; the note now says so.
2. Table 4, ParaView row: the example log names `cell.vts` and `cell.xdmf` only; `cell.h5` is now
   sourced to the output folder.
3. Table 4, `max_iterations=1` row: the explanation "one Newton iteration per solve" was inference
   and is removed.
4. §7: the "about 10 s" and "about 5 s" totals were not measured as totals; replaced by pointers
   to the kept wall times.
5. Repeated numbers removed from prose in §4, §5.5 and §9.

Not checked by the second reader (it did not have the access or the source):
- that `git status` in the parent is clean (checked by this session: `git -C ../MultiscaleMixedBallinSocket status --short` printed nothing, HEAD 2aaccad);
- that the three lazy imports of 6.2 sit in functions the API never calls (stated from reading the code, not tested);
- that the wrappers set none of the variables of 6.1 (checked by this session: no `MICRO_` or `EHL_SURFACE_STORE` in `_claude/wslrun`, `_claude/wsltest` or `scripts/ehl_env.sh`);
- that nothing was written in the parent (5.3) beyond the clean `git status`;
- the two notes on what the settings and scalar counts include (Table 3 notes);
- Rob's decisions in §0, which come from the session, not from a file.
