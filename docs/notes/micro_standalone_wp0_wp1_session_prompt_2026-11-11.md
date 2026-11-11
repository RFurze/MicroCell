# Standalone micro-cell solver: session prompt for WP0 and WP1 (11 Nov 2026)

Staged here because the new repo did not exist when this was written. File both this prompt and the
plan beside it into the new repo's `docs/notes/` with the first commit.

## Read first

- The plan, whole (about 10 KB): `micro_standalone_plan_2026-11-11.md` in this folder.
- In the parent, by section and not whole:
  - `scripts/run/run_micro_cells.py` (640 lines): the docstring, `_extract_fields` (143),
    `_write_vts` (242), `_write_xdmf` (267), `_result_row` (307), `run_cell` (332).
  - `scripts/run/run_microscale_phase_neutral.py`: `_build_phase_neutral_settings` (653),
    `MicroTaskResult` (836), `run_micro_task_phase_neutral` (989 to about 1950), `_init_worker` (303).
  - `scripts/run/micro_task_helpers.py`: `_parse_task` (321), `q_re` (408).

## Scope

WP0 (reference cells from the parent) and WP1 (verbatim copy + thin `microsolver()` API + parity
gate). Stop after WP1. WP2 to WP5 (pruning, settings collapse, tests, README) are later sessions.

## Decided (don't reopen)

D1 to D13 of the plan. Rob answered D1 to D4 on 11 Nov 2026 and approved the plan with D5 to D13 as
recommended.

## State as found: re-check before acting

| Item | Expected |
|---|---|
| Parent `RefactorRefactor/MultiscaleMixedBallinSocket` | on `main` at 2aaccad; read-only for this session |
| New repo `RefactorRefactor/MicroCell` | an empty git repo on `main`, `user.name` RFurze, `user.email` py07rf@leeds.ac.uk, the parent's `commit-msg` hook copied in |
| Workstreams row | "Standalone micro-cell solver" in `_claude/memory/project_workstreams.md` |

If `MicroCell` does not exist, or Rob gave another name, give him the setup commands below (with
his name) and wait. Rob creates the repo; you don't.

```bash
git init -b main /c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
git -C /c/Users/Robin/Documents/code/RefactorRefactor/MicroCell config user.name "RFurze"
git -C /c/Users/Robin/Documents/code/RefactorRefactor/MicroCell config user.email "py07rf@leeds.ac.uk"
cp /c/Users/Robin/Documents/code/RefactorRefactor/MultiscaleMixedBallinSocket/.git/hooks/commit-msg /c/Users/Robin/Documents/code/RefactorRefactor/MicroCell/.git/hooks/commit-msg
```

## Where you work

- Your worktree is `RefactorRefactor/MicroCell`. Everything else is a read-only reference.
- The `wsltest` / `wslrun` wrappers source the current folder's `scripts/ehl_env.sh`, so the new
  repo needs its own copy (the parent's file works unchanged: it finds its root from its own path).
- WP0 imports the parent's code. Run it from the parent's root (the parent resolves `config/` from
  the working directory), but write every output under `MicroCell/` or
  `RefactorRefactor/_to_delete/micro_standalone_2026-11-11/`, never into the parent
  (`run_micro_cells.py` takes `--out`).
- A `.gitattributes` in the new repo: `*.py`, `*.sh`, `*.yml` as `text eol=lf`; reference data
  (`*.npz`, `*.vts`, `*.h5`) as binary.

## WP0: references

- About 6 cells, per the plan: a Cup90 genrough ladder with and without contact, ripple on and
  off, one pair cell at a fixed offset, one cosine cell, one full-film cell. Show Rob the list
  (case file, variant, H, P, U, dt, offset, expected mesh) in one table before running.
- `cells.tsv` holds 6 significant figures (`_fmt`, `run_micro_cells.py:626`), which cannot carry a
  1e-12 parity. Write a small script in the new repo (`tools/make_references.py`) that calls the
  parent's worker the way `run_cell` does and saves, per cell, one `.npz` at full precision: the
  task tuple, the `MicroCase` fields, every float of `MicroTaskResult`, the `_extract_fields`
  arrays in SI, `mesh_used`, and the parent commit.
- One cell per process, so a re-targeted cached solver (`reset_task`) cannot enter a reference.
  If you want several cells per process, show first that it changes nothing.
- Not checked in the plan: whether the worker can run a pair cell at a set offset without the
  contact table. If it cannot, the pair reference is the solver's coupled `P_asp` before table
  composition; say which in the log.
- Before the run: ask Rob whether WSL is free and state the expected time (minutes; the first
  run JIT-compiles forms). At most 8 workers. Run through `wslrun`.
- A NaN in a reference must come with a status; otherwise the reference is a failure.

## WP1: verbatim copy + API

- Copy the closure of files listed in the plan's size table unchanged except import paths, under
  the plan's layout names or the original names (merging files is WP2/WP3, not now). Do not copy
  either `__init__.py` that imports the legacy solver or `pipeline` state.
- `microcell/api.py`: `microsolver()`, `Zeta` (with `Zeta.from_task`), `MicroSettings`,
  `MicroResult`, written from the body of `run_micro_task_phase_neutral` without the solver cache,
  the two mesh tiers, the contact table, xwarm, dphi sampling and the env overrides. At WP1
  `MicroSettings` may wrap the copied settings dataclasses; collapsing them is WP4.
- `microcell/fields.py`: `_extract_fields`, `_write_vts`, `_write_xdmf`.
- Apply the quadrature-degree pin and the dolfin parameters the parent's worker applies
  (`_init_worker`, `apply_macro_process_settings`, `pin_micro_worker`) before the first form.
- `tests/test_parity.py`: each reference cell through `microsolver()` at the reference's
  `mesh_used`; homogenised outputs and fields to relative 1e-12. The script form exits
  0 = PASS, 1 = FAIL, 2 = INCOMPLETE.
- One example script that runs a cell with `save_paraview=True`.

## Gate

Parity PASS on every reference cell. If it fails: report the number, one cause and one fix, and
wait. Don't loosen the tolerance or drop a cell.

## Stop and ask Rob before

- the first commit in the new repo (author `RFurze <py07rf@leeds.ac.uk>`, no Claude attribution);
- any WSL run over about 5 minutes or on more than 8 workers;
- adding a setting or input that the plan's API does not list;
- writing anything inside the parent clone or another worktree.

Never push; Rob pushes.

## Done looks like

- `MicroCell` holds the copied modules, `api.py`, `fields.py`, the references, `tools/`,
  `tests/test_parity.py`, one example, `scripts/ehl_env.sh`, `environment.yml`, a short README with
  the provenance (parent commit per file), and `docs/notes/` with the plan, this prompt and the log.
- `docs/notes/micro_standalone_wp0_wp1_implementation_log_2026-11-11.md` (or the day written):
  what was built, the commits, the reference table, the parity numbers each stated once in a table
  with their source file, numbered departures from the plan, open items, the file list, and the
  `second-reader` corrections (run it before the log reaches Rob).
- The workstreams row updated: stage, latest record, next step, what waits on Rob.
- A final summary to Rob: what changed, what he ran or must run (exact commands, times), what is
  open, the decisions he must take.
