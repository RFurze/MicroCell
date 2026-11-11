# Standalone micro-cell solver: session prompt for WP3, prune the helpers (11 Nov 2026)

Written by the WP2 session on Rob's go-ahead of 11 Nov 2026 ("start a new session on WP3").
The removal candidates below were collected by name and by import, not verified: step 1 of this
session is to verify them.

## Read first

- `docs/notes/micro_standalone_plan_2026-11-11.md`: "Decisions" (D2, D4, D5, D6, D10), "Work
  packages" (WP3), "Repo layout (target)".
- `docs/notes/micro_standalone_wp2_implementation_log_2026-11-11.md`: §0, §6 (departures), §7,
  §8 (open items 1 and 2), and `..._files/pruning_candidates.md`.
- `tests/test_parity.py` (the gate and its rule), `microcell/api.py`,
  `microcell/roughness/frontend.py`.
- The helper modules by function (grep `^def \|^class `); don't read them whole.

## Scope

WP3 only: prune the roughness and shear helpers the copied solver does not need. Stop after WP3.
WP4 (settings and results collapse) and WP5 (tests, examples, README) are later sessions.

In scope: `microcell/roughness/` (all files), `shear_reduce.py`, `fluid_shear.py`,
`contact_shear.py`, `contact_sample.py`, `pressure.py`, `ideal_ht.py`, `surface_ht.py`,
`contact.py`, `elastic_contact.py`; in `api.py`, the solver classes and `micro_diagnostics.py`
only the imports and calls that follow a removed helper; `tests/`; the README's provenance note.

Not in scope: the physics of the solver classes; the settings and results dataclasses and the
`legacy` keys (WP4); file renames or merges toward the plan's target layout (list them as
candidates); the references; `tools/make_references.py` (it runs in the parent, on the parent's
modules).

## Decided (don't reopen)

- D1 to D13 of the plan (Rob, 11 Nov 2026).
- WP2, Rob 11 Nov 2026: the c5 reference was re-made with `k_spring = 0` and the spring removed;
  departures 6.2 (full-film `contact_model` tag `off`), 6.5 (`pair_branch` flag) and 6.7
  (logging disabled at import) accepted; `ContactSettings.nu` and `.k_asp_exponent` are removed
  at WP4, not now.
- Kept by D2: generated Gaussian surfaces (one surface, or a stationary + translating pair), a
  user-supplied height array, the analytic cosine cell. Kept by D4: one offset per call.

## State as found: re-check before acting

| Item | Expected (11 Nov 2026) |
|---|---|
| `MicroCell` | `main`, working tree clean; the commit that files this prompt is the base (record its SHA) |
| Gate | `13 passed`; 8 of 8 cells PASS, 1188 of 1188 comparisons exact (run it before the first edit) |
| Sizes | `microcell/roughness/` 4151 lines in 15 files; the shear, pressure and film helpers as `wc -l` gives them (record a before table) |
| Parent `MultiscaleMixedBallinSocket` | read-only; not needed for WP3 |
| Workstreams row | "Standalone micro-cell solver" in `_claude/memory/project_workstreams.md` |

## Step 1: inventory, before any removal

For every module in scope list each top-level function, class and constant with who uses it:
reachable from `microsolver` (through `api.py`, the solver classes, `micro_diagnostics.py`,
`fields.py`), used only by `tests/`, `examples/` or `tools/`, or used by nothing in the repo.
Keep the table as a TSV in the log's files folder. A name is dead only when a grep of the whole
repo shows no reader that is itself kept.

## Removal groups

One group per commit, in this order. The contents are what the inventory confirms, not this
list; a name here that turns out to be used stays, and is reported.

| # | Group | Candidates (unverified) |
|---|---|---|
| 1 | Macro-frame and batch code in the shear modules | `fluid_shear.py`: `shear_predictions`, `shear_record`, `legitimate_nan_rows`, `macro_frames`, `takes_shear_frames`, `with_shear_frames`, `export_arrays`, `metadata_json`. `contact_shear.py`: `allocate_arrays`, `legitimate_nan_rows`, `contact_shear_predictions`, `contact_shear_record`, `diagnostic_state_mismatch`, `export_arrays`, `metadata_json` |
| 2 | Remains of the removed contact laws | `UNSUPPORTED_MODELS` and its branches and `finish_unsupported` (`contact_shear.py`, `contact_sample.py`); the Hertz and summit descriptor entries in `ideal_ht.py`; `effective_modulus`, `CONTACT_MODELS` in `contact.py` |
| 3 | Macro-side helpers in `pressure.py` | `require_definition`, `level_constraint_of`, `require_level_constraint`, `effective_level_constraint` (`average_pressure_ledgers` is called by `api.py` and stays) |
| 4 | Roughness plumbing behind the front end | `case.py`, `assets.py`, `prepare.py`, `targets.py`, `report.py`, `spec.py`, `modes.py`: whatever `Roughness` (`frontend.py`), `runtime.py`, `pair.py`, `generation.py`, `evaluator.py` and the parts of `stats.py` and `quantiles.py` they call do not need |
| 5 | The environment read | `EHL_SURFACE_STORE` and the surface store in `roughness/case.py` (D10: no environment variables) |

Rules for every group:

- Kept by the plan: generation, evaluator, runtime, pair, and the parts of stats they call; the
  per-cell shear reduction. A whole file that becomes empty of kept code is `git rm`.
- Group 4 may need the front end rewired (for example `Roughness.generated` calling the
  generator directly instead of through `prepare` and `spec`). That is allowed only if every
  array a `Roughness` constructor returns is unchanged: `test_generated_surface_reproduces_reference`
  must pass, and add the same check for any other constructor path you rewire, before the rewiring.
- Group 5: if removing the store changes what a caller sees (an on-disk cache that no longer
  exists, a generation time paid on every call), measure it and ask Rob before removing.
- Remove functions, classes, constants and branches that are dead after the group, and nothing
  else. Code that looks dead but is outside the table goes in the log as a candidate.
- Nothing is deleted outside git; before editing a note in place, put a copy in `_to_delete/`.

## Gate, after every group

- `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from `MicroCell/` (runs without
  asking; about 10 s, about 30 s after a form change).
- All 8 cells PASS at relative 1e-12; the three negative controls and
  `test_generated_surface_reproduces_reference` still pass.
- The references are not regenerated.
- WP3 removes helpers, not physics, so no UFL form should change and every comparison should stay
  exact. Any non-zero difference must be below 1e-12 and attributed in the log to a named
  change; a difference you cannot attribute is a failure.
- If the gate fails: report the number, one cause and one fix, and wait. Don't loosen the
  tolerance, drop a cell or change a reference.
- Once at the end: `_claude/wslrun python tests/test_parity.py --out <log files folder>/parity_report.tsv`
  and `_claude/wslrun python examples/run_cell.py` (each about 10 s).

## Commits

Commit in `MicroCell` after each group passes the gate: author `RFurze <py07rf@leeds.ac.uk>`, no
Claude attribution, message `MicroCell WP3: <what was removed>; log docs/notes/<log file>`.
Never push, amend or rebase; Rob pushes. Git runs in Git Bash, never from WSL.

## Stop and ask Rob before

- removing anything a reference cell, a test or an example uses;
- changing what a `Roughness` constructor returns, or its arguments;
- adding a setting or input;
- any change that moves a result by more than round-off;
- any WSL run over about 5 minutes;
- writing outside `MicroCell/` (except the workstreams row and `RefactorRefactor/_to_delete/`).

Put open questions to Rob in one round, recommended option first, and carry on with the groups
that do not depend on the answer.

## Done looks like

- The inventory TSV, and groups 1 to 5 removed or reported as not removable, each with its
  commit SHA.
- `docs/notes/micro_standalone_wp3_implementation_log_<date>.md` with a §0 of at most 40 lines
  and a body under about 25 KB (detail in its `_files/` folder): one table of lines per file
  before and after (`wc -l` at the base and at the final commit); one table of parity per group
  (cells passed, comparisons exact of total, largest relative difference, source file); changed
  test expectations; numbered departures; candidates for further pruning, including renames and
  merges toward the target layout; open items; the file list; the `second-reader` corrections
  (run it before the log reaches Rob).
- The README's provenance note says which files were pruned or removed in WP3.
- The workstreams row updated: stage, latest record, next step (WP4), what waits on Rob.
- A final summary to Rob: what changed, the commands to repeat the gate with expected times,
  what is open, the decisions he must take.
