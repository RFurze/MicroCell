# Standalone micro-cell solver: session prompt for WP4, settings and results (11 Nov 2026)

Written by the WP2 session on Rob's go-ahead of 11 Nov 2026 ("create a new sub session for WP4").
The lists below were collected by grep at 83597af, not verified: step 1 of this session is to
verify them.

## Read first

- `docs/notes/micro_standalone_plan_2026-11-11.md`: "Decisions" (D3, D7, D10, D11), "Target API"
  (the `out` table and the `MicroSettings` field list), "Work packages" (WP4).
- `docs/notes/micro_standalone_wp3_implementation_log_2026-11-11.md`: §0, §6, §8, and
  `..._files/pruning_candidates.md` (B7, E, F, G).
- `docs/notes/micro_standalone_wp2_implementation_log_2026-11-11.md`: §0 only, and
  `..._files/pruning_candidates.md`.
- `microcell/api.py` whole (638 lines); `tests/test_parity.py` (`settings_from_reference`,
  `run_cell`, the three negative controls).
- The settings dataclasses by grep (`^class \|^@dataclass` in `microcell/micro_phase_neutral_class*.py`);
  don't read the solver classes whole.

## Scope

WP4 only: one settings object, one result object, a status for every NaN. Stop after WP4.
WP5 (invariant tests, examples, README) is a later session.

In scope: `microcell/api.py`; the settings dataclasses and their readers in the three solver
class files; the result-field helpers in `fluid_shear.py`, `contact_shear.py`, `shear_reduce.py`,
`contact_sample.py`, `pressure.py`, `micro_diagnostics.py`; `microcell/__init__.py`; `tests/`;
`examples/run_cell.py`; the README where it names a removed field.

Not in scope: the physics (no UFL form, no solve order, no tolerance changes); the roughness
package; the references; `tools/make_references.py` (it runs in the parent, on the parent's
modules); file renames or merges toward the plan's target layout unless Rob says so (question 6).

## Decided (don't reopen)

- D1 to D13 of the plan (Rob, 11 Nov 2026).
- WP2 (Rob, 11 Nov 2026): `ContactSettings.nu` and `.k_asp_exponent` are removed at WP4;
  departures 6.2, 6.5, 6.7 of the WP2 log accepted.
- WP3 (Rob, 11 Nov 2026): departures 6.3, 6.4, 6.5 of the WP3 log accepted; nothing is restored.

## State as found: re-check before acting

| Item | Expected (11 Nov 2026) |
|---|---|
| `MicroCell` | `main`, working tree clean; `origin/main` at 83597af (Rob pushed); the commit that files this prompt is the base (record its SHA) |
| Gate | `13 passed`; 8 of 8 cells PASS, 1188 of 1188 comparisons exact (run it before the first edit) |
| Settings | `Zeta`, `ContactSettings`, `RippleSettings`, `MicroSettings` (with a `legacy` dict and `LEGACY_DEFAULTS`: `beta`, `delta_h`, `eps_solid`, `eps_smooth`, `patch_pin`, `patch_pin_kappa`) in `api.py`; `MicroPhysicalParameters`, `MicroSolverSettings`, `PhaseNeutralSettings` (base and mixed), `PhaseNeutralContactSettings` in the solver class files; `api._solver_settings` maps the first set to the second |
| Results | `MicroResult` has `status`, `reason` and 18 named outputs, plus `fields`, `info` and `raw`; `raw` holds the outputs under the parent's `MicroTaskResult` names (`api._failed_raw`, `api._RAW_DEFAULTS`); `tests/test_parity.py` compares `out.raw[key]` |
| Not built | `n_steps=None` (march from 0, D3) raises `NotImplementedError` |
| Parent `MultiscaleMixedBallinSocket` | read-only; not needed for WP4 |
| Workstreams row | "Standalone micro-cell solver" in `_claude/memory/project_workstreams.md` |

## Step 1: inventory, before any edit

Three TSVs in the log's files folder.

1. Settings: every field of the nine containers above (and each `LEGACY_DEFAULTS` key): its
   default, who sets it, who reads it (`file:line`), its value in each of the 8 references, and
   whether two containers give it different defaults (for example `spos_form`: `"sqrt"` in
   `MicroSettings`, `"spline"` in `MicroSolverSettings`).
2. Results: every `raw` key: who writes it, who reads it (`MicroResult` field, `info`, the
   gate, nothing), whether it is constant over the 8 references, whether it can be NaN.
3. NaN paths: every place a NaN can enter a `MicroResult` (homogenised output, `info`, a field
   array), the condition, and the status that reports it today (the overall `status`, a
   per-quantity status such as `shear_asperity_status`, or none).

## Step 2: one round of questions to Rob

After the inventory, put the open decisions to Rob in one message, recommended option first,
with the inventory's evidence in a table. Don't build the parts that depend on an answer; carry
on with the parts that don't (groups 1 and the 1:1 part of 2). Expected questions (add what the
inventory turns up):

1. The six `legacy` keys: named `MicroSettings` fields (D10: every setting is an explicit
   field) or fixed constants in the solver, per key. They are not in the plan's field list, so
   each named field is a new setting and needs his yes.
2. The `MicroSettings` defaults (now the parent's `GenSa40-ripple` case): which case's values
   are "current production"; give the table of defaults with their sources.
3. Outputs that are NaN on an OK result (full-film asperity and total shear, with
   `shear_asperity_status = "unavailable"`): keep NaN with a per-quantity status, or another
   rule. Rob's standing rule: a NaN must map to a status code, otherwise it is a failure.
4. The global-x shear outputs (`gx_*`), NaN with status `no_frame` in every cell because a cell
   has no macro frame (WP3 candidates B7): remove them with the `rotation=` path, or keep.
5. `n_steps=None` (D3): build it in WP4, leave it for WP5, or drop it from the API.
6. Renames and merges toward the target layout (WP3 candidates F): a pure-move commit at the
   end of WP4, at WP5, or not at all.

## Work groups

One group per commit, in this order. The contents are what the inventory confirms.

| # | Group | What done looks like |
|---|---|---|
| 1 | Decided removals | `ContactSettings.nu` and `.k_asp_exponent` gone, with their readers and the notes that say "kept until WP4" |
| 2 | Settings collapse | One definition and one default per setting. The solver classes take what `microsolver` builds from `Zeta`, `t_end`, the roughness and `MicroSettings`; `MicroSettings.legacy`, `legacy_value`, `LEGACY_DEFAULTS` and `api._solver_settings`'s remapping are gone. Whether the solver-side dataclasses are removed or kept as one internal container is your call; report it. `MicroSettings.check()` rejects every value the solver would refuse |
| 3 | Results collapse | `MicroResult.raw`, `api._RAW_DEFAULTS` and `api._failed_raw` gone: each output is written once, into a named `MicroResult` field or `info`. A failed result is built in one place. Fields that Rob's answers remove are removed with their writers |
| 4 | Status for every NaN | One table in code that says, for each output that can be NaN, which status reports it. A new test: for every reference cell, and for a forced `FAILED_NEWTON`, `FAILED_RIPPLE` and `FAILED_CONTACT` (monkeypatched, as the negative controls are), every NaN in the homogenised outputs, `info` and the field arrays has a non-OK status by that table, and no output is NaN without one |

Rules for every group:

- No physics change. If a default differs between two containers, the value each reference
  cell ran with must be the value it runs with after the collapse.
- No new setting, no changed default and no changed `microsolver` argument without Rob's yes.
- The parity test will need a name map (reference name, as stored, to the new field or `info`
  key) once `raw` is gone. Write the map in `tests/test_parity.py` as one table.
- Nothing is deleted outside git; before editing a note in place, put a copy in `_to_delete/`.
- Code that looks dead but is outside the group goes in the log as a candidate.

## Gate, after every group

- `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from `MicroCell/` (runs
  without asking; about 10 s, about 30 s after a form change).
- All 8 cells PASS at relative 1e-12, and every comparison exact: WP4 moves values between
  containers, so no number should change. A non-zero difference must be below 1e-12 and
  attributed in the log to a named change; one you cannot attribute is a failure.
- The comparison count may fall only by quantities that leave the result. Account for it per
  cell: compared at base (1188 in total) = compared now + no longer compared, with each dropped
  name listed once with its reason (always NaN, constant, bookkeeping). A quantity that is
  finite and varies over the references is not dropped without Rob's yes.
- The three negative controls and `test_generated_surface_reproduces_reference` still pass. A
  control that tested a removed thing (for example a `legacy` key) is rewritten to test its
  replacement, and listed under changed test expectations.
- The references are not regenerated.
- If the gate fails: report the number, one cause and one fix, and wait. Don't loosen the
  tolerance, drop a cell or change a reference.
- Once at the end: `_claude/wslrun python tests/test_parity.py --out <log files folder>/parity_report.tsv`
  and `_claude/wslrun python examples/run_cell.py` (each about 10 s).

## Commits

Commit in `MicroCell` after each group passes the gate: author `RFurze <py07rf@leeds.ac.uk>`, no
Claude attribution, message `MicroCell WP4: <what changed>; log docs/notes/<log file>`.
Never push, amend or rebase; Rob pushes. Git runs in Git Bash, never from WSL.

## Stop and ask Rob before

- adding a setting or input, changing a default, or changing `microsolver`'s arguments;
- removing an output that is finite in any reference cell;
- any change that moves a result by more than round-off;
- changing what a status code means, or adding one beyond what group 4 needs;
- any WSL run over about 5 minutes;
- writing outside `MicroCell/` (except the workstreams row and `RefactorRefactor/_to_delete/`).

## Done looks like

- The three inventory TSVs; Rob's answers recorded with their date; groups 1 to 4 built or
  reported as waiting on an answer, each with its commit SHA.
- `docs/notes/micro_standalone_wp4_implementation_log_<date>.md` with a §0 of at most 40 lines
  and a body under about 25 KB (detail in its `_files/` folder): the settings table before and
  after (field, default, source of the default); the results table before and after; the NaN
  and status table; parity per group (cells passed, comparisons exact of total, the count
  account, largest relative difference, source file); lines per file before and after; changed
  test expectations; numbered departures; candidates left in place; open items; the file list;
  the `second-reader` corrections (run it before the log reaches Rob).
- The README's call example and settings list match the code.
- The workstreams row updated: stage, latest record, next step (WP5), what waits on Rob.
- A final summary to Rob: what changed, the commands to repeat the gate with expected times,
  what is open, the decisions he must take.
