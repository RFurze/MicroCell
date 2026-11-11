# Standalone micro-cell solver: session prompt for WP5, layout, tests, examples, README (11 Nov 2026)

Written by the WP2 session on Rob's go-ahead of 11 Nov 2026 ("Start WP5"). The lists below were
collected by grep at 3d71e1f and from the parent at 2aaccad, not verified: step 1 of this
session is to verify them. WP5 is the last work package of the plan.

## Read first

- `docs/notes/micro_standalone_plan_2026-11-11.md`: "Decisions" (D3, D8), "Target API", "Repo
  layout (target)", "Work packages" (WP5), "Verification", "Known traps to carry over".
- `docs/notes/micro_standalone_wp4_implementation_log_2026-11-11.md`: §0, §2 (answers 5 and
  6), §5, §10, §11.
- `docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/pruning_candidates.md`
  (A, B, D, E, F, H) and the WP2 log's `..._files/pruning_candidates.md`.
- `README.md` whole (193 lines); `microcell/api.py` by function; `examples/run_cell.py`;
  `tests/test_parity.py` (the gate's rule and its negative controls).
- The four parent test files below, from `git show 2aaccad:<path>` in
  `../MultiscaleMixedBallinSocket` (read-only; run git there in Git Bash, never from WSL).

## Scope

In scope: file renames and merges inside `microcell/`; `n_steps=None` in `microsolver`; new
files in `tests/`; `examples/`; `README.md`; `docs/notes/provenance.tsv` if the renames need
a new column; the dead-code candidates, only on Rob's yes (question 3).

Not in scope: the physics (no UFL form, tolerance or solve order changes); the settings and
their defaults; the result layout and the status codes; the references;
`tools/make_references.py` (it runs in the parent, on the parent's modules).

## Decided (don't reopen)

- D1 to D13 of the plan (Rob, 11 Nov 2026).
- WP4 answers (Rob, 11 Nov 2026): the renames toward the target layout are the first commit of
  WP5 (answer 6); `n_steps=None` is built in WP5 (answer 5); the GenSa40-ripple defaults stay;
  departures 9.2 and 9.5 of the WP4 log kept.
- The accepted departures of WP2 (6.2, 6.5, 6.7) and WP3 (6.3, 6.4, 6.5).

## State as found: re-check before acting

| Item | Expected (11 Nov 2026) |
|---|---|
| `MicroCell` | `main`, working tree clean; `origin/main` at 83597af unless Rob has pushed since; the commit that files this prompt is the base (record its SHA) |
| Gate | `60 passed` (about 17 s); 8 of 8 cells PASS, 1100 of 1100 comparisons exact (run it before the first edit) |
| `microcell/` | 26 files, 8413 lines (`wc -l microcell/*.py microcell/roughness/*.py`; record a before table) |
| Tests | `tests/test_parity.py`, `test_results.py`, `test_settings.py` |
| Examples | `examples/run_cell.py` only |
| `n_steps=None` | `api.py` raises `NotImplementedError` |
| Parent tests at 2aaccad | `tests/ripple/test_ripple_micro_fenics.py` (482 lines, 15 tests), `tests/elastic_contact/test_fft_micro_fenics.py` (244, 8), `tests/micro_fluid_mean/test_fluid_mean_fenics.py` (284, 13), `tests/test_micro_phase_neutral_steady.py` (672, 10; taken to be the plan's "steady conservation checks": confirm) |
| Workstreams row | "Standalone micro-cell solver" in `_claude/memory/project_workstreams.md` |

## Step 1: inventory, before any edit

1. Layout: for each target file of the plan, the present files that go into it, their lines,
   and every import of them in `microcell/`, `tests/`, `examples/`, `tools/`. Say where the
   plan's layout no longer fits the tree (for example `roughness/pair.py`, one function).
2. Parent tests: for each of the 46 tests in the four files, what it builds and checks, and
   whether it can run here: yes as it is, yes with renamed settings, or no because it tests
   something removed (name the removed thing). Expected values are copied, never re-derived.
3. `n_steps=None`: what the solver's `run()` does with it today, the step count and the wall
   time of a full march for the example cell, measured on a short window and stated as an
   estimate beyond it.

## Step 2: one round of questions to Rob

After the inventory, one message, recommended option first, evidence in a table. Carry on with
what does not depend on an answer. Expected questions (add what the inventory turns up):

1. The layout where the plan and the tree disagree (one `solver.py` or three files; where
   `pair.py` and `quantiles.py` go).
2. Parent tests that cannot run here: leave them out with the reason recorded, or another rule.
3. The dead-code candidates (WP4 log §10, WP3 candidates A, B, D, E, the WP2 list): remove the
   ones the inventory shows unread, as a last group under the gate, or leave them.
4. Anything `n_steps=None` needs that is a choice (a cap on the step count, what `info`
   reports).

## Work groups

One group per commit, in this order.

| # | Group | What done looks like |
|---|---|---|
| 1 | Renames and merges | The target layout, by `git mv` where a file is renamed. Code moves verbatim: only import lines, module docstrings and `__all__` change. A check script in the log's files folder shows that the sorted non-import code lines of `microcell/` are the same before and after. README layout and provenance tables name the new files and what each came from |
| 2 | `n_steps=None` | The march from t = 0 to `t_end` (D3), with its own test: a window short enough to run in under a minute, and an expectation that does not come from the same code path (for example agreement with `n_steps=N` where the two are the same problem by construction; say what makes them the same). The docstring and README state the cost |
| 3 | Invariant tests | The parent tests that can run here, adapted to the new names, in `tests/`. Expected numbers and tolerances unchanged. A test that fails after adaptation is reported, not edited to pass |
| 4 | Examples | One script per roughness kind: generated, pair, array, cosine. Each runs end to end in under a minute, prints its outputs with units, and writes its ParaView files under `examples/output/` |
| 5 | README | The model in one page, the call, the conventions (D8 velocities, D3 window), the settings and result tables, how to run, the provenance table. Every number and name in it checked against the code |
| 6 | Dead-code candidates | Only if Rob says yes to question 3 |

Rules for every group:

- No new setting, changed default, changed `microsolver` argument, status code or result field
  without Rob's yes.
- A second parent-derived surface test (WP3 candidates H) belongs to group 3 if it can be
  made exact; if it holds only to round-off, report the number and ask.
- Nothing is deleted outside git; before editing a note in place, put a copy in `_to_delete/`.
- README prose: plain, short sentences; frictional torque is `\mathcal{T}`, never plain T.

## Gate, after every group

- `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from `MicroCell/` (runs
  without asking). Write it to a log in the log's files folder once the copied tests are in,
  and read the count line only: `grep -E '(passed|failed).* in [0-9.]+s' <log>`.
- All 8 cells PASS at relative 1e-12 and 1100 of 1100 comparisons exact, after every group.
  WP5 changes no number, so any non-zero difference is a failure to report.
- The negative controls, `test_generated_surface_reproduces_reference`, `tests/test_results.py`
  and `tests/test_settings.py` still pass.
- The references are not regenerated.
- The first run of the copied tests JIT-compiles forms. Before any WSL run you expect to take
  over about 5 minutes, ask Rob whether WSL is free; nothing in WP5 should run over 10 minutes.
- If the gate or a copied test fails: report the number, one cause and one fix, and wait.
  Don't loosen a tolerance, drop a cell or a test, or change a reference.
- Once at the end: `_claude/wslrun python tests/test_parity.py --out <log files folder>/parity_report.tsv`
  and each example through `_claude/wslrun`.

## Commits

Commit in `MicroCell` after each group passes the gate: author `RFurze <py07rf@leeds.ac.uk>`, no
Claude attribution, message `MicroCell WP5: <what changed>; log docs/notes/<log file>`.
Never push, amend or rebase; Rob pushes. Git runs in Git Bash, never from WSL.

## Stop and ask Rob before

- adding a setting, an input, a status code or a result field, or changing a default;
- changing an expected number or a tolerance of a copied test;
- any change that moves a result by more than round-off;
- removing code outside group 1's moves without his yes to question 3;
- any WSL run over about 5 minutes;
- writing outside `MicroCell/` (except the workstreams row and `RefactorRefactor/_to_delete/`).

## Done looks like

- The inventories; Rob's answers recorded with their date; groups 1 to 5 built (6 if he said
  yes), each with its commit SHA.
- `docs/notes/micro_standalone_wp5_implementation_log_<date>.md` with a §0 of at most 40 lines
  and a body under about 25 KB (detail in its `_files/` folder): the layout before and after
  with lines per file; the moved-lines check; parity per group (cells passed, comparisons exact
  of total, source file); the parent tests copied, adapted and left out, with reasons; test
  counts and times per group; the `n_steps=None` test and its measured cost; the examples with
  their printed outputs and times; numbered departures; candidates left in place; open items;
  the file list; the `second-reader` corrections (run it before the log reaches Rob).
- What Rob must check himself: the examples' `.vts` and `.xdmf` open in ParaView (plan,
  "Verification"). Give him the paths.
- The workstreams row updated: stage, latest record, next step, what waits on Rob.
- A final summary to Rob: what changed, the commands to repeat the gate and the examples with
  expected times, what is open, the decisions he must take.
