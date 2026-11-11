# Standalone micro-cell solver: session prompt for WP2, prune dead physics (11 Nov 2026)

Staged here; file it into `MicroCell/docs/notes/` with your first commit.

## Read first

- `MicroCell/docs/notes/micro_standalone_plan_2026-11-11.md`: "Decisions" (D1, D5, D6, D10) and
  "Work packages" (WP2).
- `MicroCell/docs/notes/micro_standalone_wp0_wp1_implementation_log_2026-11-11.md`: §0, §3 (the
  parity rule and what it compares), §5 (departures 5.2 and 5.7), §6 (open items 6.1, 6.2).
- `MicroCell/tests/test_parity.py` and `MicroCell/microcell/api.py`.
- The three solver files by section (grep the targets below); don't read
  `micro_phase_neutral_class.py` (3501 lines) whole.

## Scope

WP2 only: remove the physics and hooks D1 dropped from the copied solver. Stop after WP2.
WP3 (roughness and shear helpers), WP4 (settings and results collapse) and WP5 are later sessions.

Files in scope: `microcell/micro_phase_neutral_class.py`, `micro_phase_neutral_class_mixed.py`,
`micro_phase_neutral_class_contact.py`, `micro_diagnostics.py`, `contact.py`, `api.py`
(the `legacy` keys and tags that go with a removed option), `tests/`, the README's provenance
note. Not in scope: `microcell/roughness/`, the shear modules, `ripple.py`, `elastic_contact.py`,
`fields.py`, file renames or merges, the references.

## Decided (don't reopen)

- D1 to D13 of the plan (Rob, 11 Nov 2026); departures 5.7 to 5.9 accepted by Rob 11 Nov 2026.
- Kept by D1: `fft_elastic_v1` contact, ripple on and off, `whole_cell` and `fluid_mean_v1`,
  `spline` and `sqrt`, full-film mode, the single-surface and pair branches, steady and transient
  solves, `pattern_flux_contact` both ways, `patch_pin`, the Newton damping settings.

## State as found: re-check before acting

| Item | Expected (11 Nov 2026) |
|---|---|
| `MicroCell` | `main` at c96ad39, pushed to `origin/main` by Rob, working tree clean |
| Parity | `tests/test_parity.py` passes on 8 of 8 reference cells (run it before the first edit) |
| Parent `MultiscaleMixedBallinSocket` | read-only; not needed for WP2 |
| Workstreams row | "Standalone micro-cell solver" in `_claude/memory/project_workstreams.md` |

## Removal groups

One group per commit, in this order. Occurrence counts are from `grep -c` at c96ad39 and only
show where the work is.

| # | Group | Where (count) |
|---|---|---|
| 1 | Dead lazy imports and `__main__` blocks (open item 6.2) | base (1), mixed (1 + the `config` import) |
| 2 | Spring deformation: `k_spring`, `spring_pointwise`, `_solve_with_spring`, the spring loop and its constants | base (71), mixed (8), contact (4), diagnostics (1), api (2) |
| 3 | Contact models `legacy_patch_v1` and `hertz_summit_v1`, with the settings only they read (`contact_E`, `use_reduced_modulus`, `overclosure_measure` if so) and `cosine_summits` / `hertz_response` | contact class (14), `contact.py` (8), api (10), base (1), diagnostics (1) |
| 4 | Roelands and compressibility stubs (`roelands_eta_dim`, `rho_nd`, `eta_nd` where they are constant 1) | base (5), mixed (3), contact (1), diagnostics (2) |
| 5 | `gauge_lift` and the `c2` cavitation form | base (4 + 4), mixed (1) |
| 6 | Export hooks inside the solver: `export_vtk`, the XDMF series, `post_process`, `output_dir`; the ignored constructor arguments `mesh_h` and `k` | base (24), api (1) |
| 7 | Warm-start plumbing: `set_pair_warm`, `pair_warm_record`, `_pair_warm_check`, `warm_start_snapshot`, `warm_start_state` | base (46), api (1) |
| 8 | Table-only code: `contact_only_load` and whatever only the contact table reads | contact class (5), base (4), `contact.py` (3), api (2) |
| 9 | Environment reads in the solver (open item 6.1): `MICRO_DEBUG_LOG`, `MICRO_NONDIM_DIAG`, `MICRO_UREF_FLOOR_OMEGA`, `MICRO_EPSP_ND_CAP`, `MICRO_SPOS_FORM` | base (5) |

Rules for every group:

- Before removing an option, show from the references that no cell uses it: the reference's
  stored setting equals the value the pruned code now hard-wires (`k_spring` 0, `gauge_lift`
  False, `contact_model` `fft_elastic_v1` on mixed cells, and so on). If a reference uses it, stop
  and ask.
- Group 9 freezes each value at what the code used with the variable unset. `MICRO_SPOS_FORM`
  goes; `spos_form` stays a setting. `EHL_SURFACE_STORE` in `roughness/case.py` is WP3.
- Group 8: the pair branch is selected today by the `contact_table_v1` tag the API sets
  (departure 5.2). Keep the pair branch. If the tag can be replaced by a plain flag without
  changing results, do it and say so; if not, leave it and list it for WP4.
- Remove an input the API passed through only for a removed option (`MicroSettings.legacy`
  keys); leave the other `legacy` keys for WP4.
- Remove settings fields, methods and branches that are dead after the group, and nothing else.
  Code that looks dead but is not in the table goes in the log as a candidate for Rob, not out.
- Nothing is deleted outside git: a removed file is `git rm`; the history keeps it.

## Gate, after every group

- `_claude/wsltest tests/ -q -p no:cacheprovider -rfE --tb=short` from `MicroCell/` (runs without
  asking; seconds, or minutes after a form change because the forms recompile).
- All 8 cells PASS at relative 1e-12, the negative control still passes.
- The references are never regenerated.
- The parity test compares the copied settings field by field (44 on a mixed cell). When a
  setting is removed, the test must not skip it silently: keep an explicit list of removed
  settings with the frozen value, and check the reference's stored value against it. Record the
  changed expectation and why in the log.
- A group that changes a UFL form may move results by round-off. Any non-zero difference must be
  below 1e-12 and attributed in the log to the named form change. A difference you cannot
  attribute is a failure.
- If the gate fails: report the number, one cause and one fix, and wait. Don't loosen the
  tolerance, drop a cell or change a reference.
- Once at the end: `_claude/wslrun python tests/test_parity.py --out <log files folder>/parity_report.tsv`
  for the kept report, and `python examples/run_cell.py`.

## Commits

Commit in `MicroCell` after each group passes the gate: author `RFurze <py07rf@leeds.ac.uk>`, no
Claude attribution, message `MicroCell WP2: <what was removed>; log docs/notes/<log file>`.
Never push, amend or rebase; Rob pushes.

## Stop and ask Rob before

- removing anything a reference cell uses, or anything not in the table;
- adding a setting or input;
- any change that moves a result by more than round-off;
- any WSL run over about 5 minutes;
- writing outside `MicroCell/` (except the workstreams row and `RefactorRefactor/_to_delete/`).

## Done looks like

- Groups 1 to 9 removed or reported as not removable, each with its commit SHA.
- `docs/notes/micro_standalone_wp2_implementation_log_<date>.md` with a §0 of at most 40 lines;
  one table of lines per file before and after (`wc -l` at c96ad39 and at the final commit);
  one table of parity per group (cells passed, largest relative difference, source file); the
  changed test expectations; numbered departures; candidates for further pruning; open items;
  the file list; the `second-reader` corrections (run it before the log reaches Rob).
- The workstreams row updated: stage, latest record, next step (WP3), what waits on Rob.
- A final summary to Rob: what changed, the commands to repeat the gate with expected times,
  what is open, the decisions he must take.
