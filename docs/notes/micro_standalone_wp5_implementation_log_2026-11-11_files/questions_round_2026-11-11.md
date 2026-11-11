# WP5 step 2: the one round of questions to Rob (11 Nov 2026)

State: asked 11 Nov 2026 in the WP5 session. **No answer yet.** Groups 1, 2, 6 and every commit
wait on the answers. Recommended option first. Evidence files are beside this file.

| Q | Question | A (recommended) | B | Evidence |
|---|---|---|---|---|
| 1 | Layout where the plan and the tree disagree | Three solver files by `git mv`: `solver_base.py`, `solver_mixed.py`, `solver_contact.py`. Merge to `samplers.py`, `film.py`, `contact.py`, `shear.py` as the plan says. Leave the `roughness/` file names, `form_compiler.py` and `process_settings.py` as they are | One `solver.py`: the alias import lines between the three classes go and two assignment lines are added, so it is not a verbatim move | `inventory_layout_files_base_d3b88ae.tsv`, `inventory_layout_imports_base_d3b88ae.tsv`, `inventory_layout_collisions_base_d3b88ae.tsv` (no name collisions in the four merges) |
| 2 | Parent tests that cannot run here | Copy the 34 that run with renamed settings, and the halves that remain of the 3 "in part". Leave the 9 out, each with its reason in the log. Adapted tests set the parent's test-time values explicitly (`tests/parent_fixtures.py`). T6 and T7 keep their `slow` mark, registered in `pyproject.toml`, and run in the gate. `tests/test_micro_phase_neutral_steady.py` is taken as the plan's "steady conservation checks" | Also run the patch-free test with a monkeypatched module constant and the moving-generated-surface test with a `Roughness.generated` surface (changed fixtures) | `inventory_tests_base_d3b88ae.tsv` (46 rows) |
| 3 | Dead-code candidates as group 6 | Yes, narrow: only names with no reader. `SAMPLER_DEFINITION`, `h_geom_min_bound`, `_print_timing_summary` and the branches under `PRINT_NEWTON` / `PRINT_TIMING` / `TRACE_NEWTON`, `FLUID_DEFORMATION_MODES` / `deformation_model_tag`, the alias rebinding in the mixed class file, unused `as_dict`s, `err_*_legacy`, unused imports. `reset_task`, `record_homog`, `auto_solve`, `PATCH_PIN` stay (the copied tests read them) | No group 6; the candidates stay listed in the log | `inventory_candidates_base_d3b88ae.tsv` (32 rows, reader counts) |
| 4 | `n_steps=None`: a cap, and what `info` reports | No cap. `info["n_steps"]` reports the count marched. The cost goes in the docstring and the README. No new setting or `info` entry | Refuse above a fixed step count | `march_cost_base_d3b88ae.tsv` |

Numbers behind Q4 (measured, `march_cost_base_d3b88ae.tsv`; the last row is inferred):

| Item | Value |
|---|---|
| Steps for `t_end = 0.02 s` at mesh 100, `Ux = 0.02 m/s` (CFL rule) | 16000 |
| Example cell, 11 steps from t = 0 | 0.91 s |
| Example cell, 21 steps from t = 0 | 1.31 s |
| Full march of the example cell (inferred: about 0.04 s per step, stationary surface) | about 640 s |

Found while preparing, outside WP5's scope, not acted on:

- A translating generated surface (Sa 40 nm, H 130 nm, contact on) fails Newton at 300 iterations
  with residual 1.0654e-06 against the tolerance 1e-6 (`probe_translating_base_d3b88ae.tsv`). The
  failed result reports `newton_iters = 0`. The examples avoid that cell.
- With an integer `n_steps` larger than the window holds, `info["n_steps"]` reports the count
  asked for, not the count marched (existing behaviour; compared with the references by the
  parity gate).

## Rob's answers (11 Nov 2026)

"Yes, A for all": the first (recommended) option of each question.

| Question | Answer | What it means |
|---|---|---|
| Q1 layout | A | three solver files by `git mv`; merges to `samplers.py`, `film.py`, `contact.py`, `shear.py`; `roughness/` file names, `form_compiler.py` and `process_settings.py` stay |
| Q2 tests | A | copy the 34 tests that run with the settings renamed and the halves of the 3 "in part"; the 9 left out are listed with reasons in the log; the parent's test-time values are set explicitly; T6 and T7 keep their `slow` mark, registered in `pyproject.toml`, and run in the gate |
| Q3 group 6 | A | yes, narrow: only names with no reader |
| Q4 `n_steps=None` | A | no cap; `info["n_steps"]` reports the count marched; the cost is in the docstring and the README; no new setting or `info` entry |
