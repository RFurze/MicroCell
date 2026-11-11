# Standalone microscale solver: extraction plan (plan only, 11 Nov 2026)

## Context

Rob wants the microscale cell solver of the mixed-lubrication model as a standalone, compact git
repository with one call, `outputs = microsolver(zeta, t_end, roughness, ...)`: one cell in, the
homogenised outputs and the pressure and film fields out, ParaView files on a boolean.
The parent repo (`MultiscaleMixedBallinSocket`, `main` at 2aaccad) is not changed; code is copied out
and pruned.

It makes sense, and most of it exists already. `scripts/run/run_micro_cells.py` runs single cells
through the production worker and writes their fields for ParaView. The standalone repo is that
script's job without the worker, pool, case and macro machinery around it.

## State as found (read 11 Nov 2026, parent at 2aaccad)

- Production path for one task: `run_micro_task_phase_neutral`
  (`scripts/run/run_microscale_phase_neutral.py:989`) builds a film callback, a
  `MicroPhysicalParameters` and a settings object, then runs `MicroPhaseNeutralContactSolver`
  (mixed) or `MicroPhaseNeutralSolver` (full film): `solver.run(Tf, n_steps, cfl_target)`, then
  `spatial_homogenisation()` and `compute_asperity_load()`.
- Class chain: `MicroPhaseNeutralSolver` -> `MicroPhaseNeutralMixedSolver` ->
  `MicroPhaseNeutralContactSolver`.
- Mesh: `dolfin.UnitSquareMesh(n, n)` with a periodic CG1 space; no mesh files, no netgen.
- No production output holds the micro fields; `run_micro_cells.py:_extract_fields` reads them from
  the solver's final state.
- Not on the production path: `micro_mixed_transient_class.py` (refused at launch),
  `micro_phase_neutral_LC_class.py`, `generated_ht_cell.py`, `ideal_ht_test.py`, `generate_mesh.py`.
- The helper modules the solver imports (`ehl/ripple.py`, `elastic_contact.py`, `contact.py`,
  `pressure.py`, `shear_reduce.py`, `fluid_shear.py`, `contact_shear.py`, `ehl/roughness/*`,
  `utils/form_compiler.py`) import only numpy and each other.

Source sizes (`wc -l` at 2aaccad):

| Group | Files | Lines |
|---|---|---|
| Solver classes | `micro_phase_neutral_class.py`, `_mixed.py`, `_contact.py` | 3501 + 859 + 770 |
| Samplers and film callbacks | `micro_diagnostics.py`, `contact_sample.py`, `ideal_ht.py`, `surface_ht.py` | 476 + 402 + 275 + 88 |
| Contact, ripple, pressure, shear helpers | `ehl/elastic_contact.py`, `ripple.py`, `contact.py`, `pressure.py`, `shear_reduce.py`, `fluid_shear.py`, `contact_shear.py` | 365 + 312 + 93 + 173 + 253 + 397 + 602 |
| Roughness package | `ehl/roughness/*.py` (14 files) | 3970 |
| Worker (to be replaced, not copied) | `run_microscale_phase_neutral.py`, `micro_task_helpers.py` | 2846 + 454 |
| Field extraction and writers | `scripts/run/run_micro_cells.py` | 640 |

## Decisions

Rob's answers of 11 Nov 2026:

- **D1 Physics:** production path plus switches. Keep `fft_elastic_v1` contact, ripple on/off, both
  level constraints (`whole_cell`, `fluid_mean_v1`), `spline` and `sqrt` cavitation forms, a
  full-film mode. Drop spring / `k_spring` / `spring_pointwise`, `legacy_patch_v1`,
  `hertz_summit_v1`, `MicroMixedSolver`, the LC class, the Roelands and compressibility stubs,
  `gauge_lift`, the `c2` form.
- **D2 Roughness:** generated Gaussian surfaces (one surface, or a stationary + translating pair),
  a user-supplied height array, and the analytic cosine cell.
- **D3 Time:** the production window. `n_steps=3` solves the last `n_steps` CFL steps ending at
  `t_end`; `n_steps=None` marches from 0. Homogenised outputs are the end state.
- **D4 Two surfaces:** one offset `(o_x, o_y)` per call; the caller loops to average. The contact
  table, xwarm and dphi sampling are dropped.

Taken as recommended unless Rob says otherwise:

- **D5 Method:** copy, then prune under a parity gate; not a rewrite. Each pruning stage must leave
  the reference cells' outputs unchanged.
- **D6 Parity rule:** homogenised outputs and fields of the reference cells equal the parent's to a
  relative 1e-12 (bitwise expected; the tolerance covers import-order effects only). References are
  made once from the parent at 2aaccad with `run_micro_cells.py` and stored in the new repo.
- **D7 Inputs in SI, named:** `Zeta` is a small dataclass, not the 19-row vector. `Zeta.from_task()`
  maps the parent's task tuple for parity work.
- **D8 Velocity convention kept:** `Ux, Uy` are the mean (entrainment) velocities as in the macro
  task; the cell slides at `2 Ux`. Stated in the docstring.
- **D9 One mesh per call:** `mesh_n` argument; no two-tier coarse/fine escalation, no predictor.
- **D10 No environment variables, no solver cache, no pool, no dedup, no kNN fill.** Every setting
  is an explicit field of one `MicroSettings` dataclass with the current production values as
  defaults. An optional `solver=` handle lets a caller reuse a built solver across calls.
- **D11 Failures:** a failed solve returns a result with a status code and reason; NaN appears only
  with a non-OK status. No exception swallowing beyond the two the worker catches today
  (`RuntimeError`, `ContactResolutionError`).
- **D12 Environment:** the same `ehl-fenics` conda env (FEniCS 2019.1.0, python 3.11); run in WSL.
- **D13 Repo:** a fresh repo with no history, created by Rob beside the worktrees (suggested name
  `MicroCell`); the README records the parent commit each file came from.

## Target API

```python
from microcell import microsolver, Zeta, Roughness, MicroSettings

zeta = Zeta(H=160e-9, P=40.8e6, Ux=0.02, Uy=0.0, dpdx=0.0, dpdy=0.0,
            Hdot=0.0, Pdot=0.0, Uxdot=0.0, Uydot=0.0, dpdx_dot=0.0, dpdy_dot=0.0)
rough = Roughness.generated(Sa=40e-9, corr_length=3.26e-7, seed=1000)   # or .pair(...), .array(z), .cosine(Ah, kx, ky)
out = microsolver(zeta, t_end=0.02, roughness=rough, settings=MicroSettings(),
                  n_steps=3, offset=(0.0, 0.0), save_paraview=False, output_dir=None)
```

`out` (a dataclass):

| Part | Contents |
|---|---|
| `status`, `reason` | OK / FAILED_NEWTON / FAILED_CONTACT / NO_FLUID ... |
| homogenised | `dQ` (2), `dP`, `Q` (2), `Pst`, `P_f`, `Fst`, `tau` (fluid, asperity, total), `P_asp`, `contact_frac`, `A_mech_frac`, `k_asp`, extrema (`p_max`, `p_min`, `h_max`, `h_min`), `yield_flag` |
| `fields` | dict of `(n+1, n+1)` arrays in SI: `p_fluid`, `p_contact`, `p_total`, `film`, `film_rigid`, `surface_height`, `u_ripple`, `u_contact`, `chi_fluid`, plus `x`, `y` |
| `info` | Newton and contact iterations, ripple passes, residuals, wall time, `mesh_n`, `t_start`, `dt` |

`save_paraview=True` writes `<output_dir>/cell.vts` (pure-numpy VTK writer) and `cell.xdmf/.h5`
(FE mesh, with the DG0 fluid indicator); with `n_steps` > 1 one file set per solved step plus a PVD.

`MicroSettings` fields (defaults = current production values): `mesh_n`, `xmax`, `ymax`, `eta0`,
`rho0`, `penalty_gamma`, `h_min`, `spos_form`, `level_constraint`, `pattern_flux_contact`,
`jacobian_reg_eps`, Newton tolerances and damping, `contact` (E*, nu, tol, maxit, sigma_yield,
mu, k_asp mode and exponent), `ripple` (on, E*, tol, maxpass, omega), `cfl_target`,
`roughness_velocity_ratio`.

## Repo layout (target)

```
MicroCell/
  README.md  environment.yml  pyproject.toml
  microcell/
    api.py            microsolver(), Zeta, MicroSettings, MicroResult   (new, ~300 lines)
    solver.py         the three solver classes, pruned (one file or base/mixed/contact)
    samplers.py       micro_diagnostics.py + contact_sample.py
    film.py           ideal_ht.py + surface_ht.py
    contact.py        ehl/contact.py + elastic_contact.py
    ripple.py         ehl/ripple.py
    pressure.py  shear.py   ehl/pressure.py; shear_reduce + fluid_shear + contact_shear (cell parts)
    roughness/        generator (IAAFT), evaluator, runtime, pair; Roughness front end
    fields.py         _extract_fields, _write_vts, _write_xdmf from run_micro_cells.py
  tests/              parity references + copied invariant tests
  examples/           one script per roughness kind
```

## Work packages

Each WP ends with the parity check (WP1 onward) and one or more commits.

- **WP0 References (in the parent, read-only run).** Choose about 6 reference cells: genrough
  Cup90 film ladder with and without contact, ripple on and off, one pair cell at a fixed offset,
  one cosine cell, one full-film cell. Run `run_micro_cells.py` at 2aaccad into `_to_delete/`;
  keep `cells.tsv`, the `.vts` fields and `run_info.json` as the references.
  The pair cell needs the worker called with an explicit offset and no table; confirm that
  `run_micro_cells.py` can do this, otherwise the pair reference is the solver's coupled
  `P_asp` read before table composition.
- **WP1 Verbatim copy + thin API.** Copy the closure of files unchanged except import paths; write
  `api.py` from the body of `run_micro_task_phase_neutral` (film callback, parameters, settings,
  run, homogenise, asperity load, `q_re`, dQ, dP) without the cache, the tiers, the table and the
  env overrides. Write `fields.py`. Gate: parity on all references.
- **WP2 Prune dead physics (D1).** Remove spring, legacy and hertz contact, Roelands and
  compressibility stubs, `gauge_lift`, `c2`, export hooks inside the solver (`export_vtk`,
  `post_process`), `__main__` blocks, the xwarm/pair-warm methods. Gate: parity.
- **WP3 Prune the helpers.** Roughness: keep generation, evaluator, runtime, pair, the parts of
  stats they call; drop case/assets/prepare/targets/report/spec plumbing behind a small `Roughness`
  front end. Shear: keep the per-cell reduction, drop macro-frame batch code. Gate: parity.
- **WP4 Settings and results.** Collapse `MicroCase` + the three settings dataclasses +
  `MicroPhysicalParameters` into `MicroSettings` and `Zeta`; `MicroTaskResult` (about 90 positional
  fields) into `MicroResult`. Status codes for every NaN path. Gate: parity, plus a test that
  every NaN in a result has a non-OK status.
- **WP5 Tests, examples, README.** Copy the invariant tests that build a cell directly
  (`tests/ripple/test_ripple_micro_fenics.py`, `tests/elastic_contact/test_fft_micro_fenics.py`,
  `tests/micro_fluid_mean/test_fluid_mean_fenics.py`, the steady conservation checks) and adapt
  them to the new names. README: the model in one page, the call, the conventions (D8, D3), the
  provenance table.

Known traps to carry over:

- `microscale/src/functions/__init__.py` imports the legacy solver, and `pipeline/__init__.py`
  imports state/checkpoint/timing; neither `__init__` is copied.
- The quadrature degree pin (`utils/form_compiler.set_at_import(2)`, `pin_micro_worker`) must be
  applied before the first form is built, or results differ.
- Generated-surface cases at `h_min` near 1e-11 need the contact tolerance 1e-11; it becomes the
  `MicroSettings` default, not an env var.
- The parent's worker applies macro process settings in pool workers; the references must be made
  the way `run_micro_cells.py` makes them, and `api.py` must set the same dolfin parameters.

## Verification

- `tests/test_parity.py`: every reference cell, homogenised outputs and fields, relative 1e-12;
  exit 0 = PASS, 1 = FAIL, 2 = INCOMPLETE (a reference missing).
- The copied invariant tests (seconds to a few minutes each; the first run JIT-compiles forms).
- `examples/*.py` run end to end and the `.vts` / `.xdmf` open in ParaView (Rob checks).
- No run in this plan exceeds about 10 minutes; nothing goes to Calder.
- Expected repo size after WP4: to be measured, not promised; the copy at WP1 is about 12 500
  lines (the table above without the worker), and the pruning stages report their own counts.

## Waits on Rob

- Create the empty repo and folder (D13) and say its name; go-ahead for its first commit.
- Go-ahead to run WP0's reference cells in WSL (a few minutes, one core).
- Any of D5-D13 he wants changed.
- On approval this plan is filed in the new repo as `docs/notes/micro_standalone_plan_2026-11-11.md`
  and a row is added to `project_workstreams.md`; implementation is a separate session from a
  session-prompt doc.

## Not checked

- Whether `run_micro_cells.py` can run a pair cell at a set offset without the contact table (WP0).
- The line counts of what survives pruning.
- The step count of a full-window march (about 1e4 at n = 60, U = 0.02 m/s, t_end = 0.02 s) is an
  estimate from the CFL rule, not a measurement.
