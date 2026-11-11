# MicroCell

The micro-cell solver of Rob Furze's transient multiscale mixed-lubrication model, as a
standalone Python call: one periodic rough cell, a macro state in, homogenised corrections
and fields out.  FEniCS 2019.1.0 (conda env `ehl-fenics`, `environment.yml`), run in WSL/Linux.

Extracted from `MultiscaleMixedBallinSocket` at commit `2aaccad`.  State: WP4 of
`docs/notes/micro_standalone_plan_2026-11-11.md` (the physics and hooks the plan dropped are
pruned from five solver files in WP2, the helpers the solve does not reach are pruned or
removed in WP3, and the settings and results are collapsed in WP4: each setting is defined
once, each output is written once; all three are marked in the provenance table; the other
files are verbatim copies).

## The call

```python
from microcell import microsolver, Zeta, Roughness, MicroSettings, ContactSettings, RippleSettings

zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)                 # SI; Ux, Uy are mean (entrainment) velocities
rough = Roughness.generated(Sa=40e-9, seed=1000)         # or .array(z, Lx, Ly), .pair(a, b), .cosine(Ah, kx, ky)
settings = MicroSettings(mesh_n=100, contact=ContactSettings(mu=0.8), ripple=RippleSettings())
out = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3)
out = microsolver(zeta, roughness=rough, steady=True)    # a steady solve, default settings

out.status, out.reason                                   # "OK", or a FAILED_* code and why
out.dQ, out.dP, out.P_asp, out.contact_frac              # homogenised outputs (18 named, SI)
out.fields["p_fluid"], out.fields["film"]                # (n+1, n+1) arrays [i_x, j_y], SI
out.info["newton_iters"], out.info["shear_asperity_status"]   # everything else, each once
```

- `steady=False` (default): the last `n_steps` CFL steps of the window ending at `t_end` are
  solved; the outputs are the end state.  `zeta` is the state at t = 0 of the window, so the
  end state is `zeta + rate * t_end`.  `steady=True`: one steady solve; rates and `t_end`
  are not used.
- The cell's surfaces slide at `2 Ux`.  Height arrays are gap perturbations `z[j_y, i_x]`
  (`z < 0` is an asperity).
- A stationary + translating pair takes one `offset=(o_x, o_y)` per call; loop to average.
- `save_paraview=True` writes `cell.vts` and `cell.xdmf` (+ `.h5`) of the end state (the files
  also carry `p_fluid_masked`, which `out.fields` does not).

Example: `python examples/run_cell.py`.

### Settings

Every setting is a field of `MicroSettings` (or of its `contact` / `ripple` part), defined once
in `microcell/api.py` with its one default: the parent's `GenSa40-ripple` case.
`MicroSettings.check()` refuses, before a mesh is built, every value the solver would refuse.

| Setting | Default | Meaning |
|---|---|---|
| `mesh_n` | 100 | cells per side of the micro mesh |
| `xmax`, `ymax` | 5e-6 | cell size [m] |
| `eta0` | 1e-3 | viscosity [Pa s] (isoviscous) |
| `penalty_gamma` | 1e10 | cavitation penalty |
| `h_min` | 1.0331e-11 | contact onset film [m] |
| `E_star` | 203775203775.2038 | reduced two-body modulus [Pa]: the one modulus of the contact law and the ripple |
| `delta_h` | 0.0 | width of the contact mask transition [m] (0: sharp) |
| `eps_solid` | 1e-8 | mobility factor inside contact |
| `eps_smooth` | 1e-3 | smoothing of the cavitation switch [Pa] |
| `spos_form` | `"sqrt"` | `"sqrt"` or `"spline"` |
| `level_constraint` | `"whole_cell"` | pressure-level constraint (`"fluid_mean_v1"`: over the fluid region) |
| `pattern_flux_contact` | True | flux pattern in the contact region |
| `jacobian_reg_eps` | -1e-7 | shift of the Jacobian (0: none) |
| `relaxation_parameter`, `max_iterations`, `abs_tolerance`, `rel_tolerance` | 1.0, 300, 1e-6, 1e-6 | Newton solve |
| `cfl_target` | 1.0 | CFL number of the transient steps |
| `roughness_velocity_ratio` | None | None: the roughness's own ratio |
| `contact` | `ContactSettings()` | `None` is the full-film mode |
| `contact.tol`, `.maxit` | 1e-11, 500 | contact solve |
| `contact.sigma_yield` | 250e6 | yield flag threshold [Pa] |
| `contact.mu` | 0.8 | asperity friction coefficient |
| `contact.k_asp_mode` | `"closure"` | `"off"` or `"closure"` |
| `ripple` | `RippleSettings()` | `None` turns the ripple deformation off |
| `ripple.tol`, `.maxpass`, `.omega` | 1e-6, 12, 1.0 | ripple fixed point |

Not settings: the Newton damping and backtracking values, `BETA` and the patch pin are fixed
values at the top of `microcell/solver_base.py`.

### Result

- `out.status` is `OK`, `FAILED_NEWTON`, `FAILED_RIPPLE`, `FAILED_CONTACT` or
  `FAILED_NONFINITE`.  A failed result has every homogenised output NaN and no fields; its
  `info` has the same entries as an OK result's.
- Each output is in one place: a named output, an array of `out.fields`, or an entry of
  `out.info` (statuses and tags, counters, the pressure ledger, the ripple and contact numbers,
  the sampled shear statistics, the call, and the field diagnostics in `info["fields"]`).
- On an OK result an output is NaN only where the table `microcell.NAN_STATUS` ties it to a
  per-quantity status in `info` that says why (`pressure_extrema_status`,
  `shear_extrema_status`, `shear_asperity_status`, `shear_total_status`: `no_fluid`,
  `phase_invalid`, `unavailable`).  A full-film cell's `tau_asperity` and `tau_total` are NaN
  with `unavailable`.  A NaN the table does not explain gives `FAILED_NONFINITE`.
- The parent's global-x shear envelopes are not produced: a standalone cell has no macro frame.

## Running

```bash
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
source scripts/ehl_env.sh
python -m pytest tests/ -q -p no:cacheprovider      # the parity gate and the other tests, about 20 s
python tests/test_parity.py                         # the gate as a script: exit 0 PASS, 1 FAIL, 2 INCOMPLETE
```

The parity gate (`tests/test_parity.py`) solves each reference cell in `tests/references/`
through `microsolver()` and compares every homogenised output and field with the parent's
value to a relative 1e-12.  The references were made from the parent at `2aaccad` by
`tools/make_references.py`.  `tests/test_settings.py` covers `MicroSettings.check()`;
`tests/test_results.py` covers the layout of a failed result and the NaN-to-status rule.

## Layout

| Path | Contents |
|---|---|
| `microcell/api.py` | `microsolver()`, `Zeta`, `MicroSettings`, `MicroResult` (new; written from the parent's worker `run_micro_task_phase_neutral`) |
| `microcell/fields.py` | field extraction and the ParaView writers (ported from the parent's `scripts/run/run_micro_cells.py`) |
| `microcell/solver_base.py` | `MicroPhaseNeutralSolver`: the full-film cell, the Newton solve, the time stepping, the ripple loop |
| `microcell/solver_mixed.py` | `MicroPhaseNeutralMixedSolver`: the contact mask and the fluid-mean level constraint |
| `microcell/solver_contact.py` | `MicroPhaseNeutralContactSolver`: the elastic contact solve in the cell |
| `microcell/samplers.py` | fluid pressure and traction samples, contact samples (was `micro_diagnostics.py` + `contact_sample.py`) |
| `microcell/film.py` | the film callbacks: the cosine cell and the generated surfaces (was `ideal_ht.py` + `surface_ht.py`) |
| `microcell/contact.py` | the periodic elastic (FFT) contact solve (was `contact.py` + `elastic_contact.py`) |
| `microcell/ripple.py` | the ripple operator and its acceptance checks |
| `microcell/pressure.py` | the pressure ledger and the pressure extrema |
| `microcell/shear.py` | shear traction: reduction, fluid, asperity and total (was `shear_reduce.py` + `fluid_shear.py` + `contact_shear.py`) |
| `microcell/roughness/` | `generation.py` and `quantiles.py` (the Gaussian IAAFT generator), `evaluator.py`, `runtime.py` and `pair.py` (the surface field and its motion), `frontend.py` (`Roughness`, new) |
| `microcell/form_compiler.py`, `microcell/process_settings.py` | the process state set before the first form |
| `tests/` | the parity gate and its references |
| `tools/` | `make_references.py` (run from the parent's root), `show_references.py` |
| `docs/notes/` | the plan, the session prompts and the implementation logs |

## Provenance

Every file below started as the parent's file at commit
`2aaccad2724c051c0f774cbe7b19b5183edf4720`, unchanged except for import paths (`ehl.` /
`microscale.src.functions.` / `utils.` -> `microcell.`).  The blob is the parent's git blob id
(first 10 characters); the full ids and the parent's line counts are in
`docs/notes/provenance.tsv`.

The five files marked "pruned" were cut in WP2 (11 Nov 2026) and are no longer the parent's
text: the spring film coupling (`k_spring`), the `legacy_patch_v1` and `hertz_summit_v1` contact laws, the
Roelands and compressibility stubs, `gauge_lift`, the `c2` cavitation form, the solver's export
hooks, the warm-start plumbing, the contact-table code and the solver's environment reads are
gone.  What went, group by group, with the parity after each:
`docs/notes/micro_standalone_wp2_implementation_log_2026-11-11.md`.

WP3 (11 Nov 2026) pruned the helpers.  The files marked "pruned in WP3" lost the code the
micro solve does not reach: the macro-frame and batch code of the shear modules, the remains
of the removed contact laws, the macro-side helpers of `pressure.py`, the surface store and its
`EHL_SURFACE_STORE` read, and in `roughness/` everything but the Gaussian IAAFT generator, the
Fourier evaluator, the in-memory surface field and the translating surface's kinematics
(`ideal_ht.py`: a docstring only).  The eight files marked "removed in WP3" are no longer in
the repo; their rows stay as the record of what was copied.  `Roughness.generated` calls the
generator directly and returns the same arrays, bit for bit.  What went, group by group, with
the parity after each: `docs/notes/micro_standalone_wp3_implementation_log_2026-11-11.md`.

WP4 (11 Nov 2026) collapsed the settings and the results.  The solver classes read `Zeta` and
`MicroSettings` directly (their own settings and parameter classes are gone), and the global-x
shear envelopes with their macro-frame rotation are removed.  That touched the files already
marked, and two that were still the parent's: `shear_reduce.py` (marked "pruned in WP4") and
`surface_ht.py` (marked "edited in WP4": two lines, the solver attribute it reads).  What
changed, group by group, with the parity after each:
`docs/notes/micro_standalone_wp4_implementation_log_2026-11-11.md`.

WP5 (11 Nov 2026) put the files into the plan's layout.  The three solver files are renamed.
Nine files are merged into four (`samplers.py`, `film.py`, `contact.py`, `shear.py`), each
part in the order of its rows below.  The code moved verbatim: only module docstrings (the old
ones are kept inside the new one), import lines and `__all__` changed.  The paragraphs above
use the file names of their time; the table gives each file's name now, and
`docs/notes/provenance.tsv` has both (`file`, `file_since_wp5`).  The check and the parity:
`docs/notes/micro_standalone_wp5_implementation_log_2026-11-11.md`.

Still the parent's text: `ripple.py`, `form_compiler.py`,
`process_settings.py`, `scripts/ehl_env.sh`, and the `elastic_contact.py` part of `contact.py`.

| File | Parent path | Parent blob |
|---|---|---|
| `microcell/solver_base.py` (pruned; renamed in WP5) | `microscale/src/functions/micro_phase_neutral_class.py` | ca5040d93a |
| `microcell/solver_mixed.py` (pruned; renamed in WP5) | `microscale/src/functions/micro_phase_neutral_class_mixed.py` | 8feecdbaff |
| `microcell/solver_contact.py` (pruned; renamed in WP5) | `microscale/src/functions/micro_phase_neutral_class_contact.py` | eea045cf74 |
| `microcell/samplers.py`, first part (pruned; merged in WP5) | `microscale/src/functions/micro_diagnostics.py` | a09479ce6b |
| `microcell/samplers.py`, second part (pruned in WP3; merged in WP5) | `microscale/src/functions/contact_sample.py` | 4f77a66bac |
| `microcell/film.py`, first part (pruned in WP3; merged in WP5) | `microscale/src/functions/ideal_ht.py` | f314c7af1b |
| `microcell/film.py`, second part (edited in WP4; merged in WP5) | `microscale/src/functions/surface_ht.py` | 6775f693ea |
| `microcell/contact.py`, first part (pruned, and again in WP3; merged in WP5) | `ehl/contact.py` | 1ac7dff6ec |
| `microcell/contact.py`, second part (merged in WP5) | `ehl/elastic_contact.py` | 95ed9d78bf |
| `microcell/ripple.py` | `ehl/ripple.py` | a869e99b30 |
| `microcell/pressure.py` (pruned in WP3) | `ehl/pressure.py` | 1c7da166fc |
| `microcell/shear.py`, first part (pruned in WP4; merged in WP5) | `ehl/shear_reduce.py` | ca9e9f9984 |
| `microcell/shear.py`, second part (pruned in WP3; merged in WP5) | `ehl/fluid_shear.py` | d927ff9942 |
| `microcell/shear.py`, third part (pruned in WP3; merged in WP5) | `ehl/contact_shear.py` | e659bf06ee |
| `microcell/roughness/__init__.py` (pruned in WP3) | `ehl/roughness/__init__.py` | 8b51a21a22 |
| `microcell/roughness/assets.py` (removed in WP3) | `ehl/roughness/assets.py` | bfe9e91419 |
| `microcell/roughness/case.py` (removed in WP3) | `ehl/roughness/case.py` | c17eb1b970 |
| `microcell/roughness/evaluator.py` (pruned in WP3) | `ehl/roughness/evaluator.py` | 7e9c2e37b5 |
| `microcell/roughness/generation.py` (pruned in WP3) | `ehl/roughness/generation.py` | e504b70d2a |
| `microcell/roughness/modes.py` (removed in WP3) | `ehl/roughness/modes.py` | f536399412 |
| `microcell/roughness/pair.py` (pruned in WP3) | `ehl/roughness/pair.py` | 4ccdc3b411 |
| `microcell/roughness/prepare.py` (removed in WP3) | `ehl/roughness/prepare.py` | a295e10e72 |
| `microcell/roughness/quantiles.py` (pruned in WP3) | `ehl/roughness/quantiles.py` | b6404d1533 |
| `microcell/roughness/report.py` (removed in WP3) | `ehl/roughness/report.py` | d07a6018a7 |
| `microcell/roughness/runtime.py` (pruned in WP3) | `ehl/roughness/runtime.py` | 5bf5215837 |
| `microcell/roughness/spec.py` (removed in WP3) | `ehl/roughness/spec.py` | f8692c4625 |
| `microcell/roughness/stats.py` (removed in WP3) | `ehl/roughness/stats.py` | 0fab7913cb |
| `microcell/roughness/targets.py` (removed in WP3) | `ehl/roughness/targets.py` | 63449dd2ef |
| `microcell/form_compiler.py` | `utils/form_compiler.py` | 583df10d92 |
| `microcell/process_settings.py` | `macroscale/src/functions/process_settings.py` | 3893447028 |
| `scripts/ehl_env.sh` | `scripts/ehl_env.sh` | 1c7dca4e8f |

Written from parent code at the same commit, not copied:

| File | Parent source | Parent blob |
|---|---|---|
| `microcell/api.py` | `scripts/run/run_microscale_phase_neutral.py` (`run_micro_task_phase_neutral`, `_build_phase_neutral_settings`), `scripts/run/micro_task_helpers.py` (`_parse_task`, `q_re`) | ecc05c13f6, 98ec12d1b5 |
| `microcell/fields.py` | `scripts/run/run_micro_cells.py` (`_extract_fields`, `_write_vts`, `_write_xdmf`) | c154ba5308 |
