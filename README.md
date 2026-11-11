# MicroCell

The micro-cell solver of Rob Furze's transient multiscale mixed-lubrication model, as a
standalone Python call: one periodic rough cell, a macro state in, homogenised corrections
and fields out.  FEniCS 2019.1.0 (conda env `ehl-fenics`, `environment.yml`), run in WSL/Linux.

Extracted from `MultiscaleMixedBallinSocket` at commit `2aaccad`.  State: WP5, the last work
package of `docs/notes/micro_standalone_plan_2026-11-11.md`.  WP2 pruned the physics and hooks
the plan dropped.  WP3 pruned or removed the helpers the solve does not reach.  WP4 collapsed
the settings and results: each setting is defined once, each output is written once.  WP5 put
the files into the plan's layout and added the march from t = 0, the parent's invariant tests
and the examples.  The provenance table marks what each work package did to each file.

## Getting started

**What you need.**  Linux, or WSL2 on Windows (legacy FEniCS 2019.1.0 has no native Windows
build); `git`; and conda (Miniforge is enough).  Every command below is typed in a Linux or
WSL shell.

**1. Download the repository.**

```bash
git clone https://github.com/RFurze/MicroCell.git
cd MicroCell
```

The repository is private: GitHub asks for a login with access to it.  With an SSH key on
your GitHub account, `git clone git@github.com:RFurze/MicroCell.git` does the same.

**2. Create the environment** (once per machine; it downloads FEniCS from conda-forge).

```bash
conda env create -f environment.yml
```

This makes a conda environment named `ehl-fenics`: Python 3.11, FEniCS 2019.1.0, numpy, scipy,
h5py, matplotlib and pytest.

**3. Start the environment** (in every new shell).

```bash
source scripts/ehl_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The script must be sourced, not run.  It activates `ehl-fenics`, changes to the repo root, puts
the repo on `PYTHONPATH` and sets the FEniCS form cache (`~/.cache/dijitso`).  It prints
`ehl env ready` with the paths it set.  The second line keeps each solve on one thread.
Without the script, `conda activate ehl-fenics` followed by `export PYTHONPATH=$PWD` from the
repo root does the same job.

**4. Check the install.**

```bash
python -m pytest tests/ -q -p no:cacheprovider
```

Every test should pass.  The first run on a machine compiles the FEniCS forms and takes
minutes; later runs take about 30 s.

**5. Launch a case.**

```bash
python examples/run_cell.py
```

This solves one cell (a generated Gaussian surface, Sa 40 nm, in mixed contact) over a 0.02 s
window and once steady.  It prints one row per solve (status, `dP`, `P_asp`, contact fraction,
Newton iterations, wall time) and exits 0 when both are OK.  The fields are written to
`examples/output/gensa40_h130/`: open `cell.vts` or `cell.xdmf` in ParaView.  The other
examples are listed under "The call".

**6. Run your own case.**  Save this as `my_cell.py` in the repo root and run `python my_cell.py`:

```python
from microcell import microsolver, Zeta, Roughness, MicroSettings

zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)              # film [m], pressure [Pa], mean velocity [m/s]
rough = Roughness.generated(Sa=40e-9, seed=1000)      # or .array(...), .pair(...), .cosine(...)
out = microsolver(zeta, t_end=0.02, roughness=rough, settings=MicroSettings(),
                  save_paraview=True, output_dir="output/my_cell")   # output/ is not tracked

print(out.status, out.reason)
print("dQ", out.dQ, "dP", out.dP, "P_asp", out.P_asp, "contact_frac", out.contact_frac)
```

Check `out.status` first: a failed solve returns NaN outputs and says why in `out.reason`.
The inputs, the conventions and every setting are described under "The call".

## The model

One periodic rough cell, `xmax` by `ymax`, stands for one point of the bearing.  The macro
state `zeta` gives the cell its mean film `H`, its pressure `P`, the mean (entrainment)
velocity `U = (Ux, Uy)`, the macro pressure gradient `G = (dpdx, dpdy)`, and the rate of each.
One surface is at rest in the cell frame and the other slides at `2 U`.

**Film.**  `h = H + Hdot t + z + u_r + u_c`.

- `z` is the roughness, a gap perturbation (`z < 0` is an asperity).  The roughness of the
  stationary surface is fixed in the cell.  The roughness of the translating surface moves
  with it, at `2 U`.
- `u_r` is the ripple deformation and `u_c` the contact deformation.  Both have zero mean,
  and each is zero when its part is off.

**Pressure.**  `p = P + Pdot t + G . (x - x_c) + lambda + w`, with `x_c` the cell centre.
`w` is periodic.  `lambda` is one number: the level of the cell's pressure against the macro
pressure.  Both are unknowns.

**Flow.**  The Reynolds equation on the cell, isoviscous with viscosity `eta0`:

    div( h^3 / (12 eta0) grad p ) = div( U h ) + dh/dt

The cell mean of `dh/dt` is subtracted, which a periodic problem needs to be solvable.  The
time derivative is a backward difference on CFL steps: first order on the first step,
second order (BDF2) after it.  A transient solve starts from a seed solve at the start of
its window.  A steady solve drops `dh/dt` and the rates.

**Cavitation.**  A penalty: the term `penalty_gamma * spos(-p)` pushes negative pressure
back to zero.  `spos` is a smoothed positive part of width `eps_smooth`.  The `"spline"` form
is zero at positive pressure; the `"sqrt"` form leaves `eps_smooth^2 / (4 p)` there.  `lambda`
is closed by the same term: it is proportional to the cell mean of `spos(-p)`, so a cell that
does not cavitate has no level shift from it.

**Level constraint.**  `"whole_cell"`: `w` has zero mean over the cell, so the cell-mean
pressure is `P + lambda`.  `"fluid_mean_v1"`: the mean is taken over the fluid region, so
the fluid-region mean pressure is `P + lambda`.  The two are the same in a full-film cell.

**Contact** (`settings.contact`; `None` is the full-film mode).  The surfaces touch where
the gap would fall below `h_min`.  One periodic elastic half-space problem (FFT, modulus
`E_star`) on the mesh nodes gives the contact pressure and `u_c`.  It is elastic only:
`sigma_yield` sets `yield_flag` when the peak contact pressure passes it and never limits
the pressure.  The flow is switched off in the contact region (its mobility is multiplied by
`eps_solid`).  With `pattern_flux_contact`, the contact patch of a translating surface
carries its film with it.  The asperity shear is `mu` times the contact pressure, along the
sliding direction.

**Ripple** (`settings.ripple`; `None` is off).  `u_r` is the zero-mean periodic half-space
response (modulus `E_star`) to the cell's own fluid pressure.  It is held fixed during a
pass and updated between passes.  A pass is accepted when `max|H[s] - u_r| <= ripple.tol * H`
(`H[s]` the response to that pass's pressure), the residual has at least halved since the
pass before, and the contact set at the end state did not change.  There are at most
`ripple.maxpass` passes.

**Outputs.**  All are the end state, in SI.

| Output | Meaning |
|---|---|
| `Q` | cell-mean flux [m^2/s] |
| `dQ` | `Q - Q_Reynolds`, with `Q_Reynolds = U H - H^3 G / (12 eta0)` of the smooth film at the end state |
| `P_f` | fluid pressure integrated over the fluid region, per cell area [Pa] |
| `dP` | `P_f - P` at the end state [Pa] |
| `P_asp` | asperity load per cell area [Pa] |
| `k_asp` | closure tangent `-dP_asp/dH` [Pa/m] |
| `contact_frac`, `A_mech_frac` | obstructed area fraction; mechanical contact area fraction |
| `tau_fluid` | mean over the cell area of `eta0 (2 U) / h + (h / 2) grad p` in the fluid region [Pa] |
| `tau_asperity`, `tau_total` | sampled asperity shear mean, and fluid + asperity [Pa] |

The frictional torque $\mathcal{T}$ of a bearing is a macroscale integral of the shear
stress.  This package returns the cell's shear stress only.

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

Conventions:

- **Velocities.**  `Ux`, `Uy` are the mean (entrainment) velocities.  The moving surface
  slides at `2 Ux`, `2 Uy`.
- **Window.**  `zeta` is the state at t = 0, so the end state is `zeta + rate * t_end`.  The
  outputs are the end state.  The CFL step is
  `dt_cfl = cfl_target * min(xmax, ymax) / mesh_n / (2 max|U|)` (`t_end / 10` when U = 0).
  - `n_steps=N` (default 3): the last `N` CFL steps ending at `t_end` are solved, from
    `t_start = t_end - N dt_cfl`.  If they do not fit in `[0, t_end]`, the solve starts at
    t = 0 and takes `ceil(t_end / dt_cfl)` equal steps.
  - `n_steps=None`: the march from t = 0, in `ceil(t_end / dt_cfl)` equal steps.
    `info["n_steps"]` reports the count.  There is no cap, and the cost grows with the
    count: the default settings at `t_end = 0.02` s and `Ux = 0.02` m/s give 16000 steps,
    about 11 minutes for the cell of `examples/run_cell.py` (an estimate from runs of 11 and
    21 steps, about 0.04 s per step).
  - `info["t_start"]` and `info["dt"]` report the window solved.
- **Steady.**  `steady=True`: one steady solve.  Rates, `t_end` and `n_steps` are not used.
- **Roughness.**  Height arrays are gap perturbations `z[j_y, i_x]` (`z < 0` is an asperity).
  `moving=False` puts a surface on the stationary side, `moving=True` on the translating
  side.  A stationary + translating pair takes one `offset=(o_x, o_y)` per call; loop to
  average.
- **ParaView.**  `save_paraview=True` writes `cell.vts` and `cell.xdmf` (+ `.h5`) of the end
  state (the files also carry `p_fluid_masked`, which `out.fields` does not).

Examples, one per kind of roughness.  Each prints its outputs with units and writes ParaView
files under `examples/output/` (not tracked):

| Script | Cell | Output folder |
|---|---|---|
| `examples/run_cell.py` | a generated Gaussian surface, Sa 40 nm, mixed contact with ripple; transient and steady | `gensa40_h130/` |
| `examples/pair_cell.py` | a stationary + translating pair, composite Sa 10 nm, with an offset | `pair_sa10_h25/` |
| `examples/array_cell.py` | a height array (one pocket), full film; on the stationary and on the translating surface | `pocket_stationary/`, `pocket_translating/` |
| `examples/cosine_cell.py` | the analytic cosine cell, full film, with a film rate; transient and steady | `cosine_h120/` |

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

From the repo root, with the environment started as in "Getting started":

```bash
source scripts/ehl_env.sh
python -m pytest tests/ -q -p no:cacheprovider      # the parity gate and the other tests, about 30 s
python tests/test_parity.py                         # the gate as a script: exit 0 PASS, 1 FAIL, 2 INCOMPLETE
python examples/run_cell.py                         # each example: under 10 s, exit 0 OK, 1 a solve failed
```

The parity gate (`tests/test_parity.py`) solves each reference cell in `tests/references/`
through `microsolver()` and compares every homogenised output and field with the parent's
value to a relative 1e-12.  The references were made from the parent at `2aaccad` by
`tools/make_references.py`.

The other tests:

| File | Covers |
|---|---|
| `tests/test_settings.py` | `MicroSettings.check()` |
| `tests/test_results.py` | the layout of a failed result and the NaN-to-status rule |
| `tests/test_march.py` | `n_steps=None`: the march from t = 0 |
| `tests/test_micro_phase_neutral_steady.py` | the parent's steady checks of the full-film cell (the plan's "steady conservation checks") |
| `tests/test_fft_micro_fenics.py` | the parent's checks of the elastic contact solve in the cell |
| `tests/test_fluid_mean_fenics.py` | the parent's checks of the fluid-mean level constraint |
| `tests/test_ripple_micro_fenics.py` | the parent's checks of the ripple deformation |

The four files of the parent's tests are copied with the settings renamed and the expected
numbers and tolerances unchanged.  They set the parent's test-time values explicitly
(`tests/parent_fixtures.py`).  The tests of the parent that need something this repo does
not have are left out; the list, with reasons, is in
`docs/notes/micro_standalone_wp5_implementation_log_2026-11-11.md`.  The two tests marked
`slow` run in the default suite.

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
| `tests/` | the parity gate and its references, and the other tests (see Running) |
| `examples/` | one script per kind of roughness; `examples/output/` is not tracked |
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
`docs/notes/provenance.tsv` has both (`file`, `file_since_wp5`).  After the move, WP5 removed
names that nothing reads: the Newton prints, the timing summary and the Newton trace
(`solver_base.py`), a back-compat alias (`solver_mixed.py`), `SAMPLER_DEFINITION` and an
`as_dict` (`samplers.py`), an `as_dict` (`shear.py`), `FLUID_DEFORMATION_MODES` and
`deformation_model_tag` (`ripple.py`, marked "pruned in WP5"), `h_geom_min_bound`
(`roughness/runtime.py`), the two legacy error metrics of the generator
(`roughness/generation.py`), and three unused imports.  The checks and the parity:
`docs/notes/micro_standalone_wp5_implementation_log_2026-11-11.md`.

Still the parent's text: `form_compiler.py`, `process_settings.py`, `scripts/ehl_env.sh`, and
the `elastic_contact.py` part of `contact.py`.

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
| `microcell/ripple.py` (pruned in WP5) | `ehl/ripple.py` | a869e99b30 |
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
