# MicroCell

The microscale cell solver of a transient multiscale mixed-lubrication model, as one Python
call: one periodic rough cell, a macroscale state in, homogenised corrections and fields out.
FEniCS 2019.1.0 is required, in Linux or WSL.  The code was extracted from
`MultiscaleMixedBallinSocket` (the parent) at commit `2aaccad`.

## Getting started

Linux or WSL2 is required, as FEniCS 2019.1.0 has no native Windows build, with `git` and
conda (Miniforge is enough).  The repository is private, so GitHub asks for a login.

```bash
git clone https://github.com/RFurze/MicroCell.git
cd MicroCell
conda env create -f environment.yml                # once per machine: creates the env ehl-fenics
source scripts/ehl_env.sh                          # in every new shell; sourced, not run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/ -q -p no:cacheprovider     # every test should pass
python examples/run_cell.py                        # one cell, transient and steady
```

`scripts/ehl_env.sh` activates `ehl-fenics`, changes to the repository root, puts it on
`PYTHONPATH` and sets the FEniCS form cache.  The exports keep each solve on one thread.  The
first run on a machine compiles the FEniCS forms and takes minutes; later test runs take
about 30 s.

`examples/run_cell.py` prints one row per solve and writes `cell.vts` and `cell.xdmf` to
`examples/output/gensa40_h130/` for ParaView.

## The model

One periodic rough cell, `xmax` by `ymax`, stands for one point of the bearing.  The
macroscale state `zeta` gives the cell its mean film `H`, its pressure `P`, the mean
(entrainment) velocity `U = (Ux, Uy)`, the macroscale pressure gradient `G = (dpdx, dpdy)`
and the rate of each.  One surface is at rest in the cell frame and the other slides at `2 U`.

**Film.**  `h = H + Hdot t + z + u_r + u_c`, where:

- `z` is the roughness, a gap perturbation (`z < 0` is an asperity).  The roughness of the
  stationary surface is fixed in the cell and that of the translating surface moves at `2 U`.
- `u_r` is the ripple deformation and `u_c` the contact deformation.  Both have zero mean,
  and each is zero when its part is off.

**Pressure.**  `p = P + Pdot t + G . (x - x_c) + lambda + w`, where `x_c` is the cell centre,
`w` is periodic and `lambda` is the level of the cell pressure against the macroscale
pressure.  `w` and `lambda` are the unknowns.

**Flow.**  The isoviscous Reynolds equation on the cell, with viscosity `eta0`:

    div( h^3 / (12 eta0) grad p ) = div( U h ) + dh/dt

The cell mean of `dh/dt` is subtracted, as a periodic problem is only solvable with zero net
source.  The time derivative is a backward difference on CFL steps: first order on the first
step and second order (BDF2) after it.  A transient solve starts from a seed solve at the
start of its window.  A steady solve drops `dh/dt` and the rates.

**Cavitation.**  A penalty term `penalty_gamma * spos(-p)` pushes negative pressure back to
zero, where `spos` is a smoothed positive part of width `eps_smooth`.  The `"spline"` form is
zero at positive pressure; the `"sqrt"` form leaves `eps_smooth^2 / (4 p)` there.  `lambda`
is proportional to the cell mean of `spos(-p)`, so a cell that does not cavitate has no level
shift from it.

**Level constraint.**  With `"whole_cell"`, `w` has zero mean over the cell and the cell-mean
pressure is `P + lambda`.  With `"fluid_mean_v1"`, the mean is taken over the fluid region.
The two are the same in a full-film cell.

**Contact** (`settings.contact`; `None` is the full-film mode).  The surfaces touch where the
film would fall below `h_min`.  One periodic elastic half-space problem (FFT, modulus
`E_star`) on the mesh nodes gives the contact pressure and `u_c`.  The model is elastic only:
`sigma_yield` sets `yield_flag` when the peak contact pressure passes it and never limits
the pressure.  The flow is switched off in the contact region (mobility multiplied by
`eps_solid`).  With `pattern_flux_contact`, the contact patch of a translating surface
carries its film with it.  The asperity shear is `mu` times the contact pressure, along the
sliding direction.

**Ripple** (`settings.ripple`; `None` is off).  `u_r` is the zero-mean periodic half-space
response (modulus `E_star`) to the fluid pressure of the cell.  It is held fixed during a
pass and updated between passes, for at most `ripple.maxpass` passes.  A pass is accepted
when `max|H[s] - u_r| <= ripple.tol * H` (`H[s]` is the response to the pressure of that
pass), the residual has at least halved since the previous pass, and the contact set at the
end state is unchanged.

**Outputs.**  All are the end state, in SI.

| Output | Meaning |
|---|---|
| `Q` | cell-mean flux [m^2/s] |
| `dQ` | `Q - Q_Reynolds`, where `Q_Reynolds = U H - H^3 G / (12 eta0)` is the smooth-film flux |
| `P_f` | fluid pressure integrated over the fluid region, per cell area [Pa] |
| `dP` | `P_f - P` [Pa] |
| `P_asp` | asperity load per cell area [Pa] |
| `k_asp` | closure tangent `-dP_asp/dH` [Pa/m] |
| `contact_frac`, `A_mech_frac` | obstructed area fraction; mechanical contact area fraction |
| `tau_fluid` | cell-area mean of `eta0 (2 U) / h + (h / 2) grad p` over the fluid region [Pa] |
| `tau_asperity`, `tau_total` | sampled asperity shear mean; fluid + asperity [Pa] |

The frictional torque $\mathcal{T}$ of a bearing is a macroscale integral of the shear
stress.  This package returns the shear stress of the cell only.

## The call

```python
from microcell import microsolver, Zeta, Roughness, MicroSettings, ContactSettings, RippleSettings

zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)                 # film [m], pressure [Pa], mean velocity [m/s]
rough = Roughness.generated(Sa=40e-9, seed=1000)         # or .array(z, Lx, Ly), .pair(a, b), .cosine(Ah, kx, ky)
settings = MicroSettings(mesh_n=100, contact=ContactSettings(mu=0.8), ripple=RippleSettings())
out = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3,
                  save_paraview=True, output_dir="output/my_cell")
out = microsolver(zeta, roughness=rough, steady=True)    # a steady solve, default settings

out.status, out.reason                                   # "OK", or a FAILED_* code and the reason
out.dQ, out.dP, out.P_asp, out.contact_frac              # homogenised outputs (18 named, SI)
out.fields["p_fluid"], out.fields["film"]                # (n+1, n+1) arrays [i_x, j_y], SI
out.info["newton_iters"], out.info["shear_asperity_status"]   # counters, statuses, diagnostics
```

`out.status` should be checked first: a failed solve returns NaN outputs and gives the reason
in `out.reason`.

- **Velocities.**  `Ux`, `Uy` are the mean (entrainment) velocities.  The moving surface
  slides at `2 Ux`, `2 Uy`.
- **Window.**  `zeta` is the state at t = 0 and the outputs are the end state,
  `zeta + rate * t_end`.  The CFL step is
  `dt_cfl = cfl_target * min(xmax, ymax) / mesh_n / (2 max|U|)` (`t_end / 10` when U = 0).
  - `n_steps=N` (default 3) solves the last `N` CFL steps ending at `t_end`.  If they do not
    fit in `[0, t_end]`, the solve starts at t = 0 and takes `ceil(t_end / dt_cfl)` equal steps.
  - `n_steps=None` marches from t = 0 in `ceil(t_end / dt_cfl)` equal steps, with no cap.
    The default settings at `t_end = 0.02` s and `Ux = 0.02` m/s give 16000 steps, estimated
    at about 11 minutes for the cell of `examples/run_cell.py`.
  - `info["n_steps"]`, `info["t_start"]` and `info["dt"]` report the window solved.
- **Steady.**  `steady=True` is one steady solve.  Rates, `t_end` and `n_steps` are not used.
- **Roughness.**  Height arrays are gap perturbations `z[j_y, i_x]` (`z < 0` is an asperity).
  `moving=False` puts a surface on the stationary side and `moving=True` on the translating
  side.  A stationary + translating pair takes one `offset=(o_x, o_y)` per call; an average
  over offsets is a loop in the calling script.
- **ParaView.**  `save_paraview=True` writes `cell.vts` and `cell.xdmf` (+ `.h5`) of the end
  state to `output_dir`.  The files also carry `p_fluid_masked`, which `out.fields` does not.

Each example prints its outputs with units.  The first four write ParaView files under
`examples/output/` (not tracked).

| Script | Cell |
|---|---|
| `examples/run_cell.py` | generated Gaussian surface, Sa 40 nm, mixed contact with ripple; transient and steady |
| `examples/pair_cell.py` | stationary + translating pair, composite Sa 10 nm, with an offset |
| `examples/array_cell.py` | height array (one pocket), full film; stationary and translating |
| `examples/cosine_cell.py` | analytic cosine cell, full film, with a film rate; transient and steady |
| `examples/use_as_library.py` | the package imported from a script in any folder: a function returning a tuple, called over four films |
| `examples/source_term.py` | a source term added to the Reynolds equation, checked against its analytic pressure |

### Settings

Every setting is a field of `MicroSettings`, or of its `contact` or `ripple` part, defined
once in `microcell/api.py`.  `MicroSettings.check()` refuses invalid values before a mesh is
built.

| Setting | Default | Meaning |
|---|---|---|
| `mesh_n` | 100 | mesh divisions per side of the cell |
| `xmax`, `ymax` | 5e-6 | cell size [m] |
| `eta0` | 1e-3 | viscosity [Pa s] (isoviscous) |
| `penalty_gamma` | 1e10 | cavitation penalty |
| `h_min` | 1.0331e-11 | contact onset film [m] |
| `E_star` | 203775203775.2038 | reduced two-body modulus [Pa], used by the contact and the ripple |
| `delta_h` | 0.0 | width of the contact mask transition [m] (0: sharp) |
| `eps_solid` | 1e-8 | mobility factor inside contact |
| `eps_smooth` | 1e-3 | smoothing width of the cavitation switch [Pa] |
| `spos_form` | `"sqrt"` | `"sqrt"` or `"spline"` |
| `level_constraint` | `"whole_cell"` | `"whole_cell"` or `"fluid_mean_v1"` |
| `pattern_flux_contact` | True | flux pattern in the contact region |
| `jacobian_reg_eps` | -1e-7 | shift of the Jacobian (0: none) |
| `relaxation_parameter`, `max_iterations`, `abs_tolerance`, `rel_tolerance` | 1.0, 300, 1e-6, 1e-6 | Newton solve; the parent's Sa 8, Sa 10 and Wang cases use `abs_tolerance=1e-5` |
| `cfl_target` | 1.0 | CFL number of the transient steps |
| `roughness_velocity_ratio` | None | None: the ratio of the roughness itself |
| `contact` | `ContactSettings()` | `None` is the full-film mode |
| `contact.tol`, `.maxit` | 1e-11, 500 | contact solve |
| `contact.sigma_yield` | 250e6 | yield flag threshold [Pa] |
| `contact.mu` | 0.8 | asperity friction coefficient |
| `contact.k_asp_mode` | `"closure"` | `"off"` or `"closure"` |
| `ripple` | `RippleSettings()` | `None` turns the ripple deformation off |
| `ripple.tol`, `.maxpass`, `.omega` | 1e-6, 12, 1.0 | ripple fixed point |

The Newton damping and backtracking values, `BETA` and the patch pin are fixed values at the
top of `microcell/solver_base.py`, not settings.

### Result

- `out.status` is `OK`, `FAILED_NEWTON`, `FAILED_RIPPLE`, `FAILED_CONTACT` or
  `FAILED_NONFINITE`.  A failed result has every homogenised output NaN and no fields.  Its
  `info` has the same entries as that of an OK result.
- Each output is in one place: a named output, an array of `out.fields`, or an entry of
  `out.info` (statuses, counters, the pressure ledger, the ripple and contact numbers, the
  sampled shear statistics, the call, and the field diagnostics in `info["fields"]`).
- On an OK result an output is NaN only where `microcell.NAN_STATUS` ties it to a status in
  `info` (`pressure_extrema_status`, `shear_extrema_status`, `shear_asperity_status`,
  `shear_total_status`, with values `no_fluid`, `phase_invalid`, `unavailable`).  In a
  full-film cell `tau_asperity` and `tau_total` are NaN with `unavailable`.  Any other NaN
  gives `FAILED_NONFINITE`.

## Changing the equations

The equations are UFL forms.  `microcell/solver_base.py` holds the full-film problem and
`microcell/solver_mixed.py` overrides the parts that change with contact.

| What | Where (method of the solver class) |
|---|---|
| Pressure `p`, mobility `K = omega h^3 mask`, advective flux `U h mask`, cavitation term `spos(-p)` | `_setup_pressure_expressions` (the mixed class replaces the transport and the cavitation weight) |
| Film `h`, the fluid indicator and the contact mask | `_setup_film_and_mask` |
| Residual `R = Reynolds + cavitation - squeeze` and its Jacobian | `_build_forms` (base only) |
| Squeeze term `dh/dt` and its mean | `_squeeze_residual_terms`, `_update_squeeze_mean` (base and mixed) |
| Scales of the non-dimensional problem | `_nondim_scalars` |
| Newton solve for `(w, lambda)` | `_newton_solve`, `_schur_step`, `_closure_G` |
| Homogenised outputs | `spatial_homogenisation` (base, mixed and contact) |

The Jacobian of the field residual is derived automatically (`derivative(R, w)`), so a new
term in `R` needs no hand-written linearisation with respect to `w`.

The forms are non-dimensional and the equation is solved as twice the one in "The model".
Lengths are scaled by the cell size, the film by `H`, the pressure by `p_scale` and time by
`t0 = 2 Lx / Uref`, with `U_nd = 2 U / Uref` and `omega = H^2 p_scale / (6 eta0 Lx Uref)`.
A rate in m/s that stands beside `dh/dt` is multiplied by `t0 / H`.

**Worked example: a source term.**  With a source `s` [m/s] the equation is

    div( h^3 / (12 eta0) grad p ) = div( U h ) + dh/dt - s

The change is `dh_dt - s_nd` where `_build_forms` passes `dh_dt` to
`_squeeze_residual_terms`, with `s_nd = s t0 / H` a UFL expression of the unit-cell
coordinates `self.x`, `self.y`.  `examples/source_term.py` does this from outside the package
and agrees with the analytic pressure of a sinusoidal source in a flat film to better than
1e-6 of the amplitude.

The following points apply to any change:

- **Solvability.**  The cell is periodic, so the net source over the cell must be zero.  The
  squeeze term has its cell mean subtracted at every step (`_update_squeeze_mean`), and a
  source routed through it is treated the same.  A source added elsewhere needs its own
  subtraction.  The mean part of a source belongs to the macroscale equation.
- **Steady solves.**  The squeeze term is switched off in a steady solve (`transient_on`),
  and a source routed through it is switched off with it.
- **Pressure-dependent terms.**  `lambda` is solved beside `w` by a hand-written Schur step.
  Its derivative forms (`_c_form`, `_g_form_raw`, `_closure_G`, `_closure_d_scalar`) cover
  the cavitation term only.  A term that depends on the pressure value, such as
  compressibility or a pressure-dependent viscosity, must be added to them.  A term that
  depends on the film, the position, the time or `grad p` need not.
- **Contact.**  The mixed class overrides `_setup_pressure_expressions`,
  `_squeeze_residual_terms`, `_update_squeeze_mean` and the closure, so a change to one of
  those is made in both classes.
- **Outputs.**  `Q`, `dQ`, `dP` and the shear are computed from the standard terms in
  `spatial_homogenisation` and do not include a new term.
- **Time-dependent coefficients** are dolfin `Constant`s assigned in `_set_time_level`.
- **Parity.**  A change to an existing form moves the reference cells.  The existing path
  should stay identical when the new term is off; `tests/test_parity.py` checks this.

## Tests

```bash
python -m pytest tests/ -q -p no:cacheprovider      # all tests, about 30 s
python tests/test_parity.py                         # the parity gate alone: exit 0 PASS, 1 FAIL, 2 INCOMPLETE
```

The parity gate solves each reference cell in `tests/references/` through `microsolver()`
and compares every homogenised output and field with the value from the parent to a relative
1e-12.  The references were made from the parent at `2aaccad` by `tools/make_references.py`.

## Layout

| Path | Contents |
|---|---|
| `microcell/api.py` | `microsolver()`, `Zeta`, `MicroSettings`, `MicroResult` |
| `microcell/fields.py` | field extraction and the ParaView writers |
| `microcell/solver_base.py` | `MicroPhaseNeutralSolver`: the full-film cell, the Newton solve, the time stepping, the ripple loop |
| `microcell/solver_mixed.py` | `MicroPhaseNeutralMixedSolver`: the contact mask and the fluid-mean level constraint |
| `microcell/solver_contact.py` | `MicroPhaseNeutralContactSolver`: the elastic contact solve in the cell |
| `microcell/samplers.py` | fluid pressure, traction and contact samples |
| `microcell/film.py` | film callbacks: the cosine cell and the generated surfaces |
| `microcell/contact.py` | periodic elastic (FFT) contact solve |
| `microcell/ripple.py` | ripple operator and its acceptance checks |
| `microcell/pressure.py` | pressure ledger and pressure extrema |
| `microcell/shear.py` | shear traction: reduction, fluid, asperity and total |
| `microcell/roughness/` | the Gaussian surface generator, the surface field and its motion, and `Roughness` |
| `microcell/form_compiler.py`, `microcell/process_settings.py` | process state set before the first form |
| `tests/` | the parity gate, its references and the other tests |
| `examples/` | example scripts; `examples/output/` is not tracked |
| `tools/` | `make_references.py` (run from the root of the parent), `show_references.py` |
| `docs/notes/` | the plan and the implementation logs |
