# MicroCell

The micro-cell solver of Rob Furze's transient multiscale mixed-lubrication model, as a
standalone Python call: one periodic rough cell, a macro state in, homogenised corrections
and fields out.  FEniCS 2019.1.0 (conda env `ehl-fenics`, `environment.yml`), run in WSL/Linux.

Extracted from `MultiscaleMixedBallinSocket` at commit `2aaccad`.  State: WP1 of
`docs/notes/micro_standalone_plan_2026-11-11.md` (the solver files are verbatim copies; pruning
and the settings collapse are later work packages).

## The call

```python
from microcell import microsolver, Zeta, Roughness, MicroSettings

zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)                 # SI; Ux, Uy are mean (entrainment) velocities
rough = Roughness.generated(Sa=40e-9, seed=1000)         # or .array(z, Lx, Ly), .pair(a, b), .cosine(Ah, kx, ky)
out = microsolver(zeta, t_end=0.02, roughness=rough, settings=MicroSettings(), n_steps=3)
out = microsolver(zeta, roughness=rough, steady=True)    # a steady solve

out.status, out.dQ, out.dP, out.P_asp, out.contact_frac  # homogenised outputs
out.fields["p_fluid"], out.fields["film"]                # (n+1, n+1) arrays [i_x, j_y], SI
```

- `steady=False` (default): the last `n_steps` CFL steps of the window ending at `t_end` are
  solved; the outputs are the end state.  `zeta` is the state at t = 0 of the window, so the
  end state is `zeta + rate * t_end`.  `steady=True`: one steady solve; rates and `t_end`
  are not used.
- The cell's surfaces slide at `2 Ux`.  Height arrays are gap perturbations `z[j_y, i_x]`
  (`z < 0` is an asperity).
- A stationary + translating pair takes one `offset=(o_x, o_y)` per call; loop to average.
- `MicroSettings()` defaults are the parent's `GenSa40-ripple` case.  `contact=None` is the
  full-film mode, `ripple=None` turns the ripple deformation off.
- `save_paraview=True` writes `cell.vts` and `cell.xdmf` (+ `.h5`) of the end state.
- A failed solve returns `status` `FAILED_NEWTON` / `FAILED_RIPPLE` / `FAILED_CONTACT` and NaN.

Example: `python examples/run_cell.py`.

## Running

```bash
cd /mnt/c/Users/Robin/Documents/code/RefactorRefactor/MicroCell
source scripts/ehl_env.sh
python -m pytest tests/ -q -p no:cacheprovider      # the parity gate, about 10 s
python tests/test_parity.py                         # the same as a script: exit 0 PASS, 1 FAIL, 2 INCOMPLETE
```

The parity gate solves each reference cell in `tests/references/` through `microsolver()` and
compares every homogenised output and field with the parent's value to a relative 1e-12.
The references were made from the parent at `2aaccad` by `tools/make_references.py`.

## Layout

| Path | Contents |
|---|---|
| `microcell/api.py` | `microsolver()`, `Zeta`, `MicroSettings`, `MicroResult` (new; written from the parent's worker `run_micro_task_phase_neutral`) |
| `microcell/fields.py` | field extraction and the ParaView writers (ported from the parent's `scripts/run/run_micro_cells.py`) |
| `microcell/roughness/frontend.py` | `Roughness` (new) |
| `microcell/*.py`, `microcell/roughness/*.py` | the solver, copied from the parent (table below) |
| `tests/` | the parity gate and its references |
| `tools/` | `make_references.py` (run from the parent's root), `show_references.py` |
| `docs/notes/` | the plan, the session prompts and the implementation logs |

## Provenance

Every file below is the parent's file at commit `2aaccad2724c051c0f774cbe7b19b5183edf4720`,
unchanged except for import paths (`ehl.` / `microscale.src.functions.` / `utils.` ->
`microcell.`).  The blob is the parent's git blob id (first 10 characters); the full ids and
line counts are in `docs/notes/provenance.tsv`.

| File | Parent path | Parent blob |
|---|---|---|
| `microcell/micro_phase_neutral_class.py` | `microscale/src/functions/micro_phase_neutral_class.py` | ca5040d93a |
| `microcell/micro_phase_neutral_class_mixed.py` | `microscale/src/functions/micro_phase_neutral_class_mixed.py` | 8feecdbaff |
| `microcell/micro_phase_neutral_class_contact.py` | `microscale/src/functions/micro_phase_neutral_class_contact.py` | eea045cf74 |
| `microcell/micro_diagnostics.py` | `microscale/src/functions/micro_diagnostics.py` | a09479ce6b |
| `microcell/contact_sample.py` | `microscale/src/functions/contact_sample.py` | 4f77a66bac |
| `microcell/ideal_ht.py` | `microscale/src/functions/ideal_ht.py` | f314c7af1b |
| `microcell/surface_ht.py` | `microscale/src/functions/surface_ht.py` | 6775f693ea |
| `microcell/contact.py` | `ehl/contact.py` | 1ac7dff6ec |
| `microcell/elastic_contact.py` | `ehl/elastic_contact.py` | 95ed9d78bf |
| `microcell/ripple.py` | `ehl/ripple.py` | a869e99b30 |
| `microcell/pressure.py` | `ehl/pressure.py` | 1c7da166fc |
| `microcell/shear_reduce.py` | `ehl/shear_reduce.py` | ca9e9f9984 |
| `microcell/fluid_shear.py` | `ehl/fluid_shear.py` | d927ff9942 |
| `microcell/contact_shear.py` | `ehl/contact_shear.py` | e659bf06ee |
| `microcell/roughness/__init__.py` | `ehl/roughness/__init__.py` | 8b51a21a22 |
| `microcell/roughness/assets.py` | `ehl/roughness/assets.py` | bfe9e91419 |
| `microcell/roughness/case.py` | `ehl/roughness/case.py` | c17eb1b970 |
| `microcell/roughness/evaluator.py` | `ehl/roughness/evaluator.py` | 7e9c2e37b5 |
| `microcell/roughness/generation.py` | `ehl/roughness/generation.py` | e504b70d2a |
| `microcell/roughness/modes.py` | `ehl/roughness/modes.py` | f536399412 |
| `microcell/roughness/pair.py` | `ehl/roughness/pair.py` | 4ccdc3b411 |
| `microcell/roughness/prepare.py` | `ehl/roughness/prepare.py` | a295e10e72 |
| `microcell/roughness/quantiles.py` | `ehl/roughness/quantiles.py` | b6404d1533 |
| `microcell/roughness/report.py` | `ehl/roughness/report.py` | d07a6018a7 |
| `microcell/roughness/runtime.py` | `ehl/roughness/runtime.py` | 5bf5215837 |
| `microcell/roughness/spec.py` | `ehl/roughness/spec.py` | f8692c4625 |
| `microcell/roughness/stats.py` | `ehl/roughness/stats.py` | 0fab7913cb |
| `microcell/roughness/targets.py` | `ehl/roughness/targets.py` | 63449dd2ef |
| `microcell/form_compiler.py` | `utils/form_compiler.py` | 583df10d92 |
| `microcell/process_settings.py` | `macroscale/src/functions/process_settings.py` | 3893447028 |
| `scripts/ehl_env.sh` | `scripts/ehl_env.sh` | 1c7dca4e8f |

Written from parent code at the same commit, not copied:

| File | Parent source | Parent blob |
|---|---|---|
| `microcell/api.py` | `scripts/run/run_microscale_phase_neutral.py` (`run_micro_task_phase_neutral`, `_build_phase_neutral_settings`), `scripts/run/micro_task_helpers.py` (`_parse_task`, `q_re`) | ecc05c13f6, 98ec12d1b5 |
| `microcell/fields.py` | `scripts/run/run_micro_cells.py` (`_extract_fields`, `_write_vts`, `_write_xdmf`) | c154ba5308 |
