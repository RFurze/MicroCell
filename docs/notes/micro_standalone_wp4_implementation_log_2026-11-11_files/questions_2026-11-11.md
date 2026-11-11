# WP4: the one round of questions to Rob (11 Nov 2026)

Base 68bc829; gate at base `13 passed in 8.95s`. Evidence: the inventory TSVs beside this file
(`inventory_settings.tsv`, `inventory_results.tsv`, `inventory_nan_paths.tsv`,
`inventory_nan_observed.tsv`), made by `inventory.py` at the base, before any edit. "Refs" are the
8 reference cells c1 to c8. Parent files are read at 2aaccad. Option A is the recommendation.

## Evidence

Table Q1. Solver-side settings the plan's `MicroSettings` does not list.

| Setting | Value in all 8 refs | Set by a parent case file | In the three tests WP5 copies | Read at |
|---|---|---|---|---|
| `delta_h` (mask transition width, m) | 0.0 | no | 0.5e-9 twice, 0.0 twice | `micro_phase_neutral_class.py:715` |
| `eps_solid` (mobility in contact) | 1e-8 | no | 1e-6 twice, 1e-8 twice | `:789` |
| `eps_smooth` (cavitation smoothing, Pa) | 1e-3 | no | 1e-3 three times | `:728` |
| `beta` (closure parameter) | 1.0 | no | 1.0 once | `:1976` |
| `patch_pin` | None (never given) | no | not set | `_mixed.py:290` |
| `patch_pin_kappa` | 1.0 (never given) | no | not set | `_mixed.py:342` |
| 7 damping knobs (`adaptive_damping`, `omega_min`, `newton_backtrack_max`, `_tol`, `_warmup_iters`, `omega_recovery_factor`, `omega_cap_shrink_factor`) | the dataclass defaults; the API cannot set them | no | not set | `:315-322` |
| `print_newton`, `print_timing` | False; the API cannot set them | no | `print_newton=False` once | `:2087`, `:2101`, `:2669` |

"Set by a parent case file": grep of `cases/` for each name, no hit. The tests are
`tests/ripple/test_ripple_micro_fenics.py`, `tests/elastic_contact/test_fft_micro_fenics.py`,
`tests/micro_fluid_mean/test_fluid_mean_fenics.py`.

Table Q2. `MicroSettings` defaults that are not the same everywhere (the rest agree, or have one source).

| Field | Default now (GenSa40-ripple case) | Parent's general default | Refs |
|---|---|---|---|
| `level_constraint` | `whole_cell` | `fluid_mean_v1` (`ehl/spec.py:888`, run default since 2 Nov) | `whole_cell` in 7, `fluid_mean_v1` in c4 |
| `spos_form` | `sqrt` | `spline` (`ehl/spec.py:368`; `:1347` warns that every multiscale run keeps `sqrt`) | `sqrt` in 6, `spline` in c5, c8 |
| `jacobian_reg_eps` | -1e-7 | 1e-10 (`ehl/spec.py:698`; -1e-7 is the pressure-accounting recipe) | -1e-7 in 5, 1e-10 in c5, c6, c8 |
| `contact.k_asp_mode` | `closure` | `off` (`ehl/spec.py:340`) | `closure` in 5, `off` in c6 |
| `contact.tol` | 1e-11 | 1e-8 (the contact settings dataclass) | 1e-11 in all 6 contact cells |
| `contact.mu` | 0.8 | 0.8 (`ehl/spec.py:336`) | 0.8 in 5, 0.4 in c4 |
| `h_min` | 1.0331e-11 | none (a case value) | 1.0331e-11 in 4, 1.7825e-11 in c4, 4e-8 in c6 |
| `contact.E_star`, `ripple.E_star` | 203775203775.2038 | none (a case value) | that value in 5, 115384615384.61537 in c4 |

Table Q4. Every NaN seen in a result (`inventory_nan_observed.tsv`; 8 OK cells and 5 forced failures).

| NaN | Cells | Overall status | Status that reports it today |
|---|---|---|---|
| every homogenised output; `fields` empty | the 5 forced failures | `FAILED_*` | the overall status |
| `tau_asperity`, `tau_total` and 5 contact-shear fractions | c5, c8 (full film) | OK | `shear_asperity_status = unavailable` |
| the 11 global-x (`gx_*`) shear statistics | all 8 | OK | `no_frame` |
| field `p_fluid_masked` at the contact vertices | c2, c3, c7 (20 of 10201), c4 (225), c6 (233) | OK | none |

Not seen, by code reading (`inventory_nan_paths.tsv` N7 to N18): `p_max`, `p_min` with
`pressure_extrema_status = no_fluid` or `phase_invalid`; `Pst_fluid` when the fluid area is 0 (no
status); the shear means with `phase_invalid`; a non-finite solution that Newton accepted, and the
`info` and field numbers that would follow (no status).

## Questions

1. **The six `legacy` keys.** A: `delta_h`, `eps_solid`, `eps_smooth` become named `MicroSettings`
   fields (three new settings, defaults 0.0, 1e-8, 1e-3: WP5's tests set two of them off-default);
   `beta`, `patch_pin`, `patch_pin_kappa` become fixed constants in the solver. B: all six fixed
   constants. C: all six named fields. In every option the 7 damping knobs stay fixed constants and
   `print_newton` / `print_timing` stay off (the API could never set them).
2. **Defaults.** A: keep the GenSa40-ripple values (no default changes in WP4). B: as A but
   `level_constraint = "fluid_mean_v1"` (the parent's run default). No reference depends on a
   default: the gate passes every field.
3. **NaN on an OK result.** A: the overall `status` stays OK when the solve converged; one table in
   code ties every output that can be NaN to a per-quantity status (`pressure_extrema_status`,
   `shear_extrema_status`, `shear_asperity_status`, `shear_total_status`: `unavailable`, `no_fluid`,
   `phase_invalid`); `Pst_fluid` is tied to `no_fluid`; a NaN the table does not explain turns the
   result into a new overall status `FAILED_NONFINITE` (all outputs NaN, as the other failures);
   `p_fluid_masked` leaves `out.fields` (it stays in the ParaView files; the gate still checks it,
   derived from `p_fluid` and `chi_fluid`). B: as A but `p_fluid_masked` stays in `fields`, with
   the rule "NaN where `chi_fluid` = 0" in the table. C: no NaN on an OK result at all (full-film
   `tau_asperity` = 0 and `tau_total` = the sampled fluid mean; a cell with no fluid gets an
   overall `NO_FLUID`): this changes compared values in c5 and c8.
4. **Global-x shear (`gx_*`).** A: remove the 11 outputs, `shear_frame` and the `rotation=` path.
   The comparison count falls by 11 per cell (88 of 1188), all NaN in every reference. The three
   shear statuses then read the local reduction's own status (`ok` or `no_support`) where the
   reference has `no_frame`; the gate checks that through a stated map. B: keep them, NaN with
   `no_frame`.
5. **`n_steps=None` (D3).** A: leave it for WP5, with its own test (no reference cell covers it).
   B: build it in WP4. C: drop it from the API.
6. **Renames and merges toward the target layout.** A: the first commit of WP5, as a pure move
   (WP5 rewrites the provenance table and adapts the copied tests once). B: a pure-move commit at
   the end of WP4. C: not at all.
7. **`MicroSettings.rho0`.** No reader anywhere (`inventory_settings.tsv`: 0 reads; isoviscous,
   incompressible). A: remove it, as `nu` and `k_asp_exponent`. B: keep it.
8. **One modulus.** The solver refuses a ripple modulus that differs from the contact modulus
   (`micro_phase_neutral_class_contact.py:109-114`); they are equal in the four cells that have
   both. A: one `MicroSettings.E_star`; `ContactSettings.E_star` and `RippleSettings.E_star` go.
   B: keep both; `check()` refuses a difference when both are on.

## What does not wait

Group 1 (`ContactSettings.nu`, `.k_asp_exponent`), and the 1:1 part of group 2 (the solver reads
`Zeta`, `MicroSettings` and the call's values directly; the six legacy keys still travel in
`MicroSettings.legacy` until question 1 is answered).
