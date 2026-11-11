# WP3: candidates for further pruning, left in place (11 Nov 2026)

State: `MicroCell` `main` at 6c047c1. Nothing here was changed. "Readers" are from a grep of
`microcell/`, `tests/`, `examples/` at 6c047c1 and from `inventory_g4.tsv`; none was run-tested.

## A. Dead by the inventory, outside the five groups

| # | Where | What | Evidence |
|---|---|---|---|
| A1 | `contact_sample.py:109` | `SAMPLER_DEFINITION` | class `none` in `inventory_g4.tsv` |
| A2 | `contact_sample.py:249` | `p1_lumped_weights` | class `none` in `inventory_g4.tsv` |

## B. Methods and branches (the inventory lists top-level names only)

| # | Where | What | Evidence |
|---|---|---|---|
| B1 | `roughness/evaluator.py` | `FourierSurface.from_modes`, `.identity`, `.kx`, `.ky`, `.variances` | no caller found by grep; `.source` is read by `tests/test_parity.py:377` only |
| B2 | `roughness/runtime.py:122` | `SurfaceField.h_geom_min_bound` | no caller found by grep |
| B3 | `shear_reduce.py:104`, `contact_sample.py:355` | `ShearReduction.as_dict`, `ContactTotalReduction.as_dict` | called only by each other |
| B4 | `contact_sample.py` | `ContactSamples.summary` | called once, to fill a report field; the reader of that field was not traced |
| B5 | `roughness/generation.py` | `iaaft(tol=...)` early-exit branch and `IAAFTResult.history`; `hpd_sorted_sample(family, shape)` arguments | `Roughness.generated` always passes `tol` 0 and `"gaussian"` |
| B6 | `roughness/generation.py` | `err_h_legacy`, `err_s_legacy` | computed by every `iaaft` call, read by nothing (removing them changes no array, saves two FFT-size passes per generated surface) |
| B7 | `fluid_shear.py`, `contact_shear.py`, `shear_reduce.py` | the global-x path: `rotation=` / `_frame`, the `gx_*` fields | every cell runs without a frame, so the fields are NaN with status `no_frame`; they are results fields, so this is WP4 |
| B8 | `ideal_ht.py` | `make_ideal_ht` two-surface and `static_moving` branches | not traced; the cosine cell uses one surface |
| B9 | `roughness/frontend.py` + `runtime.py` | merge `ArrayField` into `SurfaceField` | `SurfaceField` no longer has its own `__init__` |

## C. Not removable, as the brief asked to report

| # | Where | What | Why |
|---|---|---|---|
| C1 | `ideal_ht.py` | the descriptor tuple entries `(Ah, kx, ky, two_surface)` (the brief's "Hertz and summit descriptor entries") | read by `height` and by `micro_phase_neutral_class.py:1345` (`len(desc) >= 4 and bool(desc[3])`); only the docstring's Hertz wording was removed |
| C2 | `pressure.py` | `LEVEL_CONSTRAINT_RUN_DEFAULT` | imported by `micro_phase_neutral_class.py:57` |

## D. Stale wording (comments and docstrings only)

- `micro_phase_neutral_class_contact.py:93`: mentions the status `unsupported_model`.
- `contact_sample.py` module docstring: a sentence on `hertz_summit_v1`.
- `ideal_ht.py:5`, `:184`: `MicroMixedSolver`.
- `elastic_contact.py:10`: `ehl.contact.effective_modulus` (removed in group 2).
- Many docstrings cite the parent's paths (`ehl.…`, `docs/plans/…`, `run_micro`).
- `tools/make_references.py` still sets `EHL_SURFACE_STORE`: it runs in the parent, on the
  parent's modules (out of scope).

## E. Unused imports (pre-existing, outside WP3's scope)

`unused_imports_g4.txt`: `ideal_ht.py:190` `sin`; the `from dolfin import *` lines and
`MicroPhysicalParameters`, `MicroSolverSettings`, `PeriodicBoundary` in the solver class files.

## F. Renames and merges toward the plan's target layout

| Target | From |
|---|---|
| `shear.py` | `shear_reduce.py` (253 lines) + `fluid_shear.py` (148) + `contact_shear.py` (269) |
| `samplers.py` | `contact_sample.py` (395) with the samplers of `micro_diagnostics.py` |
| `film.py` | `ideal_ht.py` (275) + `surface_ht.py` (88) |
| `contact.py` | `contact.py` (12, two names) + `elastic_contact.py` (365) |
| `roughness/kinematics` into `runtime.py` | `roughness/pair.py` (29, one function) |
| `roughness/generation.py` | absorb `roughness/quantiles.py` (107, one public function) |

Line counts: `lines_after_6c047c1.tsv`.

## G. Carried from WP2 (its `pruning_candidates.md`)

`h_base_nd`, the `contact_model` field, the alias `MicroPhaseNeutralSolver`, and the rest of
that list are unchanged by WP3.

## H. For WP5

A second parent-derived surface test: reference c4's two stored surfaces are
`Roughness.generated(Sa=10e-9/sqrt(2), seed=1000)` and `seed=1001` to round-off, not bit for
bit (`generated_c4_seeds.tsv`). That was so before WP3; the gate's surface test covers c1 only.
