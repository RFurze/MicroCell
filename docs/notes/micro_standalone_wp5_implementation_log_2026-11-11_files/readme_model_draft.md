# README draft for group 5: "The model" (not in the README yet)

Draft written before Rob's answers to Q1-Q4. It names no file that group 1 may rename. Each
statement carries the place in the code it was read from; the lines marked CHECK are still to be
read against the code before this goes into `README.md`.

## The model

One periodic rough cell, `xmax` by `ymax`, stands for one point of the bearing. The macro state
`zeta` gives the cell its mean film `H`, its pressure `P`, the mean (entrainment) velocity
`(Ux, Uy)`, the macro pressure gradient `(dpdx, dpdy)`, and the rate of each.

**Film.** The gap is `h = H + Hdot t + z + u_r + u_c`. `z` is the roughness, a gap perturbation
(`z < 0` is an asperity). `u_r` is the ripple deformation and `u_c` the contact deformation, both
zero when their part is off. The surfaces slide at `2 Ux`; the roughness on the moving surface
translates with it.
(`microcell/roughness/frontend.py:11`; solver `h_base_nd`, `_ripple_write_field`. CHECK: how `u_c`
enters the film, and the pattern speed for `velocity_ratio`.)

**Pressure.** The fluid pressure is split as `p = P + grad P . (x - x_c) + lambda + w`. `w` is
periodic with zero mean, and `lambda` is one number: the level of the cell's pressure against
the macro pressure. Both are unknowns.
(solver module docstring, `P_nd = P_re_nd + phi_c_nd + lambda_nd + w`.)

**Flow.** `w` solves the Reynolds equation on the cell, isoviscous with viscosity `eta0`:

    div( h^3 / (12 eta0) grad p ) = div( U h ) + dh/dt

The time derivative is second-order backward (BDF2) on a CFL step. A steady solve drops it.
(`_build_forms`: `inner(F_adv_nd - K_nd * gradP, gradv)`, `K_nd ~ h^3`, `F_adv_nd = U_nd h`,
`bdf_a0..a2`.)

**Cavitation.** A penalty: the term `penalty_gamma * spos(-p)` pushes negative pressure back to
zero. `spos` is a smoothed positive part of width `eps_smooth` (`spos_form`: `"sqrt"` or
`"spline"`). `lambda` is closed by the same term: `lambda` equals the cell mean of the penalty,
so it is zero in a cell that does not cavitate.
(`R_cav`, `negP_pos`; closure `G = BETA * lambda_nd - gamma_nd * mean(spos(-P_nd))`, `BETA = 1`.)

**Level constraint.** `"whole_cell"`: the mean of `p` over the whole cell is `P + lambda`.
`"fluid_mean_v1"`: the mean over the fluid region is `P + lambda` (mixed and contact cells).
(solver module comment on `level_constraint`.)

**Contact** (`settings.contact`, or `None` for full film). Where the rigid film falls below
`h_min` the surfaces touch. The contact pressure and `u_c` come from a periodic elastic
half-space solve (FFT, modulus `E_star`). The fluid equation is masked in the contact region.
The outputs are the asperity load per cell area `P_asp`, the contact fraction, the closure
tangent `k_asp = -dP_asp/dH`, and the asperity shear `mu * p_contact`.
(CHECK each against the contact class and the contact shear module.)

**Ripple** (`settings.ripple`, or `None`). `u_r` is the zero-mean periodic half-space response
(modulus `E_star`) to the cell's own fluid pressure. It is held fixed during a pass and updated
between passes until `max|H[s] - u_r| <= ripple.tol * H`, at most `ripple.maxpass` passes.
(`_ripple_run`.)

**Outputs.** All are the end state, in SI.

| Output | Meaning | Read from |
|---|---|---|
| `Q` | cell-mean flux [m^2/s] | `MicroResult` |
| `dQ` | `Q - Q_Reynolds`, with `Q_Reynolds = U H - H^3 grad P / (12 eta0)` at the end state | `reynolds_flux`, `api.py` |
| `dP` | `P_f - P` at the end state, `P_f` the fluid pressure over the cell area [Pa] | `api.py`, `dP=float(ledger["P_f"] - P_re)` |
| `P_asp` | asperity load per cell area [Pa] | `MicroResult` |
| `tau_fluid`, `tau_asperity`, `tau_total` | shear stress means [Pa] | `MicroResult` |

The frictional torque `\mathcal{T}` of a bearing is a macroscale integral of `tau_total`; this
package returns the cell's shear stress only.

## To add to "The call" in group 5

- D8: `Ux`, `Uy` are the mean (entrainment) velocities; the cell's surfaces slide at `2 Ux`.
- D3: the window. `n_steps=N` solves the last `N` CFL steps ending at `t_end`; `n_steps=None`
  marches from t = 0 (wording and cost after Rob's answer to Q4; measured numbers in
  `march_cost_base_d3b88ae.tsv`).
- The four examples and their output folders under `examples/output/`.
- The test files copied from the parent and the gate time (`gate_counts.tsv`).
