"""Changing the equations: a source term added to the Reynolds equation, checked against its
analytic solution.  The worked example of the README section "Changing the equations".

With a source ``s`` [m/s] (film volume added per unit area per unit time) the cell equation is

    div( h^3 / (12 eta0) grad p ) = div( U h ) + dh/dt - s

so ``s`` enters wherever ``dh/dt`` does.  The solver's one place for that is
``_squeeze_residual_terms(dh_dt)``; this script wraps it from outside, without editing the
package (in the repo itself the same change is ``dh_dt - s_nd`` in ``_build_forms``).  The
solver's equation is non-dimensional: a rate in m/s is multiplied by ``t0 / H``.

The check: a flat film (no roughness, full-film mode), ``s = s0 sin(2 pi x / L)``.  The pressure
is then ``p = P + s0 (12 eta0 / H^3) (L / (2 pi))^2 sin(2 pi x / L)``, 1 MPa in amplitude here.

Two things the wrapper inherits from the squeeze term (see the README section):
the cell mean of the source is subtracted, and the term is off in a steady solve.

Run in WSL:

    source scripts/ehl_env.sh
    python examples/source_term.py

exit 0 = the solved pressure matches the analytic one to 1 %, 1 = it does not or the solve failed.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from microcell import MicroSettings, Roughness, Zeta, microsolver
from microcell.solver_base import MicroPhaseNeutralSolver

L = 5e-6            # cell size [m]
H = 100e-9          # film [m]
P = 40.8e6          # macro pressure [Pa]
ETA0 = 1e-3         # viscosity [Pa s]
AMPLITUDE = 1e6     # wanted pressure amplitude [Pa]
S0 = AMPLITUDE * H ** 3 / (12.0 * ETA0) * (2.0 * math.pi / L) ** 2     # source amplitude [m/s]


def add_source(solver_cls, source):
    """Make ``solver_cls`` solve with the source ``source(x, y)`` [m/s], a UFL expression of the
    unit-cell coordinates (``x``, ``y`` in 0..1).  Returns the method it replaced."""
    original = solver_cls._squeeze_residual_terms

    def with_source(self, dh_dt):
        from dolfin import Constant
        s_nd = Constant(self.t0 / float(self.zeta.H)) * source(self.x, self.y)    # m/s -> non-dimensional
        return original(self, dh_dt - s_nd)

    solver_cls._squeeze_residual_terms = with_source
    return original


def sine_source(x, y):
    from dolfin import Constant, pi, sin
    return Constant(S0) * sin(2.0 * pi * x)


def main() -> int:
    flat = Roughness.array(np.zeros((16, 16)), L, L)
    settings = MicroSettings(xmax=L, ymax=L, eta0=ETA0, mesh_n=64, contact=None, ripple=None)
    zeta = Zeta(H=H, P=P, Ux=0.02)

    without = microsolver(zeta, t_end=0.02, roughness=flat, settings=settings, n_steps=3)
    original = add_source(MicroPhaseNeutralSolver, sine_source)
    try:
        with_s = microsolver(zeta, t_end=0.02, roughness=flat, settings=settings, n_steps=3)
    finally:
        MicroPhaseNeutralSolver._squeeze_residual_terms = original
    if not (without.ok and with_s.ok):
        print(f"solve failed: {without.status} {without.reason} / {with_s.status} {with_s.reason}")
        return 1

    x = with_s.fields["x"]                                   # [m], arrays are [i_x, j_y]
    analytic = P + AMPLITUDE * np.sin(2.0 * math.pi * x / L)
    err = float(np.max(np.abs(with_s.fields["p_fluid"] - analytic))) / AMPLITUDE
    flat_dev = float(np.max(np.abs(without.fields["p_fluid"] - P)))

    print(f"source amplitude s0            {S0:.6e} m/s")
    print(f"without the source: max|p - P| {flat_dev:.3e} Pa")
    print(f"with the source:    p range    {with_s.fields['p_fluid'].min():.6e} .. "
          f"{with_s.fields['p_fluid'].max():.6e} Pa")
    print(f"max|p - analytic| / amplitude  {err:.3e}   (pass below 1e-2)")
    print(f"dP with the source             {with_s.dP:.3e} Pa   (a zero-mean source leaves the level alone)")
    return 0 if err < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
