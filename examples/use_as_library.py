"""MicroCell used as a library from another script: import it, call it in a loop, get plain numbers back.

This file does not touch ``sys.path`` and can be copied to, and run from, any folder.  The package
only has to be importable, which either of these does:

    source /path/to/MicroCell/scripts/ehl_env.sh      # activates ehl-fenics, puts the repo on PYTHONPATH
    python /anywhere/use_as_library.py

    conda activate ehl-fenics && pip install -e /path/to/MicroCell      # once; then no PYTHONPATH (not tested)

``micro_model`` below is the pattern for a tuple-returning call: ``microsolver`` returns a
``MicroResult`` with named outputs, and a script picks the ones it wants.  The loop solves the
pendulum Sa 10 nm surface at four films (three CFL steps ending at 0.02 s each).

exit 0 = every solve OK, 1 = a solve failed.
"""
from __future__ import annotations

import sys

from microcell import MicroSettings, RippleSettings, Roughness, Zeta, microsolver

#: built once, reused by every call (the Cup90 pendulum Sa 10 nm recipe)
ROUGHNESS = Roughness.generated(Sa=10e-9, seed=1000)
SETTINGS = MicroSettings(abs_tolerance=1e-5, pattern_flux_contact=False, ripple=RippleSettings(tol=1e-4))


def micro_model(H, P, Ux, Uy=0.0, dpdx=0.0, dpdy=0.0, t_end=0.02, roughness=ROUGHNESS, settings=SETTINGS):
    """One micro cell.  Inputs in SI (``Ux``, ``Uy`` are the mean, entrainment, velocities).

    Returns ``(dQx, dQy, dP, P_asp, tau_x, tau_y, contact_frac, fields)``: the flux and pressure
    corrections [m^2/s, Pa], the asperity load per cell area [Pa], the total shear [Pa], the
    contact area fraction, and the dict of field arrays (``p_fluid``, ``film``, ... on the cell grid).
    Raises ``RuntimeError`` when the solve failed.  (In the full-film mode, ``MicroSettings(contact=None)``,
    the total shear is NaN by definition; return ``out.tau_fluid`` there.)"""
    zeta = Zeta(H=H, P=P, Ux=Ux, Uy=Uy, dpdx=dpdx, dpdy=dpdy)
    out = microsolver(zeta, t_end=t_end, roughness=roughness, settings=settings, n_steps=3)
    if not out.ok:
        raise RuntimeError(f"micro cell failed: {out.status} {out.reason}")
    return (out.dQ[0], out.dQ[1], out.dP, out.P_asp, out.tau_total[0], out.tau_total[1],
            out.contact_frac, out.fields)


def main() -> int:
    print(f"{'H [nm]':>8s}{'dQx [m^2/s]':>16s}{'dP [Pa]':>16s}{'P_asp [Pa]':>16s}{'tau_x [Pa]':>16s}"
          f"{'contact_frac':>14s}{'max p_fluid [Pa]':>18s}")
    try:
        for H in (60e-9, 40e-9, 30e-9, 26e-9):
            dQx, _dQy, dP, P_asp, tau_x, _tau_y, contact_frac, fields = micro_model(H=H, P=40.8e6, Ux=0.02)
            print(f"{H * 1e9:8.1f}{dQx:16.6e}{dP:16.6e}{P_asp:16.6e}{tau_x:16.6e}{contact_frac:14.6f}"
                  f"{fields['p_fluid'].max():18.6e}")
    except RuntimeError as err:
        print(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
