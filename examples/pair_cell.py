"""A pair of rough surfaces in mixed contact with ripple deformation.

One stationary and one translating generated Gaussian surface, each of Sa 10 nm / sqrt(2)
(composite Sa 10 nm).  The settings are those of the reference cell
``c4_wangpair10_h25_offset``; the translating surface starts displaced by ``offset``.  Solved
over a 0.02 s window (the last 3 CFL steps).  Prints the status, the homogenised outputs, the
window, the iteration counts and the wall time, and writes the fields at the end of the window
for ParaView.

Run in WSL from the repository root:

    source scripts/ehl_env.sh
    python examples/pair_cell.py                 # writes examples/output/pair_sa10_h25/

Exit 0 = the solve is OK, 1 = it failed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import (ContactSettings, MicroSettings, RippleSettings, Roughness, Zeta,    # noqa: E402
                       microsolver)

OUT = ROOT / "examples" / "output" / "pair_sa10_h25"


def main() -> int:
    sa = 10e-9 / math.sqrt(2.0)                             # Sa of each surface; the composite Sa of the pair is 10 nm
    rough = Roughness.pair(stationary=Roughness.generated(Sa=sa, seed=1000),
                           translating=Roughness.generated(Sa=sa, seed=1001))
    settings = MicroSettings(
        h_min=1.7825e-11,                                   # contact onset film [m]
        E_star=115384615384.61537,                          # reduced two-body modulus [Pa]
        level_constraint="fluid_mean_v1",
        pattern_flux_contact=False,
        abs_tolerance=1e-5,
        contact=ContactSettings(sigma_yield=450e6, mu=0.4),
        ripple=RippleSettings(tol=1e-4),
    )
    zeta = Zeta(H=25e-9, P=40.8e6, Ux=0.02)                 # Ux: mean (entrainment) velocity

    out = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3,
                      offset=(1.25e-6, 6.25e-7),            # initial displacement of the translating surface [m]
                      save_paraview=True, output_dir=OUT)

    print(f"status        {out.status} {out.reason}")
    print(f"dP            {out.dP:.6e} Pa        (fluid pressure per cell area - P)")
    print(f"P_asp         {out.P_asp:.6e} Pa        (asperity load per cell area)")
    print(f"contact_frac  {out.contact_frac:.6f}")
    print(f"Q             ({out.Q[0]:.6e}, {out.Q[1]:.6e}) m^2/s")
    print(f"tau_fluid     ({out.tau_fluid[0]:.6e}, {out.tau_fluid[1]:.6e}) Pa")
    print(f"tau_asperity  ({out.tau_asperity[0]:.6e}, {out.tau_asperity[1]:.6e}) Pa")
    print(f"h_min, h_max  {out.h_min:.6e}, {out.h_max:.6e} m")
    print(f"window        t_start {out.info['t_start']:.9e} s, dt {out.info['dt']:.6e} s, "
          f"{out.info['n_steps']} steps")
    print(f"iterations    Newton {out.info['newton_iters']}, contact {out.info['contact_iters']}, "
          f"ripple passes {out.info['ripple_passes']}")
    print(f"wall time     {out.info['wall_time']:.2f} s")
    print("ParaView files:", *out.info.get("paraview", []), sep="\n  ")
    return 0 if out.ok else 1


if __name__ == "__main__":
    sys.exit(main())
