"""The analytic cosine cell: amplitude 42.3 nm, one period per cell in x and y, on the stationary
surface, full film (no contact, no ripple), with the film closing at 100 nm/s.  Solved over a
0.02 s window (the last 3 CFL steps) and once steady (a steady solve carries no rates), with the
transient end-state fields written for ParaView.

Run in WSL from the repo root:

    source scripts/ehl_env.sh
    python examples/cosine_cell.py               # writes examples/output/cosine_h120/

exit 0 = both solves OK, 1 = a solve failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import MicroSettings, Roughness, Zeta, microsolver    # noqa: E402

OUT = ROOT / "examples" / "output" / "cosine_h120"


def main() -> int:
    rough = Roughness.cosine(Ah=42.3e-9, kx=1, ky=1)        # stationary pattern (moving=False)
    settings = MicroSettings(mesh_n=60, contact=None, ripple=None)      # full film; the cell is settings.xmax x ymax
    zeta = Zeta(H=120e-9, P=40.8e6, Ux=0.02, Hdot=-1e-7)    # Ux is the mean (entrainment) velocity

    transient = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3,
                            save_paraview=True, output_dir=OUT)
    steady = microsolver(zeta, roughness=rough, settings=settings, steady=True)

    print(f"{'':12s}{'status':>8s}{'dP [Pa]':>14s}{'Qx [m^2/s]':>16s}{'tau_fluid_x [Pa]':>18s}"
          f"{'p_min [Pa]':>14s}{'p_max [Pa]':>14s}{'h_min [m]':>14s}{'Newton':>8s}{'wall [s]':>10s}")
    for name, out in (("transient", transient), ("steady", steady)):
        print(f"{name:12s}{out.status:>8s}{out.dP:14.3e}{out.Q[0]:16.6e}{out.tau_fluid[0]:18.6e}"
              f"{out.p_min:14.6e}{out.p_max:14.6e}{out.h_min:14.6e}"
              f"{out.info['newton_iters']:8d}{out.info['wall_time']:10.2f}")
    print("ParaView files:", *transient.info.get("paraview", []), sep="\n  ")
    return 0 if (transient.ok and steady.ok) else 1


if __name__ == "__main__":
    sys.exit(main())
