"""A surface given as a height array: one Gaussian pocket, 100 nm deep, in an otherwise flat
surface, full film (no contact, no ripple).  The same array is solved once on the stationary
surface and once on the translating surface, over a 0.02 s window (the last 3 CFL steps), with
the end-state fields of each written for ParaView.

The array is ``z[j_y, i_x]`` [m] on the periodic cell, in the gap convention: the film is
``H + z``, so a pocket is positive.

Run in WSL from the repo root:

    source scripts/ehl_env.sh
    python examples/array_cell.py                # writes examples/output/pocket_stationary/ and pocket_translating/

exit 0 = both solves OK, 1 = a solve failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import MicroSettings, Roughness, Zeta, microsolver    # noqa: E402

OUT = ROOT / "examples" / "output"
L = 5e-6                                                    # cell size [m]


def pocket(n: int = 64, depth: float = 100e-9, width: float = 0.6e-6) -> np.ndarray:
    """``z[j_y, i_x]`` [m]: a Gaussian pocket of the given depth and standard width at the cell centre."""
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x)
    return depth * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2.0 * width ** 2))


def main() -> int:
    z = pocket()
    settings = MicroSettings(contact=None, ripple=None)     # full film
    zeta = Zeta(H=100e-9, P=1e5, Ux=0.02)                   # Ux is the mean (entrainment) velocity

    results = {}
    for name, moving in (("pocket_stationary", False), ("pocket_translating", True)):
        rough = Roughness.array(z, Lx=L, Ly=L, moving=moving)
        results[name] = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3,
                                    save_paraview=True, output_dir=OUT / name)

    print(f"{'':20s}{'status':>8s}{'dP [Pa]':>14s}{'Qx [m^2/s]':>16s}{'tau_fluid_x [Pa]':>18s}"
          f"{'p_min [Pa]':>14s}{'p_max [Pa]':>14s}{'h_min [m]':>14s}{'Newton':>8s}{'wall [s]':>10s}")
    for name, out in results.items():
        print(f"{name:20s}{out.status:>8s}{out.dP:14.3e}{out.Q[0]:16.6e}{out.tau_fluid[0]:18.6e}"
              f"{out.p_min:14.6e}{out.p_max:14.6e}{out.h_min:14.6e}"
              f"{out.info['newton_iters']:8d}{out.info['wall_time']:10.2f}")
    for name, out in results.items():
        print(f"ParaView files, {name}:", *out.info.get("paraview", []), sep="\n  ")
    return 0 if all(out.ok for out in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
