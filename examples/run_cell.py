"""One micro cell from Python: a generated Gaussian surface (Sa 40 nm) on the cup side, mixed
contact with ripple deformation, solved over a 0.02 s window and once steady, with the fields
written for ParaView.

Run in WSL from the repo root:

    source scripts/ehl_env.sh
    python examples/run_cell.py                  # writes examples/output/gensa40_h130/

exit 0 = both solves OK, 1 = a solve failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microcell import MicroSettings, Roughness, Zeta, microsolver    # noqa: E402

OUT = ROOT / "examples" / "output" / "gensa40_h130"


def main() -> int:
    rough = Roughness.generated(Sa=40e-9, seed=1000)        # 64 x 64 source grid on a 5 um cell
    settings = MicroSettings()                              # the GenSa40-ripple case's values
    zeta = Zeta(H=130e-9, P=40.8e6, Ux=0.02)                # Ux is the mean (entrainment) velocity

    transient = microsolver(zeta, t_end=0.02, roughness=rough, settings=settings, n_steps=3,
                            save_paraview=True, output_dir=OUT)
    steady = microsolver(zeta, roughness=rough, settings=settings, steady=True)

    print(f"{'':12s}{'status':>8s}{'dP [Pa]':>16s}{'P_asp [Pa]':>16s}{'contact_frac':>14s}"
          f"{'Newton':>8s}{'wall [s]':>10s}")
    for name, out in (("transient", transient), ("steady", steady)):
        print(f"{name:12s}{out.status:>8s}{out.dP:16.6e}{out.P_asp:16.6e}{out.contact_frac:14.6f}"
              f"{out.info['newton_iters']:8d}{out.info['wall_time']:10.2f}")
    print("fields:", ", ".join(sorted(transient.fields)))
    print("ParaView files:", *transient.info.get("paraview", []), sep="\n  ")
    return 0 if (transient.ok and steady.ok) else 1


if __name__ == "__main__":
    sys.exit(main())
