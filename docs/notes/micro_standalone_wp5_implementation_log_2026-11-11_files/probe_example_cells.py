"""WP5 group 4 preparation: candidate cells for the examples, each solved once and timed.
Edits nothing.  Writes probe_example_cells.tsv."""
import math
import time
from pathlib import Path

import numpy as np

from microcell import ContactSettings, MicroSettings, RippleSettings, Roughness, Zeta, microsolver

HERE = Path(__file__).resolve().parent


def dimple(n=64, L=5e-6, depth=100e-9, width=0.6e-6):
    x = (np.arange(n) + 0.0) * L / n
    X, Y = np.meshgrid(x, x)                       # z[j_y, i_x]
    r2 = (X - L / 2) ** 2 + (Y - L / 2) ** 2
    return depth * np.exp(-r2 / (2 * width ** 2))


def test_probe():
    rows = ["\t".join(["cell", "call", "status", "dP_Pa", "P_f_Pa", "P_asp_Pa", "contact_frac", "Qx_m2_s", "tau_fluid_x_Pa",
                       "p_min_Pa", "p_max_Pa", "h_min_m", "newton", "contact_it", "ripple_passes", "wall_s", "reason"])]

    def run(name, zeta, rough, st, **kw):
        t0 = time.time()
        out = microsolver(zeta, roughness=rough, settings=st, **kw)
        i = out.info
        call = ", ".join(f"{k}={v}" for k, v in kw.items())
        rows.append("\t".join(str(v) for v in [
            name, call, out.status, f"{out.dP:.6e}", f"{out.P_f:.6e}", f"{out.P_asp:.6e}", f"{out.contact_frac:.6f}",
            f"{out.Q[0]:.6e}", f"{out.tau_fluid[0]:.6e}", f"{out.p_min:.6e}", f"{out.p_max:.6e}", f"{out.h_min:.6e}",
            i["newton_iters"], i["contact_iters"], i["ripple_passes"], f"{time.time() - t0:.2f}",
            out.reason.replace("\t", " ")[:200]]))

    # pair: the Wang pair cell (reference c4): two generated surfaces, composite Sa 10 nm
    sa = 10e-9 / math.sqrt(2.0)
    pair = Roughness.pair(Roughness.generated(Sa=sa, seed=1000), Roughness.generated(Sa=sa, seed=1001))
    st_pair = MicroSettings(h_min=1.7825e-11, E_star=115384615384.61537, level_constraint="fluid_mean_v1",
                            pattern_flux_contact=False, abs_tolerance=1e-5,
                            contact=ContactSettings(sigma_yield=450e6, mu=0.4), ripple=RippleSettings(tol=1e-4))
    z_pair = Zeta(H=25e-9, P=40.8e6, Ux=0.02)
    run("pair simple zeta", z_pair, pair, st_pair, t_end=0.02, n_steps=3, offset=(1.25e-6, 6.25e-7))
    run("pair simple zeta no offset", z_pair, pair, st_pair, t_end=0.02, n_steps=3)
    run("pair H40", Zeta(H=40e-9, P=40.8e6, Ux=0.02), pair, st_pair, t_end=0.02, n_steps=3)

    # array: one Gaussian pocket in a flat surface
    full = MicroSettings(contact=None, ripple=None)
    z = dimple()
    for P in (1e5, 1e6):
        run(f"dimple stationary P={P:g}", Zeta(H=100e-9, P=P, Ux=0.02), Roughness.array(z, 5e-6, 5e-6), full,
            steady=True)
        run(f"dimple stationary P={P:g}", Zeta(H=100e-9, P=P, Ux=0.02), Roughness.array(z, 5e-6, 5e-6), full,
            t_end=0.02, n_steps=3)
        run(f"dimple translating P={P:g}", Zeta(H=100e-9, P=P, Ux=0.02), Roughness.array(z, 5e-6, 5e-6, moving=True),
            full, t_end=0.02, n_steps=3)
    run("dimple stationary P=1e5 mesh 64", Zeta(H=100e-9, P=1e5, Ux=0.02), Roughness.array(z, 5e-6, 5e-6),
        MicroSettings(mesh_n=64, contact=None, ripple=None), steady=True)

    # cosine: full film, squeeze
    st_c = MicroSettings(mesh_n=60, contact=None, ripple=None)
    zc = Zeta(H=120e-9, P=40.8e6, Ux=0.02, Hdot=-1e-7)
    for moving in (False, True):
        r = Roughness.cosine(Ah=42.3e-9, moving=moving)
        run(f"cosine moving={moving}", zc, r, st_c, t_end=0.02, n_steps=3)
        run(f"cosine moving={moving}", zc, r, st_c, steady=True)
    run("cosine low P", Zeta(H=120e-9, P=1e5, Ux=0.02), Roughness.cosine(Ah=42.3e-9), st_c, steady=True)
    (HERE / "probe_example_cells.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
