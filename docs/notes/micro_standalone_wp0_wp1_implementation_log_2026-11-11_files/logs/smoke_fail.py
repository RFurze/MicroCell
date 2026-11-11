"""The failure path: tolerances no Newton solve can meet."""
import numpy as np
from microcell import MicroSettings, Roughness, Zeta, microsolver
rough = Roughness.generated(Sa=40e-9)
for steady in (False, True):
    out = microsolver(Zeta(H=130e-9, P=40.8e6, Ux=0.02), t_end=0.02, roughness=rough,
                      settings=MicroSettings(max_iterations=3, abs_tolerance=0.0, rel_tolerance=0.0), steady=steady)
    floats = [out.dP, out.Pst, out.P_f, out.Fst, out.P_asp, out.contact_frac, out.A_mech_frac, out.k_asp,
              out.p_max, out.p_min, out.h_max, out.h_min, *out.dQ, *out.Q, *out.tau_fluid]
    print("steady" if steady else "transient", "|", out.status, "| all NaN:", bool(np.all(np.isnan(floats))),
          "| fields", len(out.fields), "| extrema", out.info["pressure_extrema_status"],
          "| newton", out.info["newton_iters"], "| reason:", out.reason[:110])
