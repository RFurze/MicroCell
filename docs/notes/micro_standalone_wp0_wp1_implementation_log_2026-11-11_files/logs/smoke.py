"""Smoke checks outside the gate: the failure path, the generated pair members, a moving cosine."""
import sys, dataclasses
sys.path.insert(0, "tests")
import numpy as np
import test_parity as tp
from microcell import MicroSettings, Roughness, Zeta, microsolver

# 1. failure path: one Newton iteration allowed
out = microsolver(Zeta(H=130e-9, P=40.8e6, Ux=0.02), t_end=0.02, roughness=Roughness.generated(Sa=40e-9),
                  settings=MicroSettings(max_iterations=1))
print("fail path:", out.status, "| reason:", out.reason[:90], "| dP", out.dP, "| P_asp", out.P_asp,
      "| fields", len(out.fields), "| extrema status", out.info["pressure_extrema_status"])

# 2. generated pair members against the Wang GenPair10 reference arrays
ref = tp.load_reference("c4_wangpair10_h25_offset")
for role, seed in (("stationary", 1000), ("translating", 1001)):
    z = Roughness.generated(Sa=10e-9 / np.sqrt(2.0), seed=seed).field().members["stationary"][0].source
    r = tp.compare_field(role, ref["surfaces"][role], np.asarray(z))
    print("pair member", role, "seed", seed, "rel", r["rel"], "ok", r["ok"], r["note"])

# 3. cosine on the moving surface, full film, steady and transient
st = MicroSettings(mesh_n=60, eta0=0.01, contact=None, ripple=None, spos_form="spline", jacobian_reg_eps=1e-10)
for steady in (False, True):
    o = microsolver(Zeta(H=120e-9, P=40.8e6, Ux=0.02), t_end=0.02, roughness=Roughness.cosine(42.3e-9, moving=True),
                    settings=st, steady=steady)
    print("moving cosine", "steady" if steady else "transient", o.status, "dQ", o.dQ, "dP", o.dP)
