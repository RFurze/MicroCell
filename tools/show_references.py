#!/usr/bin/env python
"""Print one line per reference cell (tests/references/*.npz): status, mesh, iteration
counts and the main homogenised outputs, at full precision.  Reads only the files."""
import json
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parents[1] / "tests" / "references"
cols = ("delta_P", "Pst", "P_f", "Pasp_hom", "contact_frac", "A_mech_frac", "k_asp_hom", "newton_iters",
        "contact_iters", "ripple_passes", "wall_time")
print("\t".join(("cell", "status", "mode", "mesh", "H_m", "P_Pa", "offset_m", "dQx", "dQy") + cols + ("nan_covered",)))
for f in sorted(REF.glob("*.npz")):
    with np.load(f) as d:
        meta = json.loads(str(d["meta_json"]))
        res = json.loads(str(d["result_json"]))
        task = d["task"]
    dq = np.ravel(np.asarray(res["delta_Q"], dtype=float))
    row = [meta["cell"], meta["status"], "transient" if meta["transient"] else "steady", meta["mesh_used"],
           repr(float(task[2])), repr(float(task[3])), meta["offset"], repr(float(dq[0])), repr(float(dq[1]))]
    row += [repr(res[c]) for c in cols]
    row.append(",".join(sorted({n.split("_")[0] + "_" + n.split("_")[1] for n, _ in meta["nan_covered"]})))
    print("\t".join(str(v) for v in row))
sys.exit(0)
