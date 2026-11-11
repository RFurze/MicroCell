"""WP5 step 1: why the translating generated cell B of measure_march_cost.py fails, and which
translating surface cells solve.  Edits nothing.  Writes probe_translating_<tag>.tsv."""
from pathlib import Path

from microcell import ContactSettings, MicroSettings, Roughness, Zeta, microsolver

HERE = Path(__file__).resolve().parent


def test_probe():
    tag = (HERE / "inventory_tag.txt").read_text().strip()
    rows = ["\t".join(["cell", "status", "newton_iters", "contact_iters", "ripple_passes", "wall_s", "dP_Pa",
                       "P_asp_Pa", "contact_frac", "reason"])]
    sa40 = dict(Sa=40e-9, seed=1000)
    cases = [
        ("gen Sa40 translating, H130, contact+ripple, n_steps=3", Roughness.generated(moving=True, **sa40),
         Zeta(H=130e-9, P=40.8e6, Ux=0.02), MicroSettings(), dict(t_end=0.02, n_steps=3)),
        ("gen Sa40 translating, H130, contact, no ripple, n_steps=3", Roughness.generated(moving=True, **sa40),
         Zeta(H=130e-9, P=40.8e6, Ux=0.02), MicroSettings(ripple=None), dict(t_end=0.02, n_steps=3)),
        ("gen Sa40 translating, H200, contact+ripple, n_steps=3", Roughness.generated(moving=True, **sa40),
         Zeta(H=200e-9, P=40.8e6, Ux=0.02), MicroSettings(), dict(t_end=0.02, n_steps=3)),
        ("gen Sa40 translating, H130, steady", Roughness.generated(moving=True, **sa40),
         Zeta(H=130e-9, P=40.8e6, Ux=0.02), MicroSettings(), dict(steady=True)),
        ("gen Sa40 translating, H400, full film, no ripple, n_steps=3", Roughness.generated(moving=True, **sa40),
         Zeta(H=400e-9, P=1e6, Ux=0.02), MicroSettings(contact=None, ripple=None), dict(t_end=0.02, n_steps=3)),
    ]
    for name, rough, zeta, st, kw in cases:
        out = microsolver(zeta, roughness=rough, settings=st, **kw)
        i = out.info
        rows.append("\t".join(str(v) for v in [name, out.status, i["newton_iters"], i["contact_iters"],
                                               i["ripple_passes"], f"{i['wall_time']:.2f}", f"{out.dP:.6e}",
                                               f"{out.P_asp:.6e}", f"{out.contact_frac:.6f}",
                                               out.reason.replace("\t", " ").replace("\n", " ")[:400]]))
    (HERE / f"probe_translating_{tag}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
