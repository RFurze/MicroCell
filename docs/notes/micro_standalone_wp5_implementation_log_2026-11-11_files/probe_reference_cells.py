"""WP5 group 4 preparation: what each reference cell is (zeta, window, settings, roughness), read
from tests/references through tests/test_parity.py.  Edits nothing.  Writes reference_cells.txt."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()
sys.path.insert(0, str(TREE / "tests"))


def test_dump():
    import test_parity as tp
    from microcell import Zeta

    lines = []
    for cell in tp.reference_cells():
        ref = tp.load_reference(cell)
        meta, mc = ref["meta"], ref["micro_case"]
        st = tp.settings_from_reference(ref)
        task = ref["task"]
        lines.append(f"== {cell}")
        lines.append(f"zeta      {Zeta.from_task(task)}")
        lines.append(f"window    transient={meta['transient']} t_end={task[15] if meta['transient'] else None} "
                     f"n_steps={meta['n_steps']} offset={meta['offset']}")
        lines.append(f"settings  {st}")
        if ref["surfaces"]:
            sj = ref["surface"]
            lines.append(f"surfaces  {[(k, v.shape, float(abs(v - v.mean()).mean())) for k, v in ref['surfaces'].items()]} "
                         f"cell_m={sj['cell_m']} bandwidth={sj['bandwidth']} "
                         f"offsets={ {k: m['offset_m'] for k, m in sj['members'].items()} }")
        else:
            lines.append(f"cosine    Ah={mc['Ah']} kx={mc['kx']} ky={mc['ky']} phi_y={mc['phi_y']} mode={mc['roughness_mode']}")
        r = ref["result"]
        lines.append("result    " + ", ".join(f"{k}={r[k]}" for k in sorted(r) if isinstance(r[k], (int, float, str)))[:900])
    (HERE / "reference_cells.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
