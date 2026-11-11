"""WP3 group 4: ``Roughness.generated`` before and after its rewiring (pytest form, through wsltest).

The brief allows the front end to be rewired only if every array a constructor returns is
unchanged.  This script calls ``Roughness.generated`` for the argument sets ``CASES`` (every
argument of the constructor away from its default at least once) and

  * when ``generated_before_rewire.npz`` is absent: writes it (run at a9a056d, before the rewiring);
  * otherwise compares the arrays it gets now with the stored ones, bit for bit
    (``np.array_equal``), and the cell and member role, and writes ``generated_rewire_check.tsv``.

It also records which seeds of ``Roughness.generated(Sa=10e-9/sqrt(2), seed=s)`` reproduce the two
stored surfaces of reference c4 (the parent's Wang pair), in ``generated_c4_seeds.tsv``.

    ../_claude/wsltest docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/generated_rewire_check.py -q -p no:cacheprovider

pytest's exit: 0 = written, or every array identical; 1 = a difference.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
STORE = HERE / "generated_before_rewire.npz"

CASES = {
    "default_sa40_seed1000": dict(Sa=40e-9, seed=1000),
    "sa10_seed7_corr2e-7": dict(Sa=10e-9, seed=7, corr_length=2e-7),
    "sa25_seed3_cell4um_n32_iter20": dict(Sa=25e-9, seed=3, Lx=4e-6, Ly=4e-6, n=32, n_iter=20),
    "sa12_seed1001_5x4um_n16_iter0_moving": dict(Sa=12e-9, seed=1001, Lx=5e-6, Ly=4e-6, n=16, n_iter=0, moving=True),
    "sa7_seed1001_moving": dict(Sa=10e-9 / np.sqrt(2.0), seed=1001, moving=True),
}


def _one(kw):
    from microcell import Roughness
    r = Roughness.generated(**kw)
    (role, (fs, off)), = r.field().members.items()
    return role, np.asarray(fs.source), r.cell, off


def _sha(z):
    return hashlib.sha256(np.ascontiguousarray(z, dtype="<f8").tobytes()).hexdigest()


def test_generated_before_and_after():
    got = {name: _one(kw) for name, kw in CASES.items()}
    if not STORE.exists():
        np.savez(STORE, **{name: z for name, (_r, z, _c, _o) in got.items()})
        (HERE / "generated_before_rewire.tsv").write_text(
            "case\trole\tshape\tcell_m\tsha256\n" + "".join(
                f"{n}\t{r}\t{z.shape[0]}x{z.shape[1]}\t{c[0]:g} x {c[1]:g}\t{_sha(z)}\n"
                for n, (r, z, c, _o) in got.items()), encoding="utf-8")
        return
    before = np.load(STORE)
    meta = {ln.split("\t")[0]: ln.rstrip("\n").split("\t")
            for ln in (HERE / "generated_before_rewire.tsv").read_text().splitlines()[1:]}
    rows, bad = ["case\tidentical\tmax_abs_diff\trole_and_cell_same\tsha256_after"], []
    for name, (role, z, cell, _off) in got.items():
        same = bool(np.array_equal(before[name], z))
        d = float(np.max(np.abs(before[name] - z))) if before[name].shape == z.shape else float("inf")
        m = meta[name]
        same_meta = (m[1] == role and m[3] == f"{cell[0]:g} x {cell[1]:g}" and m[4] == _sha(z))
        rows.append(f"{name}\t{same}\t{d:.3e}\t{same_meta}\t{_sha(z)}")
        if not (same and same_meta):
            bad.append(name)
    (HERE / "generated_rewire_check.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert not bad, bad


def test_c4_seeds():
    """Which seeds give the parent's two Wang-pair surfaces (reference c4), bit for bit."""
    import test_parity as tp
    from microcell import Roughness
    ref = tp.load_reference("c4_wangpair10_h25_offset")
    rows = ["role\tseed\tidentical\tmax_rel"]
    found = {}
    for role in ("stationary", "translating"):
        want = np.asarray(ref["surfaces"][role])
        for seed in list(range(1000, 1006)) + list(range(0, 6)) + [2000, 2001]:
            fs = Roughness.generated(Sa=10e-9 / np.sqrt(2.0), seed=seed).field().members["stationary"][0]
            z = np.asarray(fs.source)
            if np.allclose(z, want, rtol=1e-9, atol=0.0):
                rows.append(f"{role}\t{seed}\t{bool(np.array_equal(z, want))}\t{tp._rel(want, z):.3e}")
                found[role] = seed
                break
        else:
            rows.append(f"{role}\tnone of the seeds tried\tFalse\tnan")
    (HERE / "generated_c4_seeds.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
