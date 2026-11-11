"""WP5 group 6 check: imports with no reader, and the names group 6 removed.

Run from ``MicroCell/``:

    <RefactorRefactor>/_claude/wsltest <path to this file> -q -p no:cacheprovider

Reads every ``.py`` under ``microcell/`` of the working directory (the tree under test) with ``ast``
and writes beside this file (tag from ``check_tag.txt``, default ``current``):

  unused_imports_<tag>.tsv   one row per imported name that the importing module never reads:
                             not as a name, not as the base of an attribute, not in ``__all__``.
                             The last column counts the readers in other files (``.<name>``): a
                             row with a count above 0 is not unused.
                             ``from x import *`` and ``from __future__`` are skipped; a package
                             ``__init__.py`` is listed with ``reexport`` in the last column, since
                             its imports are the package's names.
  removed_names_<tag>.tsv    for each name group 6 removed: the lines of ``microcell/``, ``tests/``,
                             ``examples/`` and ``tools/`` that still contain it as a word.

The test passes when both files are written.  It judges nothing: the rows are read by hand.
"""
import ast
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()

REMOVED = ("SAMPLER_DEFINITION", "h_geom_min_bound", "_print_timing_summary", "TRACE_NEWTON",
           "_newton_trace", "_t_run_start", "FLUID_DEFORMATION_MODES", "deformation_model_tag",
           "as_dict", "err_h_legacy", "err_s_legacy", "err_h", "err_s")


def _tag() -> str:
    f = HERE / "check_tag.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else "current"


def unused_imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []                                   # (bound name, line, text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.append(((a.asname or a.name).split(".")[0], node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                imported.append((a.asname or a.name, node.lineno,
                                 f"from {'.' * node.level}{node.module or ''} import {a.name}"))
    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            exported |= {e.value for e in getattr(node.value, "elts", []) if isinstance(e, ast.Constant)}
    return [(name, line, text) for name, line, text in imported if name not in read and name not in exported]


def test_write_the_two_tables():
    tag = _tag()
    rows = []
    for path in sorted((TREE / "microcell").rglob("*.py")):
        rel = path.relative_to(TREE).as_posix()
        kind = "reexport" if path.name == "__init__.py" else ""
        for name, line, text in unused_imports(path):
            # a module-level import can be read from another module as ``<module>.<name>``
            # (``samplers.py`` reads ``shear.FFT_MODEL``): count the ``.<name>`` readers elsewhere
            pat = re.compile(r"\." + re.escape(name) + r"(?![A-Za-z0-9_])")
            elsewhere = sum(len(pat.findall(p.read_text(encoding="utf-8")))
                            for top in ("microcell", "tests", "examples", "tools")
                            for p in sorted((TREE / top).rglob("*.py")) if p != path)
            note = "; ".join(v for v in (kind, f"read elsewhere as .{name}: {elsewhere}") if v)
            rows.append((rel, line, name, text, note))
    with open(HERE / f"unused_imports_{tag}.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("file\tline\tname\timport\tnote\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r) + "\n")
    with open(HERE / f"removed_names_{tag}.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("name\tfile\tline\ttext\n")
        for name in REMOVED:
            pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
            hits = 0
            for top in ("microcell", "tests", "examples", "tools"):
                for path in sorted((TREE / top).rglob("*.py")):
                    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if pat.search(line):
                            hits += 1
                            fh.write(f"{name}\t{path.relative_to(TREE).as_posix()}\t{i}\t{line.strip()[:160]}\n")
            if not hits:
                fh.write(f"{name}\t-\t0\t(no line)\n")
    assert (HERE / f"unused_imports_{tag}.tsv").exists() and (HERE / f"removed_names_{tag}.tsv").exists()
