"""WP5 group 1 check: the renames and merges moved code verbatim.

Compares ``microcell/`` of the base commit (an archive, ``_to_delete/wp5_2026-11-11/base_source_d3b88ae``)
with ``microcell/`` of the tree under test (the working directory).  From every file it drops

  * the module docstring,
  * every import statement (module level or inside a function),
  * a module-level ``__all__ = ...`` assignment,
  * blank lines,

and compares what is left, as a sorted list of lines (leading whitespace kept, trailing whitespace
dropped), over the whole package.  Comments are kept, so a comment that names an old module is
unchanged and shows as stale wording, not as a difference.

Run from ``MicroCell/``:

    <RefactorRefactor>/_claude/wsltest <path to this file> -q -p no:cacheprovider

Writes beside this file (tag from ``check_tag.txt``, default ``current``):

  moved_lines_<tag>.tsv        the counts, then every line found on one side only
  moved_lines_files_<tag>.tsv  one row per file of each tree: lines, code lines compared

PASS (the test passes) = no line on one side only.  FAIL otherwise.
"""
import ast
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()
BASE = TREE / "_to_delete" / "wp5_2026-11-11" / "base_source_d3b88ae"


def code_lines(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    drop = set()
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                drop.update(range(node.lineno, node.end_lineno + 1))
    return [ln.rstrip() for i, ln in enumerate(lines, start=1) if i not in drop and ln.strip()]


def tree_lines(root: Path):
    per_file, total = [], Counter()
    for p in sorted((root / "microcell").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        kept = code_lines(p)
        per_file.append((p.relative_to(root / "microcell").as_posix(), n, len(kept)))
        total.update(kept)
    return per_file, total


def test_moved_lines():
    tag = (HERE / "check_tag.txt").read_text().strip() if (HERE / "check_tag.txt").exists() else "current"
    assert (BASE / "microcell").is_dir(), f"no base archive at {BASE}"
    base_files, base = tree_lines(BASE)
    new_files, new = tree_lines(TREE)
    only_base = sorted((base - new).elements())
    only_new = sorted((new - base).elements())

    rows = ["\t".join(["quantity", "base_d3b88ae", "tree_under_test"]),
            "\t".join(["files", str(len(base_files)), str(len(new_files))]),
            "\t".join(["lines", str(sum(f[1] for f in base_files)), str(sum(f[1] for f in new_files))]),
            "\t".join(["code lines compared", str(sum(base.values())), str(sum(new.values()))]),
            "\t".join(["code lines on this side only", str(len(only_base)), str(len(only_new))]),
            "", "\t".join(["side", "line"])]
    rows += [f"base only\t{ln}" for ln in only_base]
    rows += [f"tree only\t{ln}" for ln in only_new]
    (HERE / f"moved_lines_{tag}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    frows = ["\t".join(["tree", "file", "lines", "code_lines_compared"])]
    frows += ["\t".join(["base_d3b88ae", f, str(n), str(k)]) for f, n, k in base_files]
    frows += ["\t".join(["tree_under_test", f, str(n), str(k)]) for f, n, k in new_files]
    (HERE / f"moved_lines_files_{tag}.tsv").write_text("\n".join(frows) + "\n", encoding="utf-8")

    assert not only_base and not only_new, (f"{len(only_base)} code lines only in the base, "
                                             f"{len(only_new)} only in the tree (moved_lines_{tag}.tsv)")
