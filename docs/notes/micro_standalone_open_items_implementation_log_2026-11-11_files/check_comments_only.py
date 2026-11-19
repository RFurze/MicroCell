"""Check that a comment rewrite changed comments and docstrings only.

Each tracked ``.py`` file is parsed at the base commit and in the working tree, docstrings and
other bare string statements are removed, and the two syntax trees are compared.  For ``.sh``
files the lines that are not comments are compared.  Comment and docstring sizes before and
after are written to ``comment_counts.tsv`` beside this file.

Run from the repository root with any Python 3 (no FEniCS needed):

    python docs/notes/micro_standalone_open_items_implementation_log_2026-11-11_files/check_comments_only.py [files ...]

The base source is ``git archive 71669f7`` extracted to
``_to_delete/comment_rewrite_2026-11-19/base_source_71669f7/``.

exit 0 = PASS (code identical), 1 = FAIL (a file's code differs), 2 = INCOMPLETE (a file or the base is missing).
"""
from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "_to_delete" / "comment_rewrite_2026-11-19" / "base_source_71669f7"
SKIP = "docs/notes/"


def _is_bare_string(stmt) -> bool:
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def code_dump(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list):
                setattr(node, name, [s for s in block if not _is_bare_string(s)])
    return ast.dump(tree, include_attributes=False)


def sizes(text: str) -> tuple[int, int]:
    """(lines, characters) of comments and docstrings."""
    lines, chars = 0, 0
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            lines += 1
            chars += len(tok.string)
    for node in ast.walk(ast.parse(text)):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list):
                for s in block:
                    if _is_bare_string(s):
                        lines += s.end_lineno - s.lineno + 1
                        chars += len(s.value.value)
    return lines, chars


def sh_code(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def sh_sizes(text: str) -> tuple[int, int]:
    com = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    return len(com), sum(len(ln) for ln in com)


def main(argv: list[str]) -> int:
    if not BASE.is_dir():
        print(f"INCOMPLETE: no base source at {BASE}")
        return 2
    if argv:
        files = [a.replace("\\", "/") for a in argv]
    else:
        out = subprocess.run(["git", "ls-files", "*.py", "*.sh"], cwd=ROOT, capture_output=True, text=True, check=True)
        files = [f for f in out.stdout.split() if not f.startswith(SKIP)]
    rows, bad, missing = [], [], []
    for f in sorted(files):
        new, old = ROOT / f, BASE / f
        if not (new.is_file() and old.is_file()):
            missing.append(f)
            continue
        a, b = old.read_text(encoding="utf-8"), new.read_text(encoding="utf-8")
        try:
            if f.endswith(".sh"):
                same, sa, sb = sh_code(a) == sh_code(b), sh_sizes(a), sh_sizes(b)
            else:
                same, sa, sb = code_dump(a) == code_dump(b), sizes(a), sizes(b)
        except (SyntaxError, tokenize.TokenError) as err:
            print(f"FAIL  {f}: {type(err).__name__}: {err}")
            bad.append(f)
            continue
        if not same:
            print(f"FAIL  {f}: code differs from the base")
            bad.append(f)
        rows.append((f, len(a.splitlines()), len(b.splitlines()), sa[0], sb[0], sa[1], sb[1], "same" if same else "DIFFERS"))
    if not argv:
        with (HERE / "comment_counts.tsv").open("w", newline="\n", encoding="utf-8") as fh:
            fh.write("file\tlines_base\tlines_now\tcomment_lines_base\tcomment_lines_now\tcomment_chars_base\tcomment_chars_now\tcode\n")
            for r in rows:
                fh.write("\t".join(str(v) for v in r) + "\n")
            tot = [sum(r[i] for r in rows) for i in range(1, 7)]
            fh.write("TOTAL\t" + "\t".join(str(v) for v in tot) + "\t" + ("same" if not bad else "DIFFERS") + "\n")
    for f in missing:
        print(f"INCOMPLETE  {f}: missing in the working tree or in the base")
    print(f"{len(rows) - len(bad)} of {len(files)} files: code identical; {len(bad)} differ; {len(missing)} missing")
    return 1 if bad else (2 if missing else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
