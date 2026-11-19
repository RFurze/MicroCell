"""Check that the message tidy changed string constants only, apart from the stated code change.

Each tracked ``.py`` file under ``microcell/``, ``tests/``, ``tools/`` and ``examples/`` is parsed
at the base commit and in the working tree, every string constant is blanked and return
annotations are dropped, and the two syntax trees are compared.  One file is expected to differ:
``microcell/solver_mixed.py`` (the unused return value of ``_assemble_newton_rhs``).

Run from the repository root with any Python 3:

    python docs/notes/micro_standalone_open_items_implementation_log_2026-11-11_files/check_strings_only.py

The base source is ``git archive 3d97e5f`` extracted to
``_to_delete/message_tidy_2026-11-19/base_source_3d97e5f/``.

exit 0 = PASS, 1 = FAIL (an unexpected file differs, or the expected one does not), 2 = INCOMPLETE.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "_to_delete" / "message_tidy_2026-11-19" / "base_source_3d97e5f"
EXPECTED = {"microcell/solver_mixed.py"}


def dump(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
    return ast.dump(tree, include_attributes=False)


def main() -> int:
    if not BASE.is_dir():
        print(f"INCOMPLETE: no base source at {BASE}")
        return 2
    out = subprocess.run(["git", "ls-files", "microcell/*.py", "tests/*.py", "tools/*.py", "examples/*.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    files, differ, changed, missing = sorted(out.stdout.split()), set(), [], []
    for f in files:
        new, old = ROOT / f, BASE / f
        if not (new.is_file() and old.is_file()):
            missing.append(f)
            continue
        a, b = old.read_text(encoding="utf-8"), new.read_text(encoding="utf-8")
        if a != b:
            changed.append(f)
        if dump(a) != dump(b):
            differ.add(f)
    print(f"{len(files)} files; {len(changed)} changed: {', '.join(changed)}")
    print(f"code other than strings and return annotations differs in: {', '.join(sorted(differ)) or 'none'}")
    if missing:
        print("INCOMPLETE: missing " + ", ".join(missing))
        return 2
    return 0 if differ == EXPECTED else 1


if __name__ == "__main__":
    sys.exit(main())
