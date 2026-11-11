"""WP3: static check after each removal group (pytest form, runs through ``_claude/wsltest``).

Pure ``ast`` over ``microcell/``, ``tests/`` and ``examples/``; catches a kept function that still
reads a removed helper on a path the gate does not run:

  1. (files without a star import) a bare name read that is bound nowhere in its file (no assignment, argument, import, def,
     class, loop or comprehension target of that name) and is not a builtin;
  2. ``from microcell.x import N`` where ``x`` has no top-level ``N``;
  3. ``alias.N`` where ``alias`` is an imported ``microcell`` module without a top-level ``N``.

It also lists imports that nothing in their file reads (``unused_imports_<tag>.txt`` beside this
file; tag from ``inventory_tag.txt``, default ``base``).  Exit: pytest's (0 = no finding).

    ../_claude/wsltest docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/check_names.py -q -p no:cacheprovider
"""
import ast
import builtins
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TAG_FILE = HERE / "inventory_tag.txt"


def module_name(rel):
    return rel[:-3].replace("/", ".").replace(".__init__", "")


def scan(rel):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    me = module_name(rel)
    pkg = me if rel.endswith("__init__.py") else me.rsplit(".", 1)[0]
    bound, loads, top = set(), [], set()
    imports = {}                      # local alias -> ("from", module, name) | ("mod", module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        top.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.If, ast.Try, ast.For, ast.With)):
            for n in ast.walk(node):
                if isinstance(n, ast.alias):
                    top.add((n.asname or n.name).split(".")[0])
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    top.add(n.id)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                loads.append((n.id, n.lineno))
            else:
                bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parts = pkg.split(".")
                parts = parts[:len(parts) - (n.level - 1)]
                base = ".".join(parts + ([n.module] if n.module else []))
            for a in n.names:
                bound.add(a.asname or a.name)
                imports[a.asname or a.name] = ("from", base, a.name, n.lineno)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
                if a.asname:
                    imports[a.asname] = ("mod", a.name, None, n.lineno)
    attrs = [(n.value.id, n.attr, n.lineno) for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)]
    return dict(bound=bound, loads=loads, top=top, imports=imports, attrs=attrs)


def test_no_undefined_names():
    files = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                   for d in ("microcell", "tests", "examples") for p in (ROOT / d).rglob("*.py"))
    info = {rel: scan(rel) for rel in files}
    mod_of = {module_name(rel): rel for rel in files}
    known = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    bad, unused = [], []
    for rel in files:
        s = info[rel]
        star = any(v[2] == "*" for v in s["imports"].values())      # "from dolfin import *": check 1 is skipped
        for name, line in ([] if star else s["loads"]):
            if name not in s["bound"] and name not in known:
                bad.append(f"{rel}:{line}: undefined name {name}")
        for alias, (kind, base, name, line) in s["imports"].items():
            if kind == "from":
                target = mod_of.get(base)
                sub = mod_of.get(base + "." + name)
                if target is not None and sub is None and name != "*" and name not in info[target]["top"]:
                    bad.append(f"{rel}:{line}: {base} has no top-level {name}")
            used = any(n == alias for n, _l in s["loads"])
            if not used and not rel.endswith("__init__.py") and alias != "annotations":
                unused.append(f"{rel}:{line}: unused import {alias}")
        for alias, attr, line in s["attrs"]:
            imp = s["imports"].get(alias)
            if imp is None:
                continue
            target = mod_of.get(imp[1] + "." + imp[2]) if imp[0] == "from" else mod_of.get(imp[1])
            if target is not None and not target.endswith("__init__.py") and attr not in info[target]["top"]:
                bad.append(f"{rel}:{line}: {alias}.{attr}: {module_name(target)} has no top-level {attr}")
    tag = TAG_FILE.read_text().strip() if TAG_FILE.exists() else "base"
    (HERE / f"unused_imports_{tag}.txt").write_text("\n".join(sorted(set(unused))) + "\n", encoding="utf-8")
    assert not bad, "\n".join(sorted(set(bad)))
