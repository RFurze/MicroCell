"""WP5 step 1, inventory 1 (layout).  Reads the tree under test (the working directory), edits nothing.

Run from ``MicroCell/`` (or an archive of a commit):

    <RefactorRefactor>/_claude/wsltest <path to this file> -q -p no:cacheprovider

Writes beside this file, with the tag in ``inventory_tag.txt`` (default ``base``):

  inventory_layout_files_<tag>.tsv       one row per file of microcell/: lines, target file of the plan, top-level names
  inventory_layout_imports_<tag>.tsv     one row per import of a microcell module in microcell/, tests/, examples/, tools/
  inventory_layout_collisions_<tag>.tsv  one row per top-level name defined in two files of one merge group
  inventory_layout_aliases_<tag>.tsv     imports between two files of one merge group that rename a name (``as``):
                                         each would need an assignment, a new code line, in the merged file
"""
import ast
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path(os.getcwd()).resolve()

#: target file of the plan ("Repo layout (target)") <- present files.  "solver.py (one file)" is the
#: plan's first alternative; the three-file alternative has no merge.
GROUPS = {
    "solver.py (one file)": ["micro_phase_neutral_class.py", "micro_phase_neutral_class_mixed.py",
                             "micro_phase_neutral_class_contact.py"],
    "samplers.py": ["micro_diagnostics.py", "contact_sample.py"],
    "film.py": ["ideal_ht.py", "surface_ht.py"],
    "contact.py": ["contact.py", "elastic_contact.py"],
    "shear.py": ["shear_reduce.py", "fluid_shear.py", "contact_shear.py"],
    "roughness/runtime.py": ["roughness/runtime.py", "roughness/pair.py"],
    "roughness/generation.py": ["roughness/generation.py", "roughness/quantiles.py"],
}
TARGET = {f: t for t, fs in GROUPS.items() for f in fs}


def _module(rel: str) -> str:
    return "microcell." + rel[:-3].replace("/", ".")


def _top_names(tree: ast.Module) -> list:
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
    return names


def _imports(path: Path, pkg: str) -> list:
    """(line, module imported, names with their aliases) of every microcell import in ``path``."""
    out = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("microcell"):
                    out.append((node.lineno, a.name, [(a.name, a.asname)]))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:                                  # relative: resolve against the file's package
                base = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                mod = ".".join(base + ([mod] if mod else []))
            if not mod.startswith("microcell"):
                continue
            for a in node.names:
                # ``from microcell import x`` imports the module microcell.x when x is one
                sub = f"{mod}.{a.name}"
                is_module = (TREE / (sub.replace(".", "/") + ".py")).exists()
                out.append((node.lineno, sub if is_module else mod, [(a.name, a.asname)]))
    return out


def test_inventory_layout():
    tag = (HERE / "inventory_tag.txt").read_text().strip() if (HERE / "inventory_tag.txt").exists() else "base"
    files = sorted(p.relative_to(TREE / "microcell").as_posix() for p in (TREE / "microcell").rglob("*.py"))
    names = {}
    rows = ["\t".join(["file", "lines", "target_of_plan", "n_top_level_names", "top_level_names"])]
    total = 0
    for rel in files:
        text = (TREE / "microcell" / rel).read_text(encoding="utf-8")
        n = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        total += n
        names[rel] = _top_names(ast.parse(text))
        rows.append("\t".join([rel, str(n), TARGET.get(rel, "(not in a merge)"), str(len(names[rel])),
                               " ".join(names[rel])]))
    rows.append("\t".join(["TOTAL", str(total), "", "", ""]))
    (HERE / f"inventory_layout_files_{tag}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    imp = ["\t".join(["importer", "line", "module", "name", "alias"])]
    alias = ["\t".join(["target", "importer", "line", "module", "name", "alias"])]
    for top in ("microcell", "tests", "examples", "tools"):
        for p in sorted((TREE / top).rglob("*.py")):
            rel = p.relative_to(TREE).as_posix()
            pkg = ".".join(p.relative_to(TREE).with_suffix("").parts[:-1])
            for line, mod, pairs in sorted(_imports(p, pkg)):
                for name, asname in pairs:
                    imp.append("\t".join([rel, str(line), mod, name, asname or ""]))
                    if top != "microcell" or not asname:
                        continue
                    src = p.relative_to(TREE / "microcell").as_posix()
                    dst = mod[len("microcell."):].replace(".", "/") + ".py" if mod != "microcell" else ""
                    if src in TARGET and TARGET.get(dst) == TARGET[src] and src != dst:
                        alias.append("\t".join([TARGET[src], src, str(line), mod, name, asname]))
    (HERE / f"inventory_layout_imports_{tag}.tsv").write_text("\n".join(imp) + "\n", encoding="utf-8")
    (HERE / f"inventory_layout_aliases_{tag}.tsv").write_text("\n".join(alias) + "\n", encoding="utf-8")

    col = ["\t".join(["target", "name", "files"])]
    for target, members in GROUPS.items():
        seen = {}
        for m in members:
            for nm in set(names[m]):
                seen.setdefault(nm, []).append(m)
        for nm, where in sorted(seen.items()):
            if len(where) > 1:
                col.append("\t".join([target, nm, " ".join(where)]))
    (HERE / f"inventory_layout_collisions_{tag}.tsv").write_text("\n".join(col) + "\n", encoding="utf-8")
    assert total > 0
