"""WP3 step 1: inventory of the helper modules (who uses each top-level name).

Pure ``ast``; imports nothing from ``microcell``.  For every module in WP3's scope it lists each
top-level function, class and constant, and classifies it by who reads it:

    solver     reachable from ``microsolver``: from ``microcell/__init__.py``, ``api.py``, the three
               solver class files, ``micro_diagnostics.py``, ``fields.py`` (and ``ripple.py``,
               ``form_compiler.py``, ``process_settings.py``, which those import), through any
               chain of kept names
    tests      not ``solver``; read by ``tests/`` (directly or through names only tests reach)
    examples   not ``solver`` or ``tests``; read by ``examples/``
    tools      not above; read by ``tools/``
    dead_only  read only by names that are themselves in none of the classes above
    none       no reader in the repo

How a reader is found (conservative: it over-keeps, never over-removes)
----------------------------------------------------------------------
A top-level definition D of file A "reads" name N of module B when D's source holds the
identifier N (as a bare name, an attribute ``x.N``, or an imported name) and either B is A, or A
imports N from B, or A imports the module B (under any alias), or N appears as an attribute of
anything in D (``x.N``: linked to every in-scope module that defines N).  ``__all__`` lists and
docstrings are not readers.  Module-level statements that are not a definition or an assignment
(module-level imports included: ``from B import N as _n`` reads N) form the node ``<module>`` of
their file, alive when any name of that file is alive.

Limit: the unit is the top-level name.  A class is alive as a whole, so a method that nothing calls
keeps what it reads alive (``SurfaceField.__init__`` and the artefact store, for example).  Those
are read by hand and listed in ``inventory_methods.md``.

Run from ``MicroCell/`` (pytest form, so that it runs through ``_claude/wsltest``):

    ../_claude/wsltest docs/notes/micro_standalone_wp3_implementation_log_2026-11-11_files/inventory.py -q -p no:cacheprovider

Writes ``inventory_<tag>.tsv`` beside this file (``tag`` from the environment-free constant
``TAG`` below, edited per stage: ``base`` before the first removal, ``final`` after the last).
"""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TAG_FILE = HERE / "inventory_tag.txt"       # one word; "base" when absent

IN_SCOPE = (["microcell/roughness/" + n for n in (
    "__init__.py", "assets.py", "case.py", "evaluator.py", "frontend.py", "generation.py", "modes.py", "pair.py",
    "prepare.py", "quantiles.py", "report.py", "runtime.py", "spec.py", "stats.py", "targets.py")]
    + ["microcell/" + n for n in (
        "shear_reduce.py", "fluid_shear.py", "contact_shear.py", "contact_sample.py", "pressure.py", "ideal_ht.py",
        "surface_ht.py", "contact.py", "elastic_contact.py")])

#: every definition of these files is a root of class "solver"
SOLVER_ROOT_FILES = ("microcell/__init__.py", "microcell/api.py", "microcell/micro_phase_neutral_class.py",
                     "microcell/micro_phase_neutral_class_mixed.py",
                     "microcell/micro_phase_neutral_class_contact.py", "microcell/micro_diagnostics.py",
                     "microcell/fields.py", "microcell/ripple.py", "microcell/form_compiler.py",
                     "microcell/process_settings.py")
#: the public front end: the plan's D2 keeps every constructor
SOLVER_ROOT_NAMES = (("microcell/roughness/frontend.py", "Roughness"),)
OTHER_ROOTS = (("tests", "tests"), ("examples", "examples"), ("tools", "tools"))


def module_name(rel: str) -> str:
    return rel[:-3].replace("/", ".").replace(".__init__", "")


def identifiers(node) -> tuple:
    """(bare names, attribute names) used inside ``node``, docstrings and ``__all__`` excluded."""
    names, attrs = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            attrs.add(n.attr)
        elif isinstance(n, (ast.ImportFrom, ast.Import)):
            for a in n.names:
                names.add(a.name.split(".")[-1])
    return names, attrs


def parse(rel: str) -> dict:
    """{"defs": {name: (kind, line, names, attrs)}, "from": {name: module}, "mods": {module}}"""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    me = module_name(rel)
    pkg = me if rel.endswith("__init__.py") else me.rsplit(".", 1)[0]
    out = {"defs": {}, "from": {}, "mods": set()}

    def note_imports(node):
        for n in ast.walk(node):
            if isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    parts = pkg.split(".")
                    parts = parts[:len(parts) - (n.level - 1)]
                    base = ".".join(parts + ([n.module] if n.module else []))
                for a in n.names:
                    out["from"][a.name] = base               # from B import N
                    out["mods"].add(base + "." + a.name)       # from pkg import module
                out["mods"].add(base)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    out["mods"].add(a.name)

    note_imports(tree)
    rest_names, rest_attrs = set(), set()

    def add(name, kind, node, value=None):
        nm, at = identifiers(value if value is not None else node)
        if name in out["defs"]:                              # re-assigned: merge
            k, line, nm0, at0 = out["defs"][name]
            nm, at = nm | nm0, at | at0
            kind, node_line = k, line
        else:
            node_line = node.lineno
        out["defs"][name] = (kind, node_line, nm, at)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, "function", node)
        elif isinstance(node, ast.ClassDef):
            add(node.name, "class", node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            flat = []
            for t in targets:
                flat += [e for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])]
            simple = [t.id for t in flat if isinstance(t, ast.Name)]
            if simple and node.value is not None:
                for name in simple:
                    if name == "__all__":
                        continue
                    add(name, "constant", node, node.value)
            elif node.value is not None:
                nm, at = identifiers(node.value)
                rest_names |= nm
                rest_attrs |= at
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            nm, _at = identifiers(node)                      # a module-level import reads what it imports
            rest_names |= nm                                 # (under any alias): module code
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                         # docstring
        else:
            nm, at = identifiers(node)
            rest_names |= nm
            rest_attrs |= at
    if rest_names or rest_attrs:
        out["defs"]["<module>"] = ("module code", 0, rest_names, rest_attrs)
    return out


def build():
    files = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                   for d in ("microcell", "tests", "examples", "tools")
                   for p in (ROOT / d).rglob("*.py"))
    parsed = {rel: parse(rel) for rel in files}
    mod_of = {module_name(rel): rel for rel in files}
    scope_defs = {}                                          # name -> [in-scope rel that defines it]
    for rel in (r for r in IN_SCOPE if r in parsed):      # a file removed by a later group is skipped
        for name in parsed[rel]["defs"]:
            scope_defs.setdefault(name, []).append(rel)

    def reads(rel, dname):
        """the in-scope (rel, name) nodes that definition ``dname`` of ``rel`` reads"""
        _k, _l, names, attrs = parsed[rel]["defs"][dname]
        got = set()
        for n in names | attrs:
            for target in scope_defs.get(n, ()):
                if target == rel and n in names and n != dname:
                    got.add((target, n))
                elif target != rel and n in names and mod_of.get(parsed[rel]["from"].get(n, "")) == target:
                    got.add((target, n))
                elif target != rel and n in attrs:
                    got.add((target, n))                      # x.N: any module that defines N
                elif target == rel and n in attrs and n != dname:
                    got.add((target, n))
        return got

    edges = {(rel, d): reads(rel, d) for rel in files for d in parsed[rel]["defs"]}
    # a re-export in a package __init__ ("from .x import N") reads N
    for rel in files:
        if rel.endswith("__init__.py"):
            exp = set()
            for n, base in parsed[rel]["from"].items():
                target = mod_of.get(base)
                if target in IN_SCOPE and n in parsed[target]["defs"]:
                    exp.add((target, n))
            edges[(rel, "<reexport>")] = exp

    def closure(roots):
        seen, stack = set(), list(roots)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            rel = node[0]
            if rel in parsed and "<module>" in parsed[rel]["defs"]:
                stack.append((rel, "<module>"))               # module code runs when the file is imported
            stack += list(edges.get(node, ()))
        return seen

    solver_roots = [(rel, d) for rel in SOLVER_ROOT_FILES for d in list(parsed[rel]["defs"]) + ["<reexport>"]
                    if (rel, d) in edges]
    solver_roots += list(SOLVER_ROOT_NAMES)
    # the package __init__ of roughness is imported with any roughness module
    solver_roots += [("microcell/roughness/__init__.py", "<reexport>")]
    alive = {"solver": closure(solver_roots)}
    for cls, d in OTHER_ROOTS:
        roots = [(rel, n) for rel in files if rel.startswith(d + "/") for n in parsed[rel]["defs"]]
        alive[cls] = closure(roots)

    readers = {}
    for src, targets in edges.items():
        for t in targets:
            readers.setdefault(t, set()).add(src)
    return parsed, alive, readers


def classify(node, alive, readers):
    for cls in ("solver", "tests", "examples", "tools"):
        if node in alive[cls]:
            return cls
    return "dead_only" if readers.get(node) else "none"


def test_inventory():
    tag = TAG_FILE.read_text().strip() if TAG_FILE.exists() else "base"
    parsed, alive, readers = build()
    rows = [["module", "name", "kind", "line", "class", "also_read_by", "readers"]]
    counts = {}
    for rel in IN_SCOPE:
        if not (ROOT / rel).exists():
            continue
        for name, (kind, line, _n, _a) in sorted(parsed[rel]["defs"].items(), key=lambda kv: kv[1][1]):
            node = (rel, name)
            cls = classify(node, alive, readers)
            also = [c for c in ("tests", "examples", "tools") if node in alive[c] and c != cls]
            rd = sorted(f"{r.replace('microcell/', '')}:{d}" for r, d in readers.get(node, ()) if (r, d) != node)
            rows.append([rel, name, kind, str(line), cls, ",".join(also), "; ".join(rd)])
            counts[(rel, cls)] = counts.get((rel, cls), 0) + 1
    (HERE / f"inventory_{tag}.tsv").write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    classes = ["solver", "tests", "examples", "tools", "dead_only", "none"]
    srows = [["module"] + classes + ["total"]]
    tot = [0] * len(classes)
    for rel in IN_SCOPE:
        if not (ROOT / rel).exists():
            continue
        c = [counts.get((rel, k), 0) for k in classes]
        tot = [a + b for a, b in zip(tot, c)]
        srows.append([rel] + [str(v) for v in c] + [str(sum(c))])
    srows.append(["All"] + [str(v) for v in tot] + [str(sum(tot))])
    (HERE / f"inventory_{tag}_counts.tsv").write_text("\n".join("\t".join(r) for r in srows) + "\n", encoding="utf-8")
