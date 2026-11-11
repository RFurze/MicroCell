"""The form-compiler quadrature degree, stated per solver and pinned per process.

[Simplification phase D1, 2026-10-12]  Legacy DOLFIN reads
``parameters["form_compiler"]["quadrature_degree"]`` -- one global per process --
each time it JIT-compiles a form (every ``assemble`` / ``project`` /
``SystemAssembler`` of a UFL form without ``metadata``).  Until D1 the solver
modules set it at import, so the degree a solver compiled with depended on which
modules a process had imported last:

* the macro module set 4 (``macro_HMM_penalty_transient_EHL.py``);
* the micro package set 4 (``micro_mixed_transient_class.py``) and then 2
  (``micro_phase_neutral_class.py``, imported after it by the package ``__init__``);
* the multiscale launcher imports the macro module, then the micro package.

The D0 check (``scripts/baselines/check_quadrature_degree.py``, run 2026-10-12
under conda) measured it, by the global and by the degree a tiny form actually
compiles with: single-scale 4; multiscale macro (parent) 2; micro worker 2; the
bench order 2.  Those are the values below.  D1 changes no number: each driver now
*pins* its value at a defined point, and the solver modules' import-time settings
(``set_at_import``) leave a pinned degree alone, so no import order can move it.

One process has one global, so the multiscale macro solver and the micro solvers
in the same process share a degree (2, as they always have).  Giving the
multiscale macro forms 4 while the micro forms stay at 2 needs per-form
``form_compiler_parameters`` -- the later, re-baselined change (decision a,
Rob 2026-10-12); ``pin_multiscale`` refuses a split until then.
"""

from __future__ import annotations

#: the degree each solver's forms compile with (measured 2026-10-12, phase D0)
QDEG_MACRO_SINGLE_SCALE = 4     # single-scale macro solver (the macro module's own 4)
QDEG_MACRO_MULTISCALE = 2       # multiscale macro solver (the micro package's 2 won)
QDEG_MICRO_PHASE_NEUTRAL = 2    # MicroPhaseNeutralSolver and its contact / mixed subclasses
QDEG_MICRO_MIXED = 2            # MicroMixedSolver (its module sets 4; the package imports the PN module after it)

_PINNED: int | None = None
_PINNED_BY: str | None = None


def _fc():
    from dolfin import parameters
    return parameters["form_compiler"]


def set_at_import(degree: int, module: str) -> None:
    """What a solver module used to do at import (``parameters[...] = degree``),
    unless a driver has pinned the degree: then the pin stands."""
    if _PINNED is None:
        _fc()["quadrature_degree"] = int(degree)


def pin(degree: int, by: str) -> None:
    """Set the process's degree and keep it: later ``set_at_import`` calls are no-ops."""
    global _PINNED, _PINNED_BY
    _fc()["quadrature_degree"] = int(degree)
    _PINNED, _PINNED_BY = int(degree), by


def pin_single_scale(by: str) -> None:
    pin(QDEG_MACRO_SINGLE_SCALE, by)


def pin_multiscale(by: str) -> None:
    """The multiscale process: the macro solver and (warm-up, workers) the micro
    solvers share the global, so their stated degrees must agree."""
    if not (QDEG_MACRO_MULTISCALE == QDEG_MICRO_PHASE_NEUTRAL == QDEG_MICRO_MIXED):
        raise RuntimeError(
            f"multiscale macro degree {QDEG_MACRO_MULTISCALE} != micro {QDEG_MICRO_PHASE_NEUTRAL} / "
            f"{QDEG_MICRO_MIXED}: one process has one global degree; a split needs per-form "
            "form_compiler_parameters (not implemented)")
    pin(QDEG_MACRO_MULTISCALE, by)


def pin_micro_worker(by: str) -> None:
    """A micro pool worker (phase D5's pool initializer)."""
    if QDEG_MICRO_PHASE_NEUTRAL != QDEG_MICRO_MIXED:
        raise RuntimeError("micro phase-neutral and mixed degrees differ: one process, one global")
    pin(QDEG_MICRO_PHASE_NEUTRAL, by)


def pinned() -> tuple[int | None, str | None]:
    return _PINNED, _PINNED_BY


def check(where: str) -> int:
    """The global equals the pin (raise otherwise); returns the degree in force."""
    q = int(_fc()["quadrature_degree"])
    if _PINNED is not None and q != _PINNED:
        raise RuntimeError(f"{where}: form-compiler quadrature_degree is {q}, but {_PINNED_BY} pinned "
                           f"{_PINNED} -- something wrote the global after the pin")
    return q
