"""Form-compiler quadrature degree of the process.

dolfin reads ``parameters["form_compiler"]["quadrature_degree"]``, one value per process, each
time it compiles a form without ``metadata``.  The degree is set here, before the first form is
built.  A pinned degree is not changed by a later ``set_at_import``.
"""

from __future__ import annotations

#: Quadrature degree of each solver's forms.
QDEG_MACRO_SINGLE_SCALE = 4     # single-scale macro solver
QDEG_MACRO_MULTISCALE = 2       # multiscale macro solver
QDEG_MICRO_PHASE_NEUTRAL = 2    # MicroPhaseNeutralSolver and its mixed and contact subclasses
QDEG_MICRO_MIXED = 2            # must equal QDEG_MICRO_PHASE_NEUTRAL (``pin_micro_worker``)

_PINNED: int | None = None
_PINNED_BY: str | None = None


def _fc():
    """The dolfin form-compiler parameters."""
    from dolfin import parameters
    return parameters["form_compiler"]


def set_at_import(degree: int, module: str) -> None:
    """Sets the degree at a solver module's import, unless a degree is pinned."""
    if _PINNED is None:
        _fc()["quadrature_degree"] = int(degree)


def pin(degree: int, by: str) -> None:
    """Sets the degree and pins it.  ``by`` names the caller, for the message of ``check``."""
    global _PINNED, _PINNED_BY
    _fc()["quadrature_degree"] = int(degree)
    _PINNED, _PINNED_BY = int(degree), by


def pin_single_scale(by: str) -> None:
    """Pins the single-scale macro degree."""
    pin(QDEG_MACRO_SINGLE_SCALE, by)


def pin_multiscale(by: str) -> None:
    """Pins the degree of a process that holds the macro and the micro solvers.

    The process has one degree, so the stated degrees must agree."""
    if not (QDEG_MACRO_MULTISCALE == QDEG_MICRO_PHASE_NEUTRAL == QDEG_MICRO_MIXED):
        raise RuntimeError(
            f"multiscale macro degree {QDEG_MACRO_MULTISCALE} != micro {QDEG_MICRO_PHASE_NEUTRAL} / "
            f"{QDEG_MICRO_MIXED}: one process has one global degree")
    pin(QDEG_MACRO_MULTISCALE, by)


def pin_micro_worker(by: str) -> None:
    """Pins the micro solver degree."""
    if QDEG_MICRO_PHASE_NEUTRAL != QDEG_MICRO_MIXED:
        raise RuntimeError("micro phase-neutral and mixed degrees differ: one process, one global")
    pin(QDEG_MICRO_PHASE_NEUTRAL, by)


def pinned() -> tuple[int | None, str | None]:
    """Returns the pinned degree and who pinned it; ``(None, None)`` when nothing is pinned."""
    return _PINNED, _PINNED_BY


def check(where: str) -> int:
    """Returns the degree in force.  Raises ``RuntimeError`` if it differs from the pin."""
    q = int(_fc()["quadrature_degree"])
    if _PINNED is not None and q != _PINNED:
        raise RuntimeError(f"{where}: form-compiler quadrature_degree is {q}, but {_PINNED_BY} pinned "
                           f"{_PINNED}: the global was written after the pin")
    return q
