"""Form-compiler flags of the process, applied once before the first form is built.

The quadrature degree is set separately, in ``microcell/form_compiler.py``.
"""

from __future__ import annotations

#: Form-compiler flags.  ``optimize`` is left off: the form-level optimiser can mishandle
#: ``conditional()`` expressions such as the cavitation penalty.
MACRO_FORM_COMPILER: tuple[tuple[str, object], ...] = (
    ("cpp_optimize", True),
    ("cpp_optimize_flags", "-O2"),
)

_APPLIED = False


def apply_form_compiler_flags() -> None:
    """Writes ``MACRO_FORM_COMPILER`` into the dolfin form-compiler parameters."""
    from fenics import parameters
    for key, value in MACRO_FORM_COMPILER:
        parameters["form_compiler"][key] = value


def apply_macro_process_settings() -> None:
    """Applies the form-compiler flags and records that this process has them."""
    global _APPLIED
    apply_form_compiler_flags()
    _APPLIED = True


def applied() -> bool:
    """True once ``apply_macro_process_settings`` has run in this process."""
    return _APPLIED
