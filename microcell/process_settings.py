"""Process-wide settings the macro module makes at import (PETSc options and
form-compiler flags), stated once.

[Simplification phase D5, 2026-10-12] Moved VERBATIM from the module level of
``macro_HMM_penalty_transient_EHL.py``, which calls ``apply_macro_process_settings()``
at the same point of its import as before.  Until D5 the micro pool's spawned
workers under ``ehl run`` got these by re-running the multiscale launcher (which
imports the macro module); since D5 the pool initializer applies them explicitly
(``run_microscale_phase_neutral._init_worker``, asked by ``ehl.drivers.multi``).
Workers that still re-run their parent's ``__main__`` (a legacy launch, the bench,
the smoke test) get from it what they always got.

The D0 check (2026-10-12) measured the form-compiler flags below equal to
DOLFIN's defaults in every route, so setting them is a no-op today; they are
kept so the statement stays explicit.  The quadrature degree is not here: it is
pinned per process (``microcell/form_compiler.py``).
"""

from __future__ import annotations

# The parent also set five PETSc KSP options here (ksp_initial_guess_nonzero, ksp_gmres_restart,
# ksp_rtol, ksp_max_it, ksp_error_if_not_converged) for its macro solve.  The cell solves with a
# direct factorisation and never read them (PETSc reported all five unused at exit), so they
# are not set here.

#: form-compiler flags set at macro-module import
# NOTE: Do NOT set parameters["form_compiler"]["optimize"] = True here.
# FFC's form-level optimizer can mishandle conditional() expressions
# (like the cavitation penalty), producing incorrect element tensors.
MACRO_FORM_COMPILER: tuple[tuple[str, object], ...] = (
    ("cpp_optimize", True),
    ("cpp_optimize_flags", "-O2"),
)

_APPLIED = False


def apply_form_compiler_flags() -> None:
    from fenics import parameters
    for key, value in MACRO_FORM_COMPILER:
        parameters["form_compiler"][key] = value


def apply_macro_process_settings() -> None:
    """The form-compiler flags, and remember that this process has them."""
    global _APPLIED
    apply_form_compiler_flags()
    _APPLIED = True


def applied() -> bool:
    """Has this process applied the macro module's settings (imported it)?"""
    return _APPLIED
