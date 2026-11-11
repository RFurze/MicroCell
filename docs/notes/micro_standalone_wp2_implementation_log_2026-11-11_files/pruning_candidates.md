# WP2: candidates for further pruning, left in place

Detail of section 7 of `../micro_standalone_wp2_implementation_log_2026-11-11.md`, as the tree
stands at b18879b. Not in the brief's table, or out of WP2's scope. None was removed.

| Candidate | Where | Why it looks dead |
|---|---|---|
| `h_base_nd` | base; read by `fields.py` | the same object as `h_nd` since the spring went |
| `ContactSettings.nu`, `.k_asp_exponent` | `api.py` | unread since group 3 |
| `contact_model` settings field | contact settings | single-valued; the shear modules read it |
| `UNSUPPORTED_MODELS` and its branches | `contact_shear.py`, `contact_sample.py` | name the two removed laws (WP3 scope) |
| Hertz descriptor entries | `ideal_ht.py` | fed the removed summit law (WP3 scope; not checked in detail) |
| `effective_modulus`, `CONTACT_MODELS` | `contact.py` | no reader in the package |
| `MicroPhaseNeutralSolver = MicroPhaseNeutralMixedSolver` | end of the mixed file | back-compat alias; the API imports the classes by their own names |
| `auto_solve` | base constructor | the API always passes False |
| `logging.disable` at import, `logger` | base | §6.7; one `logger.warning` call remains and can never print |
| `_epsP_nd_capped` | base | written, never read |
| `_contact_restarts`, `_contact_restart_kinds` | contact class | counted by the rescue; the worker that printed them is not in this repo |
| `print_timing` and the timing summary | base | a settings field; WP4 |
