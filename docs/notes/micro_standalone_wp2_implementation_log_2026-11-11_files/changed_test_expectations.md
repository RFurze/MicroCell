# WP2: changed test expectations (Table 6)

Detail of section 9 of `../micro_standalone_wp2_implementation_log_2026-11-11.md`. All in
`tests/test_parity.py`, except the last row. The comparison of results is unchanged: same
fields, same tolerance.

| Change | Since | Why |
|---|---|---|
| `REMOVED_SETTINGS` (`spring_pointwise`, `use_reduced_modulus`, `overclosure_measure`, `gauge_lift`): the built settings must not carry the key, and the reference's stored value must equal the frozen value exactly | groups 2, 3, 5 | the brief: a removed setting is checked, not skipped |
| `REMOVED_UNREAD_SETTINGS` (`contact_E`, `contact_nu`, `k_asp_exponent`): the built settings must not carry the key, and the reference's `contact_model` must be `fft_elastic_v1` | group 3 | only the removed law read them, so there is no value to freeze |
| `FULL_FILM_CONTACT_TAG`: a full-film reference's result tag `legacy_patch_v1` is compared as `off` | group 3 | §6.2 |
| `REPLACED_SETTINGS` (`phase_average` -> `pair_branch`, `off` -> False, `contact_table_v1` -> True): the built settings must carry the new key with the mapped value and type, and not the old key | group 8 | §6.5 |
| `settings_from_reference` no longer passes the `legacy` keys `contact_E`, `contact_model`, `use_reduced_modulus`, `overclosure_measure`, and (b18879b) `k_spring` | groups 3, 2 | the API no longer has them |
| New test `test_removed_setting_is_checked`: with the frozen value of `spring_pointwise` set wrong, cell c8 must fail on exactly `settings.spring_pointwise` | group 2 | a negative control for the new check |
| New record `case.k_spring` in every cell (`spring_acted`): fails when the parent's spring acted on the reference cell, that is a transient full-film cell with `k_spring > 0`, or a mixed cell with `k_spring` between 0 and 1e19 | group 2 (b18879b) | the spring is removed; the stored value is checked, not passed on |
| New test `test_removed_spring_is_checked`: c5 given `k_spring = 1e20` in its case record must fail on exactly `case.k_spring` | group 2 (b18879b) | a negative control for that record |
| Reference `tests/references/c5_cosine_fullfilm.npz` re-made with `k_spring = 0` (Table 2b) | a1395f4 | Rob's decision, §2 |
