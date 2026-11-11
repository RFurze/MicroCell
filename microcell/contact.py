"""Contact identities and the shared contact error (SI), independent of FEniCS.

The macro owns mean deformation.  The parent's ``legacy_patch_v1`` and
``hertz_summit_v1`` laws are not in this repo (MicroCell WP2, 11 Nov 2026).
"""

#: periodic elastic (FFT) contact solve per micro cell, ehl.elastic_contact (16 Oct 2026)
FFT = "fft_elastic_v1"


class ContactResolutionError(RuntimeError):
    """Valid geometry but an unresolved fluid obstruction mesh."""
