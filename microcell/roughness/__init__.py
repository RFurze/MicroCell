"""``ehl.roughness`` -- the one surface module (Case API plan §6.1, §15.3).

Pure data and NumPy: generation, statistics, the canonical Fourier evaluator,
strict file inputs, prepared artefacts and surface-pair composition.  No
FEniCS and no SciPy are imported anywhere in this package; the FE adapters
(stage 2, R4) live with the micro solver and consume this package.

Modules
-------
``quantiles``  NumPy-only HPD quantile functions (replace scipy.stats)
``generation`` IAAFT port (Pérez-Ràfols & Almqvist 2019), PSD/HPD inputs,
               legacy Sm fit, the filtered Gaussian generator
``modes``      cosine mode-list surfaces (idealised profiles, the
               roughness-location study's band-limited generator)
``stats``      Sa, Sq, Sp, Sv, Sz, Ssk, Sku, Sm, summit counts (gap convention)
``evaluator``  :class:`FourierSurface` -- band-limited reconstruction at any
               grid / displacement; ``bilinear_nodal`` is the legacy path
``assets``     strict ``.npy`` + JSON inputs; immutable ``.npz`` + JSON
               artefacts with sha256 identity
``spec``       frozen per-surface and pair recipes
``pair``       stationary + translating composition, kinematics, legacy
               ``roughness_mode`` mapping
``targets``    Sa/Sq target policies and feasibility (R2)
``prepare``    recipe -> resolved array -> prepared artefact (R2)
``report``     report data (R2)
``case``       a Case's arrangement prepared into the surface store, and its
               descriptor (R3)
``runtime``    the descriptor loaded and verified in a micro worker; the
               rigid film ``fft_elastic_v1`` reads (R4); the translating
               member's motion (R5a)

The FE adapter is ``microscale/src/functions/surface_ht.py`` (R4).

Sign convention: arrays are gap perturbations ``z[j_y, i_x]``,
``h = H0 + HT*t + z``; ``z < 0`` is an asperity.
"""
from __future__ import annotations

from .evaluator import EVALUATOR_REVISION, Bandwidth, FourierSurface, bilinear_nodal
from .generation import GENERATOR_REVISION
from .stats import surface_stats

__all__ = ["EVALUATOR_REVISION", "GENERATOR_REVISION", "Bandwidth", "FourierSurface",
           "bilinear_nodal", "surface_stats"]
