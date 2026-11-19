"""The surfaces of a micro cell (NumPy only; ``microcell/film.py`` passes them to the solver).

``frontend``   ``Roughness``: a generated Gaussian surface, a height array, a stationary and
               translating pair, the analytic cosine cell
``generation`` the IAAFT generator for a Gaussian autocorrelation and height distribution
``quantiles``  the normal quantile function used by the generator
``evaluator``  ``FourierSurface``: band-limited reconstruction on any grid, at any displacement
``runtime``    ``SurfaceField``, the surfaces as the solver reads them; the undeformed film
               of the contact solve; the motion of the translating surface
``pair``       the displacement of the translating surface

Height arrays are gap perturbations ``z[j_y, i_x]`` [m]: ``h = H0 + HT*t + z``, and ``z < 0`` is
an asperity.
"""
from __future__ import annotations

from .evaluator import Bandwidth, FourierSurface

__all__ = ["Bandwidth", "FourierSurface"]
