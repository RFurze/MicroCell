"""``microcell.roughness`` -- the surfaces of a micro cell (from the parent's ``ehl.roughness``).

Pure data and NumPy: no FEniCS and no SciPy are imported anywhere in this package; the FE
adapter is ``microcell/surface_ht.py``.

Modules
-------
``frontend``   :class:`Roughness`, the front end: a generated Gaussian surface, a height array,
               a stationary + translating pair, the analytic cosine cell
``generation`` the IAAFT generator (Pérez-Ràfols & Almqvist 2019) for a Gaussian
               autocorrelation and a Gaussian height distribution
``quantiles``  the NumPy-only normal quantile function the generator uses
``evaluator``  :class:`FourierSurface` -- band-limited reconstruction at any grid / displacement
``runtime``    :class:`SurfaceField`, the member surfaces as the solver reads them; the rigid
               film ``fft_elastic_v1`` reads; the translating member's motion
``pair``       the kinematics of the translating member

The parent's recipes, targets, reports, mode lists, file inputs, prepared artefacts and surface
store (``spec``, ``targets``, ``prepare``, ``report``, ``modes``, ``assets``, ``case``) and its
statistics module (``stats``) are not in this repo (removed in MicroCell WP3, 11 Nov 2026).

Sign convention: arrays are gap perturbations ``z[j_y, i_x]``,
``h = H0 + HT*t + z``; ``z < 0`` is an asperity.
"""
from __future__ import annotations

from .evaluator import Bandwidth, FourierSurface

__all__ = ["Bandwidth", "FourierSurface"]
