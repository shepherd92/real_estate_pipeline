#!/usr/bin/env python
"""Functions calculating a distance decay."""


from abc import ABC

import numpy as np


class DistanceDecay(ABC):
    """Return a weight to be used for a given point when calculating a feature."""

    def __init__(self, parameter: float = 1.0) -> None:
        """Set parameters of the decay function."""
        self.parameter = parameter

    def __call__(self, distance: float) -> float:
        """Return the decay value."""
        raise NotImplementedError


class ConstantDecay(DistanceDecay):
    """Return a constant decay."""

    def __call__(self, distance: float) -> float:
        """Return the decay value."""
        return 1.


class PowerLawDecay(DistanceDecay):
    """Return a power law decay."""

    maximum: float = 1.  # 1 = f(x) = x^-p: in this case x = 1

    def __call__(self, distance: float) -> float:
        """Return the decay value."""
        if np.isclose(distance, 0.):
            return self.maximum

        return distance**(- self.parameter)


class ExponentialDecay(DistanceDecay):
    """Return an exponential decay."""

    def __call__(self, distance: float) -> float:
        """Return the decay value."""
        return np.exp(- distance / self.parameter)
