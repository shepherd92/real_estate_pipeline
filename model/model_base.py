#!/usr/bin/env python
"""Module to merge all data tables."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd


class ModelType(Enum):
    """Enum to denote model type."""

    LINEAR_REGRESSION = 0
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2


class Model(ABC):
    """Model to predict prices."""

    def __init__(self) -> None:
        """Construct a model."""
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(self, features, labels) -> None:
        """Fit model to the training data."""

    @abstractmethod
    def predict(self, features) -> np.ndarray:
        """Fit model to the training data."""

    @abstractmethod
    def get_feature_importances(self) -> pd.DataFrame:
        """Return the feature importances."""
