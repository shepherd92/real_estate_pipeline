#!/usr/bin/env python
"""Linear regression model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from model.model_base import Model


class LinearRegressionModel(Model):
    """Linear regression model to predict real estate prices."""

    def __init__(self) -> None:
        """Construct linear regression model."""
        super().__init__()
        self._model = LinearRegression()

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit model to the training data."""
        self._feature_names = features.columns
        self._model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Evaluate model performance."""
        predictions = self._model.predict(features)
        return predictions

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances."""
        importances = self._model.coef_
        std = np.zeros_like(importances)
        return pd.DataFrame(np.c_[importances, std], index=self._feature_names, columns=['importance', 'std'])
