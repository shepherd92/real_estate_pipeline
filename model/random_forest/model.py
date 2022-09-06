#!/usr/bin/env python
"""Random forest model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model.model_base import Model


class RandomForestModel(Model):
    """Random forest model for price regression."""

    def __init__(self) -> None:
        """Construct a random forest model."""
        super().__init__()
        self._model = RandomForestRegressor(
            n_estimators=1000,
            bootstrap=True,
            max_features='log2',
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=100
        )

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
        importances = self._model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self._model.estimators_], axis=0)
        importances_dataframe = pd.DataFrame(
            np.c_[importances, std], index=self._feature_names, columns=['importance', 'std']
        )
        sorted_importances = importances_dataframe.sort_values('importance', ascending=False)
        return sorted_importances
