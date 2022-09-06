#!/usr/bin/env python
"""Module to train models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode

from model.model_base import Model, ModelType
from model.neural_network.model import NeuralNetworkModel
from model.linear_regression.model import LinearRegressionModel
from model.random_forest.model import RandomForestModel


class Train(Step):
    """Merge all data to create the final database for training."""

    def __init__(self, model_type: ModelType) -> None:
        """Construct a training step."""
        self._model_type = model_type
        if self._model_type == ModelType.LINEAR_REGRESSION:
            self.model: Model = LinearRegressionModel()
        elif self._model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestModel()
        elif self._model_type == ModelType.NEURAL_NETWORK:
            self.model = NeuralNetworkModel()

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Merge all data to create the final database for training."""
        print('Training...', flush=True)

        data_repository['configuration'].load()
        data_repository['7_filtered']['properties'].load()
        whole_dataset: pd.DataFrame = data_repository['7_filtered']['properties'].dataframe

        print(f'Number of properties in whole dataset: {len(whole_dataset)}')

        whole_dataset.loc[:, 'price'] = np.log(whole_dataset['price'])

        features = whole_dataset.drop('price', axis=1)
        labels = whole_dataset['price']

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        data_repository['8_training']['predictions'].dataframe = \
            pd.DataFrame(np.exp(np.c_[y_test, predictions]), columns=['labels', 'predictions'])
        data_repository['8_training'].flush()

        relative_errors = (np.exp(predictions) - np.exp(y_test)) / np.exp(y_test)
        mean_relative_error = np.sqrt(np.mean(relative_errors**2))
        print(f'Relative error is {mean_relative_error * 100:.4f}%')

        predictions_figure = plot_predictions(y_test, predictions)
        data_repository['9_evaluation']['predictions_labels'].figure = predictions_figure

        feature_importances = self.model.get_feature_importances()
        feature_importances_figure = plot_feature_importances(feature_importances)
        data_repository['9_evaluation']['feature_importances'].figure = feature_importances_figure

        data_repository['9_evaluation'].flush()


def plot_predictions(y_test, predictions) -> Figure:
    """Plot predictions and labels on a scatter plot."""
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.scatter(np.exp(y_test), np.exp(predictions), c='blue', s=1)
    axis.axline([0, 0], [1, 1], color='red')
    axis.set_xlim([0, 1e8])
    axis.set_ylim([0, 1e8])
    axis.set_title("Prediction Accuracy")
    axis.set_xlabel("Real Price")
    axis.set_ylabel("Predicted Price")

    return figure


def plot_feature_importances(feature_importances: pd.DataFrame) -> Figure:
    """Plot feature importances of real estates."""
    figure, axis = plt.subplots(figsize=(8, 6), dpi=800)
    feature_importances['importance'].plot.bar(yerr=feature_importances['std'], ax=axis)
    axis.set_title("Feature Importances")
    axis.set_ylabel("Mean decrease in impurity")
    figure.tight_layout()
    return figure
