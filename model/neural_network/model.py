#!/usr/bin/env python
"""Module responsible for the training of neural networks."""

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from shap import DeepExplainer, summary_plot

from model.model_base import Model
from model.neural_network.dataset import PropertyDataset
from model.neural_network.neural_network import NeuralNetwork


BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.
EPOCHS = 500
EARLY_STOPPING = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNetworkModel(Model):
    """Class to train a neural network model."""

    def __init__(self) -> None:
        """Construct a neural network trainer."""
        super().__init__()
        self._model = NeuralNetwork(0).to(DEVICE)
        self._feature_mean = pd.Series(dtype=float)
        self._feature_std = pd.Series(dtype=float)
        self._label_mean = 0.0
        self._label_std = 0.0

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit model to the training data."""
        self._feature_names = features.columns

        assert not features.isna().any().any(), \
            f'Features contain nans: {features.columns[features.isna().any()].tolist()}'

        x_train, x_validation, y_train, y_validation = \
            train_test_split(features, labels, test_size=0.1, random_state=42)

        # normalize features
        self._feature_mean = x_train.mean()
        self._feature_std = x_train.std()
        assert not self._feature_std.isna().any()
        x_train_normalized = (features - self._feature_mean) / self._feature_std
        assert not x_train_normalized.isna().any().any(), \
            f'Normalized features contain nans: {x_train_normalized.columns[x_train_normalized.isna().any()].tolist()}'
        x_validation_normalized = (x_validation - self._feature_mean) / self._feature_std
        assert not x_validation_normalized.isna().any().any()

        # normalize labels
        self._label_mean = y_train.mean()
        self._label_std = y_train.std()
        y_train_normalized = (labels - self._label_mean) / self._label_std
        y_validation_normalized = (y_validation - self._label_mean) / self._label_std

        train_set = PropertyDataset(x_train_normalized, y_train_normalized)
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        validation_set = PropertyDataset(x_validation_normalized, y_validation_normalized)
        validation_dataloader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

        print(f'Using {DEVICE} device')

        self._model = NeuralNetwork(len(train_set._features.columns)).to(DEVICE)

        loss = nn.MSELoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        writer = SummaryWriter()

        epochs = EPOCHS
        last_test_loss = 1e20
        for epoch in range(epochs):
            print(f'\rEpoch {epoch+1}, ', end='')
            train(train_dataloader, self._model, loss, optimizer, writer, epoch)
            test_loss = test(validation_dataloader, self._model, loss, writer, epoch)

            if EARLY_STOPPING and test_loss > last_test_loss:
                print()
                break

            last_test_loss = test_loss

        background = torch.cat([sublist[0] for sublist in train_dataloader])[:1000].to(DEVICE)
        test_images = torch.cat([sublist[0] for sublist in validation_dataloader])[:1000].to(DEVICE)
        explainer = DeepExplainer(self._model, background)

        shap_values = explainer.shap_values(test_images)
        shap_values_summary = np.absolute(np.array(shap_values)).mean(axis=0)
        shap_dataframe = pd.DataFrame(
            shap_values_summary, index=self._feature_names, columns=['importance']
        ).sort_values('importance', ascending=False)

        # summary_plot_figure = summary_plot(shap_values, test_images, plot_type="bar", feature_names=self._feature_names)

        print(shap_dataframe)

        writer.flush()
        writer.close()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Evaluate model performance."""
        normalized_features = (features - self._feature_mean) / self._feature_std
        predictions = self._model.forward(torch.Tensor(normalized_features.to_numpy()).to(DEVICE))
        predictions = predictions * self._label_std + self._label_mean
        return predictions.cpu().detach().numpy().squeeze()

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances."""
        importances = np.zeros(len(self._feature_names))
        std = np.zeros_like(importances)
        return pd.DataFrame(np.c_[importances, std], index=self._feature_names, columns=['importance', 'std'])


def train(dataloader: DataLoader, model: NeuralNetwork, loss_fn, optimizer, writer, epoch) -> float:
    """Run the actual training."""
    model.train()

    total_train_loss = 0.

    for X, y in dataloader:
        features, labels = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # Compute prediction error
        prediction = model(features)

        loss = loss_fn(prediction, labels)
        assert torch.isfinite(loss)
        total_train_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    writer.add_scalar("Loss/train", total_train_loss / len(dataloader), epoch)
    print(f"Training loss: {total_train_loss / len(dataloader):>7f}; ", end='')

    return total_train_loss / len(dataloader)


def test(dataloader, model, loss_fn, writer, epoch) -> float:
    """Test."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0., 0.

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    writer.add_scalar("Loss/test", test_loss, epoch)
    print(f'Validation loss: {test_loss:>7f}', end='')

    return test_loss
