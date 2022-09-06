#!/usr/bin/env python
"""Dataset for training."""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PropertyDataset(Dataset):
    """Dataset containing all property data."""

    def __init__(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Construct the total property dataset."""
        assert not features.isna().any().any()
        features_mean = features.mean()
        features_std = features.std()
        normalized_features = (features - features_mean) / features_std
        self._features = normalized_features
        assert not self._features.isna().any().any()

        self._labels = labels

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._features)

    def __getitem__(self, index: int) -> tuple:
        """Return the item from the dataset indicated by the index."""
        features = self._features.iloc[index].to_numpy(dtype=np.float32)
        label = np.array([np.float32(self._labels.iloc[index])])
        return features, label
