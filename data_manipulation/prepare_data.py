#!/usr/bin/env python
"""Module to filter data to train with."""

import sys
from typing import Any

import numpy as np
import pandas as pd

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode


class PrepareData(Step):
    """Filter data to train with."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Filter data to train with."""
        print('Filtering data...', flush=True)

        data_repository['configuration'].load()
        parameters: dict[str, Any] = data_repository['configuration'].params['filtering_parameters']
        features_to_use: list[str] = parameters['features']
        valid_price_range: float = parameters['valid_price_range']
        column_values: dict[str, list[Any]] = parameters['column_values']
        location_boundaries: dict[str, float] = parameters['location_rectangle']

        data_repository['6_merged']['properties'].load()
        prepared_data = data_repository['6_merged']['properties'].dataframe.copy()

        prepared_data = _filter_valid_locations(
            prepared_data,
            parameters['maximum_geocoder_uncertainty']
        )
        prepared_data = _filter_column_values(prepared_data, column_values)
        prepared_data = _drop_constant_features(prepared_data)
        prepared_data = _drop_correlating_features(prepared_data, parameters['threshold_correlation'])
        prepared_data = _impute_missing_values(prepared_data)
        prepared_data = _filter_properties_with_valid_price(prepared_data, valid_price_range)
        # prepared_data = _filter_locations(prepared_data, location_boundaries)
        # filtered_data = _filter_features(filtered_data, features_to_use)

        data_repository['7_filtered']['description'].dataframe = prepared_data.describe().transpose()
        data_repository['7_filtered']['correlation'].dataframe = prepared_data.corr()
        data_repository['7_filtered']['properties'].dataframe = prepared_data
        data_repository['7_filtered'].flush()


def _impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace([-np.inf, -2, -1, np.inf], np.nan)
    imputer = MissForest()
    features = data.drop('price', axis=1)
    data_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    data_imputed['price'] = data['price']

    return data_imputed


def _drop_constant_features(data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = data.dropna(axis=1, how='all')
    nunique = filtered_data.nunique()
    cols_to_drop = nunique[nunique == 1].index
    filtered_data = filtered_data.drop(cols_to_drop, axis=1)
    return filtered_data


def _filter_valid_locations(
    data_to_filter: pd.DataFrame,
    maximum_uncertainty: float
) -> pd.DataFrame:

    filtered_data = data_to_filter.loc[data_to_filter['geocoder_location_found']]
    filtered_data = filtered_data.loc[filtered_data['geocoder_uncertainty'] < maximum_uncertainty]
    return filtered_data


def _filter_properties_with_valid_price(
    data_to_filter: pd.DataFrame,
    valid_price_range: dict[str, float]
) -> pd.DataFrame:
    data_to_filter = data_to_filter[
        (data_to_filter['price'] > valid_price_range['minimum']) &
        (data_to_filter['price'] < valid_price_range['maximum'])
    ]
    return data_to_filter


def _filter_locations(data_to_filter: pd.DataFrame, location_boundaries: dict[str, float]) -> pd.DataFrame:

    data_to_filter = data_to_filter.loc[
        (int(location_boundaries['min_latitude'] * 1e6) <= data_to_filter['geocoder_latitude']) &
        (int(location_boundaries['max_latitude'] * 1e6) >= data_to_filter['geocoder_latitude']) &
        (int(location_boundaries['min_longitude'] * 1e6) <= data_to_filter['geocoder_longitude']) &
        (int(location_boundaries['max_longitude'] * 1e6) >= data_to_filter['geocoder_longitude'])
    ]

    return data_to_filter


def _drop_correlating_features(data_to_filter: pd.DataFrame, threshold_correlation: float) -> pd.DataFrame:

    corr_matrix = data_to_filter.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    columns_to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold_correlation)
    ]

    print(f'Drop features with high correlations: {columns_to_drop}')

    filtered_data = data_to_filter.drop(columns_to_drop, axis=1)
    return filtered_data


def _filter_column_values(data_to_filter: pd.DataFrame, column_values: dict[str, list[Any]]) -> pd.DataFrame:

    for column_name, values in column_values.items():
        data_to_filter = data_to_filter.loc[data_to_filter[column_name].isin(values)]

    return data_to_filter


def _filter_features(data_to_filter: pd.DataFrame, features_to_use: list[str]) -> pd.DataFrame:

    result = data_to_filter[features_to_use]

    return result
