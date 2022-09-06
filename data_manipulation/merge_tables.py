#!/usr/bin/env python
"""Module to merge all data tables."""

from functools import reduce

import pandas as pd

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode


class MergeTables(Step):
    """Merge all data to create the final database for training."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Merge all data to create the final database for training."""
        print('Merge data tables...', flush=True)

        data_repository['configuration'].load()

        data_repository['2_geolocated']['properties'].load()
        data_repository['3_location_features']['locations'].load()
        data_repository['4_time_features']['properties'].load()
        data_repository['5_transformed']['properties'].load()

        data_frames = [
            data_repository['2_geolocated']['properties'].dataframe,
            data_repository['4_time_features']['properties'].dataframe,
            data_repository['5_transformed']['properties'].dataframe
        ]
        merged_dataframe_without_location_features = reduce(
            lambda left, right: pd.merge(left, right, left_index=True, right_index=True), data_frames
        )

        rows_with_nan_locations = merged_dataframe_without_location_features[
            ~merged_dataframe_without_location_features['geocoder_location_found']
        ]
        rows_with_valid_locations = merged_dataframe_without_location_features.loc[(
            merged_dataframe_without_location_features['geocoder_location_found']
        )]

        rows_with_location_features = pd.merge(
            rows_with_valid_locations, data_repository['3_location_features']['locations'].dataframe,
            left_on=['geocoder_latitude', 'geocoder_longitude'], right_index=True
        )
        merged_dataframe = pd.concat([rows_with_location_features, rows_with_nan_locations])

        data_repository['6_merged']['properties'].dataframe = merged_dataframe
        data_repository['6_merged'].flush()
