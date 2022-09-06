#!/usr/bin/env python
"""Module to locate given addresses."""

import pandas as pd

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode


class CalcTimeFeatures(Step):
    """Calculate time features."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Calculate time features."""
        print('Calculating time features...', flush=True)

        data_repository['configuration'].load()

        data_repository['1_scraped']['properties'].load_if_possible()
        data_repository['4_time_features']['properties'].load_if_possible()
        new_data = data_repository['1_scraped']['properties'].difference(
            data_repository['4_time_features']['properties']
        )

        if len(new_data) == 0:
            return

        data_repository['1_scraped']['query_times'].load()
        query_times = data_repository['1_scraped']['query_times'].dataframe
        query_times_with_indices = query_times.loc[query_times['identifier'].isin(new_data.index)]
        data_with_time_features = query_times_with_indices.groupby('identifier').apply(_aggregate_groups)

        data_repository['4_time_features']['properties'].add_data(data_with_time_features)
        data_repository['4_time_features'].flush()


def _aggregate_groups(group) -> pd.Series:
    result = {
        'min_timestamp': group['queried_at'].min(),
        'max_timestamp': group['queried_at'].max()
    }
    return pd.Series(result)
