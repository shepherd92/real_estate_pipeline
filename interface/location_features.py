#!/usr/bin/env python
"""Property data node."""

from pathlib import Path

import pandas as pd

from pipeline.data import ParentDataNode
from interface.dataframe_container import DataframeContainer


class LocationFeatures(DataframeContainer):
    """Class handling location features.

    Index of the dataframe is the pair of latitude and longitude coordinates.
    Both coordinates are multiplied by 1e6 and treated as integers.
    """

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path)

    def _load_content(self) -> None:
        """Load data from storage."""
        self.dataframe: pd.DataFrame = \
            pd.read_csv(self._path, index_col=[0, 1], compression='zip', header=0, dtype=self.dtypes)
        latitude = self.dataframe.index.levels[0].astype(int)
        longitude = self.dataframe.index.levels[1].astype(int)
        self.dataframe.index = self.dataframe.index.set_levels([latitude, longitude])

    def add_data_node(self, additional_data: ParentDataNode) -> None:
        """Add new data node and merge it into this one."""
        if len(additional_data) == 0:
            return

        self.load_if_possible()

        for data in additional_data.values():
            data.load()

        self.dataframe = pd.concat(
            [self.dataframe] + [data.dataframe for data in additional_data.values()], axis=0
        )
