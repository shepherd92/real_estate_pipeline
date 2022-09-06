#!/usr/bin/env python
"""General property data node."""

from __future__ import annotations

from pathlib import Path
from shutil import move
from time import time
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.data import ParentDataNode, LeafDataNode


class DataframeContainer(LeafDataNode):
    """Class handling all property data."""

    def __init__(self, path: Path, dtypes: Optional[dict[str, type] | type] = None) -> None:
        """Construct a high level data node."""
        super().__init__(path)

        self.dtypes = dtypes
        if isinstance(self.dtypes, dict):
            self._dataframe = \
                pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in self.dtypes.items()})
        else:
            self._dataframe = pd.DataFrame()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Set content of the data node."""
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set content of the data node."""
        self.loaded = True
        self._dataframe = dataframe

    def add_data(self, new_data: pd.DataFrame) -> None:
        """Add new raw data and merge it into this one."""
        if len(new_data) == 0:
            return

        self.load_if_possible()
        self.dataframe = pd.concat([self.dataframe, new_data], axis=0)

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

    def get_identifiers(self) -> np.ndarray:
        """Return existing property identifiers."""
        self.load_if_possible()
        return np.array(self.dataframe.index.to_list())

    def difference(self, old_data: DataframeContainer) -> pd.DataFrame:
        """Create a new dataframe with rows only found in self."""
        self.load_if_possible()
        old_data.load_if_possible()

        all_identifiers = self.get_identifiers()
        old_identifiers = old_data.get_identifiers()
        new_identifiers = np.setdiff1d(all_identifiers, old_identifiers, assume_unique=True)

        return self.dataframe.loc[new_identifiers]

    def _load_content(self) -> None:
        """Load data from storage."""
        if self.dtypes is not None:
            self.dataframe = pd.read_csv(
                self._path, index_col=0, compression='zip', header=0, low_memory=False, dtype=self.dtypes
            )
        else:
            self.dataframe = pd.read_csv(
                self._path, index_col=0, compression='zip', header=0, low_memory=False
            )

    def _save_content(self) -> None:
        """Save data to storage.

        If the data file exists, first move it to a temporary file.
        This way there will be no data loss if the program stops during saving.
        """
        assert self._path is not None
        temp_file_name: Optional[Path] = None

        if self._path.is_file():
            current_timestamp = str(int(time()))
            temp_file_name = self._path.parent / f'{current_timestamp}.zip'
            move(self._path, temp_file_name)

        self.dataframe.to_csv(
            self._path,
            float_format='%.6f',
            compression={'method': 'zip', 'archive_name': f"{self._path.with_suffix('.csv').name}"}
        )

        if temp_file_name is not None:
            temp_file_name.unlink()

    def _free_content(self) -> None:
        """Clear data from memory."""
        self.dataframe = pd.DataFrame()
