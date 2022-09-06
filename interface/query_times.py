#!/usr/bin/env python
"""Property data node."""

from __future__ import annotations
from pathlib import Path

import numpy as np

from interface.dataframe_container import DataframeContainer


class QueryTimesData(DataframeContainer):
    """Class handling all property data."""

    dtypes: dict[str, type] = {
        'identifier': np.int32,
        'queried_at': np.int32,
    }

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path, dtypes=self.dtypes)
