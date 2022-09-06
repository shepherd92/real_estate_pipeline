#!/usr/bin/env python
"""Property data node."""

from __future__ import annotations
from pathlib import Path
import time

import pandas as pd
from pipeline.data import LeafDataNode


class ScrapeStatistics(LeafDataNode):
    """Class handling scraping statistics to detect anomalies."""

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path)
        self._history = pd.DataFrame(columns=[], dtype=int)
        self.metrics: dict[str, int] = {
            'time_stamp':                int(time.time()),
            'all_properties_seen':       0,
            'new_properties_found':      0,
            'created_properties':        0,
            'address_not_found':         0,
            'details_not_found':         0,
            'main_parameters_not_found': 0,
            'description_not_found':     0
        }

    def __setitem__(self, key: str, item: int) -> None:
        """Make data node act like a dict."""
        self.loaded = True
        self.metrics[key] = item

    def _load_content(self) -> None:
        """Load data from storage."""
        self._history = pd.read_csv(self._path, index_col=0, compression='zip', header=0)

    def _save_content(self) -> None:
        """Save data to storage."""
        assert self._path is not None
        self.load_if_possible()

        self._history = self._history.append(self.__dict__)
        self._history.to_csv(
            self._path,
            compression={'method': 'zip', 'archive_name': f"{self._path.with_suffix('.csv').name}"}
        )

    def _free_content(self) -> None:
        """Clear data from memory."""
        self._history = pd.DataFrame()
        self.metrics = {
            'time_stamp':                int(time.time()),
            'all_properties_seen':       0,
            'new_properties_found':      0,
            'created_properties':        0,
            'address_not_found':         0,
            'details_not_found':         0,
            'main_parameters_not_found': 0,
            'description_not_found':     0
        }

    def check(self, limits: dict[str, float]) -> bool:
        """
        Check if statistics for not found attributes, containers are OK.

        If there is a too high ratio of not found fields, assertion fails.
        """
        # require minimum 100 properties before checking statistics
        if self.metrics['created_properties'] < 100:
            return True

        created_properties = float(self.metrics['created_properties'])

        main_parameters_not_found_ratio = self.metrics['main_parameters_not_found'] / created_properties
        details_not_found_ratio = self.metrics['details_not_found'] / created_properties
        description_not_found_ratio = self.metrics['description_not_found'] / created_properties

        if main_parameters_not_found_ratio < limits['main_parameters_not_found_ratio']:
            return False
        if details_not_found_ratio < limits['details_not_found_ratio']:
            return False
        if description_not_found_ratio < limits['description_not_found_ratio']:
            return False

        return True
