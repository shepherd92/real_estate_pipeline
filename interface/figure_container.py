#!/usr/bin/env python
"""General figure data node."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pipeline.data import LeafDataNode


class FigureContainer(LeafDataNode):
    """Class handling figures."""

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path)
        self._figure: Figure = plt.figure()

    @property
    def figure(self) -> Figure:
        """Set content of the data node."""
        return self._figure

    @figure.setter
    def figure(self, figure: Figure) -> None:
        """Set content of the data node."""
        self.loaded = True
        self._figure = figure

    def _load_content(self) -> None:
        """Load data from storage."""
        raise NotImplementedError

    def _save_content(self) -> None:
        """Save data to storage."""
        assert self._path is not None

        self._figure.savefig(self.path)

    def _free_content(self) -> None:
        """Clear data from memory."""
        self._figure = plt.figure()
