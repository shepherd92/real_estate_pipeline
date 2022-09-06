#!/usr/bin/env python
"""Property data node."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import json
from pipeline.data import LeafDataNode


class Configuration(LeafDataNode):
    """Class handling all property data."""

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path)
        self.params: dict[str, Any] = {}

    def _load_content(self) -> None:
        """Load data from storage."""
        assert self._path is not None
        with open(self._path, encoding='utf-8') as config_file:
            self.params = json.load(config_file)

    def _save_content(self) -> None:
        """Do nothing to prevent accidental overwriting of configuration."""
        return

    def _free_content(self) -> None:
        """Clear data from memory."""
        self.params = {}
