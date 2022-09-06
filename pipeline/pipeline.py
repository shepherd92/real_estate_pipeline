#!/usr/bin/env python
"""Pipeline with steps."""

from __future__ import annotations
from abc import ABC, abstractmethod

from pipeline.data import ParentDataNode


class Step(ABC):
    """Represent a general step in the pipeline."""

    @abstractmethod
    def __call__(self, data_repository: ParentDataNode):
        """Run step."""


class Pipeline:
    """Represent a pipeline with steps."""

    def __init__(self, data_repository: ParentDataNode) -> None:
        """Construct a pipeline."""
        self._data_repository = data_repository
        self._steps: list[Step] = []

    def get_steps(self) -> list[Step]:
        """Return steps contained in pipeline."""
        return self._steps

    def add_steps(self, steps: list[Step]) -> None:
        """Add steps to pipeline."""
        self._steps.extend(steps)

    def run(self) -> None:
        """Run whole pipeline."""
        for step in self._steps:
            step(self._data_repository)
            self._data_repository.free()
