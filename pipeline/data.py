#!/usr/bin/env python
"""Abstract data node classes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock


class DataNode(ABC):
    """Abstract data class to store data."""

    @abstractmethod
    def load(self) -> None:
        """Load data from storage."""

    @abstractmethod
    def load_if_possible(self) -> None:
        """Load data from storage if the respective path is found."""

    def reload(self) -> None:
        """Load data from storage even if it is already loaded."""
        self.free()
        self.load()

    @abstractmethod
    def flush(self) -> None:
        """Flush data to storage and free memory."""

    @abstractmethod
    def save(self) -> None:
        """Save data to storage."""

    @abstractmethod
    def free(self) -> None:
        """Clear data from memory."""

    @abstractmethod
    def delete(self) -> None:
        """Delete all children data nodes and contents from the storage."""


class ParentDataNode(dict, DataNode):
    """Data class representing a group of further data nodes."""

    def load(self) -> None:
        """Load data from storage.

        If the data is loaded, it is not loaded again.
        """
        for data in self.values():
            data.load()

    def load_if_possible(self) -> None:
        """Load data from storage if the respective path is found."""
        for data in self.values():
            data.load_if_possible()

    def save(self) -> None:
        """Save data to storage."""
        for data in self.values():
            data.save()

    def flush(self) -> None:
        """Flush data to storage and free memory."""
        for data in self.values():
            data.flush()

    def free(self) -> None:
        """Clear data from memory."""
        for data in self.values():
            data.free()

    def delete_child(self, key: str) -> None:
        """Delete a child node, also removes files from the storage."""
        self[key].delete()
        del self[key]

    def delete(self) -> None:
        """Delete all children data nodes."""
        for data in self.values():
            data.delete()


class LeafDataNode(DataNode):
    """Data node representing a leaf node in the tree structure."""

    def __init__(self, path: Path) -> None:
        """Construct empty data entity."""
        self.lock = Lock()
        self._path = path
        self.loaded = False

    @property
    def path(self) -> Path:
        """Return the path member."""
        return self._path

    def load(self) -> None:
        """Load self, ignore children."""
        if self.loaded:
            return

        assert self.path is not None, 'Path of the data node is not set, cannot load.'
        assert self.path.is_file(), 'Path does not exist, cannot reload.'

        with self.lock:
            self.loaded = True
            self._load_content()

    def load_if_possible(self) -> None:
        """Load data from storage if the respective path is found."""
        if self.loaded:
            return

        if not self.path.is_file():
            return

        with self.lock:
            self.loaded = True
            self._load_content()

    def save(self) -> None:
        """Save self, ignore children."""
        assert self.loaded, 'Data is not loaded, nothing to save.'

        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save_content()

    def save_if_loaded(self) -> None:
        """Save content to file if it is loaded."""
        if self.loaded:
            self.save()

    def flush(self) -> None:
        """Flush data to storage and free memory."""
        self.save_if_loaded()
        self.free()

    def free(self) -> None:
        """Clear data from memory."""
        with self.lock:
            self._free_content()
            self.loaded = False

    def delete(self) -> None:
        """Delete all children data nodes."""
        self.free()

        with self.lock:
            self.path.unlink()

    @abstractmethod
    def _load_content(self) -> None:
        """Load content from disk."""

    @abstractmethod
    def _save_content(self) -> None:
        """Save content to disk."""

    @abstractmethod
    def _free_content(self) -> None:
        """Free data from memory."""
