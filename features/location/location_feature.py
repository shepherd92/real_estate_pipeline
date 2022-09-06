#!/usr/bin/env python
"""Locaction features."""

from abc import ABC
from itertools import groupby
from operator import itemgetter
from typing import Callable, Optional, NamedTuple

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

from features.location.overpass_handler import (
    Location,
    OverpassQueryResult,
    location_to_point,
)

from features.location.aggregation import (
    calculate_minimum_distance,
    integrate_points,
    integrate_line_strings,
    integrate_polygons
)
from features.location.distance_decay import DistanceDecay
from features.location.tag_collection import Tag, tag_collection


class FeatureParameters(NamedTuple):
    """Represent parameters for a group of features."""

    root_name: str
    radius: float
    distance_decay: Optional[DistanceDecay] = None

    @property
    def tags(self) -> list[Tag]:
        """Return relevant tags from tag collection."""
        return tag_collection[self.root_name]


class LocationFeature(ABC):
    """Represent a location feature."""

    def __init__(self, params: FeatureParameters):
        """Construct a location feature."""
        self.root_name: str = params.root_name
        self.radius: float = params.radius
        self.tags: list[Tag] = params.tags

    def calc(self, overpass_query_result: OverpassQueryResult, location: Location) -> float:
        """Calculate feature value based on its parameters."""
        raise NotImplementedError

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return feature name."""
        return f'{self.root_name}_{int(self.radius)}_{self.__class__.__name__}'


class MinimumDistance(LocationFeature):
    """Represent a minimum distance location feature."""

    def calc(self, overpass_query_result: OverpassQueryResult, location: Location) -> float:
        """Calculate feature value based on its parameters."""
        base_point = location_to_point(location)

        found_objects = self._search(overpass_query_result, location, self.radius, self.tags)
        if not found_objects.is_empty:
            result = calculate_minimum_distance(base_point, found_objects)
        else:
            result = np.nan

        return result


class ClosestNode(MinimumDistance):
    """Represent a minimum distance location feature."""

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_nodes_around_location(location, radius, tags_to_search)


class ClosestWay(MinimumDistance):
    """Represent a minimum distance location feature."""

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_ways_around_location(location, radius, tags_to_search)


class ClosestArea(MinimumDistance):
    """Represent a minimum distance location feature."""

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_areas_around_location(location, radius, tags_to_search)


class IntegratedFeature(LocationFeature):
    """Represent an integrated location feature."""

    def __init__(self, params: FeatureParameters):
        """Construct a location feature."""
        super().__init__(params)
        assert params.distance_decay is not None
        self.distance_decay: DistanceDecay = params.distance_decay
        self._integrate: Callable[[DistanceDecay, Point, BaseGeometry], float]

    def calc(self, overpass_query_result: OverpassQueryResult, location: Location) -> float:
        """Calculate feature value based on its parameters."""
        base_point = location_to_point(location)
        same_weight_tag_groups = groupby(self.tags, itemgetter(2))

        result = 0.
        for weight, tag_group in same_weight_tag_groups:
            found_objects = self._search(overpass_query_result, location, self.radius, list(tag_group))
            if not found_objects.is_empty:
                result += weight * self._integrate(self.distance_decay, base_point, found_objects)

        return result


class IntegratedNodes(IntegratedFeature):
    """A location feature representing points in the neighborhood."""

    def __init__(self, params: FeatureParameters) -> None:
        """Construct a location feature."""
        super().__init__(params)
        self._integrate = integrate_points

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_nodes_around_location(location, radius, tags_to_search)


class IntegratedWays(IntegratedFeature):
    """A location feature representing ways in the neighborhood."""

    def __init__(self, params: FeatureParameters) -> None:
        """Construct a location feature."""
        super().__init__(params)
        self._integrate = integrate_line_strings

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_ways_around_location(location, radius, tags_to_search)


class IntegratedAreas(IntegratedFeature):
    """A location feature representing areas in the neighborhood."""

    def __init__(self, params: FeatureParameters) -> None:
        """Construct a location feature."""
        super().__init__(params)
        self._integrate = integrate_polygons

    def _search(
        self,
        overpass_query_result: OverpassQueryResult,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> BaseGeometry:
        return overpass_query_result.search_areas_around_location(location, radius, tags_to_search)
