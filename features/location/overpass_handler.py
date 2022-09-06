#!/usr/bin/env python
"""Use the overpass API to query loactions."""

from enum import Enum
from typing import NamedTuple
from urllib.error import URLError

import numpy as np
import matplotlib.pyplot as plt

from overpy import Overpass, Result, Element, Way

from shapely.ops import unary_union, polygonize
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon

from features.location.tag_collection import Tag


COS_PHI0 = np.cos(np.deg2rad(47.4967993))  # latitude of Budapest
EARTH_RADIUS = 6.371e6


class Location(NamedTuple):
    """Represent a latitude longitude coordinate pair."""

    u_latitude: int
    u_longitude: int


class MapObjectType(Enum):
    """Represent map object types."""

    NODE = 0
    WAY = 1
    RELATION = 2


class OverpassInterface:
    """Encapsulate methods for easy overpass usage."""

    def __init__(self, url: str) -> None:
        """Construct OverpassInterface."""
        self.api: Overpass = Overpass(
            url=url,
            max_retry_count=None,
            retry_timeout=None
        )
        self.response: Result = Result()

    def query_all_objects(self, location: Location, radius: float, tags_to_search: list[Tag]) -> Result:
        """Wrap query to handle exceptions."""
        query = '[out:json];'
        query += '('
        for tag in tags_to_search:
            # nodes
            query += f'node["{tag.key}"~"{tag.value}"]'
            query += f'(around:{radius}, {location.u_latitude / 1e6}, {location.u_longitude / 1e6});'
            # ways
            query += f'way["{tag.key}"~"{tag.value}"]'
            query += f'(around:{radius}, {location.u_latitude / 1e6}, {location.u_longitude / 1e6});'
            # areas
            query += f'relation["{tag.key}"~"{tag.value}"]["type"="multipolygon"]'
            query += f'(around:{radius}, {location.u_latitude / 1e6}, {location.u_longitude / 1e6});'
        query += ');'
        query += 'out body; >; out body;'

        try:
            response = self.api.query(query)
            return response
        except URLError:
            print('No Overpass API is available.')
            raise


class OverpassQueryResult:
    """Handle the overpass result object with support methods."""

    def __init__(self, response: Result):
        """Construct a Query result object."""
        self.response = response

    def search_nodes_around_location(
        self,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> MultiPoint:
        """
        Given a neighborhood, the function returns all nodes with specified tags.

        E.g.: search_nodes_around_point(point, 500, 'amenity', 'restaurant')
        """
        found_nodes = self._select_from_response(MapObjectType.NODE, tags_to_search)
        if len(found_nodes) == 0:
            return MultiPoint()

        coordinates = np.array([[float(node.lat), float(node.lon)] for node in found_nodes])
        points = _coordinates_to_points(coordinates)
        result = self._select_parts_in_search_area(location, radius, MultiPoint(points))
        assert isinstance(result, MultiPoint)

        # search for areas as well, and calculate their centroids
        polygons = self.search_areas_around_location(location, radius, tags_to_search)
        if not polygons.is_empty:
            if isinstance(polygons, Polygon):
                centroids = MultiPoint([polygons.centroid])
            elif isinstance(polygons, MultiPolygon):
                centroids = MultiPoint([polygon.centroid for polygon in polygons.geoms])
            else:
                assert False

            result = unary_union((result, centroids))

        result = _ensure_multipart_geometry(result)
        assert isinstance(result, MultiPoint)
        return result

    def search_ways_around_location(
        self,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> MultiLineString:
        """
        Given a point and a neighborhood, the function returns all ways with specified tags.

        E.g.: search_ways_around_point(point, 500, 'highway', 'cafe|bar|restaurant')
        """
        found_ways = self._select_from_response(MapObjectType.WAY, tags_to_search)
        line_strings = _convert_ways_to_line_strings(found_ways)
        result = self._select_parts_in_search_area(location, radius, line_strings)
        result = _ensure_multipart_geometry(result)

        assert isinstance(result, MultiLineString)
        return result

    def search_areas_around_location(
        self,
        location: Location,
        radius: float,
        tags_to_search: list[Tag]
    ) -> MultiPolygon:
        """Given a point and a neighborhood, the function returns all areas with specified tags.

        E.g.: search_areas_around_point(point, 500, 'amenity', 'cafe|bar|restaurant')
        """
        found_relations = self._select_from_response(MapObjectType.RELATION, tags_to_search)
        found_ways = self._select_from_response(MapObjectType.WAY, tags_to_search)

        ways_from_relations = _get_ways_from_relations(found_relations)
        polygon_borders = _convert_ways_to_line_strings(found_ways + ways_from_relations)
        polygons = MultiPolygon(polygonize(polygon_borders.geoms))
        result = self._select_parts_in_search_area(location, radius, polygons)
        result = _ensure_multipart_geometry(result)

        assert isinstance(result, MultiPolygon)
        return result

    def _select_from_response(self, object_type: MapObjectType, tags: list[Tag]) -> list:

        response_objects: list[Element] = []
        if object_type == MapObjectType.NODE:
            response_objects = self.response.nodes
        elif object_type == MapObjectType.WAY:
            response_objects = self.response.ways
        elif object_type == MapObjectType.RELATION:
            response_objects = self.response.relations

        tag_set_to_search = {(tag.key, tag.value) for tag in tags}

        found_objects = []
        for object_ in response_objects:
            if len(tag_set_to_search & set(object_.tags.items())) != 0:
                found_objects.append(object_)

        return found_objects

    @staticmethod
    def _select_parts_in_search_area(
        location: Location,
        radius: float,
        objects: BaseMultipartGeometry
    ) -> BaseMultipartGeometry:

        if objects.is_empty:
            return type(objects)()

        union = unary_union([obj if obj.is_valid else obj.buffer(0) for obj in objects.geoms])
        neighborhood_center = location_to_point(location)
        neighborhood = neighborhood_center.buffer(radius)
        result = union.intersection(neighborhood)
        result = _ensure_multipart_geometry(result)

        return result


def _get_ways_from_relations(relations) -> list[Way]:

    ways: list[Way] = []

    for relation in relations:
        for member in relation.members:
            if member.role == 'outer':
                resolved_member = member.resolve(resolve_missing=True)
                if isinstance(resolved_member, Way):
                    ways.append(resolved_member)
            elif member.role == 'inner':
                pass

    return ways


def _convert_ways_to_line_strings(ways: list[Way]) -> MultiLineString:
    """Convert a list of ways to shapely line strings.

    This function is pretty slow. It could not be vectorized.
    """
    line_strings = []
    for way in ways:
        coordinates = np.array([[float(node.lat), float(node.lon)] for node in way.get_nodes(resolve_missing=True)])
        points = _coordinates_to_points(coordinates)
        line_strings.append(LineString(points.geoms))

    return MultiLineString(line_strings)


def _ensure_multipart_geometry(objects: BaseGeometry) -> BaseMultipartGeometry:

    if isinstance(objects, (MultiPoint, MultiLineString, MultiPolygon)):
        result = objects
    elif isinstance(objects, Point):
        result = MultiPoint([objects]) if not objects.is_empty else MultiPoint()
    elif isinstance(objects, LineString):
        result = MultiLineString([objects]) if not objects.is_empty else MultiLineString()
    elif isinstance(objects, Polygon):
        result = MultiPolygon([objects]) if not objects.is_empty else MultiPolygon()
    else:
        assert False

    return result


def _convert_to_multipart(
    location: Location,
    radius: float,
    objects: BaseMultipartGeometry
) -> BaseMultipartGeometry:

    neighborhood_center = location_to_point(location)
    neighborhood = neighborhood_center.buffer(radius)
    intersection = objects.intersection(neighborhood)
    assert isinstance(intersection, type(objects))
    return intersection


def _coordinates_to_points(latlon_coordinates: np.ndarray) -> MultiPoint:
    """Convert latitude longitude coordinates to MultiPoint in Descartes coordinates."""
    descartes_coordinates = latlon_to_descartes(latlon_coordinates)
    points = MultiPoint(descartes_coordinates)
    return points


def latlon_to_descartes(coordinates: np.ndarray) -> np.ndarray:
    """Convert latitude longitude coordinates to Descartes coordinates.

    First Descartes coordinate is latitude, second is longitude.
    Use equirectangular projection.
    see: https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    """
    coordinates_with_radians = np.deg2rad(coordinates)

    x_coordinates = EARTH_RADIUS * coordinates_with_radians[:, 1] * COS_PHI0
    y_coordinates = EARTH_RADIUS * coordinates_with_radians[:, 0]

    descartes_coordinates = np.c_[x_coordinates, y_coordinates]

    return descartes_coordinates


def location_to_point(location: Location) -> Point:
    """Convert latitude longitude coordinates to Descartes coordinates.

    Use equirectangular projection.
    see: https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
    """
    x_coordinate = EARTH_RADIUS * np.deg2rad(location.u_longitude / 1e6) * COS_PHI0
    y_coordinate = EARTH_RADIUS * np.deg2rad(location.u_latitude / 1e6)

    return Point(x_coordinate, y_coordinate)


if __name__ == '__main__':

    OVERPASS_API_URL = "http://localhost:12345/api/interpreter"
    interface = OverpassInterface(OVERPASS_API_URL)
    test_location = Location(int(47.4642305602828 * 1e6), int(19.030786755247608 * 1e6))
    SEARCH_RADIUS = 800

    node_tags = [
        Tag('amenity', 'bar'),
        Tag('amenity', 'restaurant')
    ]
    way_tags = [
        Tag('highway', 'motorway'),
        Tag('highway', 'trunk'),
        Tag('highway', 'primary'),
        # Tag('highway', 'secondary'),
        # Tag('highway', 'tertiary'),
        # Tag('highway', 'unclassified'),
        # Tag('highway', 'residential'),
        Tag('highway', 'construction'),
    ]
    area_tags = [
        # Tag('leisure', 'park'),
        # Tag('landuse', 'commercial'),
        Tag('building', '*'),
    ]
    response_ = interface.query_all_objects(test_location, SEARCH_RADIUS, node_tags + way_tags + area_tags)
    query_result = OverpassQueryResult(response_)

    # res = query_result.search_nodes_around_location(test_location, SEARCH_RADIUS, node_tags)
    # res = query_result.search_ways_around_location(test_location, SEARCH_RADIUS, way_tags)
    res = query_result.search_areas_around_location(test_location, SEARCH_RADIUS, area_tags)

    neigborhood = location_to_point(test_location).buffer(SEARCH_RADIUS)

    figure, axis = plt.subplots()

    for obj in res.geoms:
        if isinstance(obj, Polygon):
            axis.plot(*obj.exterior.xy)
        elif isinstance(obj, LineString):
            axis.plot(*obj.coords.xy)
        elif isinstance(obj, Point):
            axis.scatter(*obj.coords.xy, s=10, c='r')

    axis.plot(*neigborhood.exterior.xy, c='black')
    plt.savefig('./figure.png')

    print(res)
