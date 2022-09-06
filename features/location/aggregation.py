#!/usr/bin/env python
"""Aggregate a set of shapely objects."""

import numpy as np
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon
from shapely.prepared import prep

from features.location.distance_decay import DistanceDecay, PowerLawDecay


def calculate_minimum_distance(base_point: Point, objects: BaseMultipartGeometry) -> float:
    """Calculate minimum distance of objects from the base point."""
    result = min([obj.distance(base_point) for obj in objects.geoms])
    return result


def integrate_points(distance_decay: DistanceDecay, base_point: Point, objects: MultiPoint) -> float:
    """Calculate integral of function over the points."""
    if objects.is_empty:
        return 0.

    result = sum([distance_decay(point.distance(base_point)) for point in objects.geoms])
    return result


def integrate_line_strings(distance_decay: DistanceDecay, base_point: Point, objects: MultiLineString) -> float:
    """Calculate integral of function over the line strings."""
    def get_segments(line_string: LineString) -> list[LineString]:
        """Split line string to segments."""
        return list(map(LineString, zip(line_string.coords[:-1], line_string.coords[1:])))

    if objects.is_empty:
        return 0.

    result = sum([
        distance_decay(segment.centroid.distance(base_point)) * segment.length
        for line_string in objects.geoms
        for segment in get_segments(line_string)
    ])

    return result


def integrate_polygons(distance_decay: DistanceDecay, base_point: Point, objects: MultiPolygon) -> float:
    """Calculate integral of function over the polygons."""
    resolution = 10.

    # calculate bounding box
    x_min, y_min, x_max, y_max = objects.bounds

    # construct rectangle of points
    x_grid_coordinates, y_grid_coordinates = np.round(np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    ), 4)
    grid_points = MultiPoint(list(zip(x_grid_coordinates.flatten(), y_grid_coordinates.flatten())))

    prepared_polygon = prep(objects)

    # validate each point falls inside shapes
    grid_points_in_polygon = MultiPoint(list(filter(prepared_polygon.contains, grid_points.geoms)))

    integral = integrate_points(distance_decay, base_point, grid_points_in_polygon) * resolution**2

    return integral


if __name__ == '__main__':

    distance_decay_to_be_integrated = PowerLawDecay(2.)

    base = Point(0.5, 1.5)

    pnts = [Point(1, 3), Point(2, 2)]
    lss = [LineString([[0, 1], [1, 1], [1, 2]])]
    polys = [Polygon([[1, 1], [3, 4], [5, 0]])]

    # res = integrate_points(base, pnts, distance_decay_to_be_integrated)
    # res = integrate_line_strings(base, lss, distance_decay_to_be_integrated)
    res = integrate_polygons(distance_decay_to_be_integrated, base, polys)

    print(res)
