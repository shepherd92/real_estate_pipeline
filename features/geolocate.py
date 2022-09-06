#!/usr/bin/env python
"""Module to locate given addresses."""

import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import docker

from geopy.location import Location
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode


class Geolocate(Step):
    """Geolocate properties using Nominatim and Open Street Map."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Clean property data."""
        print('Geolocation...', flush=True)

        data_repository['configuration'].load()
        data_repository['1_scraped']['properties'].load_if_possible()
        data_repository['2_geolocated']['properties'].load_if_possible()

        new_data = data_repository['1_scraped']['properties'].difference(
            data_repository['2_geolocated']['properties']
        )

        if len(new_data) == 0:
            return

        nominatim_container_name = data_repository['configuration'] \
            .params['geolocate_parameters']['nominatim_docker_container']
        nominatim_container = self._get_nominatim_container(nominatim_container_name)
        nominatim_container.start()
        time.sleep(10)
        data_with_location = new_data.progress_apply(locate, axis=1)
        nominatim_container.stop()

        data_repository['2_geolocated']['properties'].add_data(data_with_location)
        data_repository['2_geolocated'].flush()

    @staticmethod
    def _get_nominatim_container(nominatim_container_name: str) -> Any:
        client = docker.from_env()
        container = client.containers.get(nominatim_container_name)
        return container


_geocode = Nominatim(
    user_agent='geopy.geocoders.options.default_user_agent',
    domain='localhost:8080',
    scheme='http'
).geocode


_reverse = Nominatim(
    user_agent='geopy.geocoders.options.default_user_agent',
    domain='localhost:8080',
    scheme='http'
).reverse


ALLOWED_LOCATION_TYPES = {
    'boundary': ['administrative'],
    'building': ['apartments', 'construction', 'house', 'office', 'residential', 'yes'],
    'highway': [
        'cycleway', 'footway', 'living_street', 'pedestrian', 'primary',
        'secondary', 'service', 'tertiary', 'residential',
    ],
    'landuse': ['construction'],
    'place': ['house', 'neighbourhood', 'quarter', 'square', 'suburb'],
    'tourism': ['guest_house', 'apartment'],
}


def locate(row: pd.Series) -> pd.Series:
    """Locate a raw address."""
    raw_address = row['raw_address']

    result = pd.Series({
        'geocoder_location_found': False,
        'geocoder_node_count':     0,
        'geocoder_way_count':      0,
        'geocoder_area_count':     0,
        'geocoder_latitude':       0,
        'geocoder_longitude':      0,
        'geocoder_uncertainty':    np.nan
    })

    locations = _do_geocode(raw_address)
    if len(locations) == 0:
        return result

    locations_to_consider = [location for location in locations if _is_location_valid(location)]
    if len(locations_to_consider) == 0:
        locations_to_consider = locations
        assert len(locations_to_consider) != 0

    # if there are invalid locations (no valid location was found above),
    # or if there are too low rank nodes among the locations,
    # replace them with their highest rank parent location
    locations_to_consider = _replace_invalid_locations_with_parents(locations_to_consider)

    if len(locations_to_consider) == 0:
        return result

    result['geocoder_node_count'] = len([loc for loc in locations_to_consider if loc.raw['osm_type'] == 'node'])
    result['geocoder_way_count'] = len([loc for loc in locations_to_consider if loc.raw['osm_type'] == 'way'])
    result['geocoder_area_count'] = len([loc for loc in locations_to_consider if loc.raw['osm_type'] == 'relation'])

    bounding_boxes = []
    for location in locations_to_consider:
        if location.raw['osm_type'] == 'node':
            # assign bounding box to nodes based on place_rank (address rank)
            extent = _get_extent_to_rank(int(location.raw['place_rank']))
            bounding_boxes.append({
                'lat_min': float(location.latitude) - extent,
                'lat_max': float(location.latitude) + extent,
                'lon_min': float(location.longitude) - extent,
                'lon_max': float(location.longitude) + extent,
            })
        else:
            bounding_boxes.append({
                'lat_min': float(location.raw['boundingbox'][0]),
                'lat_max': float(location.raw['boundingbox'][1]),
                'lon_min': float(location.raw['boundingbox'][2]),
                'lon_max': float(location.raw['boundingbox'][3]),
            })

    centroid = _calc_centroid(bounding_boxes)

    # neglect sphere geometry
    result['geocoder_location_found'] = True
    result['geocoder_latitude'] = int(centroid['lat'] * 10. ** 6)
    result['geocoder_longitude'] = int(centroid['lon'] * 10. ** 6)
    result['geocoder_uncertainty'] = centroid['uncertainty']

    return result


def _do_geocode(address: str, attempt: int = 1) -> list[Location]:

    max_attempts = 10

    try:
        addresses = _geocode(
            address,
            timeout=2,
            addressdetails=True,
            exactly_one=False,
            language='hu,en',
            country_codes=['hu'],
            namedetails=True
        )
        return addresses if addresses is not None else []
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as exception:
        if attempt >= 3:
            print(f'Geolocation attempt {attempt}; {type(exception)}.')
        if attempt <= max_attempts:
            time.sleep(2**attempt)
            return _do_geocode(address, attempt=attempt+1)
        raise


def _is_location_valid(location: Location) -> bool:
    """Check if the location type is in the allowed types."""
    return location.raw['category'] in ALLOWED_LOCATION_TYPES and \
        location.raw['type'] in ALLOWED_LOCATION_TYPES[location.raw['category']]


def _replace_invalid_locations_with_parents(locations: list[Location]) -> list[Location]:
    adjusted_locations = []

    for location in locations:
        if _is_location_valid(location):
            adjusted_locations.append(location)
        else:
            parent = _find_closest_valid_parent(location)
            if parent is not None:
                adjusted_locations.append(parent)

    return adjusted_locations


def _get_extent_to_rank(place_rank: int) -> float:

    rank_to_extent = {
        1:	500.,
        2:	400.,
        3:	320.,
        4:	250.,  # country
        5:	200.,
        6:	165.,
        7:	140.,
        8:	120.,
        9:	100.,  # state
        10:	80.,
        11:	65.,
        12:	50.,  # county
        13:	40.,
        14:	30.,
        15:	20.,
        16:	12.,  # city
        17:	8.00,
        18:	5.00,
        19:	4.00,
        20:	3.00,
        21:	2.00,  # suburb
        22:	1.50,
        23:	1.00,
        24:	0.65,  # neighborhood
        25:	0.40,
        26:	0.25,  # square, farm, locality
        27:	0.15,  # street
        28:	0.05,
        29:	0.02,
        30:	0.01,  # POI/house number
    }

    earth_radius = 6371.0

    return np.rad2deg(rank_to_extent[place_rank] / earth_radius)


def _find_closest_valid_parent(location: Location, recursion: int = 1) -> Optional[Location]:
    """If address rank is not high enough, get the parent location.

    https://nominatim.org/release-docs/develop/customize/Ranking/
    """
    parent_place = _get_parent(location)
    if parent_place is None or recursion >= 10:
        return None

    if _is_location_valid(parent_place):
        return parent_place
    else:
        return _find_closest_valid_parent(parent_place, recursion + 1)


def _get_parent(location: Location) -> Optional[Location]:
    # https://wiki.openstreetmap.org/wiki/Zoom_levels

    location_found = False
    # start from the maximum zoom level
    for zoom_level in range(20, 0, -1):
        parent = _reverse(location.point, zoom=zoom_level)
        if location_found and parent.address != location.address:
            return parent
        if parent.address == location.address:
            location_found = True

    return None


def _calc_centroid(bounding_boxes: list[dict[str, float]]) -> dict[str, float]:

    earth_radius = 6371.

    # calculate centroid
    centers = []
    areas = []

    for bounding_box in bounding_boxes:
        area = earth_radius**2 * \
            np.deg2rad(bounding_box['lat_max'] - bounding_box['lat_min']) * \
            np.deg2rad(bounding_box['lon_max'] - bounding_box['lon_min'])
        lat_center = 0.5 * (bounding_box['lat_max'] + bounding_box['lat_min'])
        lon_center = 0.5 * (bounding_box['lon_max'] + bounding_box['lon_min'])

        centers.append([lat_center, lon_center])
        areas.append(max(area, 1.0))  # 1 ensures that all areas have weights

    weighted_center = np.average(np.array(centers), axis=0, weights=np.array(areas))

    # calculate the average distance of the points
    # shift all rectagles so that the centroid is at the origin
    # integrate r^2 = x^2 + y^2 over all rectangles
    #     r is the distance from the centroid
    # divide result with the total area
    # take the square root

    total_integrals = 0.0
    for bounding_box, area in zip(bounding_boxes, areas):

        # shift all rectagles so that the centroid is at the origin
        x_min = np.deg2rad(bounding_box['lat_min'] - weighted_center[0]) * earth_radius
        x_max = np.deg2rad(bounding_box['lat_max'] - weighted_center[0]) * earth_radius
        y_min = np.deg2rad(bounding_box['lon_min'] - weighted_center[1]) * earth_radius
        y_max = np.deg2rad(bounding_box['lon_max'] - weighted_center[1]) * earth_radius

        # integrate r^2 = x^2 + y^2 over the rectangle
        #     r is the distance from the centroid
        total_integrals += 1/3 * (
            (x_max**3 - x_min**3) * (y_max - y_min) +
            (x_max - x_min) * (y_max**3 - y_min**3)
        )

    uncertainty = np.sqrt(total_integrals / sum(areas))

    result = {
        'lat': weighted_center[0],
        'lon': weighted_center[1],
        'uncertainty': uncertainty
    }

    return result


if __name__ == '__main__':
    test_row = pd.Series(dtype=object)

    # test_row['raw_address'] = 'Szádelő utca'
    # test_row['raw_address'] = 'Székesfehérvár, Öreghegy'
    test_row['raw_address'] = 'Tétényi út, Budapest'

    res = locate(test_row)
    print(res)
