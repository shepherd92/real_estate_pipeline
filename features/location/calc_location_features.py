#!/usr/bin/env python
"""Module to locate given addresses."""

import time
from multiprocessing import Pool
from typing import Any, NamedTuple, Optional
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import docker
from tqdm import tqdm

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode

from interface.configuration import Configuration
from interface.location_features import LocationFeatures

from features.location.location_feature import LocationFeature
from features.location.overpass_handler import Location, OverpassInterface, OverpassQueryResult
from features.location.feature_parameters import all_feature_group_parameters
from features.location.location_feature_factory import create_all_feature_groups
from features.location.tag_collection import tag_collection


class PropertiesAtSameLocation(NamedTuple):
    """Represent a group of properties, that are at the same location.

    The location features only have to be calculated once for all the properties.
    """

    location: Location
    property_indices: pd.Index


class Parcel(NamedTuple):
    """Represent a "2D bin" of properties, that are approximately in one sphere rectangle.

    The OverPass API has to queried only once for the whole group.
    """

    center_location: Location
    coordinates: np.ndarray

    def __len__(self) -> int:
        """Return the number of coordinates in the parcel."""
        return len(self.coordinates)


# global variable to access all processes and prevent object pickling
MAP_OBJECTS: Optional[OverpassQueryResult] = None


class CalcLocationFeatures(Step):
    """Calculate location features using Overpass and Open Street Map."""

    def __init__(self) -> None:
        """Create all features."""
        super().__init__()

        self._interfaces: list[OverpassInterface] = []
        self._processes: int = 1
        self._next_file_index: int = 1

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Clean property data."""
        print('Calculating location features...', flush=True)

        configuration: Configuration = data_repository['configuration']
        configuration.load()
        _merge_temporary_data(data_repository)
        data_repository['2_geolocated']['properties'].load_if_possible()
        data_repository['3_location_features']['locations'].load_if_possible()

        all_coordinates = _get_distinct_coordinates(
            data_repository['2_geolocated']['properties'].dataframe
        )
        if data_repository['3_location_features']['locations'].loaded:
            old_coordinates = data_repository['3_location_features']['locations'].get_identifiers()
            all_rows = set(map(tuple, all_coordinates))
            old_rows = set(map(tuple, old_coordinates))
            new_coordinates = np.array(list(all_rows.difference(old_rows)))
        else:
            new_coordinates = all_coordinates

        if len(new_coordinates) == 0:
            return

        overpass_containers = _get_overpass_containers(data_repository)
        _start_overpass_containers(overpass_containers)

        self._interfaces = [
            OverpassInterface(api_params['url'])
            for api_params in configuration.params['calc_location_features_parameters']['overpass_apis']
        ]

        features = create_all_feature_groups(all_feature_group_parameters)
        processes = configuration.params['calc_location_features_parameters']['processes']
        bin_size = configuration.params['calc_location_features_parameters']['bin_size']

        # maximum radius must handle whole chunk
        max_radius = max([feature.radius for feature in features]) + bin_size * np.sqrt(2) / 2.
        parcels = _create_parcels(new_coordinates, bin_size)

        all_tags = []
        for sublist in tag_collection.values():
            all_tags.extend(sublist)

        with tqdm(total=len(new_coordinates)) as pbar:
            for parcel in parcels:
                response = self._interfaces[0].query_all_objects(
                    parcel.center_location, max_radius, all_tags
                )
                overpass_query_result = OverpassQueryResult(response)
                data_with_location_features = _process_parcel(parcel, overpass_query_result, features, processes)
                if len(data_with_location_features) != 0:
                    parcel_file_path = self._get_next_file_path(data_repository)
                    _add_parcel_features_to_repository(data_repository, parcel_file_path, data_with_location_features)
                pbar.update(len(parcel.coordinates))

        _stop_overpass_containers(overpass_containers)

        data_repository['3_location_features'].flush()
        _merge_temporary_data(data_repository)

    def _get_next_file_path(self, data_repository: ParentDataNode) -> Path:
        file_index = self._next_file_index
        self._next_file_index += 1

        data_directory = Path(data_repository['configuration'].params['data_directory'])

        return data_directory / '3_location_features' / 'temp' / f'{file_index}.zip'


def _get_distinct_coordinates(geolocated_properties: pd.DataFrame) -> np.ndarray:
    """Return the set of distinct locations.

    Locations are in micro degrees and treated as integers.
    """
    valid_property_coordinates = geolocated_properties[['geocoder_latitude', 'geocoder_longitude']][
        geolocated_properties['geocoder_location_found']
    ]
    distinct_coordinates_times = valid_property_coordinates.drop_duplicates()
    return distinct_coordinates_times.to_numpy()


def _create_parcels(all_coordinates: np.ndarray, bin_size: float) -> list[Parcel]:
    """Split new data to parcels.

    A chunk contains properties that are close to each other by placing them into a "2D histogram".
    This way the Overpass API is only queried once for each chunk.
    """
    lat_bin_size, lon_bin_size = _get_latlon_bin_size(bin_size)
    all_bin_indices = np.zeros_like(all_coordinates)
    all_bin_indices[:, 0] = all_coordinates[:, 0] // lat_bin_size
    all_bin_indices[:, 1] = all_coordinates[:, 1] // lon_bin_size

    # create the set of existing bin index pairs
    unique_bin_indices = np.unique(all_bin_indices, axis=0)

    parcels = [Parcel(
        Location(
            int((lat_lon_bin_index[0] + 0.5) * lat_bin_size),
            int((lat_lon_bin_index[1] + 0.5) * lon_bin_size)
        ),
        all_coordinates[np.all(all_bin_indices == lat_lon_bin_index, axis=1)])
        for lat_lon_bin_index in list(unique_bin_indices)
    ]

    sorted_parcels = list(sorted(parcels, key=len, reverse=True))

    return sorted_parcels


def _process_parcel(
    parcel: Parcel,
    map_objects: OverpassQueryResult,
    features: list[LocationFeature],
    processes: int
) -> pd.DataFrame:

    def _init_pool(_map_objects):
        """Make data accessible to all processes in the pool."""
        global MAP_OBJECTS  # pylint: disable-msg=W0603
        MAP_OBJECTS = _map_objects

    if processes == 1:
        _init_pool(map_objects)
        results = [
            _calc_location_features(features, Location(coords[0], coords[1]))
            for coords in list(parcel.coordinates)
        ]
    else:
        with Pool(processes=processes, initializer=_init_pool, initargs=(map_objects,)) as pool:
            result_promises = [
                pool.apply_async(_calc_location_features, args=(features, Location(coords[0], coords[1])))
                for coords in list(parcel.coordinates)
            ]
            results = [promise.get() for promise in result_promises]

    location_features = pd.DataFrame(results).astype({'latitude': 'int32', 'longitude': 'int32'})
    location_features.set_index(['latitude', 'longitude'], inplace=True, verify_integrity=True)

    return location_features


def _calc_location_features(features: list[LocationFeature], location: Location) -> dict[str, float]:
    """Calculate location features for a coordinate."""
    assert MAP_OBJECTS is not None

    result_row: dict[str, float] = {
        'latitude': location.u_latitude,
        'longitude': location.u_longitude
    }

    result_row |= {
        feature.name: feature.calc(MAP_OBJECTS, location)
        for feature in features
    }

    return result_row


def _add_parcel_features_to_repository(
    data_repository: ParentDataNode,
    parcel_file_path: Path,
    parcel_data: pd.DataFrame
) -> None:

    if 'temp' not in data_repository['3_location_features'].keys():
        data_repository['3_location_features']['temp'] = ParentDataNode()

    chunk_datanode = LocationFeatures(parcel_file_path)
    chunk_datanode.dataframe = parcel_data
    data_repository['3_location_features']['temp'][parcel_file_path.stem] = chunk_datanode
    chunk_datanode.flush()


def _get_overpass_containers(data_repository: ParentDataNode) -> list[Any]:
    overpass_api_params = data_repository['configuration'].params[
        'calc_location_features_parameters']['overpass_apis']
    client = docker.from_env()
    containers = [client.containers.get(params['docker_container']) for params in overpass_api_params]
    return containers


def _start_overpass_containers(overpass_containers: list[Any]) -> None:
    for container in overpass_containers:
        container.start()
    time.sleep(10)


def _stop_overpass_containers(overpass_containers: list[Any]) -> None:
    for container in overpass_containers:
        container.stop()


def _get_latlon_bin_size(bin_size: float) -> tuple[int, int]:

    hungary_west_east_size_meters = 530000.
    hungary_south_north_size_meters = 300000.
    hungary_borders = {
        'latitude': {
            'min': 45.734774,
            'max': 48.583586
        },
        'longitude': {
            'min': 16.112241,
            'max': 22.896822
        }
    }

    west_east_degrees_per_meters = \
        (hungary_borders['longitude']['max'] - hungary_borders['longitude']['min']) / hungary_west_east_size_meters
    south_north_degrees_per_meters = \
        (hungary_borders['latitude']['max'] - hungary_borders['latitude']['min']) / hungary_south_north_size_meters

    lat_bin_size = int(south_north_degrees_per_meters * bin_size * 1e6)
    lon_bin_size = int(west_east_degrees_per_meters * bin_size * 1e6)

    return lat_bin_size, lon_bin_size


def _merge_temporary_data(data_repository: ParentDataNode) -> None:
    """Merge temporary data and add it to the downloaded data frame."""
    if 'temp' not in data_repository['3_location_features'].keys():
        return

    data_repository['configuration'].load()

    data_repository['3_location_features']['locations'].add_data_node(
        data_repository['3_location_features']['temp']
    )
    data_repository['3_location_features'].flush()

    data_repository['3_location_features'].delete_child('temp')
    data_directory = Path(data_repository['configuration'].params['data_directory'])
    rmtree(data_directory / '3_location_features' / 'temp')
