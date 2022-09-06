#!/usr/bin/env python
"""Property data node."""

from pathlib import Path

from interface.dataframe_container import DataframeContainer


class GeolocatedProperties(DataframeContainer):
    """Class handling all property data."""

    dtypes: dict[str, type] = {
        'geocoder_valid_location_found': bool,
        'geocoder_node_count':           int,
        'geocoder_way_count':            int,
        'geocoder_area_count':           int,
        'geocoder_latitude':             int,
        'geocoder_longitude':            int,
        'geocoder_uncertainty':          float
    }

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path, GeolocatedProperties.dtypes)
