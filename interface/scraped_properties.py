#!/usr/bin/env python
"""Property data node."""

from pathlib import Path

import numpy as np

from interface.dataframe_container import DataframeContainer


class ScrapedProperties(DataframeContainer):
    """Class handling all property data."""

    dtypes: dict[str, type] = {
        'identifier':               np.int32,
        'main_parameters_found':    bool,
        'price':                    str,
        'floor_area':               str,
        'rooms':                    str,
        'land_area':                str,
        'minimum_rentable_area':    str,
        'subtype':                  str,
        'raw_address':              str,
        'details_found':            bool,
        'wheelchair_accessible':    str,
        'site_coverage':            str,
        'height':                   str,
        'gross_area_of_site':       str,
        'furnished':                str,
        'sewers':                   str,
        'smoking':                  str,
        'floor':                    str,
        'energy_certificate':       str,
        'year_built':               str,
        'floors_in_building':       str,
        'balcony':                  str,
        'bathroom_and_WC':          str,
        'heating':                  str,
        'gas':                      str,
        'mechanized':               str,
        'quality':                  str,
        'office_building_category': str,
        'garden_connection':        str,
        'view':                     str,
        'animal':                   str,
        'comfort':                  str,
        'ready_to_move_in':         str,
        'air_conditioning':         str,
        'lift':                     str,
        'minimal_rentable_area':    str,
        'minimal_rent_time':        str,
        'panelprogram':             str,
        'parking':                  str,
        'parking_place_price':      str,
        'parking_places':           str,
        'cellar':                   str,
        'overhead':                 str,
        'plot_ratio':               str,
        'orientation':              str,
        'roof':                     str,
        'operating_expenses':       str,
        'electricity':              str,
        'water':                    str,
        'description':              str,
        'description_found':        bool,
    }

    def __init__(self, path: Path) -> None:
        """Construct a high level data node."""
        super().__init__(path, ScrapedProperties.dtypes)
