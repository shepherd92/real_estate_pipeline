#!/usr/bin/env python
"""Tag collections for features."""

from typing import NamedTuple


class Tag(NamedTuple):
    """Represent an OSM tag."""

    key: str
    value: str
    weight: int = 1


tag_collection = {
    # ----------- ENVIRONMENT -----------
    'organized_green_area': [
        Tag('leisure', 'park'),
        Tag('leisure', 'playground'),
        Tag('leisure', 'pitch'),
        Tag('leisure', 'garden'),
        Tag('leisure', 'golf_course'),
        Tag('landuse', 'grass'),
        Tag('landuse', 'village_green'),
    ],
    'unorganized_green_area': [
        Tag('landuse', 'allotments'),
        Tag('landuse', 'forest'),
        Tag('landuse', 'meadow'),
        Tag('landuse', 'orchard'),
        Tag('landuse', 'vineyard'),
        Tag('landuse', 'greenfield'),
        Tag('natural', 'heath'),
        Tag('natural', 'scrub'),
        Tag('natural', 'grassland'),
        Tag('natural', 'wood'),
    ],
    'mountains': [
        Tag('natural', 'peak'),
        Tag('historic', 'castle'),
        Tag('tourism', 'viewpoint'),
    ],
    'water': [
        Tag('natural', 'water'),
        Tag('water', 'river'),
        Tag('water', 'lake'),
    ],
    'buildings': [
        Tag('building', '.*'),
    ],
    # 'residential_area': [
    #     Tag('landuse', 'residential'),
    # ],
    'industrial_area': [
        Tag('landuse', 'railway'),
        Tag('landuse', 'military'),
        Tag('landuse', 'industrial'),
        Tag('landuse', 'railway'),
        Tag('building', 'industrial'),
    ],
    # ----------- LEISURE -----------
    'drink': [
        Tag('amenity', 'bar'),
        Tag('amenity', 'biergarten'),
        Tag('amenity', 'cafe'),
        Tag('amenity', 'pub'),
    ],
    'food': [
        Tag('amenity', 'fast_food'),
        Tag('amenity', 'food_court'),
        Tag('amenity', 'restaurant'),
    ],
    'entertainment': [
        Tag('amenity', 'cinema'),
        Tag('amenity', 'theatre'),
        Tag('amenity', 'museum'),
    ],
    'sport': [
        Tag('amenity', 'gym'),
        Tag('leisure', 'swimming_pool'),
        Tag('leisure', 'water_park'),
        Tag('leisure', 'sports_centre'),
    ],
    # ----------- EDUCATION -----------
    'lower_education': [
        Tag('amenity', 'kindergarten'),
        Tag('amenity', 'school'),
        Tag('building', 'school'),
    ],
    'higher_education': [
        Tag('amenity', 'college'),
        Tag('amenity', 'university'),
        Tag('building', 'university'),
    ],
    # ----------- RELIGION -----------
    'religion': [
        Tag('amenity', 'place_of_worship'),
        Tag('building', 'mosque'),
        Tag('building', 'synagogue'),
    ],
    # ----------- PUBLIC SERVICE -----------
    'safety': [
        Tag('amenity', 'fire_station'),
        Tag('amenity', 'police'),
    ],
    'healthcare': [
        Tag('amenity', 'clinic', 2),
        Tag('amenity', 'doctors'),
        Tag('amenity', 'hospital', 3),
        Tag('building', 'hospital', 3),
    ],
    'other_public_service': [
        Tag('amenity', 'post_office'),
    ],
    # ----------- NOISE -----------
    'building_noise': [
        Tag('amenity', 'nightclub'),
        Tag('aeroway', 'aerodrome', 3),
        Tag('building', 'industrial', 5),
    ],
    'highway_noise': [
        Tag('highway', 'motorway', 5),
        Tag('highway', 'trunk', 5),
        Tag('highway', 'primary', 3),
        Tag('highway', 'secondary', 1),
        Tag('highway', 'tertiary', 1),
        # Tag('highway', 'unclassified'),
        # Tag('highway', 'residential'),
        # Tag('highway', 'construction'),
    ],
    'railway_noise': [
        Tag('railway', 'tram', 3),
        Tag('railway', 'rail', 5),
        Tag('railway', 'subway', 1),
    ],
    # ----------- CENTER -----------
    'hotels': [
        Tag('building', 'hotel'),
        Tag('tourism', 'hotel'),
    ],
    'settlements': [
        Tag('place', 'city', 5),
        Tag('place', 'town'),
    ],
    'finance': [
        Tag('amenity', 'atm'),
        Tag('atm', 'yes'),
        Tag('amenity', 'bank'),
    ],
    # ----------- SHOPPING -----------
    'stores': [
        Tag('shop', 'supermarket', 3),
        Tag('amenity', 'marketplace', 2),
        Tag('building', 'shop', 1),
        Tag('building', 'store', 2),
    ],
    'malls': [
        Tag('shop', 'mall'),
        Tag('shop', 'shopping_centre'),
    ],
    'shopping_area': [
        Tag('landuse', 'commercial'),
        Tag('landuse', 'retail'),
    ],
    # ----------- TRAFFIC -----------
    'private_traffic': [
        # Tag('amenity', 'fuel'),
        # Tag('amenity', 'parking'),
        Tag('highway', 'motorway_link'),
        Tag('highway', 'trunk_link'),
    ],
    'public_transportation': [
        Tag('amenity', 'bus_station'),
        Tag('highway', 'bus_stop'),
        Tag('highway', 'platform'),
        Tag('railway', 'tram_stop', 3),
        Tag('railway', 'subway_entrance', 5),
    ],
    'train': [
        Tag('building', 'train_station', 4),
        # Tag('railway', 'platform'),
        Tag('railway', 'halt', 2),
        Tag('railway', 'station', 3),
        Tag('railway', 'stop', 1),
    ],
}
