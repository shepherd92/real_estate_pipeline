#!/usr/bin/env python
"""Tag collections for features."""

from features.location.location_feature import (
    FeatureParameters, LocationFeature,
    ClosestArea, ClosestNode, ClosestWay,
    IntegratedAreas, IntegratedNodes, IntegratedWays,
)
from features.location.feature_parameters import FeatureGroupParameters


def create_all_feature_groups(all_feature_group_parameters: list[FeatureGroupParameters]):
    """Create all groups of features."""
    nested_features: list[list[LocationFeature]] = [
        create_feature_group(feature_group_parameters)
        for feature_group_parameters in all_feature_group_parameters
    ]
    features = [feature for sublist in nested_features for feature in sublist]
    return features


def create_feature_group(feature_group_parameters: FeatureGroupParameters) -> list[LocationFeature]:
    """Create a group of features."""
    def get_decay(distance):
        """Choose and create the appropriate distance decay."""
        return feature_group_parameters.decay(feature_group_parameters.decay_param(distance))

    features: list[LocationFeature] = []
    max_distance = max(feature_group_parameters.distances)

    for root_name in feature_group_parameters.node_features:
        for distance in feature_group_parameters.distances:
            features.append(IntegratedNodes(FeatureParameters(root_name, distance, get_decay(distance))))
        features.append(ClosestNode(FeatureParameters(root_name, max_distance)))

    for root_name in feature_group_parameters.way_features:
        for distance in feature_group_parameters.distances:
            features.append(IntegratedWays(FeatureParameters(root_name, distance, get_decay(distance))))
        features.append(ClosestWay(FeatureParameters(root_name, max_distance)))

    for root_name in feature_group_parameters.area_features:
        for distance in feature_group_parameters.distances:
            features.append(IntegratedAreas(FeatureParameters(root_name, distance, get_decay(distance))))
        features.append(ClosestArea(FeatureParameters(root_name, max_distance)))

    return features
