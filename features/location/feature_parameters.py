#!/usr/bin/env python
"""Tag collections for features."""

from typing import Callable, NamedTuple

from features.location.distance_decay import ConstantDecay, ExponentialDecay, PowerLawDecay

# flake8: noqa


class FeatureGroupParameters(NamedTuple):
    """Represent a set of features."""

    name: str
    node_features: list[str]
    way_features: list[str]
    area_features: list[str]
    distances: list[float]
    decay: type
    decay_param: Callable[[float], float]


all_feature_group_parameters = [
    FeatureGroupParameters('environment',
        ['mountains'], [], ['buildings', 'organized_green_area', 'unorganized_green_area', 'water', 'industrial_area'],
        [100., 500.],
        ConstantDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('leisure',
        ['drink', 'food', 'entertainment', 'sport'], [], [],
        [2000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('education',
        ['lower_education', 'higher_education'], [], [],
        [3000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('religion',
        ['religion'], [],[],
        [2000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('public_service',
        ['safety', 'healthcare', 'other_public_service'], [], [],
        [2000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('noise',
        ['building_noise'], ['highway_noise', 'railway_noise'], [],
        [500.],
        PowerLawDecay,
        lambda _ : 2.
    ),
    FeatureGroupParameters('center',
        ['settlements', 'hotels', 'finance'], [], [],
        [3000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('shopping',
        ['stores', 'malls',], [], ['shopping_area'],
        [3000.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
    FeatureGroupParameters('traffic',
        ['private_traffic', 'public_transportation', 'train'], [], [],
        [1500.],
        ExponentialDecay,
        lambda distance : distance / 5.
    ),
]
