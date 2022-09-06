#!/usr/bin/env python
"""Entry point for property analyzer."""

import argparse

import cProfile
import io
import pstats
from pathlib import Path

from pipeline.data import ParentDataNode
from pipeline.pipeline import Pipeline

from interface.configuration import Configuration

from fill_data_repository import FillDataRepository
from data_collection.scrape import Scrape
from features.calc_time_features import CalcTimeFeatures
from features.geolocate import Geolocate
from features.location.calc_location_features import CalcLocationFeatures
from data_manipulation.transform import Transform
from data_manipulation.merge_tables import MergeTables
from data_manipulation.prepare_data import PrepareData
from model.train import Train
from model.model_base import ModelType


def create_parser() -> argparse.ArgumentParser:
    """Create parser for command line arguments."""
    parser = argparse.ArgumentParser(description='Load and prepare input data for training.')
    parser.add_argument('--config_file', type=str, default='config.json', help='Path to the config json file')
    return parser


def main(params: argparse.Namespace) -> None:
    """Run the property analyzer."""
    data_repository = ParentDataNode()
    data_repository['configuration'] = Configuration(Path(params.config_file))
    data_repository['configuration'].load()

    if data_repository['configuration'].params['profile'].lower() == 'true':
        profiler = cProfile.Profile()
        profiler.enable()

    pipeline = Pipeline(data_repository)

    pipeline.add_steps([
        FillDataRepository(),  # always the first step
        Scrape(),
        CalcTimeFeatures(),
        Geolocate(),
        Transform(),
        CalcLocationFeatures(),
        MergeTables(),
        # PrepareData(),
        # Train(ModelType.NEURAL_NETWORK),
    ])

    pipeline.run()

    data_repository['configuration'].load()
    if data_repository['configuration'].params['profile'].lower() == 'true':
        profiler.disable()
        stream = io.StringIO()
        statistics = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        statistics.print_stats(50)
        print(stream.getvalue())


if __name__ == '__main__':
    argument_parser = create_parser()
    parameters = argument_parser.parse_args()
    main(parameters)
