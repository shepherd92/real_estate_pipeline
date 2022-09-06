#!/usr/bin/env python
"""Load initial data."""

from pathlib import Path

from pipeline.data import ParentDataNode
from pipeline.pipeline import Step

from interface.scrape_statistics import ScrapeStatistics
from interface.query_times import QueryTimesData

from interface.scraped_properties import ScrapedProperties
from interface.geolocated_properties import GeolocatedProperties
from interface.time_feature_properties import TimeFeatureProperties
from interface.merged_properties import MergedProperties
from interface.transformed_properties import TransformedProperties
from interface.filtered_properties import FilteredProperties
from interface.location_features import LocationFeatures
from interface.test_predictions import TestPredictions
from interface.figure_container import FigureContainer
from interface.feature_correlation import FeatureCorrelation
from interface.description import Description


class FillDataRepository(Step):
    """Merge multiple property data tables."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Load data from the root data directory."""
        data_repository['configuration'].load()

        self._fill_scrape_data(data_repository)
        self._fill_geolocated_data(data_repository)
        self._fill_location_features_data(data_repository)
        self._fill_time_features_data(data_repository)
        self._fill_transformed_data(data_repository)
        self._fill_merged_data(data_repository)
        self._fill_filtered_data(data_repository)
        self._fill_training_data(data_repository)
        self._fill_evaluation_data(data_repository)

    @staticmethod
    def _fill_scrape_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])

        data_repository['1_scraped'] = ParentDataNode()

        data_repository['1_scraped']['properties'] = \
            ScrapedProperties(data_directory / '1_scraped' / 'property_data.zip')
        data_repository['1_scraped']['query_times'] = \
            QueryTimesData(data_directory / '1_scraped' / 'query_times.zip')
        data_repository['1_scraped']['scrape_statistics'] = \
            ScrapeStatistics(data_directory / '1_scraped' / 'scrape_statistics.zip')

        temp_data_dir = data_directory / '1_scraped' / 'temp'
        if temp_data_dir.is_dir():
            data_repository['1_scraped']['temp'] = ParentDataNode()
            data_repository['1_scraped']['temp']['properties'] = ParentDataNode()
            data_repository['1_scraped']['temp']['query_times'] = ParentDataNode()

            data_repository['1_scraped']['temp']['properties'] |= \
                {
                    str(file): ScrapedProperties(file)
                    for file in (temp_data_dir / 'properties').iterdir()
                    if file.is_file()
                }

            data_repository['1_scraped']['temp']['query_times'] |= \
                {
                    str(file): QueryTimesData(file)
                    for file in (temp_data_dir / 'query_times').iterdir()
                    if file.is_file()
                }

    @staticmethod
    def _fill_geolocated_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['2_geolocated'] = ParentDataNode()
        data_repository['2_geolocated']['properties'] = \
            GeolocatedProperties(data_directory / '2_geolocated' / 'property_data.zip')

    @staticmethod
    def _fill_location_features_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['3_location_features'] = ParentDataNode()
        data_repository['3_location_features']['locations'] = \
            LocationFeatures(data_directory / '3_location_features' / 'locations.zip')

        temp_data_dir = data_directory / '3_location_features' / 'temp'
        if temp_data_dir.is_dir():
            data_repository['3_location_features']['temp'] = ParentDataNode()

            data_repository['3_location_features']['temp'] |= \
                {
                    str(file): LocationFeatures(file)
                    for file in temp_data_dir.iterdir()
                    if file.is_file()
                }

    @staticmethod
    def _fill_time_features_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['4_time_features'] = ParentDataNode()
        data_repository['4_time_features']['properties'] = \
            TimeFeatureProperties(data_directory / '4_time_features' / 'property_data.zip')

    @staticmethod
    def _fill_transformed_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['5_transformed'] = ParentDataNode()
        data_repository['5_transformed']['properties'] = \
            TransformedProperties(data_directory / '5_transformed' / 'property_data.zip')

    @staticmethod
    def _fill_merged_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['6_merged'] = ParentDataNode()
        data_repository['6_merged']['properties'] = \
            MergedProperties(data_directory / '6_merged' / 'property_data.zip')

    @staticmethod
    def _fill_filtered_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['7_filtered'] = ParentDataNode()
        data_repository['7_filtered']['properties'] = \
            FilteredProperties(data_directory / '7_filtered' / 'property_data.zip')
        data_repository['7_filtered']['correlation'] = \
            FeatureCorrelation(data_directory / '7_filtered' / 'feature_correlation.zip')
        data_repository['7_filtered']['description'] = \
            Description(data_directory / '7_filtered' / 'features.zip')

    @staticmethod
    def _fill_training_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])
        data_repository['8_training'] = ParentDataNode()

        data_repository['8_training']['predictions'] = \
            TestPredictions(data_directory / '8_training' / 'predictions.zip')

    @staticmethod
    def _fill_evaluation_data(data_repository: ParentDataNode) -> None:

        data_directory = Path(data_repository['configuration'].params['data_directory'])

        data_repository['9_evaluation'] = ParentDataNode()
        data_repository['9_evaluation']['predictions_labels'] = FigureContainer(
            data_directory / '9_evaluation' / 'predictions_labels.png'
        )
        data_repository['9_evaluation']['feature_importances'] = FigureContainer(
            data_directory / '9_evaluation' / 'feature_importances.png'
        )
