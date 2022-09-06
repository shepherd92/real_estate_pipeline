#!/usr/bin/env python
"""Package contatining ingatlan.com scraping functions and types."""

from pathlib import Path
import time
from typing import Any
from shutil import rmtree

from tqdm import tqdm
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from bs4.element import Tag

from tools.request import Request
from data_collection.property_factory import PropertyFactory
from data_collection.property_list import PropertyList
from pipeline.data import ParentDataNode
from pipeline.pipeline import Step
from interface.scraped_properties import ScrapedProperties
from interface.query_times import QueryTimesData


class Scrape(Step):
    """Encapsulate methods for scraping www.ingatlan.com."""

    def __init__(self) -> None:
        """
        Initialize Scraper object.

        Save input arguments as members.
        """
        self.unsaved_properties: list[dict[str, Any]] = []
        self.query_times: list[list[int]] = []
        self.request = Request(initial_num_of_parallel_requests=1, dynamically_adjust_parallel_requests=False)

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Scrape all property types from ingatlan.com."""
        print('Scraping...', flush=True)

        self._merge_temporary_data(data_repository)

        data_repository['configuration'].load()

        scraped_identifiers = data_repository['1_scraped']['properties'].get_identifiers()

        all_property_lists = self._create_all_property_lists(data_repository)
        property_factory = PropertyFactory()

        # divide property_lists to multiple subsets
        chunk_size = 50
        chunks = [all_property_lists[i:i + chunk_size] for i in range(0, len(all_property_lists), chunk_size)]

        for chunk in (pbar := tqdm(chunks, total=len(chunks))):
            urls = [property_list.url for property_list in chunk]
            property_list_soups = self.request(urls)
            identifiers = np.array(list(set.union(*[
                _extract_identifiers(property_list_soup)
                for property_list_soup in property_list_soups
                if property_list_soup is not None
            ])))

            # update query_times database
            queried_at = int(time.time())
            self.query_times.extend([[identifier, queried_at] for identifier in identifiers])

            unknown_identifiers = np.array(list({
                identifier
                for identifier in identifiers
                if identifier not in scraped_identifiers
            }))

            self._update_run_statistics(data_repository, identifiers, unknown_identifiers)
            pbar.set_description(
                f"Found {data_repository['1_scraped']['scrape_statistics'].metrics['new_properties_found']} " +
                f"seen {data_repository['1_scraped']['scrape_statistics'].metrics['all_properties_seen']}."
            )

            # update scraped_identifiers
            scraped_identifiers = np.concatenate((scraped_identifiers, unknown_identifiers))

            # update new properties
            new_properties = self._create_properties(data_repository, property_factory, unknown_identifiers)
            self.unsaved_properties.extend(new_properties)

            self._insert_chunk_to_repository(data_repository, last_chunk=False)

            data_repository['1_scraped']['scrape_statistics'].check(
                data_repository['configuration'].params['scrape_parameters']['processing_error_limits']
            )

        self._insert_chunk_to_repository(data_repository, last_chunk=True)  # always save the last couple of properties
        data_repository['1_scraped'].flush()

        self._merge_temporary_data(data_repository)

    def _create_all_property_lists(self, data_repository: ParentDataNode) -> list[PropertyList]:

        first_property_lists = self._create_first_property_lists(data_repository)

        print('Requesting first property lists...', end='')
        first_property_list_soups = self.request(
            [first_property_list.url for first_property_list in first_property_lists]
        )
        print('done')

        property_lists: list[PropertyList] = []

        for first_property_list, first_property_list_soup in zip(first_property_lists, first_property_list_soups):
            listing_count = _extract_listing_count(first_property_list_soup)
            property_lists_for_type = self._create_property_lists_for_type(
                data_repository, first_property_list, listing_count
            )
            property_lists.extend(property_lists_for_type)

        return property_lists

    def _create_properties(
        self,
        data_repository: ParentDataNode,
        property_factory: PropertyFactory,
        identifiers: np.ndarray
    ) -> list[dict[str, Any]]:

        # create property urls
        base_url = data_repository['configuration'].params['scrape_parameters']['base_url']
        property_urls = [f"{base_url}/{str(identifier)}" for identifier in identifiers]
        property_soups = self.request(property_urls)

        parameters = list(zip(identifiers, property_soups))
        parameters = [parameter for parameter in parameters if parameter[1] is not None]
        # new_properties = list(map(partial(property_factory.create, data_repository=data_repository), parameters))
        new_properties = [property_factory.create(data_repository, *params) for params in parameters]

        return new_properties

    def _create_first_property_lists(self, data_repository: ParentDataNode) -> list[PropertyList]:

        scrape_parameters = data_repository['configuration'].params['scrape_parameters']

        first_property_lists = [
            PropertyList(sale_or_rent, property_type, 1)
            for sale_or_rent in scrape_parameters['sale_or_rent_values']
            for property_type in scrape_parameters['property_types']
        ]

        return first_property_lists

    def _create_property_lists_for_type(
        self, data_repository: ParentDataNode, first_property_list_url: PropertyList, listing_count: int
    ) -> list[PropertyList]:

        scrape_parameters = data_repository['configuration'].params['scrape_parameters']

        property_lists_for_type = [first_property_list_url]
        num_of_pages = listing_count // scrape_parameters['listings_per_page'] + 1

        for page_num in range(2, num_of_pages + 1):
            property_lists_for_type.append(PropertyList(
                first_property_list_url.sale_or_rent,
                first_property_list_url.property_type,
                page_num
            ))

        return property_lists_for_type

    def _update_run_statistics(self, data_repository, identifiers: np.ndarray, unknown_identifiers: np.ndarray) -> None:

        data_repository['1_scraped']['scrape_statistics'].metrics['all_properties_seen'] += len(identifiers)
        data_repository['1_scraped']['scrape_statistics'].metrics['new_properties_found'] += len(unknown_identifiers)

    def _insert_chunk_to_repository(self, data_repository: ParentDataNode, last_chunk: bool) -> None:

        # check if save is necessary
        if not last_chunk and len(self.unsaved_properties) < 100:
            return

        time_stamp = str(int(time.time()))
        data_directory = Path(data_repository['configuration'].params['data_directory'])

        if 'temp' not in data_repository['1_scraped']:
            data_repository['1_scraped']['temp'] = ParentDataNode()

        if 'properties' not in data_repository['1_scraped']['temp']:
            data_repository['1_scraped']['temp']['properties'] = ParentDataNode()
        if 'query_times' not in data_repository['1_scraped']['temp']:
            data_repository['1_scraped']['temp']['query_times'] = ParentDataNode()

        if len(self.unsaved_properties) != 0:
            unsaved_properties_datanode = ScrapedProperties(
                data_directory / '1_scraped' / 'temp' / 'properties' / f'{time_stamp}.zip'
            )
            unsaved_properties = pd.DataFrame(self.unsaved_properties)
            unsaved_properties.set_index('identifier', inplace=True, verify_integrity=True)
            unsaved_properties_datanode.dataframe = unsaved_properties

            data_repository['1_scraped']['temp']['properties'][time_stamp] = unsaved_properties_datanode

            self.unsaved_properties = []

        if len(self.query_times) != 0:
            unsaved_query_times_datanode = QueryTimesData(
                data_directory / '1_scraped' / 'temp' / 'query_times' / f'{time_stamp}.zip'
            )
            query_times = pd.DataFrame(self.query_times, columns=['identifier', 'queried_at'])
            unsaved_query_times_datanode.dataframe = query_times

            data_repository['1_scraped']['temp']['query_times'][time_stamp] = unsaved_query_times_datanode

            self.query_times = []

        data_repository['1_scraped']['temp'].flush()

    @staticmethod
    def _merge_temporary_data(data_repository: ParentDataNode) -> None:
        """Merge temporary data and add it to the downloaded data frame."""
        if 'temp' not in data_repository['1_scraped'].keys():
            return

        data_repository['configuration'].load()

        data_repository['1_scraped']['properties'].add_data_node(data_repository['1_scraped']['temp']['properties'])
        data_repository['1_scraped']['query_times'].add_data_node(data_repository['1_scraped']['temp']['query_times'])
        data_repository['1_scraped'].flush()

        data_repository['1_scraped'].delete_child('temp')
        data_directory = Path(data_repository['configuration'].params['data_directory'])
        rmtree(data_directory / '1_scraped' / 'temp')


def _extract_listing_count(property_list_soup: BeautifulSoup) -> int:
    class_name = 'results__number'
    listing_count = property_list_soup.find(
        'div', {'class': class_name}
    ).attrs['data-listings-count'].replace(' ', '')
    return int(listing_count)


def _extract_identifiers(property_list_soup: BeautifulSoup) -> set[int]:
    listings = _extract_listings_from_page(property_list_soup)
    identifiers = {int(listing.attrs['data-id']) for listing in listings}
    return identifiers


def _extract_listings_from_page(property_list_soup: BeautifulSoup) -> list[Tag]:
    class_names = ['listing', 'js-listing']

    # create text for selecting divs
    text_for_selection = 'div.' + '.'.join(class_names)
    listings = property_list_soup.select(text_for_selection)
    return listings
