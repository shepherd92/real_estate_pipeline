"""Test the scraper module."""

import json
from pathlib import Path
from typing import Generator
import shutil

import pytest
from bs4 import BeautifulSoup, Tag

from tools.request import Request

from data_collection.scrape.scrape import (
    Scrape,
    _extract_listing_count,
    _extract_identifiers,
    _extract_listings_from_page
)
from data_collection.scrape.property_list import PropertyList


config_file_name = Path('config.json')
scraper_src_file = Path('steps/scrape.py')


@pytest.fixture(name='scraper')
def scraper_fix() -> Generator[Scrape, None, None]:
    """Create a clean scraper to test."""
    with open(config_file_name, encoding='utf-8') as config_file:
        config = json.load(config_file)

    save_dir = Path('scrape/test/temp')
    yield Scrape(config['scrape_parameters'], save_directory=save_dir)
    shutil.rmtree(save_dir)


@pytest.fixture(name='request')
def request_fix() -> Generator[Request, None, None]:
    """Create a request object for testing."""
    request = Request(initial_num_of_parallel_requests=2, dynamically_adjust_parallel_requests=False)
    yield request


@pytest.fixture(name='property_list')
def property_list_fix() -> Generator[PropertyList, None, None]:
    """Create a property list object for testing."""
    property_list = PropertyList('elado', 'lakas', 1)
    yield property_list


@pytest.fixture(name='property_list_soup')
def property_list_soup_fix(property_list: PropertyList, request: Request) -> BeautifulSoup:
    """Create a property list soup object for testing."""
    property_list_soup = request.request_urls([property_list.url])[0]
    yield property_list_soup


def test_files_exist():
    """Check if necessary files exist."""
    assert config_file_name.is_file()
    assert scraper_src_file.is_file()


def test_create_first_property_lists(scraper: Scrape) -> None:
    """Test creating first property lists."""
    property_lists = scraper._create_first_property_lists()
    assert len(property_lists) != 0


def test_extract_listing_count(property_list_soup: BeautifulSoup) -> None:
    """Test listing count extraction."""
    listing_count = _extract_listing_count(property_list_soup)
    assert isinstance(listing_count, int)
    assert listing_count > 100


def test_extract_listings_from_page(property_list_soup: BeautifulSoup) -> None:
    """Test listing extraction."""
    listings = _extract_listings_from_page(property_list_soup)
    assert isinstance(listings, list)
    assert all(isinstance(listing, Tag) for listing in listings)
    assert len(listings) == 20


def test_extract_identifiers(property_list_soup: BeautifulSoup) -> None:
    """Test property identifier extraction from property list soups."""
    identifiers = _extract_identifiers(property_list_soup)
    assert isinstance(identifiers, set)
    assert all(isinstance(identifier, int) for identifier in identifiers)
    assert len(identifiers) == 20
