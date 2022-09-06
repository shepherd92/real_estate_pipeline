"""Test the property list URL."""

from typing import Generator

import pytest
from bs4 import BeautifulSoup

from tools.request import Request
from scrape.property_list import PropertyList


@pytest.fixture(name='request')
def request_fix() -> Generator[Request, None, None]:
    """Create a request object for testing."""
    request = Request(initial_num_of_parallel_requests=2, dynamically_adjust_parallel_requests=False)
    yield request


@pytest.fixture(name='property_list1')
def property_list_fix() -> Generator[PropertyList, None, None]:
    """Create a property list object for testing."""
    property_list = PropertyList('elado', 'lakas', 1)
    yield property_list


def test_property_list_url(request: Request, property_list: PropertyList) -> None:
    """Test property identifier extraction from property list soups."""
    property_list_soup = request.request_urls([property_list.url])[0]
    assert isinstance(property_list_soup, BeautifulSoup)
