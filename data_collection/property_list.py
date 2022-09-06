#!/usr/bin/env python
"""Module to represent a page with a list of properties."""


class PropertyList:
    """Represent a page with a list of properties."""

    ingatlan_com_url_prefix = 'https://ingatlan.com/lista'
    page_str = '?page='

    def __init__(self, sale_or_rent: str, property_type: str, page: int) -> None:
        """Create main parameters of property list."""
        self.sale_or_rent = sale_or_rent
        self.property_type = property_type
        self.page_num = page

    @property
    def url(self) -> str:
        """Create URL for property list."""
        if self.page_num == 1:
            return f'{PropertyList.ingatlan_com_url_prefix}/{self.sale_or_rent}+{self.property_type}'

        return f'{PropertyList.ingatlan_com_url_prefix}/' + \
            f'{self.sale_or_rent}+{self.property_type}{PropertyList.page_str}{str(self.page_num)}'
