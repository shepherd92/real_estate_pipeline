#!/usr/bin/env python
"""Module to create properties."""

from typing import Any, Dict
from threading import Lock

from bs4 import BeautifulSoup
from bs4.element import Tag

from pipeline.data import ParentDataNode


main_parameter_names = {
    'Ár':                'price',
    'Ár havonta':        'price',
    'Alapterület':       'floor_area',
    'Szobák':            'rooms',
    'Telekterület':      'land_area',
    'Minimum bérelhető': 'minimum_rentable_area',
}


detail_parameter_names = {
    'Akadálymentesített':     'wheelchair_accessible',
    'Beépíthetőség':          'site_coverage',
    'Belmagasság':            'height',
    'Bruttó szintterület':    'gross_area_of_site',
    'Bútorozott':             'furnished',
    'Csatorna':               'sewers',
    'Dohányzás':              'smoking',
    'Emelet':                 'floor',
    'Energiatanúsítvány':     'energy_certificate',
    'Építés éve':             'year_built',
    'Épület szintjei':        'floors_in_building',
    'Erkély mérete':          'balcony',
    'Fürdő és wc':            'bathroom_and_WC',
    'Fűtés':                  'heating',
    'Gáz':                    'gas',
    'Gépesített':             'mechanized',
    'Ingatlan állapota':      'quality',
    'Irodaház kategóriája':   'office_building_category',
    'Kertkapcsolatos':        'garden_connection',
    'Kilátás':                'view',
    'Kisállat':               'animal',
    'Komfort':                'comfort',
    'Költözhető':             'ready_to_move_in',
    'Légkondicionáló':        'air_conditioning',
    'Lift':                   'lift',
    'Min. bérelhető terület': 'minimal_rentable_area',
    'Min. bérleti idő':       'minimal_rent_time',
    'Panelprogram':           'panelprogram',
    'Parkolás':               'parking',
    'Parkolóhely ára':        'parking_place_price',
    'Parkolóhelyek száma':    'parking_places',
    'Pince':                  'cellar',
    'Rezsiköltség':           'overhead',
    'Szintterületi mutató':   'plot_ratio',
    'Tájolás':                'orientation',
    'Tetőtér':                'roof',
    'Üzemeltetési díj':       'operating_expenses',
    'Villany':                'electricity',
    'Víz':                    'water',
}


class PropertyFactory:
    """Factory for creating a property from soups."""

    def __init__(self) -> None:
        """Construct property factory."""
        self.lock = Lock()

    def create(self, data_repository: ParentDataNode, identifier: int, soup_property: Tag) -> dict[str, Any]:
        """Create a property. Main interface function."""
        property_: Dict[str, Any] = {}

        property_['identifier'] = identifier

        main_parameters = self._extract_main_parameters(data_repository, soup_property)
        property_.update(main_parameters)

        details = self._extract_details(data_repository, soup_property)
        property_.update(details)

        description = self._extract_description(data_repository, soup_property)
        property_.update(description)

        data_repository['1_scraped']['scrape_statistics'].metrics['created_properties'] += 1

        return property_

    def _extract_main_parameters(self, data_repository: ParentDataNode, soup_property: BeautifulSoup) -> dict[str, Any]:

        # initialize result to default values
        result: Dict[str, Any] = {
            'main_parameters_found': False,
        }
        result.update({parameter: None for parameter in main_parameter_names.values()})

        # extract parameters from page title
        page_title = soup_property.find('title')  # e.g. 'Eladó tégla lakás - Siófok, Köztársaság utca 3. #3215554'
        assert page_title is not None, 'Page title not found.'
        result['subtype'] = page_title.text.split('-')[0].strip()  # e.g. 'Eladó tégla lakás'
        result['raw_address'] = \
            page_title.text.split('-')[1].split('#')[0].strip()  # e.g. 'Siófok, Köztársaság utca 3.'

        # extract main parameters
        main_parameters = soup_property.find_all('div', {'class': 'listing-property'})

        if len(main_parameters) != 0:

            result['main_parameters_found'] = True

            for soup_param in main_parameters:
                parameter_name = soup_param.find_all('span')[0].text.strip()
                parameter_value = soup_param.find_all('span')[1].text.strip()

                assert parameter_name in main_parameter_names, \
                    f'Main parameter name {parameter_name} not found in dictionary. ' + \
                    'A new parameter might have appeared.'

                result[main_parameter_names[parameter_name]] = parameter_value
        else:
            with self.lock:
                data_repository['1_scraped']['scrape_statistics'].metrics['main_parameters_not_found'] += 1

        return result

    def _extract_details(self, data_repository: ParentDataNode, soup_property) -> dict:

        result: Dict[str, Any] = {
            'details_found': False
        }
        result.update({parameter: None for parameter in detail_parameter_names.values()})

        soup_parameters = soup_property.find_all('tr', {'class': 'text-onyx'})

        if len(soup_parameters) != 0:

            result['details_found'] = True

            for soup_param in soup_parameters:
                parameter_name = soup_param.find_all('td')[0].text.strip()
                parameter_value = soup_param.find_all('td')[1].text.strip()

                assert parameter_name in detail_parameter_names, \
                    f'Detail parameter name {parameter_name} not found in detail parameters dictionary. ' + \
                    'A new parameter might have appeared.'

                result[detail_parameter_names[parameter_name]] = parameter_value

        else:
            data_repository['1_scraped']['scrape_statistics'].metrics['details_not_found'] += 1

        return result

    def _extract_description(self, data_repository: ParentDataNode, soup_property: BeautifulSoup) -> dict[str, Any]:

        result = {
            'description_found': False,
            'description': None
        }

        soup_description = soup_property.find('p', {'id': 'listing-description'})

        if soup_description is not None:
            result['description_found'] = True
            result['description'] = soup_description.text.strip()
        else:
            data_repository['1_scraped']['scrape_statistics'].metrics['description_not_found'] += 1

        return result
