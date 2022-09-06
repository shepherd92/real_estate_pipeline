#!/usr/bin/env python
"""Data cleaning for downloaded data from ingatlan.com."""

import re
import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm

from pipeline.pipeline import Step
from pipeline.data import ParentDataNode

tqdm.pandas()


EUR_HUF_EXCHANGE_RATE = 370.0


class Transform(Step):
    """Clean data by converting strings to numeric values."""

    def __call__(self, data_repository: ParentDataNode) -> None:
        """Clean property data."""
        print('Transformation...', flush=True)
        data_repository['configuration'].load()
        data_repository['1_scraped']['properties'].load_if_possible()
        data_repository['5_transformed']['properties'].load_if_possible()

        new_data: pd.DataFrame = data_repository['1_scraped']['properties'].difference(
            data_repository['5_transformed']['properties']
        )

        transformed_data = new_data.progress_apply(convert_row, axis=1)
        transformed_data.drop(columns=[
            'raw_address', 'description', 'main_parameters_found', 'details_found', 'description_found'
        ], inplace=True)
        data_repository['5_transformed']['properties'].add_data(transformed_data)
        data_repository['5_transformed'].flush()


def convert_row(row: pd.Series) -> pd.Series:
    """Convert all attributes of a row from raw data."""
    converted_row = row

    converted_row = _convert_int(converted_row, 'floor_area')
    converted_row = _convert_int(converted_row, 'land_area')
    converted_row = _convert_int(converted_row, 'gross_area_of_site')
    converted_row = _convert_price(converted_row, 'price')
    converted_row = _convert_rooms(converted_row)
    converted_row = _convert_list(converted_row, 'wheelchair_accessible', ['nem', 'igen'])
    converted_row = _convert_int(converted_row, 'site_coverage')
    converted_row = _convert_list(converted_row, 'height', ['3 m-nél alacsonyabb', '3 m-nél magasabb'])
    converted_row = _convert_list(converted_row, 'floor', [
        'szuterén', 'földszint', 'félemelet', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10 felett'])
    converted_row = _convert_list(converted_row, 'energy_certificate', [
        'nem értékelt',
        'jj', 'i', 'ii', 'hh', 'h', 'gg', 'ff', 'g', 'ee', 'f', 'dd',
        'e', 'cc', 'bb', 'd', 'c', 'b', 'aa', 'a', 'aa+', 'a+', 'aa++'])
    converted_row = _convert_year_built(converted_row)
    converted_row = _convert_list(converted_row, 'floors_in_building', [
        'földszintes', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'több mint 10'])
    converted_row = _convert_balcony(converted_row)
    converted_row = _convert_list(converted_row, 'bathroom_and_WC', [
        'külön helyiségben', 'egy helyiségben', 'külön és egyben is'])
    converted_row = _convert_heating(converted_row)
    converted_row = _convert_list(converted_row, 'gas', ['nincs', 'utcában', 'telken belül', 'van'])
    converted_row = _convert_list(converted_row, 'electricity', ['nincs', 'utcában', 'telken belül', 'van'])
    converted_row = _convert_list(converted_row, 'sewers', ['nincs', 'utcában', 'telken belül', 'van'])
    converted_row = _convert_list(converted_row, 'water', ['nincs', 'utcában', 'telken belül', 'van'])
    converted_row = _convert_list(converted_row, 'quality', [
        'befejezetlen', 'felújítandó', 'közepes állapotú',
        'jó állapotú', 'felújított', 'újszerű', 'új építésű'])
    converted_row = _convert_list(converted_row, 'office_building_category', [
        'c', 'b vagy b+', 'a vagy a+'])
    converted_row = _convert_list(converted_row, 'garden_connection', ['nem', 'igen'])
    converted_row = _convert_list(converted_row, 'view', [
        'utcai', 'udvari', 'kertre néz', 'panorámás'])
    converted_row = _convert_list(converted_row, 'comfort', [
        'komfort nélküli', 'félkomfortos', 'komfortos', 'összkomfortos', 'duplakomfortos', 'luxus'])
    converted_row = _convert_list(converted_row, 'air_conditioning', ['nincs', 'van'])
    converted_row = _convert_list(converted_row, 'lift', ['nincs', 'van'])
    converted_row = _convert_list(converted_row, 'panelprogram', ['nem vett részt', 'részt vett'])
    converted_row = _convert_parking(converted_row)
    converted_row = _convert_price(converted_row, 'parking_place_price')
    converted_row = _convert_int(converted_row, 'parking_places')
    converted_row = _convert_list(converted_row, 'cellar', ['nincs', 'van'])
    converted_row = _convert_list(converted_row, 'orientation', [
        'észak', 'északkelet', 'kelet', 'délkelet', 'dél', 'délnyugat', 'nyugat', 'északnyugat'])
    converted_row = _convert_list(converted_row, 'roof', [
        'nem tetőtéri', 'tetőtéri', 'legfelső emelet, nem tetőtéri',
        'nem beépíthető', 'beépíthető', 'beépített', 'zárószint', 'penthouse'])
    converted_row = _convert_type(converted_row)
    converted_row = _convert_list(converted_row, 'furnished', [
        'nem', 'részben', 'megegyezés szerint', 'igen'])
    converted_row = _convert_list(converted_row, 'animal', ['nem hozható', 'hozható'])
    converted_row = _convert_list(converted_row, 'smoking', ['nem megengedett', 'megengedett'])
    converted_row = _convert_list(converted_row, 'mechanized', ['nem', 'igen'])
    converted_row = _convert_plot_ratio(converted_row)
    converted_row = _convert_price(converted_row, 'overhead')
    converted_row = _convert_price(converted_row, 'operating_expenses')
    converted_row = _convert_ready_to_move_in(converted_row)
    converted_row = _convert_int(converted_row, 'minimal_rent_time')
    converted_row = _convert_int(converted_row, 'minimum_rentable_area')

    return converted_row


def _convert_int(row: pd.Series, column_name: str) -> pd.Series:
    raw_value = row[column_name]

    if raw_value == 'nincs megadva':
        row[column_name] = -1
        return row

    if isinstance(raw_value, float):
        if np.isnan(raw_value):
            row[column_name] = -2
            return row

    raw_value = str(raw_value)

    pattern = r'^-?(\d+)(,|\.)?(\d+)?.*$'
    if re.match(pattern, raw_value):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)))

    # trigger assertion if conversion was not successful
    assert isinstance(row[column_name], int), f'Could not convert to int {raw_value}'

    return row


def _convert_list(row: pd.Series, column_name: str, value_list: list[str]) -> pd.Series:
    """Convert an enumerated element to a value.

    -2: Invalid parameter in this row
    -1: parameter is not given
    """
    raw_value = row[column_name]
    row[column_name] = -2

    try:
        raw_value = float(raw_value)
    except ValueError:
        pass

    if raw_value == 'nincs megadva':
        row[column_name] = -1
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row[column_name] = -2
            return row
        else:
            raw_value = str(int(raw_value))

    assert raw_value.lower() in value_list, \
        f'Value {raw_value.lower()} (type {type(raw_value)}) not found in value list {value_list}.'

    row[column_name] = value_list.index(raw_value.lower())

    return row


def _convert_price(row: pd.Series, column_name: str) -> pd.Series:

    raw_value = row[column_name]

    if raw_value is None or str(raw_value) == 'nan':
        row[column_name] = -2
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row[column_name] = -2
            return row

    raw_value = raw_value.replace('\n', ' ')
    raw_value = str(raw_value).replace(' ', '')

    if raw_value == 'nincsmegadva' or raw_value == 'árnélkül' or raw_value == 'árnélkül/árnélkül':
        row[column_name] = -1
        return row

    # replace strings beginning with "39,9 trillió Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?trillióFt.*$'
    if re.match(pattern, raw_value):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e12)
        return row

    # replace strings beginning with "39,9 T Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?TFt.*$'
    if re.match(pattern, raw_value):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e12)
        return row

    # replace strings beginning with "39,9 milliárd Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?milliárdFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e9)
        return row

    # replace strings beginning with "177,19 mrd Ft" type substring
    pattern = r'^(\d+)(,)?(\d+)?MrdFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e9)
        return row

    # replace strings beginning with "39,9 millió Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?millióFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6)
        return row

    # replace strings beginning with "39,9 M Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?MFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6)
        return row

    # replace strings beginning with "39,9 ezer Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?ezerFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3)
        return row

    # replace strings beginning with "39,9 E Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?EFt.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3)
        return row

    # replace strings beginning with "39,9 Ft" type substrings
    pattern = r'^(\d+)(,)?(\d+)?Ft.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e0)
        return row

    # replace strings beginning with "177,19 milliárd €" type substring
    pattern = r'^(\d+)(,)?(\d+)?milliárd€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e9 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 Mrd €" type substring
    pattern = r'^(\d+)(,)?(\d+)?Mrd€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e9 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 millió €" type substring
    pattern = r'^(\d+)(,)?(\d+)?millió€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 M €" type substring
    pattern = r'^(\d+)(,)?(\d+)?M€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 ezer €" type substring
    pattern = r'^(\d+)(,)?(\d+)?ezer€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 E €" type substring
    pattern = r'.*(\d+)(,)?(\d+)?E€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3 * EUR_HUF_EXCHANGE_RATE)
        return row

    # replace strings beginning with "177,19 €" type substring
    pattern = r'.*(\d+)(,)?(\d+)?€.*$'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e0 * EUR_HUF_EXCHANGE_RATE)
        return row

    assert isinstance(row[column_name], int)

    return row


def _convert_rooms(row: pd.Series) -> pd.Series:

    rooms = row['rooms']

    if rooms is None:
        row['whole_rooms'] = -2
        row['half_rooms'] = -2
        row['rooms'] = None
        return row

    # replace strings like "5" and "5.0"
    pattern = r'^(\d+)(\.|,)?(\d+)?$'
    if re.match(pattern, str(rooms)):
        row['whole_rooms'] = int(float(re.sub(pattern, r'\1.\3', rooms)))
        row['half_rooms'] = 0
        row['rooms'] = row['whole_rooms'] + 0.5 * row['half_rooms']
        return row

    # replace strings like "5 + 2 fél"
    pattern = r'^(\d+)\s+\+\s+(\d+)\s+fél$'
    if re.match(pattern, str(rooms)):
        row['whole_rooms'] = int(re.sub(pattern, r'\1', rooms))
        row['half_rooms'] = int(re.sub(pattern, r'\2', rooms))
        row['rooms'] = row['whole_rooms'] + 0.5 * row['half_rooms']
        return row

    # replace strings like "2 fél"
    pattern = r'^(\d+)\s+fél$'
    if re.match(pattern, str(rooms)):
        row['whole_rooms'] = 0
        row['half_rooms'] = int(re.sub(pattern, r'\1', rooms))
        row['rooms'] = row['whole_rooms'] + 0.5 * row['half_rooms']
        return row

    # replace strings like "nan"
    pattern = r'^nan$'
    if re.match(pattern, str(rooms)):
        row['whole_rooms'] = -2
        row['half_rooms'] = -2
        row['rooms'] = None
        return row

    assert isinstance(row['whole_rooms'], int)
    assert isinstance(row['half_rooms'], int)
    assert isinstance(row['rooms'], float)

    return row


def _convert_year_built(row: pd.Series) -> pd.Series:

    raw_value = row['year_built']

    if raw_value is None:
        row['year_built'] = -2
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['year_built'] = -2
            return row

    if raw_value == 'nincs megadva' or raw_value == 'Nincs megadva':
        row['year_built'] = -1
        return row

    if isinstance(raw_value, int):
        row['year_built'] = raw_value
        return row
    elif raw_value.isdigit():
        row['year_built'] = int(raw_value)
        return row
    elif raw_value == '1950 előtt':
        row['year_built'] = 1930  # estimated year built
    elif raw_value == '1950 és 1980 között':
        row['year_built'] = 1965  # estimated year built
    elif raw_value == '1981 és 2000 között':
        row['year_built'] = 1990  # estimated year built
    elif raw_value == '2001 és 2010 között':
        row['year_built'] = 2005  # estimated year built
    else:
        assert False, f'Could not convert {raw_value} (type: {type(raw_value)}) in column year_built.'

    assert isinstance(row['year_built'], int)

    return row


def _convert_balcony(row: pd.Series) -> pd.Series:

    raw_value = row['balcony']

    if raw_value is None or raw_value == 'nan':
        row['balcony'] = -2
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['balcony'] = -2
    elif raw_value == 'nincs megadva':
        row['balcony'] = -1

    pattern = r'^(\d+)(,|\.)?(\d+)? .*$'
    if re.match(pattern, str(raw_value)):
        row['balcony'] = int(float(re.sub(pattern, r'\1.\3', raw_value)))

    # trigger assertion if conversion was not successful
    assert isinstance(row['balcony'], int), f'Could not convert to int: {raw_value}'

    return row


def _convert_heating(row: pd.Series) -> pd.Series:

    raw_value = row['heating']

    if raw_value is None or raw_value == 'nan':
        row['heating'] = -2
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['heating'] = -2
    elif raw_value == 'nincs megadva':
        row['heating'] = -1
    else:
        row['heating'] = 0

    heating_types = {
        'nincs':                       'no_heating',
        'cserépkályha':                'tile_stove',
        'egyéb':                       'other_heating',
        'egyéb kazán':                 'other_boiler',
        'elektromos':                  'electronic_heating',
        'falfűtés':                    'wall_heating',
        'fan-coil':                    'fan_coil',
        'gáz (cirko)':                 'gas_central_heating',
        'gáz (konvektor)':             'gas_convector',
        'gázkazán':                    'gas_boiler',
        'házközponti':                 'house_central_heating',
        'házközponti egyedi méréssel': 'house_central_heating_with_individual_measurement',
        'hőszivattyú':                 'heat_pump',
        'megújuló energia':            'renewable_energy_heating',
        'mennyezeti hűtés-fűtés':      'ceiling_cooling_heating',
        'padlófűtés':                  'underfloor_heating',
        'távfűtés':                    'district_heating',
        'távfűtés egyedi méréssel':    'district_heating_with_individual_measurement',
        'vegyes tüzelésű kazán':       'mixed_fuel_boiler'
    }

    for hungarian, english in heating_types.items():
        row[english] = 0
        if hungarian in str(raw_value):
            row[english] = 1

    return row


def _convert_parking(row: pd.Series) -> pd.Series:

    raw_value = row['parking']

    # assign default values
    row['parking'] = -2
    row['parking_price_relation'] = -1

    if raw_value is None or raw_value == 'nan':
        row['parking'] = -2
        row['parking_price_relation'] = -2
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['parking'] = -2
            row['parking_price_relation'] = -2
    elif raw_value == 'nincs megadva':
        row['parking'] = -1
        row['parking_price_relation'] = -1

    # raw_value = str(raw_value)

    parking_types = {
        'utca, közterület': 0,
        'udvari beálló': 1,
        'teremgarázs hely': 2,
        'önálló garázs': 3
    }

    for string, value in parking_types.items():
        if string in str(raw_value):
            row['parking'] = value

    parking_price_relation = {
        'benne van az árban': 0,
        'bérelhető': 1,
        'kötelező kibérelni': 2,
        'megvásárolható': 3,
        'kötelező megvenni': 4
    }

    for string, value in parking_price_relation.items():
        if string in str(raw_value):
            row['parking_price_relation'] = value

    return row


def _convert_type(row: pd.Series) -> pd.Series:

    raw_value = row['subtype']

    # -------------- sale or rent --------------

    # treat berleti jog as a separate sale_or_rent value
    pattern = r'(.*) bérleti joga átadó$'
    if re.match(pattern, str(raw_value)):
        raw_sale_or_rent_value = 'Bérleti jog'
        raw_subtype_value = re.sub(pattern, r'\1', raw_value).lower()

    # strip 'Eladó' and 'Kiadó' strings
    pattern = r'(Eladó|Kiadó) (.*)$'
    if re.match(pattern, str(raw_value)):
        raw_sale_or_rent_value = re.sub(pattern, r'\1', raw_value)
        raw_subtype_value = re.sub(pattern, r'\2', raw_value).lower()

    sale_or_rent_types = ['Eladó', 'Kiadó', 'Bérleti jog']
    row['sale_or_rent'] = sale_or_rent_types.index(raw_sale_or_rent_value)

    # -------------- subtype --------------

    # subtype values
    subtypes = [
        'tégla lakás', 'panel lakás', 'csúsztatott zsalus',
        'családi ház', 'ikerház', 'sorház', 'házrész', 'kastély', 'tanya', 'könnyűszerkezetes ház', 'vályogház',
        'lakóövezeti telek', 'üdülőövezeti telek', 'külterületi telek', 'egyéb telek',
        'önálló garázs', 'teremgarázs hely', 'beálló',
        'nyaralótelek', 'hétvégi házas nyaraló', 'üdülőházas nyaraló',
        'irodahelyiség irodaházban', 'családi házban iroda', 'lakásban iroda', 'egyéb iroda',
        'üzletházban üzlethelyiség', 'utcai bejáratos üzlethelyiség', 'udvarban', 'egyéb üzlethelyiség',
        'szálloda, hotel, panzió', 'étterem, vendéglő', 'egyéb vendéglátó egység',
        'raktárhelyiség',
        'műhely', 'telephely', 'egyéb ipari ingatlan', 'telek ipari hasznosításra',
        'tanya', 'általános mezőgazdasági ingatlan', 'termőföld, szántó', 'erdő', 'pince, présház',
        'lakóterület', 'kereskedelmi, szolgáltató terület', 'vegyes (lakó',
        'ipari terület', 'üdülőterület', 'különleges terület',
        'egészségügyi intézmény', 'iskola', 'múzeum', 'óvoda', 'szoba',
    ]

    assert raw_subtype_value in subtypes
    row['subtype'] = subtypes.index(raw_subtype_value)

    # -------------- main type --------------

    if raw_subtype_value in ['tégla lakás', 'panel lakás', 'csúsztatott zsalus']:
        main_type = 'lakas'
    elif raw_subtype_value in [
        'családi ház', 'ikerház', 'sorház', 'házrész',
        'kastély', 'tanya', 'könnyűszerkezetes ház', 'vályogház'
    ]:
        main_type = 'haz'
    elif raw_subtype_value in ['lakóövezeti telek', 'üdülőövezeti telek', 'külterületi telek', 'egyéb telek']:
        main_type = 'telek'
    elif raw_subtype_value in ['önálló garázs', 'teremgarázs hely', 'beálló']:
        main_type = 'garazs'
    elif raw_subtype_value in ['nyaralótelek', 'hétvégi házas nyaraló', 'üdülőházas nyaraló']:
        main_type = 'nyaralo'
    elif raw_subtype_value in ['irodahelyiség irodaházban', 'családi házban iroda', 'lakásban iroda', 'egyéb iroda']:
        main_type = 'iroda'
    elif raw_subtype_value in [
        'üzletházban üzlethelyiség', 'utcai bejáratos üzlethelyiség',
        'udvarban', 'egyéb üzlethelyiség'
    ]:
        main_type = 'uzlethelyiseg'
    elif raw_subtype_value in ['szálloda, hotel, panzió', 'étterem, vendéglő', 'egyéb vendéglátó egység']:
        main_type = 'vendeglato'
    elif raw_subtype_value in ['raktárhelyiség']:
        main_type = 'raktar'
    elif raw_subtype_value in ['műhely', 'telephely', 'egyéb ipari ingatlan', 'telek ipari hasznosításra']:
        main_type = 'ipari'
    elif raw_subtype_value in [
        'tanya', 'általános mezőgazdasági ingatlan',
        'termőföld, szántó', 'erdő', 'pince, présház'
    ]:
        main_type = 'mezogazdasagi'
    elif raw_subtype_value in [
        'lakóterület', 'kereskedelmi, szolgáltató terület', 'vegyes (lakó',
        'ipari terület', 'üdülőterület', 'különleges terület'
    ]:
        main_type = 'fejl-terulet'
    elif raw_subtype_value in ['egészségügyi intézmény', 'iskola', 'múzeum', 'óvoda', 'szoba']:
        main_type = 'intezmeny'
    else:
        assert False, f'Unknown subtype: {raw_subtype_value}'

    main_types = [
        'lakas', 'haz', 'telek', 'garazs', 'nyaralo', 'iroda', 'uzlethelyiseg', 'vendeglato',
        'raktar', 'ipari', 'mezogazdasagi', 'fejl-terulet', 'intezmeny'
    ]

    assert main_type in main_types
    row['main_type'] = main_types.index(main_type)

    return row


def _convert_monthly_outgoings(row: pd.Series, column_name: str) -> pd.Series:

    raw_value = row[column_name]

    if raw_value is None:
        row[column_name] = -2
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row[column_name] = -2
            return row

    raw_value = raw_value.replace("\n", " ")

    if raw_value == 'nincs megadva' or 'ár nélkül' in str(raw_value).lower():
        row[column_name] = -1
        return row

    # handle strings such as '10,1 Ft/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? Ft.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)))
        return row

    # handle strings such as '10,1 E Ft/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? E Ft.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3)
        return row

    # handle strings such as '10,1 M Ft/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? M Ft.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6)
        return row

    # handle strings such as '4,29 Mrd Ft/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? Mrd Ft.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e9)
        return row

    # handle strings such as '10,1 €/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? €.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * EUR_HUF_EXCHANGE_RATE)
        return row

    # handle strings such as '10,1 E €/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? E €.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e3 * EUR_HUF_EXCHANGE_RATE)
        return row

    # handle strings such as '10,1 M €/hó'
    pattern = r'.*(\d+)(\.|,)?(\d+)? M €.*'
    if re.match(pattern, str(raw_value)):
        row[column_name] = int(float(re.sub(pattern, r'\1.\3', raw_value)) * 1e6 * EUR_HUF_EXCHANGE_RATE)
        return row

    assert isinstance(row[column_name], int)

    return row


def _convert_plot_ratio(row: pd.Series) -> pd.Series:

    raw_value = row['plot_ratio']

    if raw_value is None:
        row['plot_ratio'] = -2
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['plot_ratio'] = -2
            return row

    if raw_value == 'nincs megadva':
        row['plot_ratio'] = -1
        return row

    row['plot_ratio'] = int(float(raw_value) * 100)
    assert isinstance(row['plot_ratio'], int)

    return row


def _convert_ready_to_move_in(row: pd.Series) -> pd.Series:
    """Return month from queried at date."""
    raw_value = row['ready_to_move_in']

    if raw_value is None:
        row['ready_to_move_in'] = -2
        return row
    elif isinstance(raw_value, float):
        if np.isnan(raw_value):
            row['ready_to_move_in'] = -2
            return row

    if raw_value == 'nincs megadva':
        row['ready_to_move_in'] = -1
        return row

    if raw_value == 'azonnal':
        row['ready_to_move_in'] = 0
        return row

    month_translation = {
        'January':    1,
        'január':     1,
        'February':   2,
        'február':    2,
        'March':      3,
        'március':    3,
        'April':      4,
        'április':    4,
        'May':        5,
        'május':      5,
        'June':       6,
        'június':     6,
        'July':       7,
        'július':     7,
        'August':     8,
        'augusztus':  8,
        'September':  9,
        'szeptember': 9,
        'October':   10,
        'október':   10,
        'November':  11,
        'november':  11,
        'December':  12,
        'december':  12,
    }

    pattern = r'^(.*) (\d{4})\.?$'
    if re.match(pattern, str(raw_value)):
        move_in_year = int(re.sub(pattern, r'\2', raw_value))
        month_str = re.sub(pattern, r'\1', raw_value)
        move_in_month = month_translation[month_str]

        row['ready_to_move_in'] = int(datetime.datetime(move_in_year, move_in_month, 1, 0, 0).timestamp())
        return row

    pattern = r'^(\d{4})\.? (.*)$'
    if re.match(pattern, str(raw_value)):
        move_in_year = int(re.sub(pattern, r'\1', raw_value))
        month_str = re.sub(pattern, r'\2', raw_value)
        move_in_month = month_translation[month_str]

        row['ready_to_move_in'] = int(datetime.datetime(move_in_year, move_in_month, 1, 0, 0).timestamp())
        return row

    assert isinstance(row['ready_to_move_in'], int)

    return row
