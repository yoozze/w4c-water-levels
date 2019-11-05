"""Weather data retrieval module

Data archive at https://darksky.net
"""

import asyncio
import aiohttp
import argparse
import json
import os
import pandas as pd
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from utils import (
    get_data_path,
    get_google_api_key,
    get_water_definitions,
    parse_date,
    save_json
)


def parse_cli_args():
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.

    """
    parser = argparse.ArgumentParser(description='Download weather archive data from `https://darksky.net`.')
    parser.add_argument('locations', type=str, nargs='+', default=[],
                        help='Location name, list of location names or file path to locations JSON file.')
    parser.add_argument('-d', '--dates', type=str, nargs='+', default=[],
                        help='Date, date interval or list of dates, e.g. `2010-01-01` or `2010-01-01:2010-12-31`')

    return parser.parse_args()


def parse_dates(dates):
    """Parse given list of dates or date intervals.

    Parameters
    ----------
    dates : list[str]
        List of dates in `%Y-%m-%d` format or date ranges in `%Y-%m-%d:%Y-%m-%d` format.

    Returns
    -------
    list[str]
        Validated (and expanded) list of dates.

    """
    date_index = pd.DatetimeIndex([], freq='D')

    for date in dates:
        if ':' in date:
            # Parse date range.
            interval = date.split(':')

            if len(interval) != 2 or interval[1] < interval[0]:
                raise ValueError('Invalid date interval format!')

            date_start = parse_date(interval[0])
            date_end = parse_date(interval[1])
            date_index = date_index.union(pd.date_range(date_start, date_end))
        else:
            # Parse single date.
            date_index = date_index.union([parse_date(date)])

    parsed_dates = []

    if len(date_index):
        parsed_dates = list(map(lambda d: str(d.date()), date_index.tolist()))
    else:
        # If no dates were provided, add current date.
        parsed_dates.append(str(datetime.now().date()))

    return parsed_dates


def parse_locations(locations):
    """Parse given list of locations.

    Parameters
    ----------
    locations : list[str]
        List of location names or path to JSON file with location info.

    Returns
    -------
    list
        Validated (and expanded) list of locations.

    """
    parsed_locations = []

    # Parse parsed_locations.
    if len(locations) == 1 and os.path.isfile(locations[0]):
        with open(locations[0], 'r', encoding='utf-8') as json_file:
            parsed_locations = json.load(json_file)
    else:
        api_key = get_google_api_key

        if api_key:
            for location in locations:
                query = {
                    'address': f'{location}, Slovenija',
                    'key': api_key
                }
                url = f'''https://maps.googleapis.com/maps/api/geocode/json?{urllib.parse.urlencode(query)}'''
                response = urllib.request.urlopen(url)
                location_info = json.loads(response.read().decode('utf-8'))

                if location_info['status'] == 'OK':
                    result = location_info['results'][0]
                    new_location = {
                        'name': result['address_components'][0]['long_name'],
                        'lat': result['geometry']['location']['lat'],
                        'lng': result['geometry']['location']['lng'],
                    }
                    parsed_locations.append(new_location)
                    print(f'''Location found: "{new_location['name']}" at [{new_location['lat']}, {new_location['lng']}]''')
                else:
                    print(f'Could not find coordinates for location "{location}"!')
        else:
            print('Could not retrieve locations with google geolocation API.')

    return parsed_locations


def time_to_seconds(time_str):
    """Convert given time string to seconds.

    Parameters
    ----------
    time_str : str
        Time string, e.g. 10:22am or 9:5pm

    Returns
    -------
    int
        Given time in seconds.

    """
    time_str = time_str.strip()
    suffix = time_str[-2:]
    seconds = 0
    for i, t in enumerate(time_str[:-2].split(':')):
        seconds += int(t) * 60**(2 - i)

    if suffix == 'pm':
        seconds += 12 * 60**2

    return seconds


async def fetch_forecast(lat, lng, date, units='ca12', lang='en'):
    """Fetch DarSky forecast for given location and date.

    Parameters
    ----------
    lat : float
        Location latitude.

    lng : float
        Location longitude.

    date : str
        Forecast date.

    units : str
        Forecast units settings.

    lang : str
        Forecast language settings.

    Returns
    -------
    str
        Forecast HTML.

    """
    url = f'https://darksky.net/details/{lat},{lng}/{date}/{units}/{lang}'
    connector = aiohttp.TCPConnector(limit=15)

    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(url, timeout=60 * 60) as response:
            result = await response.text(encoding='utf-8')
            print(f'Reading: {url}')
            return date, result


def parse_forecast(html, date):
    """Parse DarSky forecast for given location and date.

    Parameters
    ----------
    html : str
        Forecast HTML.

    date : str
        Forecast date.

    Returns
    -------
    dict
        Forecast data dictionary.

    """

    # Parse HTML.
    sunrise_re = r'<img src="/images/sunrise\.png" width="28" height="30" />\s*<span class="time">([^>]*)</span>'
    sunset_re = r'<img src="/images/sunset\.png" width="28" height="30" />\s*<span class="time">([^>]*)</span>'
    precip_type_re = r'<span class="label swip">([^>]*)</span>'
    precip_value_re = r'<span class="num swip">([^>]*)</span>'
    precip_unit_re = r'<span class="unit swap">([^>]*)</span>'
    precip_re = rf'<div class="precipAccum swap">\s*{precip_type_re}.+?{precip_value_re}\s*{precip_unit_re}.+?</div>'
    script_re = r'<script>(.+?)</script>'
    html_match = re.search(rf'{sunrise_re}.+?{sunset_re}.+?{precip_re}.+?{script_re}', html, re.DOTALL)

    if not html_match:
        return None

    # Parse JavaScript.
    js_re = r'var hours = (.+?),\s*?startHour.+?forecastTime = (\d+).+?tz_offset = (-?\d+)'
    js_match = re.search(js_re, html_match.group(6), re.DOTALL)

    forecast_time = int(js_match.group(2))
    tz_offset = int(js_match.group(3))
    date_time = int(parse_date(date).replace(tzinfo=timezone.utc).timestamp()) - tz_offset * 3600
    sunrise_time = date_time + time_to_seconds(html_match.group(1))
    sunset_time = date_time + time_to_seconds(html_match.group(2))
    precipitation_type = html_match.group(3).lower()
    precipitation = ' '.join([html_match.group(4).strip(), html_match.group(5).strip()]).strip()
    hourly_forecast = json.loads(js_match.group(1))

    # Split precipitation to value and unit if possible.
    # precip_match = re.match(r'((?:\d+)(?:\.(?:\d+))?) (\w+)', precipitation)
    # if precip_match:
    #     precipitation = [float(precip_match.group(1)), precip_match.group(2)]

    forecast = {
        'date': date,
        'time': forecast_time,
        'precipitation_type': precipitation_type,
        'precipitation': precipitation,
        'sunrise_time': sunrise_time,
        'sunset_time': sunset_time,
        'hourly': hourly_forecast
    }

    return forecast


def save_forecast(forecasts, file_dir, file_name):
    """

    Parameters
    ----------
    forecasts : dict[str, object]
        Dictionary of forecasts where keys are dates and values are forecast dictionaries.

    file_dir : str
        Directory name.

    file_name : str
        File name (without extension).

    Returns
    -------

    """
    if not forecasts:
        return

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(os.path.join(file_dir, f'{file_name}.json'), 'w') as file:
        json.dump(forecasts, file)


def partition_dates(dates):
    """Partition dates by years.

    Parameters
    ----------
    dates : list[str]
        Dates list.

    Returns
    -------
    dict[str, list[str]]
        Dictionary of dates, where keys are date years and values are date lists.

    """
    partitions = {}
    for date in dates:
        year = date[0:4]
        if year not in partitions:
            partitions[year] = []
        partitions[year].append(date)

    return partitions


def extend_metadata(metadata, dir_name, location):
    """Extend metadata for given directory with location data.

    Parameters
    ----------
    metadata : dictionary
        Metadata dictionary.

    dir_name : str
        Directory name.

    location : dict
        location dictionary.

    Returns
    -------
    None

    """
    water_defs = get_water_definitions()
    for water_body in [d['body'] for d in water_defs.values()]:
        if water_body in location:
            ref = location[water_body]
            if water_body not in metadata[dir_name]:
                metadata[dir_name][water_body] = location if isinstance(ref, list) else [ref]
            else:
                if isinstance(ref, list):
                    metadata[dir_name][water_body].extend(ref)
                else:
                    metadata[dir_name][water_body].append(ref)


def crawl(args):
    """Crawl weather data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Metadata of crawled data.

    """
    metadata = {}
    locations = parse_locations(args.locations)
    dates = parse_dates(args.dates)
    dates_by_years = partition_dates(dates)
    loop = asyncio.get_event_loop()

    start_time = time.time()
    request_count = 0
    for index, location in enumerate(locations):
        # Initialize location directory.
        dir_name = f'{location["lat"]}-{location["lng"]}'
        location_dir = get_data_path('weather', 'raw', dir_name)
        dir_exists = os.path.exists(location_dir)

        # Skip locations that were already downloaded.
        if dir_exists:
            print(f'''Directory "{location_dir}" already exist. Skipping location "{location['name']}"!''')
            extend_metadata(metadata, dir_name, location)
            continue

        for year, year_dates in dates_by_years.items():
            # Asynchronously download forecasts.
            year_forecasts = loop.run_until_complete(
                asyncio.gather(
                    *[fetch_forecast(location['lat'], location['lng'], date) for date in year_dates]
                )
            )
            request_count += len(year_dates)

            # Parse forecasts.
            forecasts = {}
            for date, forecast_html in year_forecasts:
                forecast = parse_forecast(forecast_html, date)
                if forecast:
                    forecasts[date] = forecast
                else:
                    print(f'''Could not get forecast for "{location['name']}" on {date}!''')

            if forecasts:
                save_forecast(forecasts, location_dir, str(year))
                metadata[dir_name] = {
                    'name': location['name'],
                    'lat': location['lat'],
                    'lng': location['lng']
                }
                extend_metadata(metadata, dir_name, location)

    end_time = time.time()
    print(f'Finished after {end_time - start_time}s and {request_count} requests!')
    loop.close()

    return metadata


def main():
    args = parse_cli_args()
    metadata = crawl(args)

    # Save metadata.
    if metadata:
        save_json(metadata, get_data_path('weather', 'data.json'))


if __name__ == '__main__':
    main()
