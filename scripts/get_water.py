"""Water data retrieval module

Data archive at http://vode.arso.gov.si:
http://vode.arso.gov.si/hidarhiv/pov_arhiv_tab.php (for surface water)
http://vode.arso.gov.si/hidarhiv/pod_arhiv_tab.php (for underground water)
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from lib.utils import (
    get_data_path,
    read_from_url,
    load_settings,
    save_json
)


def find_select(name, html):
    """Find select with given `name` from given HTML.

    Parameters
    ----------
    name : str
        Select's `name` attribute.
    html : str
        HTML haystack.

    Returns
    -------
    str or None
        HTML of the first select with given name if search was successful, `None` otherwise.

    """
    select = re.search(rf'<select[^>]+name="{name}"[^>]*>.*?</select>', html, re.DOTALL)

    return select.group(0) if select else None


def get_select_options(html):
    """Get list of select options from given select HTML.

    Parameters
    ----------
    html : str
        Select HTML.

    Returns
    -------
    list[tuple[str, str]]
        List of select options, where each option is represented by a tuple of option value and option content,
        if options were found, empty list otherwise.

    """
    options = re.findall(r'<option[^>]+value="(.*?)"[^>]*>(.*?)</option>', html if html else '', re.DOTALL)

    return list(map(lambda option: (option[0], option[1].strip()), options))


def get_watercourses(html):
    """Get list of watercourses from given data archive web page HTML.

    Parameters
    ----------
    html : str
        HTML of a data archive web page with a watercourse select.

    Returns
    -------
    list[tuple[str, str]]
        List of watercourses, where each watercourses is represented by tuple of watercourse value
        and watercourses name, if watercourse select was found, empty list otherwise.

    """
    return get_select_options(find_select('p_vodotok', html))


def get_stations(html):
    """Get list of watercourse stations from given watercourse web page HTML.

    Parameters
    ----------
    html : str
        HTML of a watercourse web page with a station select.

    Returns
    -------
    list[tuple[str, str]]
        List of watercourse stations, where each station is represented by tuple of station value and station name,
        if station select was found, empty list otherwise.

    """
    return get_select_options(find_select('p_postaja', html))


def get_years(html):
    """Get list of years when measurements were taken in selected station from given station HTML.

    Parameters
    ----------
    html : str
        HTML of a station web page with a year select.

    Returns
    -------
    list[tuple[str, str]]
        List of years when measurements were taken in selected station if year select was found,
        empty list otherwise.

    """
    return get_select_options(find_select('p_od_leto', html))


def filter_dictionary(dictionary, keys):
    """Filters `dictionary` by given keys.

    Parameters
    ----------
    dictionary : dict
        Dictionary to be filtered.
    keys : list[str]
        List of filter keys.

    Returns
    -------
    dict
        New dictionary with selected keys.

    """
    return {key: value for key, value in dictionary.items() if key in keys}


def filter_tuples(tuples, keys=None):
    """Filters `tuples` by given keys if provided.

    Parameters
    ----------
    tuples : list[tuple(str, str)]
        List of tuples.
    keys : list[str]
        List of filter keys.

    Returns
    -------
    dict
        New dictionary generated from given (filtered) tuples.

    """
    return {key: value for key, value in tuples if key in keys} if keys else dict(tuples)


def parse_cli_args(archives):
    """Parse CLI arguments.

    Parameters
    ----------
    archives : Iterable[str]
        List of archive names.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.

    """
    parser = argparse.ArgumentParser(description='Download water archive data from `http://vode.arso.gov.si`.')
    parser.add_argument('-a', '--archive', type=str, nargs='+', default=archives,
                        choices=archives,
                        help='Water archive: `surface` or `ground`')
    parser.add_argument('-w', '--watercourse', type=str, nargs='+', default=[],
                        help='Watercourse name or list of watercourse names, e.g. `Ljubljansko polje`')
    parser.add_argument('-s', '--station', type=str, nargs='+', default=[],
                        help='Station name or list of station names, e.g. `85054`')

    return parser.parse_args()


def crawl(archives, args):
    """Crawl selected archives.

    Parameters
    ----------
    archives : dict[str, str]
        Dictionary of archive URLs.

    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Metadata of crawled data.

    """
    metadata = {}

    # Only crawl selected archives.
    archives = filter_dictionary(archives, args.archive)

    start_time = time.time()
    request_count = 0
    for archive_key, archive_url in archives.items():
        # Get list of watercourses.
        watercourses = get_watercourses(read_from_url(archive_url))
        request_count += 1

        if not watercourses:
            print(f'Could not retrieve "{archive_key}" watercourses!')
            continue

        metadata[archive_key] = {}

        # Only crawl selected watercourses.
        watercourses = filter_tuples(watercourses, args.watercourse)

        for index, watercourse in enumerate(watercourses.keys()):
            # Build watercourse query.
            watercourse_query = {
                'p_vodotok': watercourse
            }

            # Get list of stations for selected watercourse.
            stations = get_stations(read_from_url(archive_url, watercourse_query))
            request_count += 1

            if not stations:
                print(f'Could not retrieve stations for "{watercourse}" watercourse!')
                continue

            # Initialize watercourse directory.
            dir_name = str(index + 1).zfill(5)
            watercourse_dir = get_data_path('water', 'raw', archive_key, dir_name)
            dir_exists = os.path.exists(watercourse_dir)

            metadata[archive_key][watercourse] = {
                'dir': dir_name,
                'stations': {}
            }

            # Only crawl selected stations.
            stations = filter_tuples(stations, args.station)

            for station_id, station_name in stations.items():
                # Build station query.
                station_query = watercourse_query.copy()
                station_query.update({
                    'p_postaja': station_id
                })

                # Get list of years when measurements were taken.
                years = get_years(read_from_url(archive_url, station_query))
                request_count += 1

                if not years:
                    print(f'Could not retrieve years for "{station_name}" station!')
                    continue

                # Build file URL.
                min_year = years[0][0]
                max_year = years[-1][0]
                file_query = station_query.copy()
                file_query.update({
                    'p_od_leto': min_year,
                    'p_do_leto': max_year,
                    'b_oddo_CSV': ''
                })
                file_url = f'{archive_url}?{urllib.parse.urlencode(file_query)}'
                file_path = os.path.join(watercourse_dir, f'{station_id}.csv')

                # Create data directory if it doesn't exist.
                if not dir_exists:
                    os.makedirs(watercourse_dir)
                    dir_exists = True

                print(f'Downloading: {file_url}')
                urllib.request.urlretrieve(file_url, os.path.join(watercourse_dir, file_path))
                request_count += 1
                print(f'Saved to: {file_path}')

                metadata[archive_key][watercourse]['stations'][station_id] = {
                    'name': station_name,
                    'min_year': min_year,
                    'max_year': max_year
                }

            if not len(metadata[archive_key][watercourse]['stations']):
                del metadata[archive_key][watercourse]

        if not len(metadata[archive_key]):
            del metadata[archive_key]

    end_time = time.time()
    print(f'Finished after {end_time - start_time}s and {request_count} requests!')

    return metadata


def main():
    archives = {
        'surface': 'http://vode.arso.gov.si/hidarhiv/pov_arhiv_tab.php',
        'ground': 'http://vode.arso.gov.si/hidarhiv/pod_arhiv_tab.php'
    }
    args = parse_cli_args(archives.keys())
    metadata = crawl(archives, args)

    # Save metadata.
    if metadata:
        save_json(metadata, get_data_path('water', 'data.json'))


if __name__ == '__main__':
    main()
