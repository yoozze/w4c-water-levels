"""Water data preparation module"""

import csv
import json
import os
import re
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from utils import (
    connect_to_db,
    get_data_path,
    get_water_definitions,
    read_from_url
)


def load_metadata(data):
    """Load metadata.

    Parameters
    ----------
    data : str
        Data subdirectory name.

    Returns
    -------
    dict or None
            Metadata dictionary if metadata file was found, `None` otherwise.

    """
    file_path = get_data_path(data, 'data.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    else:
        print('Failed to load metadata!')
        return None


def create_tables(connection):
    """Create database tables.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite database connection.

    Returns
    -------
    None

    """
    cursor = connection.cursor()

    # Surface water: watercourses
    # cursor.execute('DROP TABLE IF EXISTS watercourses')
    cursor.execute('''CREATE TABLE IF NOT EXISTS watercourses (
                          id integer PRIMARY KEY,
                          location_id integer NOT NULL,
                          name text NOT NULL UNIQUE,
                          FOREIGN KEY (location_id) REFERENCES locations(id)
                      )''')
    # cursor.execute('DROP TABLE IF EXISTS watercourse_stations')
    cursor.execute('''CREATE TABLE IF NOT EXISTS watercourse_stations (
                          id integer PRIMARY KEY,
                          watercourse_id integer NOT NULL,
                          location_id integer NOT NULL,
                          name text NOT NULL,
                          FOREIGN KEY (watercourse_id) REFERENCES watercourses(id),
                          FOREIGN KEY (location_id) REFERENCES locations(id)
                      )''')
    # cursor.execute('DROP TABLE IF EXISTS watercourse_measurements')
    cursor.execute('''CREATE TABLE IF NOT EXISTS watercourse_measurements (
                          id integer PRIMARY KEY,
                          station_id integer NOT NULL,
                          date string NOT NULL,
                          flow real,
                          level real,
                          FOREIGN KEY (station_id) REFERENCES watercourse_stations(id)
                      )''')

    # Ground waters: aquifers
    # cursor.execute('DROP TABLE IF EXISTS aquifers')
    cursor.execute('''CREATE TABLE IF NOT EXISTS aquifers (
                          id integer PRIMARY KEY,
                          location_id integer NOT NULL,
                          name text NOT NULL UNIQUE,
                          FOREIGN KEY (location_id) REFERENCES locations(id)
                      )''')
    # cursor.execute('DROP TABLE IF EXISTS aquifer_stations')
    cursor.execute('''CREATE TABLE IF NOT EXISTS aquifer_stations (
                          id integer PRIMARY KEY,
                          aquifer_id integer NOT NULL,
                          location_id integer NOT NULL,
                          name text NOT NULL,
                          FOREIGN KEY (aquifer_id) REFERENCES aquifers(id),
                          FOREIGN KEY (location_id) REFERENCES locations(id)
                      )''')
    # cursor.execute('DROP TABLE IF EXISTS aquifer_measurements')
    cursor.execute('''CREATE TABLE IF NOT EXISTS aquifer_measurements (
                          id integer PRIMARY KEY,
                          station_id integer NOT NULL,
                          date string NOT NULL,
                          altitude real,
                          level real,
                          FOREIGN KEY (station_id) REFERENCES aquifer_stations(id)
                      )''')

    # Locations
    # cursor.execute('DROP TABLE IF EXISTS locations')
    cursor.execute('''CREATE TABLE IF NOT EXISTS locations (
                          id integer PRIMARY KEY,
                          name text NOT NULL,
                          lat real NOT NULL,
                          lng real NOT NULL
                      )''')

    # Weather
    cursor.execute('DROP TABLE IF EXISTS weather')
    cursor.execute('''CREATE TABLE IF NOT EXISTS weather (
                          id integer PRIMARY KEY,
                          location_id integer NOT NULL,
                          time integer,
                          day_time integer,
                          precipitation real,
                          snow_accumulation real,
                          temperature_avg real,
                          temperature_min real,
                          temperature_max real,
                          cloud_cover_avg real,
                          cloud_cover_min real,
                          cloud_cover_max real,
                          dew_point_avg real,
                          dew_point_min real,
                          dew_point_max real,
                          humidity_avg real,
                          humidity_min real,
                          humidity_max real,
                          pressure_avg real,
                          pressure_min real,
                          pressure_max real,
                          uv_index_avg integer,
                          uv_index_min integer,
                          uv_index_max integer,
                          precipitation_probability_avg real,
                          precipitation_probability_min real,
                          precipitation_probability_max real,
                          precipitation_intensity_avg real,
                          precipitation_intensity_min real,
                          precipitation_intensity_max real,
                          FOREIGN KEY (location_id) REFERENCES locations(id)
                      )''')


def get_water_index_map(archive, header):
    """Generates mapping from water measurements column names to indices of the given header.

    Parameters
    ----------
    archive : str
        Archive name.

    header : list
        List of column headings.

    Returns
    -------
    dict or None
        Water measurements column names to header indices map if at least one mapping exists, `None` otherwise.

    """
    column_re = {
        'surface': {
            'flow': 'pretok',
            'level': 'vodostaj'
        },
        'ground': {
            'altitude': 'nivo',
            'level': 'vodostaj'
        }
    }
    column_map = {key: -1 for key in column_re[archive].keys()}
    empty = True

    # Do regex search of every db column for every CSV file column heading.
    for i, column in enumerate(header):
        for column_name in column_re[archive].keys():
            if re.search(column_re[archive][column_name], column, re.IGNORECASE):
                column_map[column_name] = i
                empty = False

    return None if empty else column_map


def get_water_value_map(row, column_names_map):
    """Generates mapping from water measurements column names to values of the given CSV row.

    Parameters
    ----------
    row : Sized
        One row of values from CSV file.

    column_names_map : dict
        Water measurements column names to indices map.

    Returns
    -------
    list or None
        Water measurements column names to row values map as specified by `column_map`.

    """
    column_values_map = column_names_map.copy()
    row_length = len(row)
    empty = True

    for key, index in column_names_map.items():
        # Check if non-empty value exist for given index.
        if -1 < index < row_length:
            value = row[index].strip()
            if value:
                column_values_map[key] = value
                empty = False
                continue
        # Else NULL is inserted in db.
        column_values_map[key] = 'NULL'

    return None if empty else column_values_map


def populate_water_measurements(cursor, archive, directory, station):
    """Populate water measurements table for selected `archive`, `directory` and `stations`.

    Parameters
    ----------
    cursor : sqlite3.Cursor
        SQLite database cursor.

    archive : str
        Archive name.

    directory : str
        File system directory name for selected archive.

    station : int
        Station ID (base file name) for selected directory.

    Returns
    -------
    bool
        `True` if measurements table was successfully populated, `False` otherwise.

    """
    csv_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'water',
        'raw',
        archive,
        directory,
        f'{station}.csv'
    )

    with open(csv_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader)
        column_names_map = get_water_index_map(archive, header)

        if not column_names_map:
            return False

        water_body = get_water_definitions(archive)['body']

        for row in reader:
            column_values_map = get_water_value_map(row, column_names_map)
            if column_values_map:
                date = datetime.strptime(row[0], '%d.%m.%Y').date()
                data_columns = ', '.join(column_values_map.keys())
                data_values = ', '.join(column_values_map.values())
                cursor.execute(f'''INSERT INTO {water_body}_measurements (station_id, date, {data_columns})
                                   VALUES ({station}, '{str(date)}', {data_values})''')

        return True


def clean_name(name):
    """Cleans given name.

    Parameters
    ----------
    name : str
        Name to be cleaned.

    Returns
    -------
    str
        Clean name.

    """
    mapping = {
        '[': 'š'
    }
    is_upper = name.isupper()
    cleaned_name = ''.join(map(lambda c: mapping[c] if c in mapping else c, name))

    return cleaned_name.upper() if is_upper else cleaned_name


def get_station_data():
    """Get list of hydrological stations from `http://www.arso.gov.si`.

    Returns
    -------
    DataFrame
        Hydrological station data (ID, name, watercourse, lat, lng, ...)

    """
    url = 'http://www.arso.gov.si/vode/podatki/arhiv/Spisek_postaj.xlsx'
    path = get_data_path('water', 'stations.xlsx')

    if not os.path.exists(path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, 'wb') as file:
            file.write(read_from_url(url, decode=False))

    df = pd.read_excel(path, sheet_name='VSE_VP')

    return df


def populate_water_tables(connection):
    """Populate watercourse and aquifer related data tables.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite database connection.

    Returns
    -------
    None

    """
    metadata = load_metadata('water')
    cursor = connection.cursor()

    # Check if tables are already populated.
    cursor.execute('SELECT count(*) FROM watercourses')
    watercourse_count = cursor.fetchone()[0]
    cursor.execute('SELECT count(*) FROM aquifers')
    aquifer_count = cursor.fetchone()[0]

    if watercourse_count and aquifer_count:
        print('Water tables already populated!')
        return

    station_data = get_station_data()

    for archive in metadata.keys():
        print(f'{archive}-water:'.upper())
        water_body = get_water_definitions(archive)['body']

        # 1. Populate watercourses/aquifers:
        stations = {}
        for water_body_name in metadata[archive].keys():
            print(f'\tPopulating {water_body}: "{water_body_name}"')
            cursor.execute(f'''INSERT INTO {water_body}s(location_id, name)
                               VALUES (0, '{water_body_name}')''')
            water_body_id = cursor.lastrowid

            # 2. Populate watercourse_stations/aquifer_stations:
            for station_id in metadata[archive][water_body_name]['stations']:
                station_name = clean_name(metadata[archive][water_body_name]['stations'][station_id]['name'])

                if station_id in stations:
                    # Prefer watercourses/aquifer with more stations
                    current_len = len(metadata[archive][water_body_name]['stations'])
                    previous_len = len(metadata[archive][stations[station_id]]['stations'])

                    if current_len < previous_len:
                        print(f'\t\tStation already exists: {station_id} - "{station_name}" ("{water_body_name}")')
                        continue
                    else:
                        cursor.execute(f'''DELETE 
                                           FROM {water_body}_stations
                                           WHERE id = {station_id}''')
                        print(f'\t\tRemoved station: {station_id} - "{station_name}" from "{stations[station_id]}")')

                stations[station_id] = water_body_name
                print(f'\t\tPopulating station: {station_id} - "{station_name}"')

                # Insert station location if station data exists.
                location_id = 0
                station_row = station_data.query(f'ŠIFRA == "{station_id}"')
                if not station_row.empty:
                    index = station_row.index[0]
                    lat = station_row.at[index, 'LAT']
                    lng = station_row.at[index, 'LON']
                    if not np.isnan(lat) and not np.isnan(lng):
                        name = f"{station_row.at[index, 'VODOMERNA POSTAJA']} ({station_row.at[index, 'VODOTOK']})"
                        cursor.execute(f'''INSERT INTO locations(name, lat, lng)
                                           VALUES ('{name}', {lat}, {lng})''')
                        location_id = cursor.lastrowid

                # Insert station.
                cursor.execute(f'''INSERT INTO {water_body}_stations(id, {water_body}_id, location_id, name)
                                   VALUES ({station_id}, {water_body_id}, {location_id}, '{station_name}')''')

                # 3. Populate watercourse_measurements/aquifer_measurements:
                if not populate_water_measurements(cursor, archive, metadata[archive][water_body_name]['dir'],
                                                   station_id):
                    cursor.execute(f'''DELETE 
                                       FROM {water_body}_stations
                                       WHERE id = {station_id}''')
                    print(f'\t\tRemoved station with useless data: {station_id} - "{station_name}"')

        # Remove empty watercourses/aquifers.
        cursor.execute(f'''SELECT w.id, w.name
                           FROM {water_body}s w
                           WHERE NOT EXISTS (
                               SELECT s.id 
                               FROM {water_body}_stations s 
                               WHERE w.id = s.{water_body}_id
                           )''')

        for row in cursor.fetchall():
            cursor.execute(f'''DELETE 
                               FROM {water_body}s
                               WHERE id = {row[0]}''')
            print(f'\tRemoved empty {water_body}: "{row[1]}"')


def populate_locations(connection):
    """Populate locations data table.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite database connection.

    Returns
    -------
    None

    """
    print('Populating locations...')
    cursor = connection.cursor()
    with open(get_data_path('locations', 'locations.json'), 'r', encoding='utf-8') as json_file:
        locations = json.load(json_file)

    for station_id, location in locations.items():
        cursor.execute(f'''SELECT id 
                           FROM watercourse_stations 
                           WHERE id = {station_id}''')

        if len(cursor.fetchall()):
            cursor.execute(f'''INSERT INTO locations(name, lat, lng)
                               VALUES ('{location['name']}', {location['lat']}, {location['lng']})''')
            cursor.execute(f'''UPDATE watercourse_stations
                               SET location_id = {cursor.lastrowid}
                               WHERE id = {station_id}''')


def is_forecast_number(key, forecast):
    """Check if given forecast dictionary contains a numeric value with provided key.

    Parameters
    ----------

    key : str
        Forecast value key.

    forecast : dict[str, Any]
        Forecast dictionary.

    Returns
    -------
    bool
        `True` if forecast dictionary contains numeric value with provided key, `False` otherwise.

    """
    return key in forecast and type(forecast[key]) in [float, int]


def populate_weather(connection):
    """Populate weather data tables.

    Parameters
    ----------
    connection : sqlite3.Connection
        SQLite database connection.

    Returns
    -------
    None

    """
    metadata = load_metadata('weather')
    cursor = connection.cursor()
    water_defs = get_water_definitions()

    # Check if tables are already populated.
    cursor.execute('SELECT count(*) FROM weather')
    weather_count = cursor.fetchone()[0]

    if weather_count:
        print('Weather tables already populated!')
        return

    print('WEATHER:')

    for dir_name, location in metadata.items():
        print(f'\tPopulating weather: "{location["name"]}".')

        # Insert location.
        cursor.execute(f'''INSERT INTO locations(name, lat, lng)
                           VALUES ('{location['name']}', {location['lat']}, {location['lng']})''')
        location_id = cursor.lastrowid

        # Set weather locations for watercourses/aquifers.
        for water_body in [d['body'] for d in water_defs.values()]:
            if water_body in location:
                cursor.execute(f'''UPDATE {water_body}s
                                   SET location_id = {location_id}
                                   WHERE name IN ('{"','".join(location[water_body])}')''')
                break

        dir_path = get_data_path('weather', 'raw', dir_name)
        for json_file_name in os.listdir(dir_path):
            json_path = os.path.join(dir_path, json_file_name)
            with open(json_path, 'r', encoding='utf-8') as json_file:
                print(f'\t\tPopulating year: {json_file_name[0:-5]}')
                year_forecasts = json.load(json_file)
                for date, date_forecast in year_forecasts.items():
                    hourly_forecasts = date_forecast['hourly']

                    if not hourly_forecasts:
                        print(f'\t\tNo hourly forecasts for {date}!')
                        continue

                    daily_forecast = {
                        'location_id': location_id,
                        'time': date_forecast['time'],
                        'day_time': date_forecast['sunset_time'] - date_forecast['sunrise_time'],
                        'precipitation': 0,
                        'snow_accumulation': 0
                    }
                    # List of value names with `avg`, `min` and `max` values
                    value_names = {
                        'temperature': 'temperature',
                        'cloud_cover': 'cloudCover',
                        'dew_point': 'dewPoint',
                        'humidity': 'humidity',
                        'pressure': 'pressure',
                        'uv_index': 'uvIndex',
                        'precipitation_probability': 'precipProbability',
                        'precipitation_intensity': 'precipIntensity'
                    }
                    # Value name counters, which indicate how many times (out of 24)
                    # certain value appears in hourly data.
                    value_counts = {k: 0 for k in value_names.keys()}

                    for value_name in value_names.keys():
                        daily_forecast[f'{value_name}_avg'] = 0.0
                        daily_forecast[f'{value_name}_min'] = float('inf')
                        daily_forecast[f'{value_name}_max'] = float('-inf')

                    # Calculate daily forecast values from hourly forecasts.
                    for hourly_forecast in hourly_forecasts:
                        for value_name in value_names.keys():
                            orig_value_name = value_names[value_name]
                            if is_forecast_number(orig_value_name, hourly_forecast):
                                daily_forecast[f'{value_name}_avg'] += hourly_forecast[orig_value_name]
                                daily_forecast[f'{value_name}_min'] = min(
                                    hourly_forecast[orig_value_name],
                                    daily_forecast[f'{value_name}_min']
                                )
                                daily_forecast[f'{value_name}_max'] = max(
                                    hourly_forecast[orig_value_name],
                                    daily_forecast[f'{value_name}_max']
                                )
                                value_counts[value_name] += 1

                        if is_forecast_number('precipAccumulation', hourly_forecast) \
                                and hourly_forecast['precipType'] == 'snow':
                            daily_forecast['snow_accumulation'] += hourly_forecast['precipAccumulation']
                        elif is_forecast_number('precipIntensity', hourly_forecast) \
                                and is_forecast_number('precipProbability', hourly_forecast):
                            daily_forecast['precipitation'] += \
                                hourly_forecast['precipIntensity'] * hourly_forecast['precipProbability']

                    for value_name, value_count in value_counts.items():
                        if value_count:
                            # Calculate average.
                            daily_forecast[f'{value_name}_avg'] = daily_forecast[f'{value_name}_avg'] / value_count
                        else:
                            # If value never appeared
                            daily_forecast[f'{value_name}_avg'] = 'NULL'
                            daily_forecast[f'{value_name}_min'] = 'NULL'
                            daily_forecast[f'{value_name}_max'] = 'NULL'

                    cursor.execute(f'''INSERT INTO weather({', '.join(daily_forecast.keys())})
                                       VALUES ({', '.join(map(lambda v: str(v), daily_forecast.values()))})''')


def create_databases():
    """Creates and populates water and weather database.

    Returns
    -------
    None

    """
    db_connection = connect_to_db()

    # Create database tables.
    create_tables(db_connection)

    # Populate water tables.
    populate_water_tables(db_connection)
    # station_data = get_station_data()
    # station = station_data.query('ŠIFRA == 30301')
    # print(station)
    # index = station.index[0]
    # lat = station.at[index, 'LAT']
    # lng = station.at[index, 'LON']
    # name = f"{station.at[index, 'VODOMERNA POSTAJA']} ({station.at[index, 'VODOTOK']})"
    # print(index, lat, lng, name)

    # Populate location tables
    # populate_locations(db_connection)

    # Populate weather tables
    populate_weather(db_connection)

    db_connection.commit()
    db_connection.close()


def main():
    print("Creating database...")
    create_databases()


if __name__ == '__main__':
    main()
