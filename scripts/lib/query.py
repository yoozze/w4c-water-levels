"""Water and weather data querying module"""

import pandas as pd
from datetime import timezone
from .utils import (
    connect_to_db,
    get_water_definitions,
    parse_date
)


def query(sql_query):
    """Execute given SQL query.

    Parameters
    ----------
    sql_query : str
        SQL query.

    Returns
    -------
    list[tuple]
        Rows of a query result or empty list when no rows are available.

    """
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(sql_query)
    result = cursor.fetchall()
    db.commit()
    db.close()
    return result


def get_weather_columns():
    """Get weather table columns.

    Returns
    -------
    list[str]
        List of all weather table columns.

    """
    column_details = query('PRAGMA table_info(weather)')
    return [cd[1] for cd in column_details]


def get_water_data(archive, station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query measurements for selected station from given water archive.

    Parameters
    ----------
    archive : str
        Archive name.

    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Measurements for selected station indexed by date.

    """
    water_defs = get_water_definitions(archive)
    if columns is None:
        columns = [water_defs['feature']]

    db = connect_to_db()
    date_query = ''
    if from_date:
        date_query += f' AND m.date >= \'{from_date}\''
    if to_date:
        date_query += f' AND m.date <= \'{to_date}\''
    df = pd.read_sql_query(f'''SELECT m.date, {', '.join(map(lambda col: f'm.{col}', columns))} 
                               FROM {water_defs['body']}_measurements m
                               INNER JOIN {water_defs['body']}_stations s on s.id = m.station_id
                               WHERE s.id = {station}{date_query}''', db)
    # Index values by date.
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Insert missing dates.
    if reindex:
        date_min = pd.to_datetime(from_date) if from_date else pd.to_datetime(f'{df.index[0].year}-01-01')
        date_max = pd.to_datetime(to_date) if to_date else pd.to_datetime(f'{df.index[-1].year}-12-31')
        df = df.reindex(pd.date_range(date_min, date_max, freq='D'))

    return df


def get_gw_data(station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query measurements for selected station from ground water archive.

    Parameters
    ----------
    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Measurements for selected station indexed by date.

    """
    return get_water_data('ground', station, from_date=from_date, to_date=to_date, columns=columns, reindex=reindex)


def get_sw_data(station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query measurements for selected station from surface water archive.

    Parameters
    ----------
    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Measurements for selected station indexed by date.

    """
    return get_water_data('surface', station, from_date=from_date, to_date=to_date, columns=columns, reindex=reindex)


def get_weather_data(archive, station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query weather data for selected station from given water archive.

    Parameters
    ----------
    archive : str
        Archive name.

    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Weather data for selected station indexed by date.

    """
    water_defs = get_water_definitions(archive)
    db = connect_to_db()
    date_query = ''
    if from_date:
        from_time = int(parse_date(from_date).replace(tzinfo=timezone.utc).timestamp())
        date_query += f' AND w.time >= \'{from_time}\''
    if to_date:
        to_time = int(parse_date(to_date).replace(tzinfo=timezone.utc).timestamp()) + 24 * 60 * 60
        date_query += f' AND w.time <= \'{to_time}\''
    
    df = pd.read_sql_query(f'''SELECT {', '.join(map(lambda col: f'w.{col}', ['time', *columns])) if len(columns) else 'w.*'}
                               FROM weather w
                               INNER JOIN locations l on l.id = w.location_id
                               INNER JOIN {water_defs['body']}s b on b.location_id = l.id
                               INNER JOIN {water_defs['body']}_stations s on s.{water_defs['body']}_id = b.id
                               WHERE s.id = {station}{date_query}''', db)
    # Index values by date.
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.round("1d")
    df.set_index('date', inplace=True)
    if 'id' in df:
        df.drop('id', 1, inplace=True)
    if 'location_id' in df:
        df.drop('location_id', 1, inplace=True)
    if 'time' in df:
        df.drop('time', 1, inplace=True)

    # Insert missing dates.
    if reindex:
        date_min = pd.to_datetime(from_date) if from_date else pd.to_datetime(f'{df.index[0].year}-01-01')
        date_max = pd.to_datetime(to_date) if to_date else pd.to_datetime(f'{df.index[-1].year}-12-31')
        df = df.reindex(pd.date_range(date_min, date_max, freq='D'))

    return df


def get_sw_weather_data(station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query weather data for selected station from surface water archive.

    Parameters
    ----------
    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Weather data for selected station indexed by date.

    """
    return get_weather_data('surface', station, from_date, to_date, columns, reindex)


def get_gw_weather_data(station, from_date=None, to_date=None, columns=None, reindex=True):
    """Query weather data for selected station from groundwater archive.

    Parameters
    ----------
    station : int
        Station ID.

    from_date : str or None
        Start date.

    to_date : str or None
        End date.

    columns : list[str] or None
        List of data columns to be retrieved.

    reindex : bool
        Indicates whether missing dates should be filled with NaNs.

    Returns
    -------
    DataFrame
        Weather data for selected station indexed by date.

    """
    return get_weather_data('ground', station, from_date, to_date, columns, reindex)


def filter_stations(archive, from_date, to_date, threshold):
    """filter hydrological stations from the given archive based on the availability of the data.

    Parameters
    ----------
    archive : str
        Archive name.

    from_date : str
        Start date.

    to_date : str
        End date.

    threshold : float
        Data availability threshold.

    Returns
    -------
    list[int]
        List of filtered station IDs.

    """
    water_defs = get_water_definitions(archive)
    time_diff = (parse_date(to_date) - parse_date(from_date)).days + 1

    # Get all stations with relevant values on given date interval.
    sql_q = f'''SELECT station_id
                FROM {water_defs['body']}_measurements
                WHERE {water_defs['feature']} IS NOT NULL AND date BETWEEN "{from_date}" AND "{to_date}"
                GROUP BY station_id
                HAVING CAST(COUNT(*) AS float) / CAST({time_diff} AS float) >= {threshold}'''

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(sql_q)
    stations = list(map(lambda t: t[0], cursor.fetchall()))
    db.close()

    return stations


def filter_gw_stations(from_date, to_date, threshold):
    """Filter hydrological stations from the ground water archive based on the availability of the data.

    Parameters
    ----------
    from_date : str
        Start date.

    to_date : str
        End date.

    threshold : float
        Data availability threshold.

    Returns
    -------
    list[int]
        List of filtered station IDs.

     """
    return filter_stations('ground', from_date, to_date, threshold)


def filter_sw_stations(from_date, to_date, threshold):
    """Filter hydrological stations from the surface water archive based on the availability of the data.

      Parameters
      ----------
      from_date : str
          Start date.

      to_date : str
          End date.

      threshold : float
          Data availability threshold.

      Returns
      -------
      list[int]
          List of filtered station IDs.

      """
    return filter_stations('surface', from_date, to_date, threshold)


def get_stations(archive, ids):
    """Get station details for given archive and station IDs.

    Parameters
    ----------
    archive : str
        Archive name.

    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    water_defs = get_water_definitions(archive)
    sql_q = f'''SELECT s.id, s.name, b.name, l.name, l.lat, l.lng
                FROM {water_defs['body']}_stations s
                INNER JOIN {water_defs['body']}s b ON b.id = s.{water_defs['body']}_id
                INNER JOIN locations l ON l.id = s.location_id
                WHERE s.id IN ({', '.join(map(lambda i: str(i), ids))})'''

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(sql_q)
    stations = cursor.fetchall()
    db.close()

    return stations


def get_sw_stations(ids):
    """Get watercourse station details for given station IDs.

    Parameters
    ----------
    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    return get_stations('surface', ids)


def get_gw_stations(ids):
    """Get aquifer station details for given station IDs.

    Parameters
    ----------
    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    return get_stations('ground', ids)


def get_weather_locations(archive, ids):
    """Get weather location details for given archive and station IDs.

    Parameters
    ----------
    archive : str
        Archive name.

    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    water_defs = get_water_definitions(archive)
    sql_q = f'''SELECT b.name, l.name, l.lat, l.lng
                FROM {water_defs['body']}_stations s
                INNER JOIN {water_defs['body']}s b ON b.id = s.{water_defs['body']}_id
                INNER JOIN locations l ON l.id = b.location_id
                WHERE s.id IN ({', '.join(map(lambda i: str(i), ids))})
                GROUP BY b.id'''

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(sql_q)
    locations = cursor.fetchall()
    db.close()

    return locations


def get_sw_weather_locations(ids):
    """Get watercourse weather location details for given station IDs.

    Parameters
    ----------
    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    return get_weather_locations('surface', ids)


def get_gw_weather_locations(ids):
    """Get groundwater weather location details for given archive and station IDs.

    Parameters
    ----------
    archive : str
        Archive name.

    ids : list[int]
        List of station IDs.

    Returns
    -------
    list[tuple]

    """
    return get_weather_locations('ground', ids)


def main():
    gw_ids = filter_gw_stations(from_date='1900-01-01', to_date='2020-01-01', threshold=0)
    gw_stations = get_sw_stations(gw_ids)
    gw_weather_locations = get_sw_weather_locations(gw_ids)
    print(gw_stations)
    print(gw_weather_locations)
    print(get_sw_data(5078, from_date='2010-01-01', to_date='2010-01-10', columns=None, reindex=True))

    sw_ids = filter_sw_stations(from_date='1900-01-01', to_date='2020-01-01', threshold=0)
    sw_stations = get_gw_stations(sw_ids)
    sw_weather_locations = get_sw_weather_locations(sw_ids)
    print(sw_stations)
    print(sw_weather_locations)
    print(get_gw_data(5050, from_date='2010-01-01', to_date='2010-01-10', columns=None, reindex=True))
    # sql_q = f'''DELETE 
    #             FROM locations 
    #             WHERE id > 637'''
    # db = connect_to_db()
    # cursor = db.cursor()
    # cursor.execute(sql_q)
    # db.commit()
    # db.close()
    pass


if __name__ == '__main__':
    main()
