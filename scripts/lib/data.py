# Global
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# Local
from .log import Log as log
from .query import (
    get_stations,
    get_weather_columns,
    get_water_data,
    get_weather_data
)
from .utils import (
    get_range
)

class Data():
    def __init__(self, config):
        self.config = config
        self.sensors = None
        self.data = None
        
        # Initialize discretization method.
        disc_method = self.config.get('disc_method')
        disc_params = disc_method.get('params', {})
        self.discretizer = disc_method['class'](**disc_params)

    def get_sensors(self):
        if not self.sensors:
            # Check sensor config.
            sensor_ids = self.config.get('sensors')
            if not sensor_ids:
                log.warning('No sensors provided!')
                return self.sensors

            # Check sensor availability.
            water_type = self.config.get('water_type')
            sensors = get_stations(water_type, sensor_ids)

            if not sensors:
                log.warning('No sensors found!')
                return self.sensors

            self.sensors = {
                s[0]: {
                    'name': s[1],
                    'waterbody_name': s[2],
                    'location': s[3],
                    'latitude': s[4],
                    'longitude': s[5]
                } for s in sensors
            }

            available_sensor_ids = list(self.sensors.keys())
            log.info(f'{len(sensors)}/{len(sensor_ids)} sensors available.')

            if len(available_sensor_ids) < len(sensor_ids):
                log.warning(f'Unavailable sensors: {", ".join([str(id) for id in sensor_ids if id not in available_sensor_ids])}')

        return self.sensors


    def query(self, sensor):
        sensors = self.get_sensors()

        if sensor not in list(sensors.keys()):
            return None
        
        # Query db.
        log.info(f'Retrieving data for {sensor} ...')

        date_from = self.config.get('date_from')
        date_to = self.config.get('date_to')

        columns = self.config.get('features')
        if not columns:
            columns = get_weather_columns()[3:]

        water_type = self.config.get('water_type')
        water_df = get_water_data(water_type, sensor, date_from, date_to)
        weather_df = get_weather_data(water_type, sensor, date_from, date_to, columns)

        return water_df, weather_df


    def fuse(self, water_df, weather_df):
        log.info('Fusing data ...')

        # Add new column for water level difference.
        water_df['level_diff'] = water_df['level'] - water_df['level'].shift(1)

        # Join water level and weather data.
        # Drop first row where `level_diff` is not defined
        dataset = pd.concat([water_df, weather_df], axis=1)[1:]
        return dataset


    def imputate(self, df):
        nan_count = df.isna().sum().sum()
        if nan_count:
            log.warning(f'Interpolating {nan_count} missing values ...')
            df.interpolate(method='spline', order=2, inplace=True)
            nan_count = df.isna().sum().sum()
            if nan_count:
                log.warning(f'Filling {nan_count} missing values with 0 ...')
                df.fillna(0, inplace=True)
                # df.fillna(method='ffill', inplace=True)


    def discretize_array(self, array):        
        return self.discretizer.fit_transform(array.reshape(-1, 1)).reshape(-1)


    def undiscretize_array(self, array):
        return self.discretizer.inverse_transform(array.reshape(-1, 1)).reshape(-1)


    def discretize(self, df):
        log.info('Discretizing target values ...')
        df['level_diff'] = self.discretize_array(df['level_diff'].values.astype(float))


    def shift_features(self, dataset, blacklist, max_shift):
        days = get_range(1, max_shift)
        for feature_name in list(dataset.columns):
            for i in days:
                if feature_name in blacklist:
                    continue
                dataset[f'{feature_name}_shift_{str(i)}d'] = dataset[feature_name].shift(i)


    def average_features(self, dataset, blacklist, max_average):
        days = get_range(2, max_average)
        for feature_name in list(dataset.columns):
            for i in days:
                if feature_name in blacklist:
                    continue
                dataset[f'{feature_name}_average_{str(i)}d'] = dataset[feature_name].rolling(i).sum() / i


    def construct_features(self, df):
        log.info('Shifting features ...')
        max_shift = self.config.get('max_shift')
        self.shift_features(df, ['level'], max_shift)
        
        log.info('Averaging features ...')
        max_average = self.config.get('max_average')
        self.average_features(df, ['level', 'level_diff'], max_average)

        # Drop all rows containing NaNs generated during feature construction.
        min_row = max_shift + max_average - 1
        return df.iloc[min_row:, :]


    # def save(self, path, columns=None):
        
    #     data_dir = os.path.join(path, 'data')
    #     if not os.path.exists(data_dir):
    #         os.makedirs(data_dir)

    #     for sensor_id, dataset in self.datasets.items():
    #         file_path = os.path.join(data_dir, f'{sensor_id}.csv')
    #         log.info(f'Saving data: {file_path}')
    #         with open(file_path, 'w') as file:
    #             ds = dataset if not columns else dataset.loc(axis=1)[columns]
    #             ds.to_csv(file, index_label='date', sep=',', line_terminator='\n')


    def get(self, sensor, columns=None, column_filter=None, ml_type='reg'):
        if not self.data or self.data['id'] != sensor:
            # Query data.
            data = self.query(sensor)
            
            if not data:
                return None

            # Prepare cache.
            self.data = {
                'id': sensor
            }
            water_df, weather_df = data
        
            if 'missing_values' not in self.sensors[sensor]:
                self.sensors[sensor]['missing_values'] = [
                    water_df.isna().sum().sum(),
                    weather_df.isna().sum().sum()
                ]

            # Fuse datasources.
            dataset = self.fuse(water_df, weather_df)

            # Imputate data if neccessary.
            self.imputate(dataset)
            
            # Cache raw data
            self.data['raw'] = dataset

        dataset = self.data['raw'].copy()
        cache_key = f'processed.{ml_type}'

        if cache_key not in self.data:
            # Prepare data for classification.
            if ml_type == 'cls':
                self.discretize(dataset)
        
            # Construct features.
            dataset = self.construct_features(dataset)
            missing_values = dataset.isna().sum().sum()
            if missing_values:
                log.warning(f'Missing values after feature construction: {missing_values}')

            # Cache processed data.
            self.data[cache_key] = dataset

        dataset = self.data[cache_key]

        # Filter columns if filter function is provided.
        if callable(column_filter):
            if not columns:
                columns = dataset.columns
            columns = filter(column_filter, columns)

        if not columns:
            return dataset
        else:
            return dataset.loc(axis=1)[columns]


def main():
    pass


if __name__ == '__main__':
    main()
