# Common 
import argparse
import os
import json
import math
import numpy as np

#Local
from lib.log import Log as log

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters


def save_figure(plt, file_path):
    plt.savefig(file_path, dpi = 300, bbox_inches='tight')


def get_resources_path(path):
    resources_path = os.path.join(path, 'resources')
    if not os.path.exists(resources_path):
        os.makedirs(resources_path)

    return resources_path


def get_results(path):
    results_path = os.path.join(path, 'results.json')
    results_json = None
    config_path = os.path.join(path, 'config.json')
    config_json = None

    if os.path.isfile(results_path) and os.path.isfile(config_path):
        with open(results_path, 'r', encoding='utf-8') as results_file:
            results_json = json.load(results_file)
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config_json = json.load(config_file)
    else:
        return None

    return results_json, config_json


def evaluate_sensors(results, sensor_wl=None, horizon_wl=None, method_wl=None):
    evaluation = {}
    print(horizon_wl)
    print(method_wl)
    # Sensors:
    sensors = results.get('sensors')
    print(list(sensors.keys()))
    print(len(list(sensors.keys())))
    if sensor_wl:
        print(sensor_wl)
        print(len(sensor_wl))
    sensors_len = len(sensors) if not sensor_wl else len(sensor_wl)
    for sensor, sensor_results in sensors.items():
        if sensor_wl and sensor not in sensor_wl:
            continue
        
        # Types:
        ml_types = sensor_results.get('types')
        for ml_type, type_results in ml_types.items():
            if not evaluation.get(ml_type):
                evaluation[ml_type] = {}

            if not evaluation[ml_type].get(sensor):
                evaluation[ml_type][sensor] = {}

            # Horizons:
            horizons = type_results.get('horizons')
            horizon_len = len(horizons) if not horizon_wl else len(horizon_wl)
            for horizon, horizon_results in horizons.items():
                if horizon_wl and horizon not in horizon_wl:
                    continue

                # Methods:
                methods = horizon_results.get('methods')
                for method, method_results in methods.items():
                    if method_wl and method not in method_wl:
                        continue

                    if not evaluation[ml_type][sensor].get(method):
                        evaluation[ml_type][sensor][method] = {}

                    if not evaluation[ml_type][sensor][method].get(horizon):
                        evaluation[ml_type][sensor][method][horizon] = {
                            'r2': 0.0
                        }
                    
                    # CV:
                    cv = method_results.get('cv')
                    cvi_len = len(cv)

                    for cvi_results in cv.values():
                        evaluation_results = cvi_results.get('evaluation')
                        evaluation[ml_type][sensor][method][horizon]['r2'] += evaluation_results['r2_score'] / cvi_len

    return evaluation


def evaluate(results, sensor_wl=None, horizon_wl=None, method_wl=None):
    evaluation = {}
    sensors_len = 0
    horizon_len = 0
    cvi_len = 0

    # 1st pass: averages
    # ==================

    # Sensors:
    sensors = results.get('sensors')
    sensors_len = len(sensors) if not sensor_wl else len(sensor_wl)
    # print(sensors_len)
    for sensor, sensor_results in sensors.items():
        if sensor_wl and sensor not in sensor_wl:
            continue
        
        # Types:
        ml_types = sensor_results.get('types')
        for ml_type, type_results in ml_types.items():
            if not evaluation.get(ml_type):
                evaluation[ml_type] = {}

            # Horizons:
            horizons = type_results.get('horizons')
            horizon_len = len(horizons) if not horizon_wl else len(horizon_wl)
            for horizon, horizon_results in horizons.items():
                if horizon_wl and horizon not in horizon_wl:
                    continue
                
                # Methods:
                methods = horizon_results.get('methods')
                for method, method_results in methods.items():
                    if method_wl and method not in method_wl:
                        continue

                    if not evaluation[ml_type].get(method):
                        evaluation[ml_type][method] = {
                            'horizons': {},
                            'times': {
                                'tt': 0.0,
                                'tp': 0.0
                            }
                        }

                    if not evaluation[ml_type][method]['horizons'].get(horizon):
                        evaluation[ml_type][method]['horizons'][horizon] = {
                            'r2': 0.0,
                            'r2_sd': 0.0
                        }
                    
                    # CV:
                    cv = method_results.get('cv')
                    cvi_len = len(cv)

                    for cvi_results in cv.values():
                        modelling_results = cvi_results.get('modelling')
                        evaluation_results = cvi_results.get('evaluation')
                        
                        evaluation[ml_type][method]['horizons'][horizon]['r2'] += evaluation_results['r2_score'] / (sensors_len * cvi_len)
                        evaluation[ml_type][method]['times']['tt'] += modelling_results['modelling_time'] / (sensors_len * cvi_len * horizon_len)
                        evaluation[ml_type][method]['times']['tp'] += modelling_results['predicting_time'] / (sensors_len * cvi_len * horizon_len)

    # 2nd pass: standard deviations
    # =============================

    for sensor, sensor_results in sensors.items():
        if sensor_wl and sensor not in sensor_wl:
            continue
        
        # Types:
        ml_types = sensor_results.get('types')
        for ml_type, type_results in ml_types.items():

            # Horizons:
            horizons = type_results.get('horizons')
            for horizon, horizon_results in horizons.items():
                if horizon_wl and horizon not in horizon_wl:
                    continue

                # Methods:
                methods = horizon_results.get('methods')
                for method, method_results in methods.items():
                    if method_wl and method not in method_wl:
                        continue
                    
                    # CV:
                    cv = method_results.get('cv')
                    cvi_len = len(cv)
                    for cvi_results in cv.values():
                        evaluation_results = cvi_results.get('evaluation')                        
                        evaluation[ml_type][method]['horizons'][horizon]['r2_sd'] += (evaluation_results['r2_score'] - evaluation[ml_type][method]['horizons'][horizon]['r2']) ** 2

    n = sensors_len * cvi_len
    for ml_type in evaluation.keys():
        for method in evaluation[ml_type].keys():
            for horizon in evaluation[ml_type][method]['horizons'].keys():
                evaluation[ml_type][method]['horizons'][horizon]['r2_sd'] = math.sqrt(evaluation[ml_type][method]['horizons'][horizon]['r2_sd'] / (n - 1))
                        

    return evaluation                


def plot_methods_by_horizon_abs(path, results, water_type, sensor_wl=None, horizon_wl=None, method_wl=None):
    method_map = {
        'reg': {
            'LinearRegression': 'LinearRegression',
            'RandomForestRegressor': 'RandomForestR',
            'GradientBoostingRegressor': 'GradientBoostingR',
            'MLPRegressor': 'MLP-R',
            'KNeighborsRegressor': 'KNeighborsR',
            'RegressionHoeffdingTree': 'HoeffdingTreeR'
        },
        'cls': {
            'LogisticRegression': 'Logistic Regression',
            'RandomForestClassifier': 'RandomForestC',
            'KNeighborsClassifier': 'KNeighborsC',
            'Perceptron': 'Perceptron',
            'GaussianNB': 'GaussianNB',
            'HoeffdingTree': 'HoeffdingTreeC'
        }
    }

    resources_path = get_resources_path(path)
    
    # Sensors:
    sensors = results.get('sensors')

    for sensor, sensor_results in sensors.items():
        if sensor_wl and sensor not in sensor_wl:
            continue
        
        # Types:
        ml_types = sensor_results.get('types')
        for ml_type, type_results in ml_types.items():

            # Horizons:
            horizons = type_results.get('horizons')
            for horizon, horizon_results in horizons.items():
                if horizon_wl and horizon not in horizon_wl:
                    continue

                # Methods:
                methods = horizon_results.get('methods')
                for method, method_results in methods.items():
                    if method_wl and method not in method_wl:
                        continue
                    
                    # CV:
                    cv = method_results.get('cv')
                    cvi_len = len(cv)
                    for i, cvi_results in enumerate(cv.values()):
                        if i < cvi_len - 1:
                            continue

                        modelling_results = cvi_results.get('modelling')
                        y_pred = modelling_results['y_pred'] if ml_type == 'reg' else modelling_results['y_pred_reg']
                        y_test = modelling_results['y_test'] if ml_type == 'reg' else modelling_results['y_test_reg']
                        ya_test = modelling_results['ya_test']
                        # days_len = len(y_pred)
                        days_len = 200
                        days = list(range(days_len))
                        h = int(horizon)

                        # ya_pred_h = np.zeros(len(ya_test))
                        ya_pred_h = np.zeros(days_len)
                        for i, abs_val in enumerate(ya_test):
                            if i + h >= days_len:
                                break
                            ya_pred_h[i + h] = abs_val + sum(y_pred[i:i + h])

                        # Plot predicted values with prediction horizon and real values.
                        unit = 'm' if water_type == 'gw' else 'cm'

                        days_len_max = min(len(ya_pred_h) - h, days_len)
                        fig, ax = plt.subplots() 
                        fig.set_size_inches(4, 2)
                        ax.plot(days[:days_len_max], ya_test[h:days_len_max + h], label="True value")
                        ax.plot(days[:days_len_max], ya_pred_h[h:days_len_max + h], label="Prediction")
                        # ax.legend(loc=1, borderaxespad=1)
                        plt.xlabel('Time (days)')
                        plt.ylabel(f'Water level ({unit})')
                        plt.title(f'{method_map[ml_type][method]} (h={h})')
                        save_figure(plt, os.path.join(resources_path, f'abs_{water_type}_{ml_type}_h{horizon}_{method}.png'))
                        save_figure(plt, os.path.join(resources_path, f'abs_{water_type}_{ml_type}_h{horizon}_{method}.svg'))
                        plt.close()
                        # plt.show()


def plot_methods_by_horizon_rel(path, results, water_type, sensor_wl=None, horizon_wl=None, method_wl=None):
    method_map = {
        'reg': {
            'LinearRegression': 'LinearRegression',
            'RandomForestRegressor': 'RandomForestR',
            'GradientBoostingRegressor': 'GradientBoostingR',
            'MLPRegressor': 'MLP-R',
            'KNeighborsRegressor': 'KNeighborsR',
            'RegressionHoeffdingTree': 'HoeffdingTreeR'
        },
        'cls': {
            'LogisticRegression': 'Logistic Regression',
            'RandomForestClassifier': 'RandomForestC',
            'KNeighborsClassifier': 'KNeighborsC',
            'Perceptron': 'Perceptron',
            'GaussianNB': 'GaussianNB',
            'HoeffdingTree': 'HoeffdingTreeC'
        }
    }

    resources_path = get_resources_path(path)
    
    # Sensors:
    sensors = results.get('sensors')

    for sensor, sensor_results in sensors.items():
        if sensor_wl and sensor not in sensor_wl:
            continue
        
        # Types:
        ml_types = sensor_results.get('types')
        for ml_type, type_results in ml_types.items():

            # Horizons:
            horizons = type_results.get('horizons')
            for horizon, horizon_results in horizons.items():
                if horizon_wl and horizon not in horizon_wl:
                    continue

                # Methods:
                methods = horizon_results.get('methods')
                for method, method_results in methods.items():
                    if method_wl and method not in method_wl:
                        continue
                    
                    # CV:
                    cv = method_results.get('cv')
                    cvi_len = len(cv)
                    for i, cvi_results in enumerate(cv.values()):
                        if i < cvi_len - 1:
                            continue

                        modelling_results = cvi_results.get('modelling')
                        y_pred = modelling_results['y_pred'] if ml_type == 'reg' else modelling_results['y_pred_reg']
                        y_test = modelling_results['y_test'] if ml_type == 'reg' else modelling_results['y_test_reg']

                        days_len = len(y_pred)
                        days = list(range(days_len))
                        h = int(horizon)

                        # Plot predicted values with prediction horizon and real values.
                        unit = 'm' if water_type == 'gw' else 'cm'
                        fig, ax = plt.subplots() 
                        fig.set_size_inches(10, 5)
                        ax.plot(days[:days_len], y_test[:days_len], label="True value")
                        ax.plot(days[:days_len], y_pred[:days_len], label="Prediction")
                        # ax.legend(loc=1, borderaxespad=1)
                        plt.xlabel('Time (days)')
                        plt.ylabel(f'Water level change ({unit})')
                        plt.title(f'{method_map[ml_type][method]} (h={h})')
                        save_figure(plt, os.path.join(resources_path, f'rel_{water_type}_{ml_type}_h{horizon}_{method}.png'))
                        save_figure(plt, os.path.join(resources_path, f'rel_{water_type}_{ml_type}_h{horizon}_{method}.svg'))
                        plt.close()
                        # plt.show()


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Analyse experiment results at given location path.')
    parser.add_argument('path', type=str, help='Path to experiment results directory')

    return parser.parse_args()


def main():
    args = parse_cli_args()
    results = get_results(args.path)

    if not results:
        print(f'Failed to load results from: {args.path}')
        return

    # analyse(*results, args.path, [])

    # GW:
    sensors = ['85065', '85012', '85064', '85030']
    sensors = ['85065', '85012', '85064']
    sensors = ['85065', '85064', '85030']
    sensors = ['85065', '85064']

    # SW:
    sensors = ['2250', '2530', '2620', '3200', '3250', '3400', '4200', '4230', '4270', '4450', '4515', '4520', '4570', '4575', '4650', '4770', '5040', '5078', '5330', '5425', '5500', '6060', '6068', '6200', '6220', '6300', '6340', '6350', '6385', '6400', '8454', '8565']
    sensors = ['2530', '2620', '4200', '4230', '4270', '4515', '4520', '4570', '4575', '5040', '5078', '5330', '5425', '5500', '6060', '6068', '6200', '6220', '6300', '6340', '8454', '8565']

    # All:
    sensors = []

    # evaluation = evaluate(results[0], sensor_wl=sensors)
    # print(json.dumps(evaluation, indent=2))

    # sensor_evaluation = evaluate_sensors(results[0], sensor_wl=sensors, horizon_wl=['3'], method_wl=['RandomForestRegressor', 'RegressionHoeffdingTree'])
    # print(json.dumps(sensor_evaluation, indent=2))

    # Regression
    plot_methods_by_horizon_abs(args.path, results[0], 'gw', sensor_wl=['85065'], horizon_wl=['1', '3', '5'], method_wl=['LinearRegression', 'GradientBoostingRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'RegressionHoeffdingTree'])
    plot_methods_by_horizon_abs(args.path, results[0], 'sw', sensor_wl=['5078'], horizon_wl=['1', '3', '5'], method_wl=['LinearRegression', 'GradientBoostingRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'RegressionHoeffdingTree'])
    plot_methods_by_horizon_rel(args.path, results[0], 'gw', sensor_wl=['85065'], horizon_wl=['1', '3', '5'], method_wl=['LinearRegression', 'GradientBoostingRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'RegressionHoeffdingTree'])
    plot_methods_by_horizon_rel(args.path, results[0], 'sw', sensor_wl=['5078'], horizon_wl=['1', '3', '5'], method_wl=['LinearRegression', 'GradientBoostingRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'RegressionHoeffdingTree'])

    # Classification
    plot_methods_by_horizon_abs(args.path, results[0], 'gw', sensor_wl=['85065'], horizon_wl=['1', '3', '5'], method_wl=['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', 'Perceptron', 'GaussianNB', 'HoeffdingTree'])
    plot_methods_by_horizon_abs(args.path, results[0], 'sw', sensor_wl=['5078'], horizon_wl=['1', '3', '5'], method_wl=['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', 'Perceptron', 'GaussianNB', 'HoeffdingTree'])
    plot_methods_by_horizon_rel(args.path, results[0], 'gw', sensor_wl=['85065'], horizon_wl=['1', '3', '5'], method_wl=['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', 'Perceptron', 'GaussianNB', 'HoeffdingTree'])
    plot_methods_by_horizon_rel(args.path, results[0], 'sw', sensor_wl=['5078'], horizon_wl=['1', '3', '5'], method_wl=['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', 'Perceptron', 'GaussianNB', 'HoeffdingTree'])



if __name__ == '__main__':
    main()
