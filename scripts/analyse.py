# Common 
import argparse
import os
import json

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


def analyse(results, config, path):
    resources_path = get_resources_path(path)
    
    sensors = results.get('sensors')
    if not sensors:
        return

    evaluation = {}

    # 1. For each sensor:
    # ===================
    for sensor, sensor_results in sensors.items():
        # print(sensor)

        ml_types = sensor_results.get('types')
        # if not ml_types:
        #     continue

        # 2. For each type:
        # =================
        for ml_type, type_results in ml_types.items():
            # print(ml_type)

            if not evaluation.get(ml_type):
                evaluation[ml_type] = {}

            horizons = type_results.get('horizons')
            # if not horizons:
            #     continue

            # 3. For each horizon:
            # ====================
            for horizon, horizon_results in horizons.items():
                # print(horizon)

                if not evaluation[ml_type].get(horizon):
                    evaluation[ml_type][horizon] = {}

                feature_selection = horizon_results.get('feature_selection')
                if feature_selection:
                    evaluation[ml_type][horizon]['feature_selection'] = feature_selection

                methods = horizon_results.get('methods')
                # if not methods:
                #     continue

                # 4. For each method:
                # ===================
                methods_len = len(methods)
                for method, method_results in methods.items():
                    # print(method)

                    if not evaluation[ml_type][horizon].get(method):
                        evaluation[ml_type][horizon][method] = {
                            'evaluation': {}
                        }

                    if not evaluation[ml_type][horizon][method].get(sensor):
                        evaluation[ml_type][horizon][method][sensor] = {
                            'evaluation': {}
                        }

                    cv = method_results.get('cv')
                    # if not cv:
                    #     continue

                    # 5. For each cross-validation iteration:
                    # =======================================
                    cvi_len = len(cv)
                    for cvi, cvi_results in cv.items():
                        # print(cvi)

                        if not evaluation[ml_type][horizon][method][sensor].get(cvi):
                            evaluation[ml_type][horizon][method][sensor][cvi] = {}

                        modelling_results = cvi_results.get('modelling')
                        evaluation[ml_type][horizon][method][sensor][cvi]['modelling'] = {
                            'modelling_time': modelling_results['modelling_time'],
                            'predicting_time': modelling_results['predicting_time']
                        }

                        evaluation_results = cvi_results.get('evaluation')
                        evaluation[ml_type][horizon][method][sensor][cvi]['evaluation'] = evaluation_results

                        for ek, ev in evaluation_results.items():
                            if not evaluation[ml_type][horizon][method][sensor]['evaluation'].get(ek):
                                evaluation[ml_type][horizon][method][sensor]['evaluation'][ek] = 0.0

                            # Sensor evaluation.
                            evaluation[ml_type][horizon][method][sensor]['evaluation'][ek] += ev / cvi_len

                            if not evaluation[ml_type][horizon][method]['evaluation'].get(ek):
                                evaluation[ml_type][horizon][method]['evaluation'][ek] = 0.0

                            # Method evaluation.
                            evaluation[ml_type][horizon][method]['evaluation'][ek] += ev / cvi_len / methods_len

                        y_pred = modelling_results['y_pred']
                        y_test = modelling_results['y_test']
                        ya_test = modelling_results['ya_test']
                        days_len = len(y_pred)
                        days = list(range(days_len))

                        # Plot predicted value differences and real value differences.
                        fig, ax = plt.subplots()
                        fig.set_size_inches(10, 5)
                        ax.plot(days, y_test[:days_len], label="True value")
                        ax.plot(days, y_pred[:days_len], label="Prediction")
                        ax.legend(loc=1, borderaxespad=1)
                        plt.xlabel('Time (days)')
                        plt.ylabel('Water level change (cm)')
                        plt.title(method)
                        save_figure(plt, os.path.join(resources_path, f'{ml_type}_h{horizon}_{method}_cv{cvi}.png'))
                        plt.close()

                        # Plot predicted values and real values.
                        predicted_abs = [ya_test[0]]
                        for n, val in enumerate(y_pred[1:]):
                            predicted_abs.append(predicted_abs[n] + val)

                        fig, ax = plt.subplots() 
                        fig.set_size_inches(10, 5)
                        ax.plot(days, ya_test[:days_len], label="True value")
                        ax.plot(days, predicted_abs[:days_len], label="Prediction")
                        ax.legend(loc=1, borderaxespad=1)
                        plt.xlabel('Time (days)')
                        plt.ylabel('Water level (cm)')
                        plt.title(method)
                        save_figure(plt, os.path.join(resources_path, f'{ml_type}_h{horizon}_{method}_cv{cvi}_sum.png'))
                        plt.close()
                    


    print(json.dumps(evaluation, indent=2))




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

    
    
    analyse(*results, args.path)


if __name__ == '__main__':
    main()
